#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""LoRA training for MOSS-Image-Base (Diffusers + PEFT, flow matching).

Why a custom script: unlike Anima (kohya sd-scripts) and Z-Image /
FLUX.3 (AI Toolkit), MOSS-Image has no established third-party LoRA
trainer, so this implements the standard flow-matching LoRA recipe
directly against the model's Diffusers-style components:

- text encoder (Qwen3.5-9B) and VAE (FLUX.2 VAE) stay FROZEN
  (freezing the text encoder is also the official USERGUIDE guidance)
- LoRA is injected into the DiT (transformer) only, via PEFT
- loss: MSE between predicted and target velocity, x_t = (1-t)*x0 + t*eps
- batch size 1 so the variable bucket sizes from preprocess_images.py
  work as-is; gradient accumulation provides the effective batch

STATUS: written against the documented architecture, NOT yet run on
the training machine. Verify these points there before trusting a run
(they are the model-specific seams):
  1. The pipeline class / component attribute names (transformer, vae,
     text_encoder, tokenizer) exposed by the MOSS-Image release.
  2. How the DiT expects latents to be packed (patchify 2x2 here) and
     whether it needs latent image ids / position embeddings.
  3. encode_prompt availability and its return signature.
  4. The VAE scaling/shift factors (read from vae.config when present).
  5. The timestep convention the DiT was trained with (t in [0,1] here).

Requires: torch, diffusers, transformers, peft, accelerate, Pillow.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class PairDataset(Dataset):
    """Image + same-stem .txt caption pairs from a training-data folder."""

    def __init__(self, root: Path):
        self.items = []
        for image in sorted(root.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                caption = image.with_suffix(".txt")
                if caption.is_file():
                    self.items.append((image, caption))
        if not self.items:
            raise SystemExit(f"error: no image/caption pairs under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, caption_path = self.items[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        caption = caption_path.read_text(encoding="utf-8").strip()
        return image, caption


def image_to_tensor(image: Image.Image, device, dtype) -> torch.Tensor:
    import numpy as np

    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device, dtype)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(b, c, h, w) -> (b, (h/2)*(w/2), c*4): 2x2 patchify for the DiT."""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-Image LoRA training (flow matching).")
    parser.add_argument("--model-id", default="OpenMOSS/MOSS-Image-Base",
                        help="HF repo id or local path of MOSS-Image-Base "
                             "(verify the exact id on the training machine)")
    parser.add_argument("--dataset", default="training-data/moss")
    parser.add_argument("--output-dir", default="outputs/moss")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from diffusers import DiffusionPipeline
    from peft import LoraConfig

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    transformer, vae = pipe.transformer, pipe.vae
    # Freeze everything; only the LoRA weights on the DiT will train.
    pipe.requires_grad_(False)
    vae.eval()
    transformer.train()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights="gaussian",
        target_modules="all-linear",
    )
    transformer.add_adapter(lora_config)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    dataset = PairDataset(Path(args.dataset))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    optimizer.zero_grad()
    while step < args.steps:
        for image, caption in dataset:
            with torch.no_grad():
                pixels = image_to_tensor(image, device, dtype)
                latents = vae.encode(pixels).latent_dist.sample()
                latents = (latents - shift) * scaling
                packed = pack_latents(latents)
                if hasattr(pipe, "encode_prompt"):
                    prompt_embeds = pipe.encode_prompt(
                        prompt=caption, device=device
                    )[0]
                else:  # fallback: tokenizer + text encoder of the pipeline
                    tokens = pipe.tokenizer(
                        caption, return_tensors="pt", padding=True,
                        truncation=True,
                    ).to(device)
                    prompt_embeds = pipe.text_encoder(**tokens).last_hidden_state

            noise = torch.randn_like(packed)
            t = torch.rand((), device=device)
            x_t = (1 - t) * packed + t * noise
            target = noise - packed
            pred = transformer(
                hidden_states=x_t,
                timestep=t.expand(1),
                encoder_hidden_states=prompt_embeds.to(dtype),
            ).sample
            loss = F.mse_loss(pred.float(), target.float()) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0 or step == args.steps - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {loss.item() * args.grad_accum:.4f}")
            if step % args.save_every == 0 or step == args.steps:
                save_path = output_dir / f"okakikai-moss-step{step}"
                # PEFT adapter format; convert to a plain LoRA safetensors
                # for distribution once the loading path is confirmed.
                transformer.save_pretrained(save_path, safe_serialization=True)
                print(f"saved LoRA at step {step} -> {save_path}")
            if step >= args.steps:
                break
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

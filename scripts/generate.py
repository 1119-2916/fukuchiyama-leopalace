#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate images with a trained LoRA (per-model presets, Diffusers).

Two modes:
- normal:  --prompt "..." (repeatable via --num with incrementing seeds)
- eval:    --eval generates the fixed prompt set in
           prompts/eval_prompts.txt with fixed seeds, so checkpoints
           and models can be compared apples-to-apples.

Output mirrors the other stages: generated/<run datetime>/<model>/
plus generation_manifest.json (base model, LoRA path + SHA-256,
prompt, seed, steps, guidance, size per image).

Runs on the training machine (GPU + weights). --dry-run prints the
resolved plan without importing torch/diffusers, so the wiring can be
checked anywhere.

Model-specific seams to verify on the machine (same policy as the
MOSS trainer): the pipeline class each release exposes through
DiffusionPipeline, the exact FLUX.3 Distilled / MOSS SFT repo ids,
and whether the MOSS LoRA (saved as a PEFT adapter by
train_moss_lora.py) loads via load_lora_weights or needs conversion
to a plain safetensors first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Per-model generation presets. steps/guidance are starting points from
# docs/base-models.md (FLUX.3 split is BFL's official Base/Distilled
# role split). "base" may be overridden with --base-model or, for
# anima, the ANIMA_BASE_PATH environment variable.
MODEL_PRESETS = {
    "anima": {
        "base": None,  # set via --base-model or ANIMA_BASE_PATH
        "steps": 30, "guidance": 4.5, "prompt_style": "tags",
    },
    "moss": {
        "base": "OpenMOSS/MOSS-Image-Base",
        "sft_base": "OpenMOSS/MOSS-Image",  # generation-check variant (verify id)
        "steps": 30, "guidance": 4.0, "prompt_style": "natural",
    },
    "z-image": {
        "base": "Tongyi-MAI/Z-Image",  # Base (undistilled); never Turbo
        "steps": 28, "guidance": 4.0, "prompt_style": "natural",
    },
    "flux-klein": {
        "base": "black-forest-labs/flux-3-image-klein-base",
        "distilled_base": "black-forest-labs/flux-3-image-klein-distilled",  # verify id
        "steps": 50, "guidance": 3.5, "prompt_style": "natural",
        "distilled_steps": 16, "distilled_guidance": 1.0,
    },
}

EVAL_SEED_BASE = 1000


def snap16(value: int) -> int:
    """Snap a dimension to the 16px grid shared by all four bases."""
    return max(16, int(round(value / 16)) * 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_eval_prompts(text: str, style: str) -> list[str]:
    """Extract prompts from the [tags] / [natural] section of the file."""
    prompts, active = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            active = stripped == f"[{style}]"
            continue
        if active and stripped and not stripped.startswith("#"):
            prompts.append(stripped)
    return prompts


def resolve_base(args) -> tuple[str | None, str]:
    """Return (base_model, variant_label) after variant flags."""
    preset = MODEL_PRESETS[args.target_model]
    variant = "base"
    base = preset.get("base")
    if args.target_model == "anima":
        base = base or os.environ.get("ANIMA_BASE_PATH")
    if getattr(args, "distilled", False) and "distilled_base" in preset:
        base, variant = preset["distilled_base"], "distilled"
    if getattr(args, "sft", False) and "sft_base" in preset:
        base, variant = preset["sft_base"], "sft"
    if args.base_model:
        base, variant = args.base_model, "custom"
    return base, variant


def resolve_plan(args) -> dict:
    """Resolve everything a run needs, without touching the GPU."""
    preset = MODEL_PRESETS[args.target_model]
    base, variant = resolve_base(args)
    steps = args.steps
    guidance = args.guidance
    if variant == "distilled":
        steps = steps if steps is not None else preset["distilled_steps"]
        guidance = guidance if guidance is not None else preset["distilled_guidance"]
    steps = steps if steps is not None else preset["steps"]
    guidance = guidance if guidance is not None else preset["guidance"]

    jobs: list[dict] = []
    if args.eval:
        eval_file = Path(args.eval_file)
        prompts = parse_eval_prompts(
            eval_file.read_text(encoding="utf-8"), preset["prompt_style"]
        )
        for i, prompt in enumerate(prompts):
            jobs.append({"prompt": prompt, "seed": EVAL_SEED_BASE + i})
    elif args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(s for s in (ln.strip() for ln in lines) if s):
            jobs.append({"prompt": line, "seed": args.seed + i})
    else:
        for i in range(args.num):
            jobs.append({"prompt": args.prompt, "seed": args.seed + i})

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("generated")
    return {
        "target_model": args.target_model,
        "variant": variant,
        "base_model": base,
        "lora": args.lora,
        "lora_sha256": sha256_file(Path(args.lora)) if args.lora and Path(args.lora).is_file() else None,
        "lora_scale": args.lora_scale,
        "steps": steps,
        "guidance": guidance,
        "width": snap16(args.width),
        "height": snap16(args.height),
        "negative_prompt": args.negative,
        "run_id": run_id,
        "output_dir": str(root / run_id / args.target_model),
        "jobs": jobs,
    }


def run_generation(plan: dict) -> list[dict]:
    """Execute a resolved plan. Imports torch/diffusers lazily."""
    import torch
    from diffusers import DiffusionPipeline

    if not plan["base_model"]:
        raise SystemExit(
            "error: no base model resolved; pass --base-model "
            "(for anima, ANIMA_BASE_PATH also works)"
        )
    pipe = DiffusionPipeline.from_pretrained(
        plan["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    if plan["lora"]:
        pipe.load_lora_weights(plan["lora"], adapter_name="okakikai")
        pipe.set_adapters(["okakikai"], adapter_weights=[plan["lora_scale"]])

    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, job in enumerate(plan["jobs"]):
        generator = torch.Generator("cuda").manual_seed(job["seed"])
        kwargs = dict(
            prompt=job["prompt"],
            height=plan["height"],
            width=plan["width"],
            num_inference_steps=plan["steps"],
            guidance_scale=plan["guidance"],
            generator=generator,
        )
        if plan["negative_prompt"]:
            kwargs["negative_prompt"] = plan["negative_prompt"]
        image = pipe(**kwargs).images[0]
        name = f"{i:03d}_seed{job['seed']}.png"
        image.save(output_dir / name)
        records.append({"file": name, **job})
        print(f"[{i + 1}/{len(plan['jobs'])}] {name}")
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with a trained LoRA.")
    parser.add_argument("--target-model", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--lora", default=None,
                        help="LoRA weights file (or PEFT adapter dir); omit for base-only")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--base-model", default=None,
                        help="Override the base checkpoint (HF id or local path)")
    parser.add_argument("--distilled", action="store_true",
                        help="flux-klein only: load the LoRA on the Distilled "
                             "checkpoint (fast generation, 16 steps / CFG 1)")
    parser.add_argument("--sft", action="store_true",
                        help="moss only: generate with the SFT checkpoint "
                             "(official generation-check variant)")
    parser.add_argument("--prompt", default=None, help="Single prompt (normal mode)")
    parser.add_argument("--prompts-file", default=None,
                        help="Text file with one prompt per line")
    parser.add_argument("--eval", action="store_true",
                        help="Use the fixed evaluation prompt set")
    parser.add_argument("--eval-file", default="prompts/eval_prompts.txt")
    parser.add_argument("--num", type=int, default=1,
                        help="Images for --prompt mode (seeds increment)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Root for outputs (default: generated)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved plan without generating")
    args = parser.parse_args(argv)
    if not args.eval and not args.prompts_file and not args.prompt:
        parser.error("one of --prompt, --prompts-file or --eval is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = resolve_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    records = run_generation(plan)
    manifest = {
        **{k: v for k, v in plan.items() if k != "jobs"},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "images": records,
    }
    output_dir = Path(plan["output_dir"])
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"generated {len(records)} image(s) -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

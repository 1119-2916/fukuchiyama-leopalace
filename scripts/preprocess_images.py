#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Per-base-model image preprocessing for LoRA training.

Pipeline position (docs/normalization.md, docs/preprocessing.md):

    dataset/  --normalize_images.py-->  dataset-normalized/<run>/<model>/
              --preprocess_images.py--> dataset-preprocessed/<run>/<model>/
              --assemble_training.py--> training-data/<model>/

Normalization is model-independent hygiene (EXIF, RGB, alpha, size cap).
THIS step is the model-dependent geometry: for the chosen base model it
assigns every image an aspect-ratio bucket at the model's training
area, resizes to cover the bucket and center-crops -- exactly what
kohya / AI Toolkit would do internally at load time, but done
externally so that:

- pipelines without bucketing (plain Diffusers-style loaders, the
  MOSS-Image stack, which *errors* on non-multiple-of-16 sizes) get
  valid inputs, and
- the exact geometry of a training run is fixed, inspectable and
  recorded in preprocess_manifest.json (bucket, scale, crop box).

Rules kept from the research: never upscale (an image smaller than the
target area keeps its own size, only snapped down to the model's step
grid), dimensions always land on the model's step grid, and sources
are never modified.

Requires Pillow (``pip install Pillow``); everything else is stdlib.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment issue, clear message
    raise SystemExit("error: Pillow is required (pip install Pillow)")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# Per-model geometry presets. Sources: docs/normalization.md.
# - target_area: training area in pixels (~1024^2 for all four bases)
# - step: bucket grid. All bases need multiples of 16 (VAE 8x * patch 2);
#   Anima via AI Toolkit snaps to 32, kohya examples vary 16-64.
# - min_side / max_side: bucket side clamps from each model's range.
MODEL_PRESETS = {
    "anima": {"target_area": 1024 * 1024, "step": 32, "min_side": 256, "max_side": 2048},
    "moss": {"target_area": 1024 * 1024, "step": 16, "min_side": 256, "max_side": 2048},
    "z-image": {"target_area": 1024 * 1024, "step": 16, "min_side": 256, "max_side": 2048},
    "flux-klein": {"target_area": 1024 * 1024, "step": 16, "min_side": 256, "max_side": 2048},
}


def _snap(value: float, step: int, mode: str) -> int:
    if mode == "round":
        return max(step, int(round(value / step)) * step)
    return max(step, int(value // step) * step)  # floor


def compute_bucket(width: int, height: int, preset: dict) -> tuple[int, int]:
    """Bucket size for an image under a model preset.

    Images at or above the target area are scaled (down) to that area,
    keeping aspect ratio, with each side snapped to the step grid and
    clamped to [min_side, max_side]. Smaller images are never upscaled:
    their own size is snapped down to the grid instead.
    """
    step, area = preset["step"], preset["target_area"]
    if width * height >= area:
        scale = math.sqrt(area / (width * height))
        bucket_w = _snap(width * scale, step, "round")
        bucket_h = _snap(height * scale, step, "round")
        bucket_w = min(max(bucket_w, preset["min_side"]), preset["max_side"])
        bucket_h = min(max(bucket_h, preset["min_side"]), preset["max_side"])
        return bucket_w, bucket_h
    return _snap(width, step, "floor"), _snap(height, step, "floor")


def preprocess_one(src: Path, dst: Path, preset: dict) -> dict:
    """Resize-to-cover + center-crop one image to its bucket."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        bucket_w, bucket_h = compute_bucket(orig_w, orig_h, preset)
        record = {
            "source": src.name,
            "original_size": [orig_w, orig_h],
            "bucket": [bucket_w, bucket_h],
        }
        if (orig_w, orig_h) == (bucket_w, bucket_h):
            record["scale"] = 1.0
            record["crop_box"] = [0, 0, orig_w, orig_h]
            out = img
        else:
            cover = max(bucket_w / orig_w, bucket_h / orig_h)
            resized = (
                max(1, round(orig_w * cover)),
                max(1, round(orig_h * cover)),
            )
            out = img.resize(resized, Image.LANCZOS)
            left = (resized[0] - bucket_w) // 2
            top = (resized[1] - bucket_h) // 2
            record["scale"] = round(cover, 4)
            record["crop_box"] = [left, top, left + bucket_w, top + bucket_h]
            out = out.crop(tuple(record["crop_box"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, format="PNG")
    return record


def make_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-model image preprocessing: bucket, resize, center-crop."
    )
    parser.add_argument("input_dir",
                        help="Image directory, normally a normalize_images.py output "
                             "(dataset-normalized/<run>/<model>) or dataset itself")
    parser.add_argument("--target-model", choices=sorted(MODEL_PRESETS), required=True,
                        help="Base model to preprocess for (selects the preset)")
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", default=None,
                        help="Root for preprocessed images; files go to "
                             "<root>/<run datetime>/<target model>/ "
                             "(default root: dataset-preprocessed)")
    parser.add_argument("--run-id", default=None,
                        help="Run directory name (default: current local time "
                             "YYYYMMDD-HHMMSS)")
    parser.add_argument("--target-area", type=int, default=None,
                        help="Override the preset's target area in pixels")
    parser.add_argument("--step", type=int, default=None,
                        help="Override the preset's bucket grid step")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2
    preset = dict(MODEL_PRESETS[args.target_model])
    if args.target_area:
        preset["target_area"] = args.target_area
    if args.step:
        preset["step"] = args.step
    run_id = args.run_id or make_run_id()
    root = Path(args.output_dir) if args.output_dir else Path("dataset-preprocessed")
    output_dir = root / run_id / args.target_model

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    records, failed = [], []
    for src in images:
        relative = src.relative_to(input_dir).with_suffix(".png")
        try:
            record = preprocess_one(src, output_dir / relative, preset)
            record["source"] = str(src.relative_to(input_dir))
            record["output"] = str(relative)
            records.append(record)
        except Exception as exc:  # keep the batch going; report at the end
            failed.append({"source": str(src.relative_to(input_dir)), "error": str(exc)})

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_model": args.target_model,
        "preset": preset,
        "input_dir": str(input_dir),
        "preprocessed": len(records),
        "failed": failed,
        "images": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"preprocessed: {len(records)}, failed: {len(failed)}, output: {output_dir}")
    for failure in failed:
        print(f"FAILED: {failure['source']}: {failure['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

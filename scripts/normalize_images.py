#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize dataset images for LoRA training (hygiene only).

Research summary (docs/normalization.md): the trainers we use (kohya
sd-scripts, AI Toolkit) already do aspect-ratio bucketing, resizing and
cropping internally at load time, and they expect native aspect ratios.
Forcing squares or pre-cropping to buckets externally would only throw
away image content and fight the trainer. What trainers do NOT reliably
do -- and what therefore belongs in this external step -- is hygiene:

- bake EXIF orientation into the pixels (otherwise sideways images)
- convert to sRGB / RGB
- composite transparency onto a background (stock VAEs are 3-channel;
  a bare convert("RGB") drops alpha unpredictably)
- cap very large originals with a downscale (never upscale: trainers
  are run with bucket_no_upscale / "images are never upscaled")

Output layout mirrors the caption runs:
``dataset-normalized/<run datetime>/<target model>/<image>.png``
with a ``normalize_manifest.json`` recording, per image, the original
size/mode and every operation applied. Sources are never modified.

Requires Pillow (``pip install Pillow``); everything else is stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - environment issue, clear message
    raise SystemExit("error: Pillow is required (pip install Pillow)")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# Per-model caps. All four bases share VAE 8x + 2x2 patch = 16px stride
# and a ~1024^2 training area handled by the trainer's bucketing; the
# external cap only trims originals far above any training bucket.
# min_area_warn: below the model's documented generation/training area
# floor, training quality is doubtful -- warn, do not upscale.
TARGET_PRESETS = {
    "anima": {"max_side": 2048, "min_area_warn": 512 * 512},
    "moss": {"max_side": 2048, "min_area_warn": 512 * 512},
    "z-image": {"max_side": 2048, "min_area_warn": 512 * 512},
    "flux-klein": {"max_side": 2048, "min_area_warn": 512 * 512},
}
DEFAULT_PRESET = {"max_side": 2048, "min_area_warn": 512 * 512}


def make_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def normalize_one(
    src: Path,
    dst: Path,
    max_side: int,
    background: tuple[int, int, int],
    min_area_warn: int,
) -> dict:
    """Normalize one image; returns its manifest record."""
    record: dict = {"source": src.name, "operations": [], "warnings": []}
    with Image.open(src) as img:
        record["original_size"] = list(img.size)
        record["original_mode"] = img.mode
        # 1. Bake EXIF orientation into the pixels.
        transposed = ImageOps.exif_transpose(img)
        if transposed.size != img.size:
            record["operations"].append("exif_orientation_baked")
        img = transposed
        # 2. Composite transparency onto the background, then RGB.
        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )
        if has_alpha:
            rgba = img.convert("RGBA")
            base = Image.new("RGBA", rgba.size, background + (255,))
            img = Image.alpha_composite(base, rgba).convert("RGB")
            record["operations"].append("alpha_composited")
        elif img.mode != "RGB":
            img = img.convert("RGB")
            record["operations"].append("converted_to_rgb")
        # 3. Downscale only, never upscale.
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            record["operations"].append(f"downscaled_{w}x{h}_to_{new_size[0]}x{new_size[1]}")
        record["final_size"] = list(img.size)
        if img.size[0] * img.size[1] < min_area_warn:
            record["warnings"].append("below_min_area_for_target")
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="PNG")
    return record


def normalize_batch(
    input_dir: Path,
    output_dir: Path,
    max_side: int,
    background: tuple[int, int, int],
    min_area_warn: int,
) -> dict:
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    records, failed = [], []
    for src in images:
        relative = src.relative_to(input_dir).with_suffix(".png")
        try:
            record = normalize_one(
                src, output_dir / relative, max_side, background, min_area_warn
            )
            record["source"] = str(src.relative_to(input_dir))
            record["output"] = str(relative)
            records.append(record)
        except Exception as exc:  # keep the batch going; report at the end
            failed.append({"source": str(src.relative_to(input_dir)), "error": str(exc)})
    return {"records": records, "failed": failed}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize dataset images (EXIF/RGB/alpha hygiene + size cap)."
    )
    parser.add_argument("input_dir", nargs="?", default="dataset",
                        help="Directory containing the original images (default: dataset)")
    parser.add_argument("--target-model", choices=sorted(TARGET_PRESETS), default=None,
                        help="Base model to normalize for; sets the size cap and "
                             "names the output subdirectory")
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", default=None,
                        help="Root for normalized images; files go to "
                             "<root>/<run datetime>/<target model>/ "
                             "(default root: dataset-normalized)")
    parser.add_argument("--run-id", default=None,
                        help="Run directory name (default: current local time "
                             "YYYYMMDD-HHMMSS)")
    parser.add_argument("--max-side", type=int, default=None,
                        help="Cap for the longest side in px (downscale only; "
                             "default from the target preset: 2048)")
    parser.add_argument("--background", choices=("white", "black"), default="white",
                        help="Background color for compositing transparency "
                             "(default: white)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2
    preset = TARGET_PRESETS.get(args.target_model or "", DEFAULT_PRESET)
    max_side = args.max_side or preset["max_side"]
    background = (255, 255, 255) if args.background == "white" else (0, 0, 0)
    run_id = args.run_id or make_run_id()
    root = Path(args.output_dir) if args.output_dir else Path("dataset-normalized")
    output_dir = root / run_id / (args.target_model or "generic")

    result = normalize_batch(
        input_dir, output_dir, max_side, background, preset["min_area_warn"]
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_model": args.target_model,
        "input_dir": str(input_dir),
        "max_side": max_side,
        "background": args.background,
        "normalized": len(result["records"]),
        "failed": result["failed"],
        "images": result["records"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "normalize_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"normalized: {len(result['records'])}, failed: {len(result['failed'])}, "
          f"output: {output_dir}")
    for failure in result["failed"]:
        print(f"FAILED: {failure['source']}: {failure['error']}", file=sys.stderr)
    warned = [r for r in result["records"] if r["warnings"]]
    for record in warned:
        print(f"warning: {record['source']}: {', '.join(record['warnings'])}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

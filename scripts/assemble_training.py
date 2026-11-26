#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble a trainer-ready dataset from a chosen caption run.

Caption generation keeps every variant under
``dataset-captions/<run datetime>/<model>/``. Training, however, needs
image files and caption files side by side. This script copies the
images (from the dataset folder) and the captions of ONE selected run
into a separate training folder (default ``training-data/<model>/``),
and writes a provenance manifest saying exactly which run it came from.

Nothing is moved or modified in the source folders.

Usage:
    python3 scripts/assemble_training.py \
        dataset-captions/20261126-170840/moss
    # -> training-data/moss/{001.png, 001.txt, ..., training_manifest.json}
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def collect_by_stem(root: Path, extensions: set[str]) -> dict[Path, Path]:
    """Map relative path without suffix -> file, for files under root."""
    found: dict[Path, Path] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            found[path.relative_to(root).with_suffix("")] = path
    return found


def assemble(
    captions_dir: Path,
    images_dir: Path,
    dest: Path,
    overwrite: bool = False,
) -> dict:
    """Copy image+caption pairs into dest. Returns a summary dict."""
    captions = collect_by_stem(captions_dir, {".txt"})
    # The run manifest is metadata, not a caption.
    captions = {k: v for k, v in captions.items() if v.name != "caption_run.json"}
    images = collect_by_stem(images_dir, IMAGE_EXTENSIONS)
    if not captions:
        raise SystemExit(f"error: no caption .txt files under {captions_dir}")
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise SystemExit(
            f"error: destination {dest} is not empty; pass --overwrite to replace it"
        )
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    pairs, missing_images = [], []
    for stem, caption_path in sorted(captions.items()):
        image_path = images.get(stem)
        if image_path is None:
            missing_images.append(str(stem))
            continue
        target_image = dest / image_path.relative_to(images_dir)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        shutil.copy2(caption_path, dest / caption_path.relative_to(captions_dir))
        pairs.append(str(stem))

    manifest = {
        "assembled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "captions_source": str(captions_dir),
        "images_source": str(images_dir),
        "model_slot": captions_dir.name,
        "pairs": len(pairs),
        "captions_without_image": missing_images,
        "images_without_caption": sorted(
            str(s) for s in images.keys() - captions.keys()
        ),
    }
    # Carry over the generation run's manifest, if present, so the full
    # provenance chain (which LLM/settings made these captions) survives.
    run_manifest = captions_dir / "caption_run.json"
    if run_manifest.is_file():
        try:
            manifest["caption_run"] = json.loads(
                run_manifest.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            manifest["caption_run"] = None
    (dest / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a trainer-ready dataset from a chosen caption run."
    )
    parser.add_argument("captions_dir",
                        help="A caption run's model directory, e.g. "
                             "dataset-captions/20261126-170840/moss")
    parser.add_argument("--images", default="dataset",
                        help="Image folder to pair with the captions (default: dataset)")
    parser.add_argument("--dest", default=None,
                        help="Training folder to build "
                             "(default: training-data/<model directory name>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the destination if it already has content")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    captions_dir = Path(args.captions_dir)
    if not captions_dir.is_dir():
        print(f"error: captions directory not found: {captions_dir}", file=sys.stderr)
        return 2
    dest = Path(args.dest) if args.dest else Path("training-data") / captions_dir.name
    manifest = assemble(captions_dir, Path(args.images), dest, args.overwrite)
    print(f"assembled {manifest['pairs']} pairs -> {dest}")
    if manifest["captions_without_image"]:
        print("captions without image:", ", ".join(manifest["captions_without_image"]))
    if manifest["images_without_caption"]:
        print("images without caption:", ", ".join(manifest["images_without_caption"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

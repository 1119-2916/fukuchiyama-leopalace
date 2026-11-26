#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Preflight check: is this machine actually ready to caption / train?

Run after cloning, before anything else:

    python3 scripts/preflight.py                # training readiness, all models
    python3 scripts/preflight.py --model anima  # one model
    python3 scripts/preflight.py --with-captioning

Every check prints OK / MISSING / WARN with the exact fix. Exit code is
1 when anything required is MISSING, so it can gate scripts. It changes
nothing on the machine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MODELS = ("anima", "moss", "z-image", "flux-klein")

OK, MISSING, WARN = "OK", "MISSING", "WARN"


def count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_pairs(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.with_suffix(".txt").is_file()
    )


def check_common(repo: Path) -> list[tuple[str, str, str]]:
    checks = []
    version = sys.version_info
    checks.append((
        OK if version >= (3, 10) else MISSING, "python",
        f"{version.major}.{version.minor} (need >= 3.10)",
    ))
    try:
        import PIL  # noqa: F401

        checks.append((OK, "Pillow", "installed (needed by normalize/preprocess)"))
    except ImportError:
        checks.append((MISSING, "Pillow", "pip install -r requirements.txt"))
    images = count_images(repo / "dataset")
    checks.append((
        OK if images >= 1 else MISSING, "dataset images",
        f"{images} image(s) in dataset/ (place 30-50 illustrations there)"
        if images else "dataset/ has no images; place the illustrations there",
    ))
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            out = subprocess.run(
                [nvidia, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            checks.append((OK, "GPU", out or "nvidia-smi responded"))
        except Exception as exc:
            checks.append((WARN, "GPU", f"nvidia-smi failed: {exc}"))
    else:
        checks.append((MISSING, "GPU", "nvidia-smi not found; training needs the RTX 6000 machine"))
    return checks


def check_captioning(repo: Path) -> list[tuple[str, str, str]]:
    checks = []
    env_file = repo / ".env"
    if env_file.is_file():
        checks.append((OK, ".env", "present"))
        text = env_file.read_text(encoding="utf-8")
        model_set = any(
            line.startswith("CAPTION_MODEL=") and line.split("=", 1)[1].strip()
            for line in text.splitlines()
        )
        checks.append((
            OK if model_set else MISSING, "CAPTION_MODEL",
            "set" if model_set else ".env exists but CAPTION_MODEL is empty; "
            "set it to the LiteLLM model alias for Qwen 3.8 27B FP8",
        ))
    else:
        checks.append((MISSING, ".env", "cp .env.example .env and fill in CAPTION_MODEL"))
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:4000/v1/models", timeout=3) as resp:
            checks.append((OK, "LiteLLM", f"reachable (HTTP {resp.status})"))
    except Exception as exc:
        checks.append((MISSING, "LiteLLM",
                       f"not reachable at localhost:4000 ({exc}); start the "
                       "vLLM + LiteLLM stack on this machine first"))
    return checks


def check_model(repo: Path, model: str) -> list[tuple[str, str, str]]:
    checks = []
    pairs = count_pairs(repo / "training-data" / model)
    checks.append((
        OK if pairs >= 1 else MISSING, f"training-data/{model}",
        f"{pairs} image/caption pair(s)"
        if pairs else "empty; run caption_batch (--target-model) then "
        "assemble_training.sh for this model first",
    ))
    env = os.environ
    if model == "anima":
        sd = Path(env.get("SD_SCRIPTS_DIR", "../sd-scripts"))
        checks.append((
            OK if (sd / "anima_train_network.py").is_file() else MISSING,
            "sd-scripts",
            str(sd) if (sd / "anima_train_network.py").is_file()
            else "clone kohya-ss/sd-scripts next to the repo or set SD_SCRIPTS_DIR",
        ))
        checks.append((
            OK if env.get("ANIMA_BASE_PATH") else MISSING, "ANIMA_BASE_PATH",
            "set" if env.get("ANIMA_BASE_PATH")
            else "set ANIMA_BASE_PATH to the Anima Base checkpoint",
        ))
    elif model in ("z-image", "flux-klein"):
        ait = Path(env.get("AI_TOOLKIT_DIR", "../ai-toolkit"))
        checks.append((
            OK if (ait / "run.py").is_file() else MISSING, "ai-toolkit",
            str(ait) if (ait / "run.py").is_file()
            else "clone ostris/ai-toolkit next to the repo or set AI_TOOLKIT_DIR",
        ))
        if model == "flux-klein":
            checks.append((
                OK if env.get("HF_TOKEN") else MISSING, "HF_TOKEN",
                "set" if env.get("HF_TOKEN")
                else "FLUX.3 is gated: accept the license on Hugging Face "
                "and set HF_TOKEN",
            ))
    elif model == "moss":
        try:
            import torch  # noqa: F401

            checks.append((OK, "torch", "installed"))
        except ImportError:
            checks.append((MISSING, "torch",
                           "pip install -r requirements-train.txt "
                           "(Blackwell needs a recent CUDA build)"))
        checks.append((WARN, "moss trainer",
                       "train_moss_lora.py is unverified on hardware; "
                       "check the 5 points in its header first"))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check captioning/training readiness.")
    parser.add_argument("--model", choices=MODELS, default=None,
                        help="Check only this model (default: all)")
    parser.add_argument("--with-captioning", action="store_true",
                        help="Also check the local-LLM captioning setup")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent.parent

    sections: list[tuple[str, list]] = [("common", check_common(repo))]
    if args.with_captioning:
        sections.append(("captioning", check_captioning(repo)))
    for model in ([args.model] if args.model else MODELS):
        sections.append((model, check_model(repo, model)))

    missing = 0
    for title, checks in sections:
        print(f"[{title}]")
        for status, name, detail in checks:
            print(f"  {status:7s} {name}: {detail}")
            if status == MISSING:
                missing += 1
    print("READY" if missing == 0 else f"NOT READY: {missing} missing item(s)")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

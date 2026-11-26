#!/usr/bin/env python3
"""Batch captioning for LoRA training images using a vision-capable LLM.

Design notes
------------
* Standard library only -- no pip installs required.
* Providers:
    - ``openai-compatible`` (default): any endpoint that speaks the OpenAI
      Chat Completions API with image inputs. This covers OpenAI itself,
      a local vLLM / Ollama / LM Studio server, and most hosted gateways.
    - ``anthropic``: the Anthropic Messages API.
    - ``mock``: no network call. Returns a deterministic placeholder so the
      batch plumbing (discovery, skipping, writing, resume) can be tested
      without an API key.
* One caption per image, written next to the image with the same stem:
  ``001.png`` -> ``001.txt`` (the layout LoRA trainers expect).
* Existing captions are skipped unless ``--overwrite`` is given, so an
  interrupted run can simply be re-run to resume.
* The trigger word is prepended deterministically by this script after
  generation, instead of trusting the model to include it.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

STYLE_INSTRUCTIONS = {
    "tags": (
        "Write the caption as comma-separated descriptive tags, "
        "for example: 1girl, long hair, red ribbon, sitting, holding a cup, indoors. "
        "Order tags from the main subject to details, then the background. "
        "Do not write full sentences."
    ),
    "natural": (
        "Write the caption as one natural descriptive sentence about what is "
        "depicted. Do not use tag lists."
    ),
    "detailed": (
        "Write the caption as 2-3 natural descriptive sentences covering the "
        "subject, appearance, action, objects, and background in detail."
    ),
}

DEFAULT_BASE_URLS = {
    "openai-compatible": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

DEFAULT_MODELS = {
    "openai-compatible": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}

DEFAULT_API_KEY_ENVS = {
    "openai-compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# --------------------------------------------------------------------------
# Discovery / caption post-processing
# --------------------------------------------------------------------------

def find_images(input_dir: Path, recursive: bool = True) -> list[Path]:
    """Return image files under ``input_dir``, sorted for stable ordering."""
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def caption_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def clean_caption(raw: str) -> str:
    """Normalize a model response into a single-line caption."""
    text = raw.strip()
    # Strip a markdown code fence if the model wrapped the caption in one.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = " ".join(lines)
    # Captions are single-line: collapse all whitespace runs.
    text = " ".join(text.split())
    # Drop surrounding quotes the model sometimes adds.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def apply_trigger_word(caption: str, trigger_word: str | None) -> str:
    """Prepend the trigger word unless it is already the first token."""
    if not trigger_word:
        return caption
    trigger = trigger_word.strip()
    if not trigger:
        return caption
    first = caption.split(",", 1)[0].strip().lower()
    if first == trigger.lower():
        return caption
    return f"{trigger}, {caption}" if caption else trigger


def build_prompt(template: str, style: str, language: str) -> str:
    language_instruction = (
        "Write the caption in English."
        if language == "en"
        else "キャプションは日本語で書いてください。"
    )
    return template.format(
        style_instruction=STYLE_INSTRUCTIONS[style],
        language_instruction=language_instruction,
    )


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

def _image_data_uri(image_path: Path) -> tuple[str, str]:
    """Return (media_type, base64_data) for an image file."""
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return media_type, data


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    max_tokens: int
    timeout: float
    retries: int


def call_openai_compatible(cfg: ProviderConfig, image_path: Path, prompt: str) -> str:
    media_type, data = _image_data_uri(image_path)
    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    },
                ],
            }
        ],
    }
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    result = _post_json(
        f"{cfg.base_url.rstrip('/')}/chat/completions", payload, headers, cfg.timeout
    )
    return result["choices"][0]["message"]["content"]


def call_anthropic(cfg: ProviderConfig, image_path: Path, prompt: str) -> str:
    if not cfg.api_key:
        raise RuntimeError("Anthropic provider requires an API key")
    media_type, data = _image_data_uri(image_path)
    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"}
    result = _post_json(
        f"{cfg.base_url.rstrip('/')}/messages", payload, headers, cfg.timeout
    )
    parts = [b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"]
    return "".join(parts)


def call_mock(cfg: ProviderConfig, image_path: Path, prompt: str) -> str:
    return f"mock caption for {image_path.stem}, hand-drawn illustration"


PROVIDER_CALLS = {
    "openai-compatible": call_openai_compatible,
    "anthropic": call_anthropic,
    "mock": call_mock,
}


def generate_caption(cfg: ProviderConfig, image_path: Path, prompt: str) -> str:
    """Call the provider with retries on transient HTTP errors."""
    call = PROVIDER_CALLS[cfg.provider]
    last_error: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            return call(cfg, image_path, prompt)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == cfg.retries:
                raise
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
            last_error = exc
            if attempt == cfg.retries:
                raise
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"caption generation failed: {last_error}")


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

@dataclass
class Result:
    image: str
    status: str  # captioned | skipped | dry-run | failed
    caption_file: str | None = None
    error: str | None = None


def process_one(
    image_path: Path,
    cfg: ProviderConfig,
    prompt: str,
    trigger_word: str | None,
    overwrite: bool,
    dry_run: bool,
) -> Result:
    caption_path = caption_path_for(image_path)
    if caption_path.exists() and not overwrite:
        return Result(str(image_path), "skipped", str(caption_path))
    if dry_run:
        return Result(str(image_path), "dry-run", str(caption_path))
    try:
        raw = generate_caption(cfg, image_path, prompt)
        caption = apply_trigger_word(clean_caption(raw), trigger_word)
        if not caption:
            raise RuntimeError("model returned an empty caption")
        # Write atomically so a crash never leaves a half-written caption.
        tmp_path = caption_path.with_suffix(".txt.tmp")
        tmp_path.write_text(caption + "\n", encoding="utf-8")
        tmp_path.replace(caption_path)
        return Result(str(image_path), "captioned", str(caption_path))
    except Exception as exc:  # keep the batch going; report at the end
        return Result(str(image_path), "failed", str(caption_path), str(exc))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate LoRA training captions with a vision LLM."
    )
    parser.add_argument("input_dir", nargs="?", default="dataset",
                        help="Directory containing images (default: dataset)")
    parser.add_argument("--provider", choices=sorted(PROVIDER_CALLS),
                        default=os.environ.get("CAPTION_PROVIDER", "openai-compatible"))
    parser.add_argument("--model", default=os.environ.get("CAPTION_MODEL"),
                        help="Model name (default depends on provider; env: CAPTION_MODEL)")
    parser.add_argument("--base-url", default=os.environ.get("CAPTION_BASE_URL"),
                        help="API base URL (env: CAPTION_BASE_URL). "
                             "For a local server e.g. http://localhost:11434/v1")
    parser.add_argument("--api-key-env", default=None,
                        help="Environment variable holding the API key "
                             "(default: OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    parser.add_argument("--prompt-file", default=None,
                        help="Prompt template file with {style_instruction} and "
                             "{language_instruction} placeholders "
                             "(default: prompts/caption_default.txt next to the repo root, "
                             "falling back to a built-in prompt)")
    parser.add_argument("--style", choices=sorted(STYLE_INSTRUCTIONS), default="tags",
                        help="Caption style (default: tags)")
    parser.add_argument("--language", choices=("en", "ja"), default="en",
                        help="Caption language (default: en)")
    parser.add_argument("--trigger-word", default=os.environ.get("CAPTION_TRIGGER_WORD"),
                        help="Trigger word prepended to every caption (env: CAPTION_TRIGGER_WORD)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate captions even if the .txt already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be processed without calling the API")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (after skipping existing captions)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel requests (default: 1; raise carefully vs rate limits)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait between requests when workers=1")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--log-file", default=None,
                        help="Append per-image results as JSON Lines to this file")
    return parser.parse_args(argv)


BUILTIN_PROMPT = """\
You are captioning a hand-drawn illustration for LoRA fine-tuning of an
image generation model.

{style_instruction}
{language_instruction}

Rules:
- Describe only what differs in THIS image: the subject, appearance, pose,
  expression, clothing, objects held or nearby, action, and background.
- Do NOT describe the drawing style, medium, or artist -- those are the
  common traits the LoRA itself should learn, and naming them in every
  caption works against that.
- Do NOT include any trigger word; it is added separately.
- Do NOT guess at things that are not visible.
- Output the caption only. No explanations, no quotes, no preamble.
"""


def load_prompt_template(prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    default = Path(__file__).resolve().parent.parent / "prompts" / "caption_default.txt"
    if default.exists():
        return default.read_text(encoding="utf-8")
    return BUILTIN_PROMPT


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    api_key_env = args.api_key_env or DEFAULT_API_KEY_ENVS.get(args.provider, "")
    cfg = ProviderConfig(
        provider=args.provider,
        model=args.model or DEFAULT_MODELS.get(args.provider, ""),
        base_url=args.base_url or DEFAULT_BASE_URLS.get(args.provider, ""),
        api_key=os.environ.get(api_key_env) if api_key_env else None,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    if args.provider != "mock" and not args.dry_run and not cfg.api_key and args.provider == "anthropic":
        print(f"error: set {api_key_env} for the anthropic provider", file=sys.stderr)
        return 2

    prompt = build_prompt(load_prompt_template(args.prompt_file), args.style, args.language)
    images = find_images(input_dir, recursive=not args.no_recursive)

    todo = [
        p for p in images
        if args.overwrite or not caption_path_for(p).exists()
    ]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"images found: {len(images)}, to caption: {len(todo)}, "
          f"provider: {cfg.provider}, model: {cfg.model or '(mock)'}")

    def run_one(path: Path) -> Result:
        result = process_one(path, cfg, prompt, args.trigger_word,
                             args.overwrite, args.dry_run)
        if args.delay and args.workers == 1 and result.status == "captioned":
            time.sleep(args.delay)
        return result

    if args.workers > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_one, todo))
    else:
        results = [run_one(p) for p in todo]

    # Images that already had captions count as skipped.
    skipped_existing = len(images) - len(todo) if not args.overwrite else 0
    counts: dict[str, int] = {"skipped": skipped_existing}
    log_lines = []
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status == "failed":
            print(f"FAILED: {result.image}: {result.error}", file=sys.stderr)
        log_lines.append(json.dumps(vars(result), ensure_ascii=False))
    if args.log_file:
        with open(args.log_file, "a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + ("\n" if log_lines else ""))

    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
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
    "long": (
        "Write a long, literal caption of 5-8 sentences. Start with the "
        "composition and framing, then describe the subject, appearance, "
        "expression, pose, clothing and objects in detail, then the "
        "background and the colors of each element. Transcribe any text "
        "visible in the image exactly, in quotation marks, without "
        "translating or normalizing it."
    ),
}

# Per-base-model captioning presets, matching docs/base-models.md.
# Anima is trained on Danbooru tags; the other three are trained on
# natural-language captions (MOSS-Image on long literal ones, with exact
# transcription of visible text). The preset only fills in values the
# user did not set explicitly.
TARGET_PRESETS = {
    "anima": {"style": "tags", "language": "en", "prompt_file": None},
    "moss": {"style": "long", "language": "en", "prompt_file": "caption_moss.txt"},
    "z-image": {"style": "detailed", "language": "en", "prompt_file": None},
    "flux-klein": {"style": "detailed", "language": "en", "prompt_file": None},
}

# Styles whose captions are sentences rather than comma-separated tags.
SENTENCE_STYLES = {"natural", "detailed", "long"}

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


def caption_path_for(
    image_path: Path, input_dir: Path | None = None, output_dir: Path | None = None
) -> Path:
    """Caption file for an image.

    Default: next to the image with the same stem. With ``output_dir``,
    mirror the image's path relative to ``input_dir`` under ``output_dir``
    (used to keep per-base-model caption sets separate from the images).
    """
    if output_dir is None or input_dir is None:
        return image_path.with_suffix(".txt")
    relative = image_path.relative_to(input_dir).with_suffix(".txt")
    return output_dir / relative


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


def apply_trigger_word(
    caption: str, trigger_word: str | None, sentence: bool = False
) -> str:
    """Prepend the trigger word unless it is already at the start.

    Tag captions get ``trigger, ...``. Sentence captions get the trigger
    as its own leading sentence (``trigger. A girl ...``), because a
    comma-joined prefix would corrupt the sentence the model produced.
    """
    if not trigger_word:
        return caption
    trigger = trigger_word.strip()
    if not trigger:
        return caption
    head = caption.replace(".", ",", 1).split(",", 1)[0].strip().lower()
    if head == trigger.lower():
        return caption
    if not caption:
        return trigger
    return f"{trigger}. {caption}" if sentence else f"{trigger}, {caption}"


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
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    sentence_trigger: bool = False,
) -> Result:
    caption_path = caption_path_for(image_path, input_dir, output_dir)
    if caption_path.exists() and not overwrite:
        return Result(str(image_path), "skipped", str(caption_path))
    if dry_run:
        return Result(str(image_path), "dry-run", str(caption_path))
    try:
        raw = generate_caption(cfg, image_path, prompt)
        caption = apply_trigger_word(
            clean_caption(raw), trigger_word, sentence=sentence_trigger
        )
        if not caption:
            raise RuntimeError("model returned an empty caption")
        # Write atomically so a crash never leaves a half-written caption.
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = caption_path.with_suffix(".txt.tmp")
        tmp_path.write_text(caption + "\n", encoding="utf-8")
        tmp_path.replace(caption_path)
        return Result(str(image_path), "captioned", str(caption_path))
    except Exception as exc:  # keep the batch going; report at the end
        return Result(str(image_path), "failed", str(caption_path), str(exc))


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from ``path`` into the environment.

    Existing environment variables win, so explicit exports and CLI-related
    env settings are never overridden by a stale .env file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


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
    parser.add_argument("--target-model", choices=sorted(TARGET_PRESETS), default=None,
                        help="Base model to caption for. Fills in --style, --language "
                             "and --prompt-file from the preset in docs/base-models.md "
                             "and names the output subdirectory after the model; "
                             "explicitly given options win over the preset")
    parser.add_argument("--style", choices=sorted(STYLE_INSTRUCTIONS), default=None,
                        help="Caption style (default: tags, or the target model's preset)")
    parser.add_argument("--language", choices=("en", "ja"), default=None,
                        help="Caption language (default: en, or the target model's preset)")
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
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", default=None,
                        help="Root directory for generated captions. The actual files "
                             "go to <root>/<run datetime>/<target model>/, mirroring "
                             "the input folder structure (default root: dataset-captions)")
    parser.add_argument("--run-id", default=None,
                        help="Name of the run directory under the output root "
                             "(default: current local time as YYYYMMDD-HHMMSS). "
                             "Reusing a previous run-id resumes into that same "
                             "directory, skipping captions that already exist there")
    parser.add_argument("--in-place", action="store_true",
                        help="Write captions next to the images instead of into the "
                             "run directory structure (previous behaviour, e.g. for "
                             "assembling a trainer-ready dataset folder directly)")
    parser.add_argument("--env-file", default=".env",
                        help="Read settings from this .env file if it exists "
                             "(default: .env; existing environment variables win)")
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


def load_prompt_template(prompt_file: str | None, preset_prompt: str | None = None) -> str:
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    if preset_prompt and (prompts_dir / preset_prompt).exists():
        return (prompts_dir / preset_prompt).read_text(encoding="utf-8")
    default = prompts_dir / "caption_default.txt"
    if default.exists():
        return default.read_text(encoding="utf-8")
    return BUILTIN_PROMPT


def resolve_target(
    target_model: str | None,
    style: str | None,
    language: str | None,
) -> tuple[str, str, str | None]:
    """Apply a target-model preset under explicitly given options.

    Returns (style, language, preset_prompt_file).
    """
    preset = TARGET_PRESETS.get(target_model or "", {})
    resolved_style = style or preset.get("style") or "tags"
    resolved_language = language or preset.get("language") or "en"
    return resolved_style, resolved_language, preset.get("prompt_file")


def make_run_id(now: "datetime | None" = None) -> str:
    """Run directory name from the execution time: YYYYMMDD-HHMMSS (local)."""
    from datetime import datetime

    return (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def build_output_dir(
    output_root: str | None,
    run_id: str,
    target_model: str | None,
    style: str,
    in_place: bool,
) -> Path | None:
    """Where captions go.

    Default layout: ``<root>/<run datetime>/<target model>/`` mirroring the
    input folder structure below that. ``in_place`` returns None, which
    means captions are written next to the images instead. Without a
    target model, the model slot is named after the style.
    """
    if in_place:
        return None
    root = Path(output_root) if output_root else Path("dataset-captions")
    model_slot = target_model or f"style-{style}"
    return root / run_id / model_slot


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Load .env before parsing: several argument defaults come from the
    # environment, so the file must be read first. Honor an explicit
    # --env-file if one is present in argv.
    env_file = ".env"
    if "--env-file" in argv:
        idx = argv.index("--env-file")
        if idx + 1 < len(argv):
            env_file = argv[idx + 1]
    load_dotenv(Path(env_file))
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    style, language, preset_prompt = resolve_target(
        args.target_model, args.style, args.language
    )
    run_id = args.run_id or make_run_id()
    effective_output_dir = build_output_dir(
        args.output_dir, run_id, args.target_model, style, args.in_place
    )
    # Long captions (MOSS preset) do not fit in the default token budget.
    max_tokens = args.max_tokens
    if style == "long" and max_tokens < 800:
        max_tokens = 800

    api_key_env = args.api_key_env or DEFAULT_API_KEY_ENVS.get(args.provider, "")
    cfg = ProviderConfig(
        provider=args.provider,
        model=args.model or DEFAULT_MODELS.get(args.provider, ""),
        base_url=args.base_url or DEFAULT_BASE_URLS.get(args.provider, ""),
        api_key=os.environ.get(api_key_env) if api_key_env else None,
        max_tokens=max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    if args.provider != "mock" and not args.dry_run and not cfg.api_key and args.provider == "anthropic":
        print(f"error: set {api_key_env} for the anthropic provider", file=sys.stderr)
        return 2

    prompt = build_prompt(
        load_prompt_template(args.prompt_file, preset_prompt), style, language
    )
    images = find_images(input_dir, recursive=not args.no_recursive)
    output_dir = effective_output_dir

    todo = [
        p for p in images
        if args.overwrite or not caption_path_for(p, input_dir, output_dir).exists()
    ]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"images found: {len(images)}, to caption: {len(todo)}, "
          f"provider: {cfg.provider}, model: {cfg.model or '(mock)'}"
          + (f", target: {args.target_model}, style: {style}" if args.target_model else "")
          + (f", output: {output_dir}" if output_dir else ", output: next to images"))

    def run_one(path: Path) -> Result:
        result = process_one(path, cfg, prompt, args.trigger_word,
                             args.overwrite, args.dry_run,
                             input_dir=input_dir, output_dir=output_dir,
                             sentence_trigger=style in SENTENCE_STYLES)
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

    # Record what this run was, so every kept caption variant stays
    # identifiable and reproducible later (which settings produced it).
    if output_dir is not None and not args.dry_run:
        import hashlib
        from datetime import datetime

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "target_model": args.target_model,
            "style": style,
            "language": language,
            "provider": cfg.provider,
            "llm_model": cfg.model,
            "trigger_word": args.trigger_word,
            "prompt_file": args.prompt_file or preset_prompt or "caption_default.txt",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_dir": str(input_dir),
            "images_found": len(images),
            "counts": counts,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "caption_run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

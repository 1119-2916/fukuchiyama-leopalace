#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Post generated images to a Discord channel via webhook.

Usage:
    ./scripts/post_generated.sh --dir generated/<run>/<model> \
        --webhook "$DISCORD_WEBHOOK_URL"

The webhook URL is a credential: pass it as an argument or through
the DISCORD_WEBHOOK_URL environment variable (a .env file in the
repo root is also read). It is never printed or logged by this
script, and it must never be committed.

Behaviour:
- Collects image files (png/jpg/jpeg/webp) from --dir, sorted.
- If generation_manifest.json exists in the dir, a summary message
  (model, variant, base, LoRA, counts) is posted first.
- Images are attached in batches of up to 10 (Discord's per-message
  attachment limit). A file over 25 MB is skipped with a warning
  (Discord's upload limit for webhooks).
- 429 responses are honoured via retry_after, up to 3 retries.
- --dry-run lists what would be sent without posting anything.

Standard library only, like the other pipeline scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
BATCH_SIZE = 10
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_RETRIES = 3


def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env without overriding the shell."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def chunked(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def summarize_manifest(directory: Path) -> str | None:
    manifest_path = directory / "generation_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    lines = [
        f"生成サンプル: {m.get('target_model', '?')}"
        f"（{m.get('variant', 'base')}） run {m.get('run_id', '?')}",
        f"base: {m.get('base_model')}",
        f"LoRA: {m.get('lora') or 'なし'}"
        f"（scale {m.get('lora_scale', '-')}）",
        f"steps {m.get('steps', '-')} / CFG {m.get('guidance', '-')}"
        f" / {m.get('width', '?')}x{m.get('height', '?')}"
        f" / {len(m.get('images', []))}枚",
    ]
    return "\n".join(lines)[:2000]


def build_multipart(content: str | None, files: list[Path]) -> tuple[bytes, str]:
    """Build a multipart/form-data body for a Discord webhook POST."""
    boundary = f"----post-generated-{uuid.uuid4().hex}"
    body = bytearray()

    def add(data: bytes) -> None:
        body.extend(data)

    payload: dict = {}
    if content:
        payload["content"] = content
    if files:
        payload["attachments"] = [
            {"id": i, "filename": f.name} for i, f in enumerate(files)
        ]
    add(f"--{boundary}\r\n".encode())
    add(b'Content-Disposition: form-data; name="payload_json"\r\n')
    add(b"Content-Type: application/json\r\n\r\n")
    add(json.dumps(payload).encode("utf-8"))
    add(b"\r\n")
    for i, path in enumerate(files):
        add(f"--{boundary}\r\n".encode())
        add(
            f'Content-Disposition: form-data; name="files[{i}]"; '
            f'filename="{path.name}"\r\n'.encode()
        )
        add(b"Content-Type: application/octet-stream\r\n\r\n")
        add(path.read_bytes())
        add(b"\r\n")
    add(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_once(webhook: str, content: str | None, files: list[Path]) -> None:
    """POST one message, honouring Discord 429 retry_after."""
    body, content_type = build_multipart(content, files)
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(
            webhook,
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
            return
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < MAX_RETRIES:
                try:
                    wait = float(json.loads(error.read()).get("retry_after", 1))
                except (ValueError, json.JSONDecodeError):
                    wait = 1.0
                time.sleep(wait + 0.25)
                continue
            detail = error.read().decode("utf-8", "replace")[:300]
            raise SystemExit(
                f"error: Discord webhook returned {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise SystemExit(f"error: could not reach Discord: {error.reason}") from error


def resolve_webhook(args) -> str | None:
    load_env_file()
    return args.webhook or os.environ.get("DISCORD_WEBHOOK_URL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post generated images to Discord via webhook."
    )
    parser.add_argument("--dir", required=True,
                        help="Directory holding the generated images")
    parser.add_argument("--webhook", default=None,
                        help="Discord webhook URL (or DISCORD_WEBHOOK_URL / .env)")
    parser.add_argument("--message", default=None,
                        help="Override the first message text")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between messages (rate-limit courtesy)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be sent without posting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1
    images = collect_images(directory)
    if not images:
        print(f"error: no images found in {directory}", file=sys.stderr)
        return 1
    sendable, skipped = [], []
    for image in images:
        (skipped if image.stat().st_size > MAX_FILE_BYTES else sendable).append(image)
    for image in skipped:
        print(f"warning: skipped (>25MB): {image.name}", file=sys.stderr)

    first_message = args.message or summarize_manifest(directory)
    batches = chunked(sendable, BATCH_SIZE)

    if args.dry_run:
        print(f"dir: {directory}")
        print(f"first message: {first_message!r}" if first_message
              else "first message: (none)")
        for i, batch in enumerate(batches):
            names = ", ".join(p.name for p in batch)
            print(f"batch {i + 1}/{len(batches)}: {len(batch)} file(s): {names}")
        print(f"total: {len(sendable)} image(s) in {len(batches)} message(s)"
              + (f", {len(skipped)} skipped" if skipped else ""))
        return 0

    webhook = resolve_webhook(args)
    if not webhook:
        print("error: no webhook URL; pass --webhook or set "
              "DISCORD_WEBHOOK_URL", file=sys.stderr)
        return 1

    if first_message:
        post_once(webhook, first_message, [])
        time.sleep(args.delay)
    for i, batch in enumerate(batches):
        post_once(webhook, None, batch)
        print(f"posted batch {i + 1}/{len(batches)} ({len(batch)} image(s))")
        if i + 1 < len(batches):
            time.sleep(args.delay)
    print(f"done: {len(sendable)} image(s) posted to Discord")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

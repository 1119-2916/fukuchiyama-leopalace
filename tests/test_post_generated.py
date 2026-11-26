# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/post_generated.py (no network: dry-run + builders)."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "post_generated.py"
spec = importlib.util.spec_from_file_location("post_generated", SCRIPT)
pg = importlib.util.module_from_spec(spec)
sys.modules["post_generated"] = pg
spec.loader.exec_module(pg)


class CollectTests(unittest.TestCase):
    def test_collect_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "b.png").write_bytes(b"x")
            (d / "a.jpg").write_bytes(b"x")
            (d / "c.txt").write_text("no")
            (d / "generation_manifest.json").write_text("{}")
            names = [p.name for p in pg.collect_images(d)]
            self.assertEqual(names, ["a.jpg", "b.png"])

    def test_chunked(self):
        self.assertEqual(pg.chunked(list(range(25)), 10),
                         [list(range(10)), list(range(10, 20)), list(range(20, 25))])


class MultipartTests(unittest.TestCase):
    def test_multipart_contains_payload_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "img.png"
            f.write_bytes(b"\x89PNG-fake")
            body, content_type = pg.build_multipart("hello", [f])
            self.assertIn(b"payload_json", body)
            self.assertIn(b'"content": "hello"', body)
            self.assertIn(b'filename="img.png"', body)
            self.assertIn(b"\x89PNG-fake", body)
            self.assertTrue(content_type.startswith("multipart/form-data; boundary="))

    def test_multipart_text_only(self):
        body, _ = pg.build_multipart("only text", [])
        self.assertIn(b'"content": "only text"', body)
        self.assertNotIn(b"files[0]", body)


class ManifestSummaryTests(unittest.TestCase):
    def test_summary_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "generation_manifest.json").write_text(json.dumps({
                "target_model": "anima", "variant": "base", "run_id": "r1",
                "base_model": "base-x", "lora": "lora.safetensors",
                "lora_scale": 1.0, "steps": 30, "guidance": 4.5,
                "width": 1024, "height": 1024, "images": [{}, {}],
            }), encoding="utf-8")
            summary = pg.summarize_manifest(d)
            self.assertIn("anima", summary)
            self.assertIn("2枚", summary)

    def test_summary_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(pg.summarize_manifest(Path(tmp)))


class DryRunTests(unittest.TestCase):
    def test_main_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(12):
                (d / f"{i:03d}.png").write_bytes(b"x")
            rc = pg.main(["--dir", str(d), "--dry-run"])
            self.assertEqual(rc, 0)

    def test_main_empty_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pg.main(["--dir", tmp, "--dry-run"]), 1)

    def test_main_no_webhook_fails_without_dry_run(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.png").write_bytes(b"x")
            saved = os.environ.pop("DISCORD_WEBHOOK_URL", None)
            try:
                rc = pg.main(["--dir", str(d)])
            finally:
                if saved is not None:
                    os.environ["DISCORD_WEBHOOK_URL"] = saved
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()

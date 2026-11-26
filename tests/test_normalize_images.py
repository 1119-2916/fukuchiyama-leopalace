# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/normalize_images.py (local Pillow only, no network)."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "normalize_images.py"
spec = importlib.util.spec_from_file_location("normalize_images", SCRIPT)
ni = importlib.util.module_from_spec(spec)
sys.modules["normalize_images"] = ni
spec.loader.exec_module(ni)


class NormalizeTests(unittest.TestCase):
    def test_alpha_composited_on_white_and_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "a.png"
            Image.new("RGBA", (10, 10), (255, 0, 0, 0)).save(src)  # transparent red
            out = root / "out" / "a.png"
            record = ni.normalize_one(src, out, 2048, (255, 255, 255), 0)
            self.assertIn("alpha_composited", record["operations"])
            with Image.open(out) as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))

    def test_downscale_only_never_upscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            big = root / "big.png"
            Image.new("RGB", (3000, 1000), (10, 20, 30)).save(big)
            record = ni.normalize_one(big, root / "o1.png", 1024, (255, 255, 255), 0)
            self.assertEqual(record["final_size"], [1024, 341])
            small = root / "small.png"
            Image.new("RGB", (100, 50), (10, 20, 30)).save(small)
            record = ni.normalize_one(small, root / "o2.png", 1024, (255, 255, 255), 0)
            self.assertEqual(record["final_size"], [100, 50])
            self.assertEqual(record["operations"], [])

    def test_exif_orientation_baked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "side.jpg"
            img = Image.new("RGB", (40, 20), (200, 100, 50))
            exif = img.getexif()
            exif[274] = 6  # Orientation: rotate 90 CW when displaying
            img.save(src, exif=exif)
            record = ni.normalize_one(src, root / "out.png", 2048, (255, 255, 255), 0)
            self.assertIn("exif_orientation_baked", record["operations"])
            self.assertEqual(record["final_size"], [20, 40])

    def test_batch_layout_manifest_and_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (64, 64), (1, 2, 3)).save(images / "ok.png")
            (images / "broken.png").write_bytes(b"not an image")
            out_root = root / "norm"
            rc = ni.main([str(images), "--target-model", "moss",
                          "--output-dir", str(out_root), "--run-id", "run1"])
            self.assertEqual(rc, 1)  # one failure reported, batch continued
            model_dir = out_root / "run1" / "moss"
            self.assertTrue((model_dir / "ok.png").exists())
            manifest = json.loads((model_dir / "normalize_manifest.json").read_text())
            self.assertEqual(manifest["normalized"], 1)
            self.assertEqual(len(manifest["failed"]), 1)
            self.assertEqual(manifest["target_model"], "moss")


if __name__ == "__main__":
    unittest.main()

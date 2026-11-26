# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/preprocess_images.py (local Pillow only, no network)."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "preprocess_images.py"
spec = importlib.util.spec_from_file_location("preprocess_images", SCRIPT)
pp = importlib.util.module_from_spec(spec)
sys.modules["preprocess_images"] = pp
spec.loader.exec_module(pp)

SMALL = {"target_area": 100 * 100, "step": 16, "min_side": 32, "max_side": 512}


class BucketTests(unittest.TestCase):
    def test_square_at_area(self):
        self.assertEqual(pp.compute_bucket(200, 200, SMALL), (96, 96))

    def test_wide_image_keeps_aspect_near_area(self):
        bucket = pp.compute_bucket(400, 100, SMALL)
        self.assertEqual(bucket[0] % 16, 0)
        self.assertEqual(bucket[1] % 16, 0)
        self.assertGreater(bucket[0], bucket[1] * 3)
        self.assertLessEqual(bucket[0], SMALL["max_side"])

    def test_small_image_never_upscaled(self):
        self.assertEqual(pp.compute_bucket(50, 30, SMALL), (48, 16))

    def test_anima_step_is_32(self):
        bucket = pp.compute_bucket(3000, 2000, pp.MODEL_PRESETS["anima"])
        self.assertEqual(bucket[0] % 32, 0)
        self.assertEqual(bucket[1] % 32, 0)


class PreprocessBatchTests(unittest.TestCase):
    def test_end_to_end_layout_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (640, 480), (9, 9, 9)).save(images / "a.png")
            Image.new("RGB", (200, 900), (8, 8, 8)).save(images / "b.png")
            out_root = root / "pre"
            rc = pp.main([str(images), "--target-model", "z-image",
                          "--output-dir", str(out_root), "--run-id", "run1"])
            self.assertEqual(rc, 0)
            model_dir = out_root / "run1" / "z-image"
            manifest = json.loads((model_dir / "preprocess_manifest.json").read_text())
            self.assertEqual(manifest["preprocessed"], 2)
            for record in manifest["images"]:
                with Image.open(model_dir / record["output"]) as img:
                    self.assertEqual(list(img.size), record["bucket"])
                    self.assertEqual(img.size[0] % 16, 0)
                    self.assertEqual(img.size[1] % 16, 0)


if __name__ == "__main__":
    unittest.main()

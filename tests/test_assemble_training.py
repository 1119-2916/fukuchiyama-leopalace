# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/assemble_training.py (filesystem only, no network)."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "assemble_training.py"
spec = importlib.util.spec_from_file_location("assemble_training", SCRIPT)
at = importlib.util.module_from_spec(spec)
sys.modules["assemble_training"] = at
spec.loader.exec_module(at)


class AssembleTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        images = root / "dataset"
        run = root / "dataset-captions" / "20261126-170840" / "moss"
        images.mkdir(parents=True)
        run.mkdir(parents=True)
        for stem in ("001", "002"):
            (images / f"{stem}.png").write_bytes(b"\x89PNG")
            (run / f"{stem}.txt").write_text(f"caption {stem}\n", encoding="utf-8")
        (images / "003.png").write_bytes(b"\x89PNG")  # image without caption
        (run / "004.txt").write_text("orphan\n", encoding="utf-8")  # no image
        (run / "caption_run.json").write_text(
            json.dumps({"style": "long", "llm_model": "qwen-test"}), encoding="utf-8"
        )
        return images, run

    def test_assemble_pairs_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, run = self._fixture(root)
            dest = root / "training-data" / "moss"
            manifest = at.assemble(run, images, dest)
            self.assertEqual(manifest["pairs"], 2)
            self.assertEqual(manifest["captions_without_image"], ["004"])
            self.assertEqual(manifest["images_without_caption"], ["003"])
            self.assertTrue((dest / "001.png").exists())
            self.assertTrue((dest / "001.txt").exists())
            self.assertEqual(manifest["caption_run"]["llm_model"], "qwen-test")
            on_disk = json.loads((dest / "training_manifest.json").read_text())
            self.assertEqual(on_disk["pairs"], 2)

    def test_refuses_nonempty_dest_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, run = self._fixture(root)
            dest = root / "training-data" / "moss"
            at.assemble(run, images, dest)
            with self.assertRaises(SystemExit):
                at.assemble(run, images, dest)
            manifest = at.assemble(run, images, dest, overwrite=True)
            self.assertEqual(manifest["pairs"], 2)


if __name__ == "__main__":
    unittest.main()

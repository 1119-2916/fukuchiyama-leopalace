# SPDX-License-Identifier: AGPL-3.0-only
"""Guards for the per-model training configs and launchers.

These do not run any training; they keep the committed configs honest
(right arch, trigger word, no-upscale bucketing, Base-only checkpoints)
and make sure the launchers and the MOSS trainer at least parse.
"""

import tomllib
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class AnimaConfigTests(unittest.TestCase):
    def test_kohya_dataset_toml(self):
        data = tomllib.loads((ROOT / "configs/anima/dataset.toml").read_text())
        dataset = data["datasets"][0]
        self.assertTrue(dataset["enable_bucket"])
        self.assertTrue(dataset["bucket_no_upscale"])
        self.assertEqual(dataset["bucket_reso_steps"], 32)
        self.assertEqual(dataset["subsets"][0]["image_dir"], "training-data/anima")
        self.assertEqual(data["general"]["keep_tokens"], 1)


class AiToolkitConfigTests(unittest.TestCase):
    def _process(self, model: str) -> dict:
        data = yaml.safe_load((ROOT / f"configs/{model}/ai_toolkit.yaml").read_text())
        return data["config"]["process"][0]

    def test_z_image(self):
        proc = self._process("z-image")
        self.assertEqual(proc["model"]["arch"], "zimage")
        self.assertEqual(proc["trigger_word"], "okakikai")
        self.assertEqual(proc["network"]["linear"], 32)
        self.assertFalse(proc["train"]["train_text_encoder"])
        self.assertEqual(proc["datasets"][0]["folder_path"], "training-data/z-image")
        self.assertNotIn("Turbo", proc["model"]["name_or_path"])

    def test_flux_klein_matches_bfl_example(self):
        proc = self._process("flux-klein")
        self.assertEqual(proc["model"]["arch"], "flux3")
        self.assertEqual(
            proc["model"]["name_or_path"], "black-forest-labs/flux-3-image-klein-base"
        )
        self.assertEqual(proc["train"]["steps"], 4000)  # BFL official example
        self.assertEqual(proc["train"]["lr"], 1e-4)
        self.assertEqual(proc["network"]["linear"], 32)


class LauncherTests(unittest.TestCase):
    def test_launchers_exist_and_moss_trainer_compiles(self):
        for name in ("train_anima.sh", "train_z_image.sh",
                     "train_flux_klein.sh", "train_moss.sh"):
            self.assertTrue((ROOT / "scripts/train" / name).is_file(), name)
        source = (ROOT / "scripts/train/train_moss_lora.py").read_text()
        compile(source, "train_moss_lora.py", "exec")  # syntax check only


if __name__ == "__main__":
    unittest.main()

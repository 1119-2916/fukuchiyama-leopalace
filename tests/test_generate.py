# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/generate.py plan resolution (no torch/GPU needed)."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate.py"
spec = importlib.util.spec_from_file_location("generate", SCRIPT)
gen = importlib.util.module_from_spec(spec)
sys.modules["generate"] = gen
spec.loader.exec_module(gen)

ROOT = Path(__file__).resolve().parent.parent


def plan(*argv: str) -> dict:
    return gen.resolve_plan(gen.parse_args(list(argv)))


class PlanTests(unittest.TestCase):
    def test_flux_base_defaults(self):
        p = plan("--target-model", "flux-klein", "--prompt", "okakikai. test")
        self.assertEqual(p["steps"], 50)
        self.assertEqual(p["guidance"], 3.5)
        self.assertEqual(p["variant"], "base")
        self.assertEqual(p["jobs"], [{"prompt": "okakikai. test", "seed": 42}])

    def test_flux_distilled_switch(self):
        p = plan("--target-model", "flux-klein", "--prompt", "x", "--distilled")
        self.assertEqual(p["variant"], "distilled")
        self.assertEqual(p["steps"], 16)
        self.assertEqual(p["guidance"], 1.0)

    def test_size_snapped_to_16(self):
        p = plan("--target-model", "z-image", "--prompt", "x",
                 "--width", "1000", "--height", "1030")
        self.assertEqual(p["width"] % 16, 0)
        self.assertEqual(p["height"] % 16, 0)

    def test_eval_prompts_per_style(self):
        text = (ROOT / "prompts/eval_prompts.txt").read_text(encoding="utf-8")
        tags = gen.parse_eval_prompts(text, "tags")
        natural = gen.parse_eval_prompts(text, "natural")
        self.assertEqual(len(tags), 6)
        self.assertEqual(len(natural), 6)
        self.assertTrue(all(t.startswith("okakikai") for t in tags + natural))

    def test_eval_plan_uses_fixed_seeds(self):
        p = plan("--target-model", "anima", "--eval",
                 "--eval-file", str(ROOT / "prompts/eval_prompts.txt"))
        self.assertEqual(len(p["jobs"]), 6)
        self.assertEqual(p["jobs"][0]["seed"], gen.EVAL_SEED_BASE)
        self.assertIn("1girl", p["jobs"][0]["prompt"])

    def test_anima_base_from_env(self):
        import os

        os.environ["ANIMA_BASE_PATH"] = "/models/anima"
        try:
            p = plan("--target-model", "anima", "--prompt", "okakikai, 1girl")
            self.assertEqual(p["base_model"], "/models/anima")
        finally:
            del os.environ["ANIMA_BASE_PATH"]

    def test_dry_run_main(self):
        rc = gen.main(["--target-model", "moss", "--prompt", "okakikai. x",
                       "--dry-run", "--run-id", "t1"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

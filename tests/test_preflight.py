# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/preflight.py counting helpers (no GPU/network)."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
pf = importlib.util.module_from_spec(spec)
sys.modules["preflight"] = pf
spec.loader.exec_module(pf)


class CountTests(unittest.TestCase):
    def test_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").write_bytes(b"x")
            (root / "a.txt").write_text("cap", encoding="utf-8")
            (root / "b.jpg").write_bytes(b"x")  # no caption
            (root / "c.txt").write_text("orphan", encoding="utf-8")
            self.assertEqual(pf.count_images(root), 2)
            self.assertEqual(pf.count_pairs(root), 1)

    def test_missing_dir_counts_zero(self):
        self.assertEqual(pf.count_images(Path("/nonexistent-xyz")), 0)
        self.assertEqual(pf.count_pairs(Path("/nonexistent-xyz")), 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the batch plumbing of scripts/caption_batch.py.

These use the mock provider only -- no network, no API key.
Run: python3 -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import caption_batch as cb  # noqa: E402

# 1x1 transparent PNG
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000000020001e221bc3300000000"
    "49454e44ae426082"
)


def make_image(directory: Path, name: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)
    return path


class DiscoveryTests(unittest.TestCase):
    def test_find_images_sorted_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_image(root, "002.png")
            make_image(root, "001.jpg")
            make_image(root, "sub/003.webp")
            (root / "note.txt").write_text("not an image")
            found = [p.name for p in cb.find_images(root)]
            self.assertEqual(found, ["001.jpg", "002.png", "003.webp"])

    def test_caption_path(self):
        self.assertEqual(
            cb.caption_path_for(Path("dataset/001.png")), Path("dataset/001.txt")
        )

    def test_caption_path_with_output_dir_mirrors_structure(self):
        self.assertEqual(
            cb.caption_path_for(
                Path("dataset/sub/001.png"), Path("dataset"), Path("caps/tags")
            ),
            Path("caps/tags/sub/001.txt"),
        )


class PostProcessingTests(unittest.TestCase):
    def test_clean_caption_collapses_and_unquotes(self):
        self.assertEqual(cb.clean_caption('  "a girl,\n  sitting"  '), "a girl, sitting")

    def test_clean_caption_strips_code_fence(self):
        self.assertEqual(cb.clean_caption("```\n1girl, sitting\n```"), "1girl, sitting")

    def test_trigger_word_prepended_once(self):
        self.assertEqual(
            cb.apply_trigger_word("1girl, sitting", "mystyle"), "mystyle, 1girl, sitting"
        )
        self.assertEqual(
            cb.apply_trigger_word("mystyle, 1girl", "mystyle"), "mystyle, 1girl"
        )
        self.assertEqual(cb.apply_trigger_word("1girl", None), "1girl")


class DotEnvTests(unittest.TestCase):
    def test_load_dotenv_sets_missing_but_never_overrides(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "# comment\nCAPTION_TEST_NEW=from_file\nCAPTION_TEST_KEEP=from_file\n"
            )
            os.environ.pop("CAPTION_TEST_NEW", None)
            os.environ["CAPTION_TEST_KEEP"] = "from_shell"
            try:
                cb.load_dotenv(env_path)
                self.assertEqual(os.environ["CAPTION_TEST_NEW"], "from_file")
                self.assertEqual(os.environ["CAPTION_TEST_KEEP"], "from_shell")
            finally:
                os.environ.pop("CAPTION_TEST_NEW", None)
                os.environ.pop("CAPTION_TEST_KEEP", None)


class PresetTests(unittest.TestCase):
    def test_resolve_target_presets(self):
        self.assertEqual(cb.resolve_target("anima", None, None), ("tags", "en", None))
        self.assertEqual(
            cb.resolve_target("moss", None, None), ("long", "en", "caption_moss.txt")
        )
        self.assertEqual(cb.resolve_target("z-image", None, None)[0], "detailed")
        self.assertEqual(cb.resolve_target("flux-klein", None, None)[0], "detailed")

    def test_explicit_options_win_over_preset(self):
        self.assertEqual(cb.resolve_target("moss", "tags", None)[0], "tags")

    def test_no_target_keeps_defaults(self):
        self.assertEqual(cb.resolve_target(None, None, None), ("tags", "en", None))

    def test_run_id_format(self):
        from datetime import datetime

        self.assertEqual(
            cb.make_run_id(datetime(2026, 11, 26, 17, 8, 40)), "20261126-170840"
        )

    def test_build_output_dir_layout(self):
        self.assertEqual(
            cb.build_output_dir(None, "20261126-170840", "moss", "long", False),
            Path("dataset-captions") / "20261126-170840" / "moss",
        )
        self.assertEqual(
            cb.build_output_dir("caps", "run1", None, "tags", False),
            Path("caps") / "run1" / "style-tags",
        )
        self.assertIsNone(cb.build_output_dir(None, "run1", "moss", "long", True))

    def test_sentence_trigger(self):
        self.assertEqual(
            cb.apply_trigger_word("A girl sits.", "okakikai", sentence=True),
            "okakikai. A girl sits.",
        )
        self.assertEqual(
            cb.apply_trigger_word("okakikai. A girl sits.", "okakikai", sentence=True),
            "okakikai. A girl sits.",
        )

    def test_long_style_prompt_builds(self):
        prompt = cb.build_prompt("{style_instruction} {language_instruction}", "long", "en")
        self.assertIn("Transcribe any text", prompt)


class MockBatchTests(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_image(root, "001.png")
            make_image(root, "002.png")
            argv = [str(root), "--provider", "mock", "--trigger-word", "mystyle",
                    "--in-place"]
            self.assertEqual(cb.main(argv), 0)
            self.assertTrue(
                (root / "001.txt").read_text().startswith("mystyle, mock caption")
            )
            # Second run: everything already captioned -> still succeeds, no changes.
            before = (root / "001.txt").read_text()
            self.assertEqual(cb.main(argv), 0)
            self.assertEqual((root / "001.txt").read_text(), before)

    def test_run_directory_layout_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            out = root / "caps"
            make_image(images, "001.png")
            argv = [str(images), "--provider", "mock", "--output-dir", str(out),
                    "--target-model", "moss", "--run-id", "20261126-170840"]
            self.assertEqual(cb.main(argv), 0)
            caption = out / "20261126-170840" / "moss" / "001.txt"
            self.assertTrue(caption.exists())
            self.assertFalse((images / "001.txt").exists())
            # Re-running with the same run-id resumes into the same directory:
            # the existing caption is left untouched.
            before = caption.read_text()
            self.assertEqual(cb.main(argv), 0)
            self.assertEqual(caption.read_text(), before)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_image(root, "001.png")
            self.assertEqual(
                cb.main([str(root), "--provider", "mock", "--dry-run",
                         "--output-dir", str(root / "caps"),
                         "--run-id", "run1"]), 0
            )
            self.assertFalse((root / "001.txt").exists())
            self.assertFalse((root / "caps").exists())


if __name__ == "__main__":
    unittest.main()

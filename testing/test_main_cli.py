import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import _parse_args


class MainCliTests(unittest.TestCase):
    def test_render_defaults_to_disabled(self):
        args = _parse_args([])
        self.assertEqual(args.mode, "train")
        self.assertFalse(args.render)
        self.assertEqual(args.render_size, 512)
        self.assertEqual(args.render_step_delay, 0.05)

    def test_mode_argument(self):
        args = _parse_args(["--mode", "inference"])
        self.assertEqual(args.mode, "inference")

    def test_render_arguments(self):
        args = _parse_args([
            "--render",
            "--render-size",
            "320",
            "--render-step-delay",
            "0",
        ])
        self.assertTrue(args.render)
        self.assertEqual(args.render_size, 320)
        self.assertEqual(args.render_step_delay, 0.0)

    def test_skill_library_argument(self):
        args = _parse_args(["--skill-library", "skills_clean_001"])
        self.assertEqual(args.skill_library, "skills_clean_001")


if __name__ == "__main__":
    unittest.main()

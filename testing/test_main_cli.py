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
        self.assertIsNone(args.episodes)
        self.assertIsNone(args.seed)
        self.assertIsNone(args.eval_id)
        self.assertIsNone(args.eval_run_idx)
        self.assertIsNone(args.early_stop_patience)

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

    def test_training_loop_arguments(self):
        args = _parse_args([
            "--episodes",
            "25",
            "--early-stop-patience",
            "5",
        ])
        self.assertEqual(args.episodes, 25)
        self.assertEqual(args.early_stop_patience, 5)

    def test_reproducibility_and_eval_metadata_arguments(self):
        args = _parse_args([
            "--seed",
            "123",
            "--eval-id",
            "eval_abc",
            "--eval-run-idx",
            "4",
        ])
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.eval_id, "eval_abc")
        self.assertEqual(args.eval_run_idx, 4)

    def test_deprecated_early_stop_window_alias(self):
        args = _parse_args(["--early-stop-window", "3"])
        self.assertEqual(args.early_stop_patience, 3)


if __name__ == "__main__":
    unittest.main()

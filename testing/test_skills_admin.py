import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.skills_admin import _parse_args


class SkillsAdminCliTests(unittest.TestCase):
    def test_list_command_uses_library_option(self):
        args = _parse_args(["--library", "skills_good", "list"])
        self.assertEqual(args.library, "skills_good")
        self.assertEqual(args.command, "list")

    def test_copy_command(self):
        args = _parse_args(["copy", "skills_a", "skills_b", "--overwrite"])
        self.assertEqual(args.command, "copy")
        self.assertEqual(args.source, "skills_a")
        self.assertEqual(args.target, "skills_b")
        self.assertTrue(args.overwrite)

    def test_export_command(self):
        args = _parse_args(["--library", "skills_good", "export", "--out", "out.json"])
        self.assertEqual(args.command, "export")
        self.assertEqual(args.library, "skills_good")
        self.assertEqual(args.out, "out.json")

    def test_reembed_command(self):
        args = _parse_args(["--library", "skills_good", "reembed"])
        self.assertEqual(args.command, "reembed")
        self.assertEqual(args.library, "skills_good")
        self.assertIsNone(args.model)


if __name__ == "__main__":
    unittest.main()

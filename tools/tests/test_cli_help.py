"""`--help` after any installer subcommand prints usage and exits 0."""

import subprocess
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "bin" / "intel-skills.mjs"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(CLI), *args], capture_output=True, text=True, check=False
    )


class SubcommandHelpTest(unittest.TestCase):
    def test_subcommand_help_prints_usage(self) -> None:
        for args in (
            ("install", "--help"),
            ("install", "-h"),
            ("verify", "--help"),
            ("show", "--help"),
            ("list", "--help"),
        ):
            with self.subTest(args=args):
                result = run(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.startswith("Usage:"))
                self.assertNotIn("invalid skill name", result.stderr)

    def test_flag_value_is_not_a_help_request(self) -> None:
        result = run("verify", "linux-perf", "--path", "-h")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Usage:", result.stdout)
        self.assertIn("no such directory", result.stderr)

    def test_list_still_lists_without_help(self) -> None:
        result = run("list")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("Usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

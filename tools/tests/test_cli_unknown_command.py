"""An unrecognised installer command names itself instead of only printing usage."""

import subprocess
import sys
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "bin" / "intel-skills.mjs"


def run(*args):
    return subprocess.run(
        ["node", str(CLI), *args], capture_output=True, text=True, check=False
    )


class UnknownCommandTest(unittest.TestCase):
    def test_unknown_command_is_named_on_stderr(self):
        result = run("instal", "linux-perf")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: unknown command: instal", result.stderr)

    def test_known_command_still_works(self):
        result = run("list")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("unknown command", result.stderr)


if __name__ == "__main__":
    sys.exit(unittest.main())

"""`intel-skills --version` prints the version package.json declares."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"


def run(*args):
    return subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, check=False)


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class VersionTests(unittest.TestCase):
    def test_long_and_short_flag_print_the_package_version(self):
        declared = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]
        for flag in ("--version", "-v"):
            result = run(flag)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), declared)

    def test_usage_mentions_the_flag(self):
        self.assertIn("--version", run("--help").stdout)


if __name__ == "__main__":
    unittest.main()

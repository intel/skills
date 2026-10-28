"""`show` with no skill name prints usage instead of "invalid skill name: undefined"."""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"
SKILL = sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))[0]


def run(*args):
    return subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, check=False)


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class ShowUsageTests(unittest.TestCase):
    def test_show_without_a_name_prints_usage(self):
        result = run("show")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)
        self.assertNotIn("undefined", result.stdout + result.stderr)

    def test_show_with_only_a_flag_prints_usage(self):
        result = run("show", "--files")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)

    def test_show_with_a_name_still_prints_the_skill(self):
        result = run("show", SKILL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"name: {SKILL}", result.stdout)

    def test_show_unknown_skill_still_names_it(self):
        result = run("show", "no-such-skill")
        self.assertEqual(result.returncode, 1)
        self.assertIn("skill not found: no-such-skill", result.stderr)


if __name__ == "__main__":
    unittest.main()

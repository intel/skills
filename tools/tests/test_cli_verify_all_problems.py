"""`verify --all-problems` lists every drifted file instead of the first four."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"
SKILL = sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))[0]


def run(*args):
    return subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, check=False)


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class VerifyAllProblemsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.assertEqual(run("install", SKILL, "--dir", str(self.root)).returncode, 0)
        self.skill = self.root / SKILL

    def add_strays(self, count):
        for i in range(count):
            (self.skill / f"stray-{i}.txt").write_text("x")

    def test_default_truncates_after_four_problems(self):
        self.add_strays(6)
        result = run("verify", SKILL, "--path", str(self.skill))
        self.assertEqual(result.returncode, 1)
        self.assertIn("(+2 more; --all-problems lists them)", result.stderr)
        self.assertNotIn("stray-5.txt", result.stderr)

    def test_all_problems_lists_every_file(self):
        self.add_strays(6)
        result = run("verify", SKILL, "--path", str(self.skill), "--all-problems")
        self.assertEqual(result.returncode, 1)
        for i in range(6):
            self.assertIn(f"stray-{i}.txt is not part of this skill", result.stderr)
        self.assertNotIn("(+", result.stderr)

    def test_clean_install_still_verifies_with_flag(self):
        result = run("verify", SKILL, "--path", str(self.skill), "--all-problems")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith(f"ok {SKILL}"))

    def test_flag_is_accepted_by_verify_all(self):
        install = run("install", "--all", "--dir", str(self.root / "all"))
        self.assertEqual(install.returncode, 0, install.stderr)
        result = run("verify", "--all", "--path", str(self.root / "all"), "--all-problems")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

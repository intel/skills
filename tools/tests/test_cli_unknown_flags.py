"""`install` and `verify` refuse an option they do not know instead of ignoring it."""

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
class UnknownOptionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.target = Path(tmp.name) / "skills"

    def test_install_rejects_misspelled_flag_and_writes_nothing(self):
        result = run("install", SKILL, "--dir", str(self.target), "--forse")
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown option: --forse", result.stderr)
        self.assertFalse(self.target.exists())

    def test_install_rejects_all_combined_with_a_skill_name(self):
        result = run("install", SKILL, "--all", "--dir", str(self.target))
        self.assertEqual(result.returncode, 1)
        self.assertIn("--all", result.stderr)
        self.assertFalse(self.target.exists())

    def test_verify_rejects_unknown_flag(self):
        self.assertEqual(run("install", SKILL, "--dir", str(self.target)).returncode, 0)
        result = run("verify", SKILL, "--path", str(self.target / SKILL), "--bogus")
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown option: --bogus", result.stderr)

    def test_known_flags_still_work(self):
        self.assertEqual(run("install", SKILL, "--dir", str(self.target), "--force").returncode, 0)
        self.assertEqual(run("install", SKILL, "--dir", str(self.target), "--force").returncode, 0)
        verified = run("verify", SKILL, "--path", str(self.target / SKILL))
        self.assertEqual(verified.returncode, 0, verified.stderr)


if __name__ == "__main__":
    unittest.main()

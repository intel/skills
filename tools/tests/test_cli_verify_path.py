"""`verify --path` accepts an installed copy and names a path that is not there."""

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
class VerifyPathTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_missing_path_reports_no_such_directory(self):
        result = run("verify", SKILL, "--path", str(self.tmp / "absent"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such directory", result.stderr)

    def test_installed_copy_verifies(self):
        root = self.tmp / "skills"
        self.assertEqual(run("install", SKILL, "--dir", str(root)).returncode, 0)
        result = run("verify", SKILL, "--path", str(root / SKILL))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

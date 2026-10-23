"""`verify --path` pointing at a file says "not a directory", not a raw ENOTDIR."""

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
class VerifyPathFileTests(unittest.TestCase):
    def test_file_path_is_reported_as_not_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "SKILL.md"
            target.write_text("x", encoding="utf8")
            result = run("verify", SKILL, "--path", str(target))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a directory", result.stderr)
        self.assertNotIn("ENOTDIR", result.stderr)

    def test_missing_path_still_says_no_such_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run("verify", SKILL, "--path", str(Path(tmp) / "absent"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such directory", result.stderr)

    def test_installed_copy_still_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            installed = run("install", SKILL, "--dir", tmp)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            result = run("verify", SKILL, "--path", str(Path(tmp) / SKILL))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"ok {SKILL}", result.stdout)


if __name__ == "__main__":
    unittest.main()

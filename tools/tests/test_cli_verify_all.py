"""`intel-skills verify --all --path <dir>` checks every catalog skill under one directory."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"
NAMES = sorted(p.name for p in (ROOT / "skills").iterdir() if (p / "SKILL.md").is_file())


def run(*args):
    return subprocess.run(
        ["node", str(CLI), *args], capture_output=True, text=True, check=False
    )


class VerifyAllTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def install_all(self):
        self.assertEqual(run("install", "--all", "--dir", str(self.dir)).returncode, 0)

    def test_accepts_a_full_install(self):
        self.install_all()
        result = run("verify", "--all", "--path", str(self.dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified under", result.stdout)

    def test_names_the_altered_and_the_missing_skill(self):
        self.install_all()
        (self.dir / NAMES[0] / "SKILL.md").write_text("altered\n", encoding="utf-8")
        shutil.rmtree(self.dir / NAMES[1])

        result = run("verify", "--all", "--path", str(self.dir))
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"error: {NAMES[0]}:", result.stderr)
        self.assertIn(f"error: {NAMES[1]}:", result.stderr)
        self.assertIn(f"{len(NAMES) - 2}/{len(NAMES)} verified", result.stdout)

    def test_needs_a_path(self):
        self.assertEqual(run("verify", "--all").returncode, 1)

    def test_rejects_a_missing_directory(self):
        result = run("verify", "--all", "--path", str(self.dir / "absent"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such directory", result.stderr)


if __name__ == "__main__":
    unittest.main()

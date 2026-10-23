"""`intel-skills install a b c --dir <dir>` installs every named skill."""

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


class InstallManyTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_installs_every_named_skill(self):
        result = run("install", NAMES[0], NAMES[1], "--dir", str(self.dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.dir / NAMES[0] / "SKILL.md").is_file())
        self.assertTrue((self.dir / NAMES[1] / "SKILL.md").is_file())
        self.assertIn("2/2 installed", result.stdout)

    def test_a_bad_name_does_not_stop_the_others(self):
        result = run("install", "no-such-skill", NAMES[0], "--dir", str(self.dir))
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: no-such-skill:", result.stderr)
        self.assertTrue((self.dir / NAMES[0] / "SKILL.md").is_file())
        self.assertIn("1/2 installed", result.stdout)

    def test_a_repeated_name_is_installed_once(self):
        result = run("install", NAMES[0], NAMES[0], "--force", "--dir", str(self.dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count(f"installed {NAMES[0]} ->"), 1)

    def test_a_single_name_still_works(self):
        result = run("install", NAMES[0], "--dir", str(self.dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.dir / NAMES[0] / "SKILL.md").is_file())

    def test_needs_a_skill_name(self):
        self.assertEqual(run("install", "--dir", str(self.dir)).returncode, 1)


if __name__ == "__main__":
    unittest.main()

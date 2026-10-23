"""`install` and `verify` take their flags before or after the skill name."""

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
class FlagOrderTests(unittest.TestCase):
    def test_install_then_verify_accepts_either_flag_order(self):
        for flags_first in (False, True):
            with self.subTest(flags_first=flags_first), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "skills"
                if flags_first:
                    install = run("install", "--force", "--dir", str(target), SKILL)
                else:
                    install = run("install", SKILL, "--dir", str(target), "--force")
                self.assertEqual(install.returncode, 0, install.stderr)
                self.assertTrue((target / SKILL / "SKILL.md").is_file())

                path = str(target / SKILL)
                verify = run("verify", "--path", path, SKILL) if flags_first else run("verify", SKILL, "--path", path)
                self.assertEqual(verify.returncode, 0, verify.stderr)
                self.assertTrue(verify.stdout.startswith(f"ok {SKILL}"))

    def test_install_with_only_flags_prints_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run("install", "--dir", tmp)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""An unknown skill name in `show`/`install` names the catalog skills it resembles."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"


def run(*args):
    return subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, check=False)


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class SkillSuggestionTests(unittest.TestCase):
    def test_show_suggests_skills_containing_the_typed_name(self):
        result = run("show", "vllm-xpu")
        self.assertEqual(result.returncode, 1)
        self.assertIn("skill not found: vllm-xpu", result.stderr)
        self.assertIn("did you mean", result.stderr)
        for name in ("vllm-xpu-bench", "vllm-xpu-profile", "vllm-xpu-run"):
            self.assertIn(name, result.stderr)

    def test_install_suggests_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            result = run("install", "vllm-xpu", "--dir", str(target))
            self.assertEqual(result.returncode, 1)
            self.assertIn("vllm-xpu-run", result.stderr)
            self.assertFalse(target.exists())

    def test_no_suggestion_when_nothing_resembles_the_name(self):
        result = run("show", "zzz-nothing-like-it")
        self.assertEqual(result.returncode, 1)
        self.assertIn("skill not found: zzz-nothing-like-it", result.stderr)
        self.assertNotIn("did you mean", result.stderr)


if __name__ == "__main__":
    unittest.main()

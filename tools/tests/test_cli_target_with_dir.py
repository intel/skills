"""`install --target` is validated even when `--dir` overrides the location."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(CLI), *args], capture_output=True, text=True, check=False
    )


def first_skill() -> str:
    return run("list").stdout.splitlines()[0].split("\t")[0]


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class InstallTargetTest(unittest.TestCase):
    def test_unknown_target_with_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run("install", first_skill(), "--target", "claud-code", "--dir", tmp)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unknown target: claud-code", result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_known_target_with_dir_installs_into_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = first_skill()
            result = run("install", name, "--target", "agents", "--dir", tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / name / "SKILL.md").is_file())

    def test_dir_alone_still_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run("install", first_skill(), "--dir", tmp).returncode, 0)


if __name__ == "__main__":
    unittest.main()

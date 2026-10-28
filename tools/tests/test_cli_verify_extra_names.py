"""`verify` refuses a second skill name, or a name next to `--all`, instead of ignoring it."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

CLI = Path(__file__).resolve().parents[2] / "bin" / "intel-skills.mjs"


def run(*args):
    return subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, check=False)


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class VerifyExtraNamesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.assertEqual(run("install", "--all", "--dir", str(self.root)).returncode, 0)

    def test_second_name_is_refused(self):
        result = run("verify", "linux-perf", "xpu-discover", "--path", str(self.root / "linux-perf"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verify takes one skill name", result.stderr)

    def test_name_with_all_is_refused(self):
        result = run("verify", "linux-perf", "--all", "--path", str(self.root))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not also name one", result.stderr)

    def test_one_name_and_all_alone_still_verify(self):
        self.assertEqual(run("verify", "linux-perf", "--path", str(self.root / "linux-perf")).returncode, 0)
        self.assertEqual(run("verify", "linux-perf", "--dir", str(self.root)).returncode, 0)
        self.assertEqual(run("verify", "--all", "--path", str(self.root)).returncode, 0)


if __name__ == "__main__":
    unittest.main()

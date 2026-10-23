"""`intel-skills show <skill> --json` prints one machine-readable object; show is strict about its arguments."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"
SKILL = sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))[0]
OTHER = sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))[1]


def run(*args):
    return subprocess.run(["node", str(CLI), *args], capture_output=True, text=True, check=False)


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class ShowJsonTests(unittest.TestCase):
    def test_describes_the_skill(self):
        result = run("show", SKILL, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["name"], SKILL)
        self.assertTrue(data["description"])
        self.assertIn("SKILL.md", data["files"])
        self.assertEqual(data["files"], sorted(data["files"]))

    def test_flag_may_come_first(self):
        result = run("show", "--json", SKILL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["name"], SKILL)

    def test_rejects_unknown_options_and_extra_names(self):
        for args in (
            (SKILL, "--jsno"),
            (SKILL, OTHER),
            ("--json",),
            (SKILL, "--files", "--json"),
        ):
            with self.subTest(args=args):
                result = run("show", *args)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout.lstrip()[:1] == "{", False)

    def test_extra_name_is_named_in_the_error(self):
        result = run("show", SKILL, OTHER)
        self.assertIn(f"extra argument {OTHER}", result.stderr)

    def test_plain_show_still_prints_the_skill_body(self):
        result = run("show", SKILL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"name: {SKILL}", result.stdout)
        self.assertFalse(result.stdout.lstrip().startswith("{"))


if __name__ == "__main__":
    unittest.main()

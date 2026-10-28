"""`list` prints a skill's description without the YAML quotes around it."""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "intel-skills.mjs"


def quoted_skills():
    names = []
    for skill_md in sorted((ROOT / "skills").glob("*/SKILL.md")):
        for line in skill_md.read_text(encoding="utf8").splitlines()[:20]:
            if line.startswith("description:") and line.split(":", 1)[1].strip()[:1] in "\"'":
                names.append(skill_md.parent.name)
    return names


@unittest.skipIf(shutil.which("node") is None, "node not installed")
class ListDescriptionTests(unittest.TestCase):
    def test_quoted_descriptions_are_listed_without_quotes(self):
        names = quoted_skills()
        self.assertTrue(names, "catalog has no quoted description to exercise")
        out = subprocess.run(["node", str(CLI), "list"], capture_output=True, text=True, check=True).stdout
        rows = {line.split("\t")[0]: line.split("\t")[3] for line in out.splitlines()}
        for name in names:
            self.assertNotIn(rows[name][:1], "\"'", f"{name}: {rows[name]!r}")


if __name__ == "__main__":
    unittest.main()

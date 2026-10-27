"""`run_evals.py --skill NAME` limits validation and scoring to one skill."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "run_evals.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO_ROOT
    )


def first_skill_with_evals():
    path = sorted((REPO_ROOT / "skills").glob("*/evals/evals.json"))[0]
    return path.parent.parent.name, json.loads(path.read_text(encoding="utf-8"))


class SkillFilterTest(unittest.TestCase):
    def test_validate_one_skill_counts_only_its_cases(self):
        name, data = first_skill_with_evals()
        n = len(data["evals"])
        result = run("--validate", "--skill", name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"OK 1 eval file(s), {n} case(s)")

    def test_score_one_skill_ignores_other_skills_cases(self):
        name, data = first_skill_with_evals()
        n = len(data["evals"])
        with tempfile.TemporaryDirectory() as tmp:
            answers = Path(tmp) / "answers.json"
            answers.write_text("{}", encoding="utf-8")
            result = run("--answers", str(answers), "--skill", name, "--fail-under", "0")
        self.assertIn(f"0/{n} passed", result.stdout)

    def test_unknown_skill_is_an_error(self):
        result = run("--validate", "--skill", "no-such-skill")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL no skills/no-such-skill/evals/evals.json found", result.stderr)


if __name__ == "__main__":
    unittest.main()

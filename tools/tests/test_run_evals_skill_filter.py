"""`run_evals.py --skill NAME` limits validation and scoring to one skill."""

import json
import subprocess
import sys
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


def test_validate_one_skill_counts_only_its_cases():
    name, data = first_skill_with_evals()
    result = run("--validate", "--skill", name)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"OK 1 eval file(s), {len(data['evals'])} case(s)"


def test_score_one_skill_ignores_other_skills_cases(tmp_path):
    name, data = first_skill_with_evals()
    answers = tmp_path / "answers.json"
    answers.write_text("{}", encoding="utf-8")
    result = run("--answers", str(answers), "--skill", name, "--fail-under", "0")
    assert f"0/{len(data['evals'])} passed" in result.stdout


def test_unknown_skill_is_an_error():
    result = run("--validate", "--skill", "no-such-skill")
    assert result.returncode != 0
    assert "no-such-skill" in result.stderr

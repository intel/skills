"""compare_harbor_skill.py --report-only reports on whichever arms have finished."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "compare_harbor_skill.py"
SKILL = "dpnp-quickstart"


def make_arm(jobs: Path, arm: str, reward: float) -> None:
    job = jobs / f"{SKILL}-{arm}"
    trial = job / "task__abc"
    trial.mkdir(parents=True)
    (job / "result.json").write_text(json.dumps({"stats": {}}), encoding="utf-8")
    body = {"task_name": "task", "verifier_result": {"rewards": {"reward": reward}}}
    (trial / "result.json").write_text(json.dumps(body), encoding="utf-8")


def report_only(jobs: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--skill",
            SKILL,
            "--model",
            "m",
            "--report-only",
            "--jobs-dir",
            str(jobs),
        ],
        capture_output=True,
        text=True,
    )


class ReportOnlyTest(unittest.TestCase):
    def test_baseline_without_candidate_still_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            make_arm(jobs, "no-skill", 0.5)
            result = report_only(jobs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("`no_skill`: 0.500", result.stdout)
            self.assertNotIn("candidate − no_skill =", result.stdout)

    def test_both_arms_report_the_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            make_arm(jobs, "no-skill", 0.5)
            make_arm(jobs, "candidate", 1.0)
            result = report_only(jobs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("candidate − no_skill = +0.500", result.stdout)

    def test_malformed_trial_result_is_one_fail_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp)
            job = jobs / f"{SKILL}-no-skill"
            (job / "task__a").mkdir(parents=True)
            (job / "result.json").write_text(json.dumps({"stats": {}}), encoding="utf-8")
            (job / "task__a" / "result.json").write_text("{bad", encoding="utf-8")
            result = report_only(jobs)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(result.stderr.startswith("FAIL"), result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

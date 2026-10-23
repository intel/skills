"""check_harbor_job.py --json prints one machine-readable verdict."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "check_harbor_job.py"


def make_job(root: Path, rewards: dict[str, float | None]) -> Path:
    job = root / "job"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps({"n_total_trials": len(rewards), "stats": {}}), encoding="utf-8"
    )
    for name, reward in rewards.items():
        trial = job / f"{name}__abc"
        trial.mkdir()
        body = {"task_name": name, "trial_name": f"{name}__abc"}
        if reward is not None:
            body["verifier_result"] = {"rewards": {"reward": reward}}
        (trial / "result.json").write_text(json.dumps(body), encoding="utf-8")
    return job / "result.json"


def run(result: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), str(result), "--expected-trials", "2", *extra],
        capture_output=True,
        text=True,
    )


def strict_loads(text: str):
    def reject(token: str):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=reject)


def write_trial_text(result: Path, text: str) -> None:
    (result.parent / "b__abc" / "result.json").write_text(text, encoding="utf-8")


class CheckHarborJobJsonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def subdir(self, name: str) -> Path:
        root = self.tmp / name
        root.mkdir()
        return root

    def assert_structured_failure(self, proc, fragment: str):
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        data = strict_loads(proc.stdout)
        self.assertIs(data["ok"], False)
        self.assertTrue(any(fragment in p for p in data["problems"]), data["problems"])
        return data

    def test_json_pass(self):
        proc = run(make_job(self.tmp, {"a": 1.0, "b": 1.0}), "--json")
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertIs(data["ok"], True)
        self.assertEqual(data["problems"], [])
        self.assertEqual([t["task"] for t in data["trials"]], ["a", "b"])
        self.assertEqual(data["trials"][0]["reward"], 1.0)

    def test_json_fail_lists_problem(self):
        proc = run(make_job(self.tmp, {"a": 1.0, "b": 0.5}), "--json")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(proc.stdout)
        self.assertIs(data["ok"], False)
        self.assertTrue(any("b scored" in p for p in data["problems"]))

    def test_json_missing_reward_is_null(self):
        proc = run(make_job(self.tmp, {"a": 1.0, "b": None}), "--json")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(proc.stdout)
        self.assertIsNone(data["trials"][1]["reward"])

    def test_json_missing_job_is_structured_failure(self):
        proc = run(self.tmp / "missing.json", "--json")
        self.assertEqual(proc.returncode, 1)
        data = strict_loads(proc.stdout)
        self.assertIs(data["ok"], False)
        self.assertTrue(any("no such file" in p for p in data["problems"]))
        self.assertEqual(data["trials"], [])

    def test_json_malformed_job_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        result.write_text("{not json", encoding="utf-8")
        proc = run(result, "--json")
        self.assertEqual(proc.returncode, 1)
        data = strict_loads(proc.stdout)
        self.assertIs(data["ok"], False)
        self.assertTrue(any("not valid JSON" in p for p in data["problems"]))

    def test_json_malformed_trial_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        write_trial_text(result, "{")
        proc = run(result, "--json")
        self.assertEqual(proc.returncode, 1)
        self.assertIs(strict_loads(proc.stdout)["ok"], False)

    def test_json_non_object_job_is_structured_failure(self):
        for index, body in enumerate(("[]", "null", "3")):
            with self.subTest(body=body):
                result = make_job(self.subdir(f"job-{index}"), {"a": 1.0, "b": 1.0})
                result.write_text(body, encoding="utf-8")
                self.assert_structured_failure(run(result, "--json"), "not a JSON object")

    def test_json_non_object_trial_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        write_trial_text(result, "[1]")
        self.assert_structured_failure(run(result, "--json"), "not a JSON object")

    def test_json_directory_as_job_file_is_structured_failure(self):
        result = self.tmp / "result.json"
        result.mkdir()
        self.assert_structured_failure(run(result, "--json"), "cannot read")

    def test_json_directory_as_trial_file_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        trial_file = result.parent / "b__abc" / "result.json"
        trial_file.unlink()
        trial_file.mkdir()
        self.assert_structured_failure(run(result, "--json"), "cannot read")

    def test_json_invalid_utf8_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        result.write_bytes(b'{"n_total_trials": "\xff\xfe"}')
        self.assert_structured_failure(run(result, "--json"), "cannot read")

    def test_text_mode_non_object_job_fails_without_traceback(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        result.write_text("[]", encoding="utf-8")
        proc = run(result)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertIn("FAIL", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_json_wrong_shaped_job_stats_does_not_crash(self):
        for index, stats in enumerate(("[]", '"x"', "7", "null")):
            with self.subTest(stats=stats):
                result = make_job(self.subdir(f"stats-{index}"), {"a": 1.0, "b": 1.0})
                result.write_text(
                    '{"n_total_trials": 2, "stats": %s}' % stats, encoding="utf-8"
                )
                proc = run(result, "--json")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIs(strict_loads(proc.stdout)["ok"], True)

    def test_json_wrong_shaped_trial_fields_do_not_crash(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        (result.parent / "a__abc" / "result.json").write_text(
            json.dumps({"task_name": 5, "trial_name": ["x"], "verifier_result": []}),
            encoding="utf-8",
        )
        write_trial_text(
            result,
            json.dumps({"task_name": "b", "verifier_result": {"rewards": [1.0]}}),
        )
        proc = run(result, "--json")
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        data = strict_loads(proc.stdout)
        self.assertEqual([t["reward"] for t in data["trials"]], [None, None])
        self.assertEqual(len(data["problems"]), 2)

    def test_json_wrong_shaped_exception_info_is_an_error(self):
        for index, info in enumerate(('"boom"', "[1]", '{"exception_type": 5}')):
            with self.subTest(info=info):
                result = make_job(self.subdir(f"exc-{index}"), {"a": 1.0, "b": 1.0})
                write_trial_text(
                    result, '{"task_name": "b", "exception_info": %s}' % info
                )
                proc = run(result, "--json")
                self.assertEqual(proc.returncode, 1, info)
                data = strict_loads(proc.stdout)
                self.assertEqual(data["trials"][1]["error"], "unknown")
                self.assertTrue(
                    any("b errored: unknown" in p for p in data["problems"])
                )

    def test_text_mode_failure_message_unchanged(self):
        proc = run(self.tmp / "missing.json")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertIn("FAIL no such file", proc.stderr)

    def test_json_non_finite_reward_is_null_and_strict(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        # json.dumps writes NaN / Infinity, which Python's json module reads back.
        write_trial_text(
            result,
            '{"task_name": "b", "verifier_result": {"rewards": {"reward": NaN}}}',
        )
        (result.parent / "a__abc" / "result.json").write_text(
            '{"task_name": "a", "verifier_result": {"rewards": {"reward": Infinity}}}',
            encoding="utf-8",
        )
        proc = run(result, "--json")
        self.assertEqual(proc.returncode, 1)
        data = strict_loads(proc.stdout)
        self.assertEqual([t["reward"] for t in data["trials"]], [None, None])
        self.assertIs(data["ok"], False)

    def test_json_huge_int_reward_is_null_and_strict(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        # 400 digits parse as an int but cannot convert to float.
        huge = "1" + "0" * 400
        for sign in ("", "-"):
            with self.subTest(sign=sign):
                write_trial_text(
                    result,
                    '{"task_name": "b", "verifier_result": {"rewards": {"reward": %s%s}}}'
                    % (sign, huge),
                )
                proc = run(result, "--json")
                self.assertEqual(proc.returncode, 1)
                self.assertNotIn("Traceback", proc.stderr)
                data = strict_loads(proc.stdout)
                self.assertIsNone(data["trials"][1]["reward"])
                self.assertTrue(
                    any("b produced no reward" in p for p in data["problems"])
                )

    def test_text_mode_huge_int_reward_has_no_traceback(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        write_trial_text(
            result,
            '{"task_name": "b", "verifier_result": {"rewards": {"reward": 1%s}}}'
            % ("0" * 400),
        )
        proc = run(result)
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("b produced no reward", proc.stdout)

    def test_json_huge_int_reward_with_allow_errors_still_fails(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        write_trial_text(
            result,
            '{"task_name": "b", "verifier_result": {"rewards": {"reward": 1%s}}}'
            % ("0" * 400),
        )
        proc = run(result, "--json", "--allow-errors")
        self.assertEqual(proc.returncode, 1)
        self.assertIsNone(strict_loads(proc.stdout)["trials"][1]["reward"])

    def test_json_over_digit_limit_int_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        # Literal text: the decoder's integer digit limit is 4300 digits.
        write_trial_text(
            result,
            '{"task_name": "b", "verifier_result": {"rewards": {"reward": 1%s}}}'
            % ("0" * 5000),
        )
        data = self.assert_structured_failure(run(result, "--json"), "b__abc")
        self.assertEqual(data["trials"], [])

    def test_json_over_digit_limit_int_in_job_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        result.write_text('{"n_total_trials": 1%s}' % ("0" * 5000), encoding="utf-8")
        self.assert_structured_failure(run(result, "--json"), "result.json")

    def test_json_deeply_nested_trial_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        write_trial_text(result, "[" * 100000 + "]" * 100000)
        data = self.assert_structured_failure(run(result, "--json"), "b__abc")
        self.assertEqual(data["trials"], [])

    def test_json_deeply_nested_job_is_structured_failure(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        result.write_text(
            '{"stats": ' + "[" * 100000 + "]" * 100000 + "}", encoding="utf-8"
        )
        # Python 3.14+ parses this depth instead of raising RecursionError, so the job
        # then fails on the missing trial count rather than on a load error. Either is a
        # structured failure; a traceback or a pass is the regression.
        data = self.assert_structured_failure(run(result, "--json"), "")
        self.assertTrue(
            any("result.json" in p or "expected 2 trials" in p for p in data["problems"]),
            data["problems"],
        )

    def test_text_mode_deeply_nested_job_fails_without_traceback(self):
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        result.write_text("[" * 100000, encoding="utf-8")
        proc = run(result)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertIn("FAIL", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_allow_errors_tolerates_errored_trial(self):
        # --allow-errors is for exploratory runs: an errored trial is reported, not failed.
        result = make_job(self.tmp, {"a": 1.0, "b": 1.0})
        write_trial_text(
            result,
            json.dumps(
                {"task_name": "b", "exception_info": {"exception_type": "Timeout"}}
            ),
        )
        proc = run(result, "--json", "--allow-errors")
        self.assertEqual(proc.returncode, 0)
        data = strict_loads(proc.stdout)
        self.assertEqual(data["trials"][1]["error"], "Timeout")

    def test_default_output_unchanged(self):
        proc = run(make_job(self.tmp, {"a": 1.0, "b": 1.0}))
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(proc.stdout.rstrip().endswith("OK 2 trial(s) at or above 1.0"))


if __name__ == "__main__":
    unittest.main()

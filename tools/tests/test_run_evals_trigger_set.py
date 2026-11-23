"""`run_evals.py --validate` checks evals/trigger_eval_set.json against CONTRIBUTING.md's format.

CONTRIBUTING.md gives the file as `[{"query", "should_trigger"}]`, the list skill-creator's
run_eval.py reads, so a row it cannot read fails the gate instead of the later run.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_evals as r  # noqa: E402

EVALS = {
    "skill_name": "demo-skill",
    "evals": [
        {
            "id": "one-case",
            "prompt": "How do I start?",
            "expected_output": "It says how.",
            "checks": {"must_include": ["start"]},
        }
    ],
}
GOOD_ROWS = [
    {"query": "load me", "should_trigger": True},
    {"query": "not me", "should_trigger": False},
]


def validate(
    trigger_text: str | None, skill: str | None = None, other_text: str | None = None
) -> tuple[int, str, str]:
    """cmd_validate over a fake tree: demo-skill with this trigger file text, other-skill too."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        texts = {"demo-skill": trigger_text}
        if other_text is not None:
            texts["other-skill"] = other_text
        for name, text in texts.items():
            evals_dir = root / "skills" / name / "evals"
            evals_dir.mkdir(parents=True)
            evals = dict(EVALS, skill_name=name)
            (evals_dir / "evals.json").write_text(json.dumps(evals), encoding="utf-8")
            if text is not None:
                (evals_dir / "trigger_eval_set.json").write_text(text, encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(r, "REPO_ROOT", root), mock.patch.object(
            r, "SKILLS_DIR", root / "skills"
        ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = r.cmd_validate(skill)
        return code, out.getvalue(), err.getvalue()


def rows_with(row: dict) -> str:
    return json.dumps([*GOOD_ROWS, row])


WHERE = "skills/demo-skill/evals/trigger_eval_set.json"


class TriggerSetTest(unittest.TestCase):
    def test_well_formed_set_passes_and_is_counted(self):
        code, out, err = validate(json.dumps(GOOD_ROWS))
        self.assertEqual(code, 0, err)
        self.assertEqual(
            out.splitlines(), ["OK 1 eval file(s), 1 case(s)", "OK 1 trigger set(s), 2 request(s)"]
        )

    def test_skill_without_a_set_still_passes(self):
        code, out, err = validate(None)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), ["OK 1 eval file(s), 1 case(s)"])

    def test_string_should_trigger_fails(self):
        code, _, err = validate(rows_with({"query": "q", "should_trigger": "yes"}))
        self.assertEqual(code, 1)
        self.assertIn(f"FAIL {WHERE}: [2]: should_trigger must be true or false, got 'yes'", err)

    def test_missing_should_trigger_fails(self):
        code, _, err = validate(rows_with({"query": "q"}))
        self.assertEqual(code, 1)
        self.assertIn(f"FAIL {WHERE}: [2]: should_trigger is missing", err)

    def test_truncated_file_fails(self):
        code, _, err = validate(json.dumps(GOOD_ROWS)[:-5])
        self.assertEqual(code, 1)
        self.assertIn(f"FAIL {WHERE}: invalid JSON", err)

    def test_empty_or_missing_query_fails(self):
        for row in ({"query": "  ", "should_trigger": True}, {"should_trigger": False}):
            with self.subTest(row=row):
                code, _, err = validate(rows_with(row))
                self.assertEqual(code, 1)
                self.assertIn(f"FAIL {WHERE}: [2]: query must be a non-empty string", err)

    def test_wrong_shape_fails(self):
        cases = [
            ("[]", "top level must be a non-empty array"),
            (json.dumps({"queries": GOOD_ROWS}), "top level must be a non-empty array"),
            (json.dumps([*GOOD_ROWS, "q"]), "[2]: must be an object"),
        ]
        for text, message in cases:
            with self.subTest(text=text):
                code, _, err = validate(text)
                self.assertEqual(code, 1)
                self.assertIn(f"FAIL {WHERE}: {message}", err)

    def test_error_summary_counts_trigger_sets(self):
        _, _, err = validate(rows_with({"query": "q"}))
        self.assertIn("1 error(s) in 1 eval file(s) and 1 trigger set(s)", err)

    def test_skill_filter_checks_only_that_skills_set(self):
        broken = rows_with({"query": "q"})
        code, out, err = validate(json.dumps(GOOD_ROWS), skill="demo-skill", other_text=broken)
        self.assertEqual(code, 0, err)
        self.assertIn("OK 1 trigger set(s), 2 request(s)", out)
        code, _, err = validate(json.dumps(GOOD_ROWS), other_text=broken)
        self.assertEqual(code, 1)
        self.assertIn("FAIL skills/other-skill/evals/trigger_eval_set.json: [2]", err)


if __name__ == "__main__":
    unittest.main()

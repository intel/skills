"""Bundle declarations remain compatible with single-skill Harbor suites."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harbor_suite import skill_names, suite_id
import validate_skills
import lint_task_leakage
import compare_harbor_skill


class DeclarationTest(unittest.TestCase):
    def test_legacy_skill(self):
        declaration = {"skill": "skill-one"}
        self.assertEqual(skill_names(declaration), ["skill-one"])
        self.assertEqual(suite_id(declaration), "skill-one")

    def test_named_suite_with_one_or_many_skills(self):
        for names in (["skill-one"], ["skill-one", "skill-two"]):
            declaration = {"id": "my-bundle", "skills": names}
            self.assertEqual(skill_names(declaration), names)
            self.assertEqual(suite_id(declaration), "my-bundle")

    def test_invalid_skills(self):
        for declaration in (
            {}, {"skills": []}, {"skills": "skill-one"}, {"skills": [None]},
            {"skills": ["../escape"]}, {"skills": ["base", "base"]},
            {"skill": "base", "skills": ["base"]}, {"skill": ["base"]},
        ):
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                skill_names(declaration)

    def test_bundle_requires_valid_id(self):
        for name in (None, "", "../escape", ["bundle"]):
            with self.subTest(name=name), self.assertRaises(ValueError):
                suite_id({"id": name, "skills": ["base"]})


class SuiteValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = self.root / "tasks"
        self.instructions = self.root / "instructions"
        self.instructions.mkdir()
        self.manifest = self.root / "suites.json"
        self.task = self.tasks / "shared-task"
        for relative in validate_skills.TASK_REQUIRED + ("solution/solve.sh", "tests/test.sh"):
            target = self.task / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("placeholder", encoding="utf-8")
        (self.task / "task.toml").write_text(
            'schema_version = "1.3"\n[metadata]\nskills = ["base", "extra"]\ncovers = ["layout"]\n',
            encoding="utf-8",
        )
        self.suite = {
            "id": "bundle", "skills": ["base", "extra"],
            "capabilities": [{"id": "layout", "class": "correctness"}],
            "tasks": [{"name": "shared-task", "status": "implemented", "role": "smoke", "covers": ["layout"]}],
        }
        (self.instructions / "use-bundle.md").write_text("Use both skills.", encoding="utf-8")

    def validate(self, suites=None, catalog=None):
        self.manifest.write_text(json.dumps({"suites": suites or [self.suite]}), encoding="utf-8")
        report = validate_skills.Report()
        with patch.multiple(validate_skills, REPO_ROOT=self.root, SUITES_PATH=self.manifest,
                            TASKS_DIR=self.tasks, INSTRUCTIONS_DIR=self.instructions):
            validate_skills.check_suites(catalog or {}, {"base": "tool-skill", "extra": "tool-skill"}, set(), report)
        return report.errors

    def test_bundle_and_shared_task(self):
        other = dict(self.suite, id="other")
        (self.instructions / "use-other.md").write_text("Use both skills.", encoding="utf-8")
        self.assertEqual(self.validate([self.suite, other]), [])

    def test_legacy_suite_and_task(self):
        self.suite.pop("id")
        self.suite.pop("skills")
        self.suite["skill"] = "base"
        (self.instructions / "use-base.md").write_text("Use base.", encoding="utf-8")
        (self.task / "task.toml").write_text(
            'schema_version = "1.3"\n[metadata]\nskill = "base"\ncovers = ["layout"]\n', encoding="utf-8")
        self.assertEqual(self.validate(), [])

    def test_duplicate_task_and_suite_are_rejected(self):
        self.assertTrue(any("duplicate suite" in error for error in self.validate([self.suite, self.suite])))
        self.suite["tasks"] *= 2
        self.assertTrue(any("duplicate task" in error for error in self.validate()))

    def test_unknown_member_and_missing_task_skill(self):
        self.suite["skills"] = ["unknown"]
        self.assertTrue(any("unknown skills" in error for error in self.validate()))
        self.suite["skills"] = ["base"]
        self.assertTrue(any("not all supplied" in error for error in self.validate()))

    def test_bundle_cannot_validate_individual_member(self):
        self.assertTrue(any("bundle coverage only" in error for error in self.validate(catalog={"base": {"status": "validated"}})))

    def test_malformed_unclaimed_task_is_reported(self):
        self.suite["tasks"] = []
        (self.task / "task.toml").write_text(
            'schema_version = "1.3"\n[metadata]\nskills = "base"\ncovers = ["layout"]\n', encoding="utf-8")
        self.assertTrue(any("non-empty list" in error for error in self.validate()))


class LeakageTest(unittest.TestCase):
    def test_bundle_union_includes_suite_skills_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "task.toml").write_text('[metadata]\nskill = "base"\n', encoding="utf-8")
            (root / "instruction.md").write_text('Use `api.shared()` and `api.extra()`.\n', encoding="utf-8")
            manifest = root / "suites.json"
            manifest.write_text(json.dumps({"suites": [{
                "id": "bundle", "skills": ["base", "extra"], "tasks": [{"name": root.name}],
            }]}), encoding="utf-8")
            texts = {"base": "`api.shared()`", "extra": "`api.shared()` and `api.extra()`"}
            with patch.object(lint_task_leakage, "SUITES_PATH", manifest), \
                 patch.object(lint_task_leakage, "skill_text", side_effect=texts.get):
                result = lint_task_leakage.audit(root, set())
            self.assertEqual(result["skills"], ["base", "extra"])
            self.assertEqual(result["score"], 2)
            self.assertEqual(result["leaked_symbols"], ["api.extra", "api.shared"])

    def test_bad_metadata_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "task.toml").write_text('[metadata]\nskills = "base"\n', encoding="utf-8")
            self.assertIn("error", lint_task_leakage.audit(root, set()))


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "suites.json"
        self.instructions = self.root / "instructions"
        self.instructions.mkdir()
        self.suite = {
            "id": "bundle", "skills": ["base", "extra"], "note": "Example evaluation scope.",
            "tasks": [{"name": "first", "status": "implemented"},
                      {"name": "second", "status": "implemented"},
                      {"name": "later", "status": "planned"}],
        }
        for skill in self.suite["skills"]:
            directory = self.root / "skills" / skill
            directory.mkdir(parents=True)
            (directory / "SKILL.md").write_text(skill, encoding="utf-8")
        (self.instructions / "use-bundle.md").write_text("Use both skills.", encoding="utf-8")

    def run_comparison(self, *arguments, previous=None):
        self.manifest.write_text(json.dumps({"suites": [self.suite]}), encoding="utf-8")
        output = io.StringIO()
        with patch.multiple(compare_harbor_skill, REPO_ROOT=self.root, SUITES_PATH=self.manifest,
                            TASKS_PATH=self.root / "tasks", INSTRUCTIONS_DIR=self.instructions), \
             patch.object(sys, "argv", ["compare", "--suite", "bundle", "--model", "test", *arguments]), \
             patch.object(compare_harbor_skill, "git", return_value="abc123"), \
             patch.object(compare_harbor_skill, "snapshot_previous", side_effect=previous) as snapshot, \
             contextlib.redirect_stdout(output):
            result = compare_harbor_skill.main()
        return result, output.getvalue(), snapshot

    def test_dry_run_snapshots_all_skills_and_isolates_baseline(self):
        result, output, _ = self.run_comparison("--dry-run")
        self.assertEqual(result, 0)
        candidate = self.root / "harbor-jobs/bundle-arms/candidate"
        for skill in self.suite["skills"]:
            self.assertEqual((candidate / skill / "SKILL.md").read_text(), skill)
        commands = [line.strip() for line in output.splitlines() if line.strip().startswith("harbor run")]
        self.assertEqual(len(commands), 2)
        self.assertNotIn("--skill ", commands[0])
        self.assertNotIn("--extra-instruction-path", commands[0])
        self.assertIn("--skill harbor-jobs/bundle-arms/candidate", commands[1])
        self.assertIn("instructions/use-bundle.md", commands[1])
        for command in commands:
            self.assertIn("--include-task-name first --include-task-name second", command)
            self.assertNotIn("later", command)

    def test_previous_requires_every_member(self):
        _, output, snapshot = self.run_comparison("--dry-run", "--previous-ref", "main", previous=[True, False])
        self.assertEqual(snapshot.call_count, 2)
        self.assertIn("extra absent", output)
        self.assertNotIn("### previous", output)
        self.assertFalse((self.root / "harbor-jobs/bundle-arms/previous").exists())

    def test_complete_previous_is_run(self):
        _, output, snapshot = self.run_comparison("--dry-run", "--previous-ref", "main", previous=[True, True])
        self.assertEqual([call.args[:2] for call in snapshot.call_args_list], [("base", "abc123"), ("extra", "abc123")])
        self.assertIn("### previous", output)

    def test_duplicate_tasks_rejected(self):
        self.suite["tasks"] *= 2
        with self.assertRaisesRegex(SystemExit, "unique implemented tasks"):
            self.run_comparison("--dry-run")

    def test_report_names_bundle_and_limits_claim(self):
        for arm, reward in (("no-skill", 0), ("candidate", 1)):
            job = self.root / "harbor-jobs" / f"bundle-{arm}"
            trial = job / "first__trial"
            trial.mkdir(parents=True)
            (job / "result.json").write_text("{}", encoding="utf-8")
            (trial / "result.json").write_text(json.dumps({
                "task_name": "first", "verifier_result": {"rewards": {"reward": reward}},
            }), encoding="utf-8")
        result, output, _ = self.run_comparison("--report-only")
        self.assertEqual(result, 0)
        self.assertIn("# bundle", output)
        self.assertIn("supplied skills: `base`, `extra`", output)
        self.assertIn("Example evaluation scope", output)
        self.assertIn("not any member's individual contribution", output)


if __name__ == "__main__":
    unittest.main()
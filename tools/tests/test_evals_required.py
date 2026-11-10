"""Every skill carries evals/evals.json; an import's comes from upstream with its other files."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sync_external as s  # noqa: E402
import validate_skills as v  # noqa: E402


def errors_for(skills: dict[str, bool], imported: set[str], pending: set[str]) -> list[str]:
    """check_evals over a fake skills/ tree: name -> whether it has evals/evals.json."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, has_evals in skills.items():
            (root / name).mkdir()
            if has_evals:
                (root / name / "evals").mkdir()
                (root / name / v.EVALS_FILE).write_text("{}", encoding="utf-8")
        report = v.Report()
        with mock.patch.object(v, "SKILLS_DIR", root), mock.patch.object(
            v, "EVALS_PENDING", frozenset(pending)
        ):
            v.check_evals({name: "tool-skill" for name in skills}, imported, report)
        return report.errors


# EVALS_PENDING when the requirement was introduced. The set may lose names, never gain
# them: an import that arrives now brings its evals, so joining the list takes editing
# this test as well, where review sees it.
ORIGINAL_PENDING = frozenset(
    {
        "cuda-to-xpu-migration",
        "linux-perf",
        "llamacpp-xpu-run",
        "model-can-it-fit",
        "model-config-recommend",
        "performance-patterns",
        "phoronix-test-suite",
        "sglang-xpu-bench",
        "sglang-xpu-run",
        "torch-xpu-bench",
        "torch-xpu-profile",
        "torch-xpu-run",
        "vllm-xpu-bench",
        "vllm-xpu-profile",
        "vllm-xpu-run",
        "xpu-container-run",
        "xpu-deploy-plan",
        "xpu-discover",
        "xpu-model-type-detect",
        "xpu-port",
        "xpu-profile-unitrace",
        "xpu-runtime-preflight",
        "xpu-system-setup",
    }
)


class EvalsRequired(unittest.TestCase):
    def test_pending_only_shrinks(self) -> None:
        self.assertEqual(sorted(v.EVALS_PENDING - ORIGINAL_PENDING), [])

    def test_repository_as_it_stands_passes(self) -> None:
        skills = {p.name: "tool-skill" for p in v.SKILLS_DIR.iterdir() if p.is_dir()}
        imported = {n for n in skills if (v.SKILLS_DIR / n / v.IMPORT_MANIFEST).is_file()}
        report = v.Report()
        v.check_evals(skills, imported, report)
        self.assertEqual(report.errors, [])

    def test_validator_run_fails_a_skill_without_evals(self) -> None:
        # The run CI makes, on the real tree with one skill's file hidden: what proves
        # check_evals is wired into main() rather than only defined.
        target = v.SKILLS_DIR / "dpnp-io" / v.EVALS_FILE
        original = target.read_bytes()
        target.unlink()
        try:
            with mock.patch.object(sys, "argv", ["validate_skills.py"]), mock.patch(
                "sys.stdout"
            ), mock.patch("sys.stderr") as err:
                code = v.main()
            written = "".join(str(c.args[0]) for c in err.write.call_args_list)
        finally:
            target.write_bytes(original)
        self.assertEqual(code, 1)
        self.assertIn("skills/dpnp-io/: no evals/evals.json", written)

    def test_skill_written_here_without_evals_fails(self) -> None:
        errors = errors_for({"a": False}, imported=set(), pending=set())
        self.assertEqual(len(errors), 1)
        self.assertIn("templates/evals.json", errors[0])

    def test_import_without_evals_fails_and_points_upstream(self) -> None:
        errors = errors_for({"a": False}, imported={"a"}, pending=set())
        self.assertEqual(len(errors), 1)
        self.assertIn("upstream adds it", errors[0])

    def test_pending_import_without_evals_passes(self) -> None:
        self.assertEqual(errors_for({"a": False}, imported={"a"}, pending={"a"}), [])

    def test_pending_import_with_evals_is_stale(self) -> None:
        errors = errors_for({"a": True}, imported={"a"}, pending={"a"})
        self.assertEqual(len(errors), 1)
        self.assertIn("remove 'a' from EVALS_PENDING", errors[0])

    def test_pending_cannot_cover_a_skill_written_here(self) -> None:
        errors = errors_for({"a": False}, imported=set(), pending={"a"})
        self.assertEqual(len(errors), 1)
        self.assertIn("names a skill written here", errors[0])

    def test_pending_name_without_a_skill_is_stale(self) -> None:
        errors = errors_for({}, imported=set(), pending={"gone"})
        self.assertEqual(len(errors), 1)
        self.assertIn("'gone'", errors[0])


SKILL_MD = b"---\nname: a\ndescription: Does a.\nlicense: MIT\n---\n\nBody.\n"
ENTRY = {
    "name": "a",
    "external-repo": "https://github.com/example/skills",
    "external-commit": "0" * 40,
    "external-path": "skills/a",
    "external-license": "MIT",
}


def upstream(extra: dict[str, bytes]) -> dict[str, tuple[bytes, int]]:
    files = {"skills/a/SKILL.md": SKILL_MD, **{f"skills/a/{k}": b for k, b in extra.items()}}
    return {path: (blob, 0o644) for path, blob in files.items()}


class ImportedEvals(unittest.TestCase):
    def test_upstream_evals_are_vendored(self) -> None:
        out = s.vendored("a", ENTRY, upstream({"evals/evals.json": b"{}\n"}), {}, {"a"})
        self.assertEqual(out["evals/evals.json"], (b"{}\n", 0o644))

    def test_upstream_perf_is_still_refused(self) -> None:
        with self.assertRaises(s.SyncError):
            s.vendored("a", ENTRY, upstream({"perf/summary.md": b"x\n"}), {}, {"a"})

    def test_local_evals_beside_an_import_are_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill = Path(tmp)
            (skill / "SKILL.md").write_bytes(SKILL_MD)
            (skill / "evals").mkdir()
            (skill / "evals" / "evals.json").write_bytes(b"{}\n")
            (skill / "perf").mkdir()
            (skill / "perf" / "summary.md").write_bytes(b"x\n")
            current = s.on_disk(skill)
        wanted = s.vendored("a", ENTRY, upstream({}), {}, {"a"})
        self.assertIn("evals/evals.json", current)
        self.assertNotIn("perf/summary.md", current)
        # What differences() reports as "not part of the pinned upstream skill".
        self.assertIn("evals/evals.json", set(current) - set(wanted))


if __name__ == "__main__":
    unittest.main()

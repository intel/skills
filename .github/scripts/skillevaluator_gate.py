#!/usr/bin/env -S uv run --quiet
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///

"""Gate one skill's SkillEvaluator JSON report against .skillevaluator-baseline.yaml.

`skillevaluator validate` already has a verdict: it fails a skill on every finding a
validator counts as an error, on any scanner that did not complete, and on a quality
score below --min-score. What it has no way to express is an accepted finding. Its policy overlay remaps a *kind* of finding catalog-wide, keyed on
CATEGORY.check_name, and there is no baseline, fingerprint or per-file suppression. A
catalog with 23 imported skills needs one, because the repair for an imported body lands
upstream and arrives through a moved pin, and until then the finding is known and
reviewed rather than new.

So this script re-derives the verdict from the JSON with the baseline applied, and
nothing else:

  * every error-level finding must be matched by a `findings` entry scoped to this
    skill, or it fails. Error-level is SkillEvaluator's own call, read from the
    result's error list: every HIGH and CRITICAL, plus the MEDIUM findings a validator
    raises as errors (a Bandit MEDIUM at medium or high confidence, for one). Gating by
    severity alone would leave those failing with nothing a baseline entry could match;
  * every scanner in a result's `incomplete_scans` must be matched by an `incomplete`
    entry, or it fails -- an incomplete scan is no answer, and SkillEvaluator's own gate
    treats it as a failure too;
  * a result that failed with an error the findings do not account for (a validator that
    reported through its legacy error list rather than a finding) fails, because there
    is nothing structured to accept it by;
  * a quality score below --min-score fails. SkillEvaluator records that only as the
    QUALITY result's `passed`, with no finding or error behind it, so nothing else
    here would see it. It is not a baseline matter: the fix is in the skill;
  * a baseline entry scoped to this skill that matched nothing fails, so fixing a
    finding forces its acceptance out in the same pull request. The file is a ratchet.

Reading the JSON rather than trusting the exit code is also what separates a skill that
failed from a tool that did: SkillEvaluator raises its configuration errors (`--tiers`,
a conflicting `--profile`) as ClickException, which exits 1 exactly like a failed skill.
A missing or malformed report is therefore the tool failing, and is reported as that.

The report is also checked for the policy it ran under. A run that silently fell back to
the bare `external` profile would fail every skill on `SCHEMA.author_missing`, which reads
as 33 regressions rather than as one dropped flag.

Every severity override in that policy must also be one in SAFE_OVERRIDES. When an
override changes a finding, SkillEvaluator's apply_policy rebuilds that validator's error
list from severity alone (ValidationResult.recalculate_from_findings), so an error-level
MEDIUM such as a confident Bandit finding drops to a warning and this gate never sees it.
The overrides listed are the ones that change nothing a validator counts as an error.

Usage:

    uv run .github/scripts/skillevaluator_gate.py \\
        --report-dir reports/linux-perf --skill linux-perf --min-score 70 --annotate
    uv run .github/scripts/skillevaluator_gate.py --check-baseline

Exits 0 when the skill passes, 1 when it fails or the report cannot be trusted, and 2 on
a usage error or an unreadable baseline.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASELINE = REPO_ROOT / ".skillevaluator-baseline.yaml"
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
EXPECTED_PROFILE = "intel-skills"
GATED_SEVERITIES = ("critical", "high")
DEFAULT_MIN_SCORE = 70.0
GLOB_CHARS = set("*?[")
# Overrides that cannot erase an error at SkillEvaluator 41e5469. SCHEMA.* is resolved by
# schema.py when the finding is emitted, so apply_policy changes nothing; WORKFLOWS.* has
# no finding on a skill; LICENSE.* raises findings that are all errors already.
SAFE_OVERRIDES = frozenset({
    "SCHEMA.author_missing",
    "SCHEMA.author_format",
    "WORKFLOWS.author_format",
    "LICENSE.*",
})
SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")


@dataclass
class Entry:
    """One baseline entry, with the bookkeeping the ratchet needs."""

    section: str
    index: int
    patterns: dict[str, str]
    skills: list[str]
    reason: str
    matched: int = 0

    def label(self) -> str:
        key = self.patterns.get("check") or self.patterns.get("scanner")
        return f"{self.section}[{self.index}] ({key})"


@dataclass
class Verdict:
    errors: list[str] = field(default_factory=list)
    accepted: list[tuple[str, str]] = field(default_factory=list)


def load_baseline(path: Path, skills_on_disk: list[str]) -> list[Entry]:
    """Parse the baseline and refuse a scope that names nothing on disk."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read baseline {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"baseline {path} must be a mapping")
    unknown = set(data) - {"findings", "incomplete"}
    if unknown:
        raise ValueError(f"baseline {path} has unknown sections: {sorted(unknown)}")

    required = {"findings": ("check", "path"), "incomplete": ("scanner",)}
    optional = {"findings": ("message",), "incomplete": ()}
    entries: list[Entry] = []
    for section, keys in required.items():
        for index, raw in enumerate(data.get(section) or []):
            where = f"{section}[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{where} must be a mapping")
            allowed = set(keys) | set(optional[section]) | {"skills", "reason"}
            extra = set(raw) - allowed
            if extra:
                raise ValueError(f"{where} has unknown keys: {sorted(extra)}")
            missing = [k for k in (*keys, "skills", "reason") if not raw.get(k)]
            if missing:
                raise ValueError(f"{where} is missing {missing}")
            reason = str(raw["reason"]).strip()
            if not reason.startswith(("POLICY:", "TRACKED:")):
                raise ValueError(f"{where} reason must start with POLICY: or TRACKED:")
            skills = raw["skills"]
            if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
                raise ValueError(f"{where} skills must be a list of skill names")
            for name in skills:
                # Exact names only. A pattern would also scope skills added later, which
                # then fail as stale for an entry that never named them.
                if GLOB_CHARS & set(name):
                    raise ValueError(f"{where} scope {name!r} is a pattern; list skill names")
                if name not in skills_on_disk:
                    raise ValueError(f"{where} scope {name!r} is not a skill under skills/")
            patterns = {k: str(raw[k]) for k in (*keys, *optional[section]) if k in raw}
            entries.append(Entry(section, index, patterns, skills, reason))
    return entries


def in_scope(entry: Entry, skill: str) -> bool:
    return skill in entry.skills


def legacy_string(finding: dict[str, Any]) -> str:
    """The line SkillEvaluator writes to a result's error or warning list for a finding.

    Same shape as ValidationResult.add_structured_finding, which is what tells an
    error-level finding from a warning: only the former is in `legacy.errors`.
    """
    location = finding.get("file_path") or ""
    if finding.get("line_number"):
        location += f":{finding['line_number']}"
    severity = str(finding.get("severity", "")).upper()
    return f"[{finding.get('category', '')}-{severity}] {finding.get('message', '')} in {location}"


def is_gated(finding: dict[str, Any], legacy_errors: set[str]) -> bool:
    severity = str(finding.get("severity", "")).lower()
    return severity in GATED_SEVERITIES or legacy_string(finding) in legacy_errors


def relative_path(file_path: str, skill_dir: Path) -> str:
    """A finding's file relative to the skill, which is what `path` patterns match."""
    if not file_path:
        return ""
    try:
        return Path(file_path).resolve().relative_to(skill_dir.resolve()).as_posix()
    except ValueError:
        marker = f"/{skill_dir.name}/"
        return file_path.split(marker, 1)[1] if marker in file_path else file_path


def find_report(report_dir: Path) -> Path:
    """The newest `skillevaluator-output-<timestamp>.json` in the skill's report dir."""
    candidates = sorted(report_dir.glob("skillevaluator-output-*[0-9].json"))
    if not candidates:
        raise FileNotFoundError(f"no skillevaluator-output-*.json under {report_dir}")
    return candidates[-1]


def gated_findings(result: dict[str, Any], legacy_errors: list[str]) -> list[dict[str, Any]]:
    errors = set(legacy_errors)
    return [f for f in result.get("findings") or [] if is_gated(f, errors)]


def match_finding(scoped: list[Entry], check: str, path: str, message: str) -> Entry | None:
    return next(
        (
            e
            for e in scoped
            if e.section == "findings"
            and fnmatch.fnmatchcase(check, e.patterns["check"])
            and fnmatch.fnmatchcase(path, e.patterns["path"])
            and fnmatch.fnmatchcase(message, e.patterns.get("message", "*"))
        ),
        None,
    )


def match_incomplete(scoped: list[Entry], scanner: str) -> Entry | None:
    return next(
        (
            e
            for e in scoped
            if e.section == "incomplete" and fnmatch.fnmatchcase(scanner, e.patterns["scanner"])
        ),
        None,
    )


def accept_or_fail(verdict: Verdict, entry: Entry | None, text: str) -> bool:
    """Record `text` as accepted by `entry`, or as a failure when there is none."""
    if entry is None:
        verdict.errors.append(text)
        return False
    entry.matched += 1
    verdict.accepted.append((text, entry.reason))
    return True


def gate_findings(
    verdict: Verdict, gated: list[dict[str, Any]], scoped: list[Entry], skill: str, skill_dir: Path
) -> None:
    for finding in gated:
        check = f"{finding.get('category', '')}.{finding.get('check_name', '')}"
        path = relative_path(finding.get("file_path") or "", skill_dir)
        message = finding.get("message") or ""
        where = f"{path}:{finding.get('line_number')}" if finding.get("line_number") else path
        text = f"[{finding['severity'].upper()}] {check} {where}: {message}"
        if not accept_or_fail(verdict, match_finding(scoped, check, path, message), text):
            annotate_error(skill, finding, path, check, message)


def gate_incomplete(verdict: Verdict, result: dict[str, Any], scoped: list[Entry]) -> None:
    validator = result.get("validator", "?")
    for scanner in result.get("incomplete_scans") or []:
        entry = match_incomplete(scoped, scanner)
        text = f"{validator}: scanner '{scanner}' did not complete"
        if entry is None:
            detail = "; ".join((result.get("legacy") or {}).get("errors", [])[:3])
            text = f"{text}{f' ({detail})' if detail else ''}"
        accept_or_fail(verdict, entry, text)


def gate_unstructured(
    verdict: Verdict, result: dict[str, Any], gated: list[dict[str, Any]], legacy_errors: list[str]
) -> None:
    """A validator can fail through its legacy error list without a structured finding.

    When the result is not incomplete, those errors have nothing a baseline entry could
    match, so they fail rather than pass unread.
    """
    structured = {legacy_string(f) for f in gated}
    unstructured = [e for e in legacy_errors if e not in structured]
    if result.get("status") == "failed" and unstructured and not result.get("incomplete_scans"):
        detail = "; ".join(unstructured[:3])
        validator = result.get("validator", "?")
        verdict.errors.append(f"{validator}: {len(unstructured)} unstructured error(s): {detail}")


def gate_report_shape(verdict: Verdict, report: dict[str, Any], min_score: float) -> bool:
    """Profile and quality floor; False when there are no results to gate at all."""
    policy = report.get("policy") or {}
    profile = policy.get("profile")
    if profile != EXPECTED_PROFILE:
        verdict.errors.append(
            f"report ran under policy profile {profile!r}, not {EXPECTED_PROFILE!r}; "
            "was --policy .skillevaluator-policy.yaml dropped?"
        )
    overrides = policy.get("severity_overrides")
    if not isinstance(overrides, dict):
        verdict.errors.append("report does not record the policy's severity_overrides")
    else:
        for key in sorted(set(overrides) - SAFE_OVERRIDES):
            verdict.errors.append(
                f"severity override {key!r} is not in SAFE_OVERRIDES: SkillEvaluator rebuilds "
                "a changed validator's errors from severity alone, which can drop an "
                "error-level MEDIUM; confirm it cannot, then add it there"
            )
    results = report.get("results")
    if not isinstance(results, list) or not results:
        verdict.errors.append("report has no validator results; the run did not complete")
        return False
    for quality in report.get("quality_summary") or []:
        score = quality.get("overall_score")
        if isinstance(score, (int, float)) and score < min_score:
            verdict.errors.append(f"quality score {score} is below the minimum of {min_score:g}")
    return True


def gate(
    report: dict[str, Any], skill: str, skill_dir: Path, entries: list[Entry], min_score: float
) -> Verdict:
    verdict = Verdict()
    scoped = [e for e in entries if in_scope(e, skill)]
    if not gate_report_shape(verdict, report, min_score):
        return verdict

    for result in report["results"]:
        legacy_errors = list((result.get("legacy") or {}).get("errors") or [])
        gated = gated_findings(result, legacy_errors)
        gate_findings(verdict, gated, scoped, skill, skill_dir)
        gate_incomplete(verdict, result, scoped)
        gate_unstructured(verdict, result, gated, legacy_errors)

    gate_stale(verdict, scoped, skill)
    return verdict


def gate_stale(verdict: Verdict, scoped: list[Entry], skill: str) -> None:
    """The ratchet: an entry scoped to this skill that matched nothing fails."""
    for entry in scoped:
        if entry.matched == 0:
            verdict.errors.append(
                f"stale baseline entry {entry.label()} matched nothing in '{skill}': "
                f"remove '{skill}' from its skills: list, or the entry if it was the last"
            )


def annotate_error(skill: str, finding: dict[str, Any], path: str, check: str, message: str) -> None:
    if not ANNOTATE:
        return
    location = f"file=skills/{skill}/{path}" if path else ""
    if location and finding.get("line_number"):
        location += f",line={finding['line_number']}"
    title = f"SkillEvaluator {check}"
    fix = finding.get("suggestion") or ""
    body = f"{message}. {fix}".replace("\n", " ").replace("%", "%25")
    print(f"::error {location},title={title}::{body}" if location else f"::error title={title}::{body}")


def write_summary(skill: str, report: dict[str, Any] | None, verdict: Verdict) -> None:
    lines: list[str] = []
    icon = ":x:" if verdict.errors else ":white_check_mark:"
    lines.append(f"### {icon} {skill}")
    if report is not None:
        counts = report.get("severity_counts") or {}
        tally = ", ".join(f"{counts.get(s, 0)} {s}" for s in SEVERITY_ORDER if s in counts)
        lines.append(f"Findings (all severities, before the baseline): {tally or 'none'}")
        for quality in report.get("quality_summary") or []:
            lines.append(
                f"Quality score: **{quality.get('overall_score')}** "
                f"(grade {quality.get('grade')}, {quality.get('skill_type')})"
            )
    if verdict.errors:
        lines.append("")
        lines.append("Failing:")
        lines.extend(f"- {e}" for e in verdict.errors)
    if verdict.accepted:
        lines.append("")
        lines.append("<details><summary>Accepted by .skillevaluator-baseline.yaml "
                     f"({len(verdict.accepted)})</summary>\n")
        lines.extend(f"- {text} — {reason}" for text, reason in verdict.accepted)
        lines.append("\n</details>")
    lines.append("")
    text = "\n".join(lines)
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


ANNOTATE = False


def self_test(
    report: dict[str, Any], skill: str, skill_dir: Path, baseline: Path, on_disk: list[str], min_score: float
) -> list[str]:
    """Assert the gate still refuses each thing it exists to refuse.

    Every way this gate can break is silent: a matcher that stops matching accepts
    everything and reads exactly like a clean catalog. So it is run against a real
    report with one defect injected at a time, and each must turn the verdict red.
    Same reason the install job asserts that `verify` refuses an altered skill.
    """

    def run(mutated: dict[str, Any]) -> Verdict:
        return gate(mutated, skill, skill_dir, load_baseline(baseline, on_disk), min_score)

    problems: list[str] = []
    unmodified = run(json.loads(json.dumps(report)))
    if unmodified.errors:
        problems.append(f"the unmodified report for '{skill}' does not pass; self-test needs a passing skill")
        problems.extend(f"  {error}" for error in unmodified.errors)
        return problems

    def mutate(label: str, change) -> None:
        copy = json.loads(json.dumps(report))
        change(copy)
        if not run(copy).errors:
            problems.append(f"gate accepted a report with {label}")

    def new_high(r: dict[str, Any]) -> None:
        r["results"][0].setdefault("findings", []).append({
            "category": "SELFTEST", "severity": "high", "check_name": "injected",
            "message": "injected by --self-test", "file_path": str(skill_dir / "SKILL.md"),
        })

    def new_critical(r: dict[str, Any]) -> None:
        new_high(r)
        r["results"][0]["findings"][-1]["severity"] = "critical"

    def incomplete(r: dict[str, Any]) -> None:
        r["results"][0].setdefault("incomplete_scans", []).append("selftest-scanner")

    def profile(r: dict[str, Any]) -> None:
        r.setdefault("policy", {})["profile"] = "external"

    def unsafe_override(r: dict[str, Any]) -> None:
        r.setdefault("policy", {}).setdefault("severity_overrides", {})["CODE_RISK.*"] = "low"

    def no_results(r: dict[str, Any]) -> None:
        r["results"] = []

    def medium_error(r: dict[str, Any]) -> None:
        finding = {
            "category": "SELFTEST", "severity": "medium", "check_name": "injected",
            "message": "injected by --self-test", "file_path": str(skill_dir / "SKILL.md"),
        }
        result = r["results"][0]
        result.setdefault("findings", []).append(finding)
        result.setdefault("legacy", {}).setdefault("errors", []).append(legacy_string(finding))
        result["status"] = "failed"

    def unstructured(r: dict[str, Any]) -> None:
        result = r["results"][0]
        result["status"] = "failed"
        result["incomplete_scans"] = []
        result.setdefault("legacy", {}).setdefault("errors", []).append("[SELFTEST] unstructured error")

    def low_quality(r: dict[str, Any]) -> None:
        if not r.get("quality_summary"):
            r["quality_summary"] = [{}]
        r["quality_summary"][0]["overall_score"] = min_score - 1

    mutate("an unaccepted HIGH finding", new_high)
    mutate("an unaccepted CRITICAL finding", new_critical)
    mutate("an unaccepted MEDIUM finding SkillEvaluator counts as an error", medium_error)
    mutate("an unaccepted incomplete scanner", incomplete)
    mutate("the bare external profile", profile)
    mutate("a severity override not in SAFE_OVERRIDES", unsafe_override)
    mutate("no validator results", no_results)
    mutate("an unstructured validator error", unstructured)
    mutate("a quality score below the minimum", low_quality)

    # Stale entries: dropping every gated finding and incomplete scan must fail when the
    # baseline scopes an entry to this skill, because that entry then matches nothing.
    if any(in_scope(e, skill) for e in load_baseline(baseline, on_disk)):
        def clean(r: dict[str, Any]) -> None:
            for result in r["results"]:
                errors = set((result.get("legacy") or {}).get("errors") or [])
                result["findings"] = [f for f in result.get("findings") or [] if not is_gated(f, errors)]
                result["incomplete_scans"] = []
        mutate("a stale baseline entry", clean)
    else:
        problems.append(f"'{skill}' has no baseline entry, so the stale-entry check was not exercised")
    return problems


def main(argv: list[str] | None = None) -> int:
    global ANNOTATE
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report-dir", type=Path,
                        help="Directory `validate -o` wrote this skill's reports into.")
    parser.add_argument("--skill", help="Skill directory name under skills/.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help="Quality score below which the skill fails; pass the "
                        "--min-score validate ran with (default: %(default)s).")
    parser.add_argument("--check-baseline", action="store_true",
                        help="Only check that the baseline parses and that every scope "
                        "names a skill on disk. Runs on every change, so deleting a skill "
                        "the baseline names fails in that pull request, not the next one.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--annotate", action="store_true",
                        help="Emit GitHub ::error annotations for unaccepted findings.")
    parser.add_argument("--self-test", action="store_true",
                        help="Inject one defect at a time into this (passing) report and "
                        "fail unless the gate refuses every one. Needs a skill that has a "
                        "baseline entry, so the stale-entry ratchet is exercised too.")
    args = parser.parse_args(argv)
    ANNOTATE = args.annotate

    skills_dir = args.skills_dir.resolve()
    on_disk = sorted(p.name for p in skills_dir.iterdir() if (p / "SKILL.md").is_file())
    try:
        entries = load_baseline(args.baseline, on_disk)
    except ValueError as exc:
        print(f"::error title=SkillEvaluator baseline::{exc}" if ANNOTATE else f"error: {exc}", file=sys.stderr)
        return 2
    if args.check_baseline:
        print(f"baseline ok: {len(entries)} entries, every scope names a skill under skills/")
        return 0

    if not args.report_dir or not args.skill:
        parser.error("--report-dir and --skill are required unless --check-baseline is given")
    skill_dir = skills_dir / args.skill
    if not (skill_dir / "SKILL.md").is_file():
        print(f"error: {skill_dir} is not a skill", file=sys.stderr)
        return 2

    try:
        report_path = find_report(args.report_dir)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        verdict = Verdict(errors=[f"SkillEvaluator produced no usable report: {exc}"])
        if ANNOTATE:
            print(f"::error title=SkillEvaluator error::{args.skill}: {exc}")
        write_summary(args.skill, None, verdict)
        return 1

    if args.self_test:
        # Injected findings are expected to fail; annotating them would put errors on
        # the diff for defects that exist only in memory.
        ANNOTATE = False
        problems = self_test(report, args.skill, skill_dir, args.baseline, on_disk, args.min_score)
        for problem in problems:
            print(f"::error title=SkillEvaluator gate self-test::{problem}" if args.annotate else problem)
        if not problems:
            print(f"self-test ok: the gate refused every injected defect in '{args.skill}'")
        return 1 if problems else 0

    verdict = gate(report, args.skill, skill_dir, entries, args.min_score)
    write_summary(args.skill, report, verdict)
    return 1 if verdict.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

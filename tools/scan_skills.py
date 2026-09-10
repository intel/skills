#!/usr/bin/env python3
"""Security scan of every skill in skills/, via NVIDIA SkillSpector.

What this adds that the rest of the gate does not: validate_skills.py asks whether a
skill is well-formed and carries none of five content shapes written out by hand here.
SkillSpector asks a different question — is this safe to install — against 71 patterns
in 17 categories it maintains, plus an AST pass over the scripts a skill ships, YARA
signatures, and an OSV.dev lookup for the dependencies a skill names. The two overlap
in one place (curl-pipe-shell) and diverge everywhere else, which is the reason to run
both rather than fold one into the other.

This wrapper exists because SkillSpector scans one skill at a time and this repository
publishes 33 of them. It runs them concurrently, applies one shared policy baseline,
turns the per-skill JSON into a table a reviewer can read, and decides the exit code.

Static analysis only (--no-llm). The semantic stage needs a model credential, and a
check that cannot run in a fork is a check that does not gate a contribution — the same
reason the rest of the level 1 gate is keyless. Every report this prints therefore says
`static` in its scan mode: a low score here is not the same claim as a low score from a
full scan, and the summary says so rather than letting the number imply it.

Two thresholds, because the two kinds of skill in this catalog can act on a finding in
different ways. A skill written here can be fixed in the pull request that reports the
problem, so it is held to SkillSpector's LOW band: 20 or below, which is where all of
them already sit. An imported skill is upstream's text, kept byte-for-byte so
sync_external.py --check still passes, and editing it here would break the thing that
makes an import worth having; its repair has to land upstream and arrive through a moved
pin. So an import is held to the scanner's own install boundary — 50, above which
SkillSpector says DO_NOT_INSTALL — and its findings below that are reported rather than
failed. --strict-imports collapses the two, and is what a pin bump should be reviewed
with: the moment upstream's text changes here is the moment to read its findings again.

A scan that could not complete fails whichever kind it was. Findings this catalog has
reviewed and accepted are suppressed by .skillspector-baseline.yaml, which carries a
reason per rule; anything not in that file is either new or untriaged, and both are worth
a human's attention.

Unlike the rest of tools/, this needs SkillSpector installed — it is not stdlib-only and
not part of the offline local gate:

    pip install "skillspector @ git+https://github.com/NVIDIA/skillspector.git@<commit>"
    python3 tools/scan_skills.py

The commit CI pins lives in .github/workflows/skillspector.yml. Pin the same one
locally: the baseline suppresses findings by rule id, so a different scanner version can
report a rule this catalog has never triaged, and it is better to find that out in CI
than to have two people disagree about what a clean scan means.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_BASELINE = REPO_ROOT / ".skillspector-baseline.yaml"

# Written beside SKILL.md by tools/sync_external.py, and the same marker the mentions
# check and the link check use to decide that a body belongs to another team.
IMPORT_MARKER = ".source.json"

# Both thresholds are SkillSpector's own band edges rather than numbers picked here:
# 0-20 LOW/SAFE, 21-50 MEDIUM/CAUTION, 51-80 HIGH, 81+ CRITICAL, and HIGH upwards is
# what it calls DO_NOT_INSTALL (nodes/report.py). A skill written here has to stay in
# the band the scanner calls safe. An import has to stay out of the band the scanner
# says not to install, and the CAUTION band between the two is reported instead —
# see the module docstring for why the two differ. Gating on the score rather than on
# the exit code keeps the threshold visible and shows how much headroom is left.
DEFAULT_MAX_SCORE = 20
DEFAULT_MAX_SCORE_IMPORTED = 50

# Only the interesting half of the exit contract. 0 and 1 both mean the scan ran;
# 2 means it did not, and that is the case this script must not read as a pass.
EXIT_SCAN_FAILED = 2


@dataclass
class Result:
    """One skill's scan, or the reason there isn't one."""

    name: str
    imported: bool = False
    score: int = 0
    severity: str = ""
    recommendation: str = ""
    active: list[dict] = field(default_factory=list)
    suppressed: int = 0
    scanner_version: str = ""
    llm_used: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def origin(self) -> str:
        return "imported" if self.imported else "authored"


def skill_dirs(names: list[str]) -> list[Path]:
    """The skills to scan: every directory shipping a SKILL.md, or the named ones.

    Enumerating the tree rather than reading skills.yaml is deliberate — the two are
    already required to agree by validate_skills.py check 10, so parsing the catalog
    here would only add a second place for that agreement to be wrong.
    """
    if names:
        chosen = []
        for name in names:
            path = SKILLS_DIR / name
            if not (path / "SKILL.md").is_file():
                sys.exit(f"FAIL no skill named {name!r} under {SKILLS_DIR}")
            chosen.append(path)
        return chosen
    return sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


def scan(path: Path, baseline: Path | None, timeout: int) -> Result:
    """Scan one skill and parse its JSON report."""
    imported = (path / IMPORT_MARKER).is_file()
    cmd = ["skillspector", "scan", str(path), "--no-llm", "--format", "json"]
    if baseline is not None:
        cmd += ["--baseline", str(baseline)]

    env = dict(os.environ)
    # The static pass logs one WARNING per skipped semantic analyzer, three per skill,
    # and none of them is news: --no-llm is what asked for them to be skipped.
    env.setdefault("SKILLSPECTOR_LOG_LEVEL", "ERROR")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result(path.name, imported, error=f"scan timed out after {timeout}s")

    if proc.returncode >= EXIT_SCAN_FAILED:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return Result(
            path.name,
            imported,
            error=f"skillspector exited {proc.returncode}: "
            + (detail[-1] if detail else "no output"),
        )

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return Result(path.name, imported, error=f"report was not JSON: {exc}")

    risk = report.get("risk_assessment", {})
    metadata = report.get("metadata", {})
    result = Result(
        name=report.get("skill", {}).get("name", path.name),
        imported=imported,
        score=risk.get("score", 0),
        severity=risk.get("severity", ""),
        recommendation=risk.get("recommendation", ""),
        active=report.get("issues", []),
        suppressed=report.get("suppressed_count", 0),
        scanner_version=metadata.get("skillspector_version", ""),
        llm_used=bool(metadata.get("llm_available")),
    )
    # execution_successful is the scanner saying its own pipeline finished. A report
    # that admits it is partial must not be scored as if it were complete.
    if report.get("execution_successful") is False:
        result.error = "scan did not complete (execution_successful=false)"
    return result


def limit_for(result: Result, max_score: int, max_score_imported: int) -> int:
    """The threshold this skill is held to — see the module docstring."""
    return max_score_imported if result.imported else max_score


def render(results: list[Result], max_score: int, max_score_imported: int) -> str:
    """The table and the tally, as markdown that also reads as plain text."""
    lines = []
    versions = {r.scanner_version for r in results if r.scanner_version}
    mode = "full (LLM)" if any(r.llm_used for r in results) else "static only (--no-llm)"
    bar = (
        f"fail above {max_score}"
        if max_score == max_score_imported
        else f"fail above {max_score} (authored here) / {max_score_imported} (imported)"
    )
    lines.append(
        f"SkillSpector {', '.join(sorted(versions)) or 'unknown'} — "
        f"{len(results)} skill(s), scan mode: {mode}, {bar}"
    )
    lines.append("")
    lines.append("| skill | origin | score | severity | active | suppressed | limit |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(results, key=lambda r: (-r.score, r.name)):
        if not r.ok:
            lines.append(f"| {r.name} | {r.origin} | — | ERROR | {r.error} | | |")
            continue
        lines.append(
            f"| {r.name} | {r.origin} | {r.score} | {r.severity} | {len(r.active)} | "
            f"{r.suppressed} | {limit_for(r, max_score, max_score_imported)} |"
        )

    tally: collections.Counter = collections.Counter()
    for r in results:
        for issue in r.active:
            tally[(issue.get("id", "?"), issue.get("severity", "?"))] += 1
    if tally:
        lines.append("")
        lines.append("Findings left active — every one of these is untriaged:")
        lines.append("")
        lines.append("| rule | severity | count | skills |")
        lines.append("| --- | --- | --- | --- |")
        for (rule, severity), count in tally.most_common():
            where = sorted(
                f"{r.name}*" if r.imported else r.name
                for r in results
                if any(i.get("id") == rule for i in r.active)
            )
            shown = ", ".join(where[:4]) + (f" +{len(where) - 4}" if len(where) > 4 else "")
            lines.append(f"| {rule} | {severity} | {count} | {shown} |")
        if any(r.imported and r.active for r in results):
            lines.append("")
            lines.append(
                "`*` marks an imported body: the repair lands upstream and arrives here "
                "through a moved pin, so the entry to make is TRACKED, not a fix in place."
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skills", nargs="*", help="skill names; default is all of them")
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="suppression baseline (default: .skillspector-baseline.yaml); '' to ignore it",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=DEFAULT_MAX_SCORE,
        help=f"fail a skill written here scoring above this (default: {DEFAULT_MAX_SCORE})",
    )
    parser.add_argument(
        "--max-score-imported",
        type=int,
        default=DEFAULT_MAX_SCORE_IMPORTED,
        help=(
            "fail an imported skill scoring above this "
            f"(default: {DEFAULT_MAX_SCORE_IMPORTED}); an import's repair lands upstream"
        ),
    )
    parser.add_argument(
        "--strict-imports",
        action="store_true",
        help="hold imported skills to --max-score too; run this when a pin moves",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="concurrent scans (default: min(8, cpu count))",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="per-skill timeout in seconds"
    )
    parser.add_argument("--json", metavar="PATH", help="write the aggregate report here")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the table and exit 0 whatever it says",
    )
    parser.add_argument(
        "--summary-title",
        default="Skill security scan",
        help="heading for the $GITHUB_STEP_SUMMARY section this writes",
    )
    args = parser.parse_args()

    if shutil.which("skillspector") is None:
        print(
            "FAIL skillspector is not on PATH. Install the commit CI pins:\n"
            "     see .github/workflows/skillspector.yml",
            file=sys.stderr,
        )
        return 1

    baseline = Path(args.baseline) if args.baseline else None
    if baseline is not None and not baseline.is_file():
        print(f"FAIL baseline not found: {baseline}", file=sys.stderr)
        return 1

    imported_limit = (
        args.max_score if args.strict_imports else args.max_score_imported
    )

    targets = skill_dirs(args.skills)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(
            pool.map(lambda p: scan(p, baseline, args.timeout), targets)
        )

    table = render(results, args.max_score, imported_limit)
    print(table)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {
                        "skill": r.name,
                        "origin": r.origin,
                        "limit": limit_for(r, args.max_score, imported_limit),
                        "score": r.score,
                        "severity": r.severity,
                        "recommendation": r.recommendation,
                        "active": r.active,
                        "suppressed_count": r.suppressed,
                        "scanner_version": r.scanner_version,
                        "error": r.error,
                    }
                    for r in sorted(results, key=lambda r: r.name)
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # The step summary is where a reviewer reads this: the log is 33 scans deep and the
    # table is the only part worth finding.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"## {args.summary_title}\n\n{table}\n")

    failed = [
        r
        for r in results
        if not r.ok or r.score > limit_for(r, args.max_score, imported_limit)
    ]
    if failed:
        print("", file=sys.stderr)
        for r in failed:
            limit = limit_for(r, args.max_score, imported_limit)
            where = "an imported skill" if r.imported else "a skill written here"
            reason = r.error or (
                f"score {r.score} > {limit} for {where} ({r.recommendation})"
            )
            print(f"FAIL {r.name}: {reason}", file=sys.stderr)
        print(
            f"\n{len(failed)} skill(s) of {len(results)} need triage. Read the finding, "
            "then either fix the skill or add a reasoned rule to "
            ".skillspector-baseline.yaml — never a generated fingerprint.",
            file=sys.stderr,
        )
        if any(r.imported for r in failed):
            print(
                "An imported body cannot be edited here — sync_external.py --check "
                "compares it to the pinned commit byte for byte. Either land the fix "
                "upstream and move the pin, or record a TRACKED rule naming the "
                "upstream change that would remove it.",
                file=sys.stderr,
            )
        # --report-only says what it found and leaves the decision to the reader, so
        # the FAIL lines above are printed either way and only the exit code differs.
        return 0 if args.report_only else 1

    authored = sum(1 for r in results if not r.imported)
    print(
        f"\nOK {len(results)} skill(s) — {authored} authored here, none above "
        f"{args.max_score}; {len(results) - authored} imported, none above "
        f"{imported_limit}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

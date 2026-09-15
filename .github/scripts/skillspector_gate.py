#!/usr/bin/env -S uv run --quiet
#
# This file is derived from a workflow copied from [https://github.com/amd/skills]
# and is licensed under the MIT License.
# See THIRD-PARTY-PROGRAMS.txt in the root directory for the full text.
#
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Gate one skill's SkillSpector JSON report against the threshold its origin is held to.

Suppression is not done here. `skillspector scan --baseline` applies
`.skillspector-baseline.yaml` itself (scoped to this skill by
`.github/scripts/skillspector_baseline.py`), drops the findings its rules match, keeps
each rule's `reason` with them, and recomputes the risk score over what is left. So this
script reads a report in which `issues` is already the active set and `suppressed_count`
is the accepted one, and it decides two things: whether the score clears the limit, and
what to say about the findings that remain.

Reading the JSON rather than the markdown report is deliberate, and not a style choice.
The markdown per-finding block carries `**Message:**`, which is the *pattern name*
("Privileged Container / Container Escape"), and no matched text at all. Every `PE5`
finding in this catalog therefore has an identical message, while the four shapes this
repository accepts (`--device /dev/*`, `--ipc=host`, `--net=host`, `--network host`) and
the two it will not (`--privileged`, `nsenter`) sit in the same rule, the same file and
often the same paragraph -- measured, in `sglang-xpu-run/SKILL.md` (6 accepted, 6 active)
and `xpu-container-run/SKILL.md` (14 accepted, 4 active). Nothing keyed on the markdown
message can tell those apart. The JSON carries `finding` (the matched text) and
`location.start_line`, which is what makes the distinction expressible.

Two thresholds, because the two kinds of skill can act on a finding differently. A skill
written in this repository can be fixed in the pull request that reports the problem, so
it is held to 20 -- SkillSpector's own LOW/`SAFE` band. An imported skill is upstream's
text, kept byte-for-byte by `tools/sync_external.py --check`, so editing it here would
break the thing that makes an import worth having; its repair lands upstream and arrives
through a moved pin. It is held to 50, the boundary above which the scanner itself says
`DO_NOT_INSTALL`. Neither number was picked here; both are band edges from the scanner's
own report code. `.source.json`, written beside `SKILL.md` by `tools/sync_external.py`, is
the origin marker -- the same one the mentions check and the link check already use to
decide that a body belongs to another team.

An active HIGH or CRITICAL finding under the threshold is reported, not failed: it is
worth a reviewer's eyes every time it moves, which is why the baseline leaves it active,
but blocking on it would make "bump an import's pin" mean "bump the pin and then wait for
upstream". The score is the gate; the findings are the reading material.

`risk_assessment.recommendation` is deliberately not read. It is `CAUTION` for all 33
skills in this catalog, including ones scoring 0 with nothing active, because the
reference resolver counts filenames mentioned in prose as unresolved.
`risk_assessment.max_issue_severity` is computed over the active findings only and is the
signal that field looks like it should be.

Usage:

    uv run .github/scripts/skillspector_gate.py \
        --report reports/linux-perf.json \
        --skill linux-perf

Exits 0 when the scan ran and the score is within the limit, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"

# Written beside SKILL.md by tools/sync_external.py for a skill copied from another
# repository. Its absence is what "authored here" means.
IMPORT_MARKER = ".source.json"

# SkillSpector's own band edges (0-20 LOW, 21-50 MEDIUM, 51-80 HIGH, 81+ CRITICAL); see
# the module docstring for which origin is held to which and why.
DEFAULT_MAX_SCORE = 20
DEFAULT_MAX_SCORE_IMPORTED = 50

# Findings worth an annotation on the diff. Everything active is listed in the log and
# the step summary regardless.
NOTABLE_SEVERITIES = ("CRITICAL", "HIGH")

# GitHub renders at most ten annotations of each level per step and silently discards the
# rest, so the count that did not fit is printed instead of left to be inferred.
ANNOTATION_LIMIT = 10


def _data(text: str) -> str:
    """Escape a workflow command's message. GitHub's own encoding, not a guess."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _prop(text: str) -> str:
    """Escape a workflow command property, where a comma or colon would end it."""
    return _data(text).replace(":", "%3A").replace(",", "%2C")


def _one_line(text: str, width: int = 120) -> str:
    """Collapse a matched snippet to something that fits on one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def summarize(issue: dict) -> tuple[str, str, str, int, str]:
    """(severity, rule, file, line, matched text) for one active finding."""
    location = issue.get("location") or {}
    return (
        str(issue.get("severity", "")).upper(),
        str(issue.get("id", "?")),
        str(location.get("file", "")),
        int(location.get("start_line") or 0),
        _one_line(issue.get("finding", "")),
    )


def write_summary(lines: list[str]) -> None:
    """Append to the job summary when running in Actions; a no-op locally."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", required=True, type=Path, help="JSON report path.")
    parser.add_argument("--skill", required=True, help="Skill the report is for.")
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=f"Directory containing skill folders (default: {DEFAULT_SKILLS_DIR}).",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=DEFAULT_MAX_SCORE,
        help=f"Threshold for a skill authored here (default: {DEFAULT_MAX_SCORE}).",
    )
    parser.add_argument(
        "--max-score-imported",
        type=int,
        default=DEFAULT_MAX_SCORE_IMPORTED,
        help="Threshold for an imported skill "
        f"(default: {DEFAULT_MAX_SCORE_IMPORTED}).",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Emit ::error/::warning annotations so findings render on the diff.",
    )
    args = parser.parse_args(argv)

    skill_dir = args.skills_dir / args.skill
    if not (skill_dir / "SKILL.md").is_file():
        print(f"No skill named {args.skill!r} under {args.skills_dir}", file=sys.stderr)
        return 1

    if not args.report.is_file():
        print(f"Report not found: {args.report}", file=sys.stderr)
        return 1
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{args.report}: report was not JSON ({exc})", file=sys.stderr)
        return 1

    imported = (skill_dir / IMPORT_MARKER).is_file()
    origin = "imported" if imported else "authored"
    limit = args.max_score_imported if imported else args.max_score

    risk = report.get("risk_assessment") or {}
    score = int(risk.get("score") or 0)
    band = str(risk.get("severity") or "")
    worst = str(risk.get("max_issue_severity") or "NONE")
    active = [issue for issue in (report.get("issues") or []) if isinstance(issue, dict)]
    suppressed = int(report.get("suppressed_count") or 0)

    # The scanner saying its own pipeline did not finish. A partial report must not be
    # scored as if it were complete, however low the score it managed to compute.
    incomplete = report.get("execution_successful") is False

    headline = (
        f"{args.skill}: score {score}/{limit} ({origin}, band {band}), "
        f"worst active {worst}, {len(active)} active, {suppressed} suppressed"
    )
    print(headline)

    notable: list[tuple[str, str, str, int, str]] = []
    for issue in active:
        severity, rule, file, line, finding = summarize(issue)
        where = f"{file}:{line}" if line else file
        print(f"  {severity:8} {rule:6} {where}  {finding}")
        if severity in NOTABLE_SEVERITIES:
            notable.append((severity, rule, file, line, finding))

    failures: list[str] = []
    if incomplete:
        failures.append("the scan did not complete (execution_successful=false)")
    if score > limit:
        failures.append(
            f"score {score} is above the limit of {limit} for an {origin} skill"
        )

    if args.annotate:
        level = "error" if failures else "warning"
        budget = ANNOTATION_LIMIT
        # A failure has to be visible on the diff even when nothing below is annotated:
        # the per-finding annotations cover HIGH/CRITICAL only, so a skill that crosses
        # its limit on MEDIUM findings alone would otherwise be a red check whose reason
        # lives in the step summary. Anchored on SKILL.md because the score is the
        # skill's, not one finding's, and it takes one of the level's ten slots.
        if failures:
            print(
                f"::error file=skills/{args.skill}/SKILL.md,title="
                f"{_prop('SkillSpector gate')}::"
                f"{_data(headline + ' -- ' + '; '.join(failures))}"
            )
            budget -= 1
        # Over-threshold findings are the ones a reader has to act on, so they get the
        # slots GitHub will render; under-threshold notables follow.
        for severity, rule, file, line, finding in notable[:budget]:
            location = f"file=skills/{args.skill}/{file}"
            if line:
                location += f",line={line}"
            print(
                f"::{level} {location},title="
                f"{_prop(f'SkillSpector {rule} ({severity})')}::"
                f"{_data(f'{args.skill}: {finding}')}"
            )
        dropped = len(notable) - budget
        if dropped > 0:
            print(
                f"::notice::{_data(f'{dropped} further {level} annotation(s) for {args.skill} were not rendered; GitHub shows {ANNOTATION_LIMIT} per level per step. The full list is in the step log and the job summary.')}"
            )

    detail = [f"  - `{s}` `{r}` `{f}{f':{line}' if line else ''}` {t}" for s, r, f, line, t in notable]
    if failures:
        reason = "; ".join(failures)
        print(f"FAIL {args.skill}: {reason}", file=sys.stderr)
        write_summary(
            [f"### :x: {args.skill}: {reason}", "", f"- {headline}", *detail]
        )
        return 1

    mark = ":warning:" if notable else ":white_check_mark:"
    note = (
        f"{len(notable)} active HIGH/CRITICAL finding(s) below the {origin} limit "
        "of " + str(limit) + " -- reported, not blocking"
        if notable
        else f"within the {origin} limit of {limit}"
    )
    write_summary([f"### {mark} {args.skill}: {note}", "", f"- {headline}", *detail])
    print(f"PASS {args.skill}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

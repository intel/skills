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
a human's attention. A rule there may name the skills it applies to, which SkillSpector's
own format cannot express — this script is what enforces that, and baselines() is where
the reason it has to lives.

Three ways the result is read, because a log nobody opens is not a report. The table and
the tally go to the step summary always. --annotate puts each finding on its file and
line as a workflow command, so it renders in Files changed, and --detail names the skills
a change touched and prints their findings in full. --sarif writes one merged SARIF file
for GitHub code scanning: an alert per active finding, with history across commits and a
diff view of the ones a pull request introduced. A finding the baseline accepts is left
out of that file rather than uploaded as a dismissed alert, because code scanning ignores
SARIF suppressions — see merge_sarif.

Unlike the rest of tools/, this needs SkillSpector installed — it is not stdlib-only and
not part of the offline local gate. It reads YAML through pyyaml, which is the scanner's
own dependency rather than a new one:

    commit=$(sed -n 's/.*SKILLSPECTOR_COMMIT: \\([0-9a-f]\\{40\\}\\).*/\\1/p' \\
      .github/workflows/skillspector.yml)
    pip install "skillspector @ git+https://github.com/NVIDIA/skillspector.git@${commit}"
    python3 tools/scan_skills.py

Run the scan through this script rather than the CLI directly. The baseline's `skills:`
scoping is enforced here, and SkillSpector ignores fields it does not know, so pointing
the CLI at the shared file applies every rule to whatever is scanned — suppressing more
than CI does, which is the wrong direction for a check to be wrong in.

The commit CI pins lives in .github/workflows/skillspector.yml. Pin the same one
locally: the baseline suppresses findings by rule id, so a different scanner version can
report a rule this catalog has never triaged, and it is better to find that out in CI
than to have two people disagree about what a clean scan means.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # from the scanner's own dependencies (pyyaml), see the module docstring

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_BASELINE = REPO_ROOT / ".skillspector-baseline.yaml"

# Written beside SKILL.md by tools/sync_external.py, and the same marker the mentions
# check and the link check use to decide that a body belongs to another team.
IMPORT_MARKER = ".source.json"

# The one field in the baseline that SkillSpector does not define. A rule carrying it is
# applied only to the skills it names, and this script is what enforces that — see
# baselines() for why the scoping has to live here rather than in the scanner.
SCOPE_FIELD = "skills"

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

# GitHub renders at most ten annotations of each level per step and silently discards
# the rest, so the count that did not fit is printed instead of left to be inferred.
ANNOTATION_LIMIT = 10


def _data(text: str) -> str:
    """Escape a workflow command's message. GitHub's own encoding, not a guess."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _prop(text: str) -> str:
    """Escape a workflow command property, where a comma or colon would end it."""
    return _data(text).replace(":", "%3A").replace(",", "%2C")


@dataclass
class Result:
    """One skill's scan, or the reason there isn't one."""

    name: str
    imported: bool = False
    score: int = 0
    # Two different questions, and the table shows both. `severity` is the band the score
    # falls in, so it says how close this skill is to its threshold. `worst` is
    # max_issue_severity, which the scanner computes over the findings that survived the
    # baseline, so it says how bad the worst thing still on the table is. They disagree,
    # and the disagreement is the point: xpu-profile-unitrace scores 20 (LOW band) while
    # carrying an unsuppressed HIGH. Printing only the band reads as reassurance the scan
    # did not give. `recommendation` is deliberately not read — it is CAUTION for all 33
    # skills, including ones with a score of 0 and nothing active, because the reference
    # resolver counts prose filenames as unresolved and fails it closed from SAFE.
    severity: str = ""
    worst: str = ""
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


def in_scope(rule: dict, skill: str) -> bool:
    """Does this rule apply to this skill? A rule naming no skills applies to all."""
    scope = rule.get(SCOPE_FIELD)
    if scope is None:
        return True
    return any(fnmatch.fnmatch(skill, pattern) for pattern in scope)


def baselines(path: Path, targets: list[Path], work: Path) -> dict[str, Path]:
    """One baseline file per skill, holding only the rules that name it.

    SkillSpector's rule schema is `id`, `path`, `message` and `reason` (suppression.py),
    and `path` is matched against a finding's file *relative to the skill* — `SKILL.md`,
    `scripts/foo.py`. There is no field for the skill, so every rule in one shared file
    reaches all 33 of them: an entry accepting `exec()` in xpu-port's verifier would also
    accept it in a skill nobody has written yet. Measured on this tree, that is not
    hypothetical — a new skill shipping one `subprocess` call scores 0 with nothing active
    under a blanket AST4 entry, and 9 with the finding active once the entry names the two
    skills it was written for.

    So the scoping is done here, with a `skills:` list the scanner does not define, and
    each skill is scanned against a file holding only its own rules. Two alternatives were
    tried and neither works: the scanner refuses a shared baseline for a recursive
    multi-skill scan ("scan each sub-skill with its own baseline"), and a per-skill file
    inside skills/<name>/ fails sync_external.py --check on any imported skill, which is
    where nearly every finding is — "not part of the pinned upstream skill".

    One caveat this cannot fix: SkillSpector ignores fields it does not know, so running
    the CLI directly against the shared file applies every rule to whatever is scanned,
    which suppresses *more* than CI does. That is why CONTRIBUTING.md documents this
    script as the way to run the scan, and why the baseline says so at the top.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules") or []

    # A scope naming a skill that does not exist silences nothing and will keep silencing
    # nothing after the typo is forgotten, so it fails here rather than reading as cover.
    known = {p.name for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()}
    for position, rule in enumerate(rules, start=1):
        for pattern in rule.get(SCOPE_FIELD) or []:
            if not any(fnmatch.fnmatch(name, pattern) for name in sorted(known)):
                sys.exit(
                    f"FAIL {path.name}: rule {position} ({rule.get('id', '?')}) is scoped "
                    f"to {pattern!r}, which matches no skill under {SKILLS_DIR}"
                )

    out: dict[str, Path] = {}
    for target in targets:
        scoped = {key: value for key, value in data.items() if key != "rules"}
        scoped["rules"] = [
            {key: value for key, value in rule.items() if key != SCOPE_FIELD}
            for rule in rules
            if in_scope(rule, target.name)
        ]
        written = work / f"{target.name}.baseline.yaml"
        written.write_text(yaml.safe_dump(scoped, sort_keys=False), encoding="utf-8")
        out[target.name] = written
    return out


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


def scan(
    path: Path, baseline: Path | None, timeout: int, sarif_dir: Path | None = None
) -> Result:
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
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return Result(
            path.name,
            imported,
            error=f"skillspector exited {proc.returncode}: "
            + (tail[-1] if tail else "no output"),
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
        worst=risk.get("max_issue_severity", ""),
        active=report.get("issues", []),
        suppressed=report.get("suppressed_count", 0),
        scanner_version=metadata.get("skillspector_version", ""),
        llm_used=bool(metadata.get("llm_available")),
    )
    # execution_successful is the scanner saying its own pipeline finished. A report
    # that admits it is partial must not be scored as if it were complete.
    if report.get("execution_successful") is False:
        result.error = "scan did not complete (execution_successful=false)"

    # A second pass for SARIF, because the CLI writes one format per run and the score
    # this gate needs is in the JSON while the shape code scanning reads is in the
    # SARIF. Only asked for when the result is going to be uploaded, so a fork's pull
    # request still pays for one pass. Its own findings are the same ones; a failure
    # here is not allowed to fail the gate, because the gate already has its answer.
    if sarif_dir is not None and result.ok:
        sarif_cmd = [
            arg if arg != "json" else "sarif" for arg in cmd
        ] + ["--output", str(sarif_dir / f"{path.name}.sarif")]
        try:
            subprocess.run(
                sarif_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Caught for the reason stated above: without this the exception leaves
            # scan(), passes through the thread pool, and takes all 33 results with it —
            # a slow second pass would lose the answer the first pass already had. The
            # missing file is reported by the merge instead.
            pass
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
    # `worst active` is the severity of the worst finding the baseline did not accept, and
    # it is in the table because the score's band cannot answer that question: a skill can
    # sit in the LOW band and still carry an unsuppressed HIGH, and reading LOW as safety
    # is exactly the mistake a gate should not invite.
    lines.append(
        "| skill | origin | score | band | worst active | active | suppressed | limit |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(results, key=lambda r: (-r.score, r.name)):
        if not r.ok:
            lines.append(f"| {r.name} | {r.origin} | — | ERROR | — | {r.error} | | |")
            continue
        lines.append(
            f"| {r.name} | {r.origin} | {r.score} | {r.severity} | {r.worst or '—'} | "
            f"{len(r.active)} | {r.suppressed} | "
            f"{limit_for(r, max_score, max_score_imported)} |"
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


def merge_sarif(sarif_dir: Path, out: Path) -> tuple[int, set[str]]:
    """One SARIF file for the catalog, with paths GitHub can find.

    Three fixes are needed on the per-skill files. SkillSpector reports a path relative
    to the skill it was pointed at (`SKILL.md`), and code scanning resolves paths from
    the repository root, so every URI gains its `skills/<name>/` prefix — without it the
    alert lands on nothing and renders nowhere. 33 files is more than the 20 SARIF
    uploads GitHub accepts for one commit, so the runs are merged into one: same tool
    driver, rules unioned by id, results concatenated.

    And a suppressed finding is dropped rather than carried. SkillSpector writes each one
    out with the `reason` from .skillspector-baseline.yaml in
    `suppressions[].justification`, which reads like it should arrive as a dismissed
    alert holding the sentence that accepted it. Measured against the API, it does not:
    code scanning ignores `suppressions` on an uploaded SARIF whether the kind is
    `external` or `inSource`, and every accepted finding becomes an open alert. On this
    tree that is 197 alerts nobody is going to act on burying the 17 that want reading,
    which is how a security tab stops being read at all. The reasons stay in the baseline
    file, and the counts stay in the table — `suppressed` is a column in it.

    Returns the number of findings written and the skills they were read from. The caller
    checks that second value against the skills that were scanned, because the count of
    findings cannot: dropping the suppressed ones means a skill whose findings were all
    accepted contributes nothing either way, so a missing or truncated per-skill file
    looks exactly like a clean skill. Silently uploading fewer alerts than the table
    reports is the one failure this stage must not have.
    """
    driver: dict = {}
    rules: dict[str, dict] = {}
    results: list[dict] = []
    parsed: set[str] = set()
    for path in sorted(sarif_dir.glob("*.sarif")):
        skill = path.stem
        try:
            run = json.loads(path.read_text(encoding="utf-8"))["runs"][0]
        except (json.JSONDecodeError, KeyError, IndexError, OSError):
            continue
        parsed.add(skill)
        driver = driver or run.get("tool", {}).get("driver", {})
        for rule in run.get("tool", {}).get("driver", {}).get("rules", []):
            rules.setdefault(rule.get("id", ""), rule)
        for result in run.get("results", []):
            if result.get("suppressions"):
                continue
            for location in result.get("locations", []):
                artifact = location.get("physicalLocation", {}).get(
                    "artifactLocation", {}
                )
                if "uri" in artifact:
                    artifact["uri"] = f"skills/{skill}/{artifact['uri']}"
            results.append(result)

    driver = dict(driver)
    driver["rules"] = list(rules.values())
    out.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "$schema": (
                    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/"
                    "sarif-2.1/schema/sarif-schema-2.1.0.json"
                ),
                "runs": [{"tool": {"driver": driver}, "results": results}],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(results), parsed


def detail(results: list[Result], names: set[str]) -> str:
    """Every active finding in the named skills, in full.

    The table says how many; this says which, where, and what the scanner wants done
    about it. Scoped to the skills a pull request touched, because a reviewer reading a
    diff of two skills should not have to find their two findings among the catalog's.
    """
    chosen = [r for r in results if r.name in names and r.active]
    if not chosen:
        return ""
    lines = ["", "### Findings in the skills this change touches", ""]
    for r in chosen:
        lines.append(f"<details><summary><code>{r.name}</code> — {len(r.active)} "
                     f"finding(s), score {r.score} {r.severity}</summary>")
        lines.append("")
        for issue in r.active:
            loc = issue.get("location") or {}
            where = loc.get("file") or "?"
            line_no = loc.get("start_line")
            lines.append(
                f"**`{issue.get('id', '?')}` {issue.get('severity', '?')}** "
                f"(confidence {issue.get('confidence', '?')}) — "
                f"`skills/{r.name}/{where}`"
                + (f":{line_no}" if line_no else "")
            )
            lines.append("")
            matched = (issue.get("finding") or "").strip()
            if matched:
                lines.append(f"> matched: `{matched[:200]}`")
                lines.append("")
            for label, key in (("What it says", "explanation"), ("Remediation", "remediation")):
                text = (issue.get(key) or "").strip()
                if text:
                    lines.append(f"{label}: {text}")
                    lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def annotate(
    results: list[Result],
    max_score: int,
    max_score_imported: int,
    first: set[str] | None = None,
) -> None:
    """Put findings on the diff, using GitHub's workflow commands.

    A log line is read by whoever opens the log. An annotation is read by whoever opens
    the pull request: it renders against the file and line in Files changed, provided
    that file is part of the diff. Error where the skill is over its threshold, warning
    where it is under it — same distinction the exit code makes.

    The ten GitHub will render go to what a reviewer of this change can act on: a skill
    over its threshold first, then the skills the change touched (*first*, the only ones
    whose files are in the diff at all), and only then the rest by score.
    """
    first = first or set()
    budget = ANNOTATION_LIMIT
    dropped = 0

    def priority(r: Result) -> tuple:
        over = not r.ok or r.score > limit_for(r, max_score, max_score_imported)
        return (0 if over else 1, 0 if r.name in first else 1, -r.score, r.name)

    for r in sorted(results, key=priority):
        over = r.score > limit_for(r, max_score, max_score_imported)
        level = "error" if over or not r.ok else "warning"
        if not r.ok:
            print(f"::error file=skills/{r.name}/SKILL.md,line=1,"
                  f"title=SkillSpector scan failed::{_data(r.error)}")
            continue
        for issue in r.active:
            if budget <= 0:
                dropped += 1
                continue
            budget -= 1
            loc = issue.get("location") or {}
            path = f"skills/{r.name}/{loc.get('file') or 'SKILL.md'}"
            line_no = loc.get("start_line") or 1
            rule = issue.get("id", "?")
            matched = (issue.get("finding") or issue.get("pattern") or "").strip()
            body = (
                f"{issue.get('severity', '?')} {rule}: {matched[:160]} — "
                f"{(issue.get('explanation') or '').strip()[:300]} "
                f"[skill scores {r.score}, limit "
                f"{limit_for(r, max_score, max_score_imported)}]"
            )
            print(
                f"::{level} file={path},line={line_no},"
                f"title={_prop(f'SkillSpector {rule} ({r.name})')}::{_data(body)}"
            )
    if dropped:
        # GitHub renders only the first ANNOTATION_LIMIT of each level, so say out loud
        # what did not get one rather than let the diff imply there was nothing else.
        print(
            f"::notice::{dropped} further finding(s) have no annotation — GitHub renders "
            f"at most {ANNOTATION_LIMIT} per step. The step summary lists all of them."
        )


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
    parser.add_argument(
        "--detail",
        metavar="SKILL",
        nargs="*",
        default=[],
        help="print every active finding in these skills in full, under the table",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="emit GitHub workflow commands so findings land on the diff",
    )
    parser.add_argument(
        "--sarif",
        metavar="PATH",
        help="also write the active findings here as one merged SARIF file, for "
        "code scanning (costs a second pass over every skill)",
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
    incomplete: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        sarif_dir = work / "sarif" if args.sarif else None
        if sarif_dir is not None:
            sarif_dir.mkdir()
        # Each skill is scanned against a baseline holding only the rules that name it.
        scoped = baselines(baseline, targets, work) if baseline is not None else {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results = list(
                pool.map(
                    lambda p: scan(
                        p, scoped.get(p.name, baseline), args.timeout, sarif_dir
                    ),
                    targets,
                )
            )
        if sarif_dir is not None:
            count, parsed = merge_sarif(sarif_dir, Path(args.sarif))
            expected = {p.name for p, r in zip(targets, results) if r.ok}
            incomplete = sorted(expected - parsed)
            print(
                f"Wrote {args.sarif}: {count} active finding(s) from {len(parsed)} of "
                f"{len(expected)} scanned skill(s). Findings the baseline accepts are "
                "left out — code scanning ignores SARIF suppressions."
            )

    table = render(results, args.max_score, imported_limit)
    if args.detail:
        table += "\n" + detail(results, set(args.detail))
    print(table)

    if args.annotate:
        annotate(results, args.max_score, imported_limit, set(args.detail))

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
                        "max_issue_severity": r.worst,
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
                f"score {r.score} > {limit} for {where}, worst active finding "
                f"{r.worst or 'unknown'}"
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

    # A skill that was scanned but produced no readable SARIF would arrive at code
    # scanning as a skill with nothing to say, and a green run would then claim a coverage
    # it does not have. Failing is the only honest answer, and it is separate from the
    # threshold failures above: the table is right and the alerts are not.
    if incomplete:
        print("", file=sys.stderr)
        print(
            f"FAIL the merged SARIF is missing {len(incomplete)} scanned skill(s): "
            f"{', '.join(incomplete)}. Code scanning would show fewer alerts than the "
            "table reports. Re-run, or drop --sarif to gate on the table alone.",
            file=sys.stderr,
        )

    if failed or incomplete:
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

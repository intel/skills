#!/usr/bin/env python3
"""Run the three-arm Level 2 differential for one skill and report the delta.

    python3 tools/compare_harbor_skill.py --skill dpnp-quickstart \\
        --model "<model>" --attempts 3 --previous-ref main

Three arms, identical in every respect except the skill:

    no_skill    the task alone. The baseline, never optional.
    previous    the skill as it exists at --previous-ref. Skipped when the skill
                is new at that ref, which is the honest answer for a first run.
    candidate   the skill in the working tree.

The gate is candidate - no_skill. A skill that does not beat the no-skill arm has
not shown that it does anything, however well written it is.

Why a wrapper instead of three harbor invocations by hand:

  * Both skill arms are snapshotted before any arm starts — the candidate copied
    out of the working tree, 'previous' exported with `git archive`. An edit made
    while the run is in flight then cannot change what was compared, and a
    half-hour comparison is worth more than the convenience of editing during it.
  * The arms are constructed from one set of parameters, so they cannot silently
    differ in attempts, model, task list, or timeouts. A differential where the
    arms differ in two things measures neither.
  * The task list comes from evaluation/harbor/suites.json, so an arm cannot be
    run on a subset of the suite by accident.

Harbor owns execution, the environment, and the result format. This script owns
the comparison and refuses to state a delta it cannot stand behind: if any arm
lost a trial to an error, the delta is reported as unusable rather than adjusted.

Use --dry-run to print the commands without starting anything. Nothing here needs
credentials except the arms themselves, which need whatever your --agent needs.

Every task is `no-network`, and that baseline is in force while harbor installs the
agent inside the environment, so a scored arm needs two narrow holes: the agent's
installation hosts (built in for --agent claude-code) and, via --allow-agent-host,
the inference endpoint. Nothing wider — a public baseline would let the no_skill arm
read the documentation the skill is made of.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from check_harbor_job import collect_trials, reward_of

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITES_PATH = REPO_ROOT / "evaluation" / "harbor" / "suites.json"
TASKS_PATH = REPO_ROOT / "evaluation" / "harbor" / "tasks"
INSTRUCTIONS_DIR = REPO_ROOT / "evaluation" / "harbor" / "instructions"

BASELINE = "no_skill"
ARMS = (BASELINE, "previous", "candidate")

# Every task declares `network_mode = "no-network"`, and that baseline applies from
# the moment the environment starts — which is *before* harbor installs the agent
# inside it. So a scored arm cannot run without punching two holes in it:
#
#   the setup hosts   harbor's claude-code agent apt-gets curl and procps, then
#                     fetches the CLI from downloads.claude.ai. With no network the
#                     trial dies in _setup_agent with `E: Unable to locate package
#                     curl` — an agent that never started, not a skill that failed.
#   the model host    the agent's own inference endpoint, which is the one thing it
#                     must reach while it works.
#
# Everything else stays blocked, and that is the point: with a `public` baseline the
# no_skill arm could read the same vendor documentation the skill contains, and the
# differential would measure the internet instead of the skill.
CLAUDE_CODE_SETUP_HOSTS = (
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "downloads.claude.ai",
    "claude.ai",
)


def die(message: str) -> None:
    sys.exit(f"FAIL {message}")


def rel(path: Path) -> str:
    """A repo-relative POSIX path, for a command that must read the same everywhere.

    Every harbor argument goes through this. On Windows str(Path) yields
    backslashes, which a Linux runner would take as part of the filename, so a
    command copied out of a report or a CI log would fail there and nowhere else.
    """
    return path.relative_to(REPO_ROOT).as_posix()


def remove_tree(path: Path) -> None:
    """rmtree that tolerates a read-only file or a briefly held directory handle.

    On Windows a sync client or scanner can hold a handle just long enough for
    rmdir to fail with a permission error; retrying succeeds. Without this the
    snapshot step fails on the second run in a checkout that happens to live in a
    synced folder, which looks like a bug in the comparison rather than the disk.
    """
    def clear_readonly(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    for attempt in range(5):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=clear_readonly)
            return
        except OSError:
            time.sleep(0.3 * (attempt + 1))
    if path.exists():
        die(f"cannot remove {rel(path)}; close anything holding it and re-run")


def git(*arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=not binary,
    )
    if result.returncode != 0:
        return None
    return result.stdout if binary else result.stdout.strip()


def suite_tasks(skill: str) -> list[str]:
    """Implemented task names for a skill, from the manifest rather than the disk.

    Reading the manifest and not tasks/ is deliberate: a task directory that no
    suite claims must not quietly join a measurement. validate_skills.py already
    fails when the two disagree.
    """
    try:
        document = json.loads(SUITES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read {SUITES_PATH.name}: {exc}")
    for suite in document.get("suites") or []:
        if suite.get("skill") != skill:
            continue
        if suite.get("evaluated_here") is False:
            die(f"{skill} is evaluated elsewhere; suites.json says so")
        names = [
            task["name"] for task in suite.get("tasks") or [] if task.get("status") == "implemented"
        ]
        if not names:
            die(f"{skill} has no implemented tasks in suites.json")
        return names
    die(f"no suite for {skill!r} in {SUITES_PATH.name}")
    return []


def snapshot_candidate(skill: str, destination: Path) -> None:
    shutil.copytree(REPO_ROOT / "skills" / skill, destination / skill)


def snapshot_previous(skill: str, ref: str, destination: Path) -> bool:
    """Export skills/<skill>/ at ref. False when it does not exist there."""
    archive = git("archive", "--format=tar", ref, f"skills/{skill}", binary=True)
    if not archive:
        return False
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(f"skills/{skill}/")]
        if not members:
            return False
        # filter="data" refuses absolute paths, "..", links, and device nodes. The
        # archive comes from our own git history, but an extractor that trusts its
        # input is the kind of thing a scanner flags and a reviewer has to re-argue.
        tar.extractall(destination, members=members, filter="data")
    (destination / "skills" / skill).rename(destination / skill)
    remove_tree(destination / "skills")
    return True


def environment_hosts(args: argparse.Namespace) -> list[str]:
    """Hosts reachable from environment start, so agent installation can succeed.

    The claude-code defaults are included unless --no-setup-hosts says otherwise:
    without them the failure is a stack trace inside harbor's agent installer, which
    reads like a broken harness rather than a missing allowlist.
    """
    hosts: list[str] = []
    if args.agent == "claude-code" and not args.no_setup_hosts:
        hosts += list(CLAUDE_CODE_SETUP_HOSTS)
    hosts += list(args.allow_environment_host)
    return list(dict.fromkeys(hosts))


def build_command(
    args: argparse.Namespace, arm: str, tasks: list[str], skill_dir: Path | None, job: str
) -> list[str]:
    command = [args.harbor, "run", "--path", rel(TASKS_PATH)]
    for task in tasks:
        command += ["--include-task-name", task]
    command += [
        "--agent",
        args.agent,
        "--model",
        args.model,
        "--n-attempts",
        str(args.attempts),
        "--job-name",
        job,
        "--jobs-dir",
        args.jobs_dir,
        "--n-concurrent",
        str(args.n_concurrent),
        "--yes",
    ]
    # Both hole lists are built here, in the one place every arm goes through, for
    # the same reason the model and attempt count are: an allowlist that differs
    # between arms is a second difference, and a differential with two differences
    # measures neither.
    for host in environment_hosts(args):
        command += ["--allow-environment-host", host]
    for host in args.allow_agent_host:
        command += ["--allow-agent-host", host]
    if skill_dir is not None:
        command += [
            "--skill",
            rel(skill_dir),
            "--extra-instruction-path",
            rel(INSTRUCTIONS_DIR / f"use-{args.skill}.md"),
        ]
    return command


# What a trial cost, beyond its reward. Ordered as the report prints them.
EFFORT_FIELDS = ("seconds", "output_tokens", "input_tokens", "cache_tokens", "cost_usd")


def parse_stamp(value: str) -> datetime:
    """A Harbor timestamp.

    Trial-level stamps end in `Z` and job-level ones do not, so a single strptime
    format fails on one of them: `'2026-08-26T19:43:42.311321' does not match
    '%Y-%m-%dT%H:%M:%S.%fZ'`. Strip the suffix instead of carrying two formats.
    """
    return datetime.fromisoformat(value.rstrip("Z"))


def phase_seconds(phase: dict) -> float | None:
    started, finished = phase.get("started_at"), phase.get("finished_at")
    if not (started and finished):
        return None
    try:
        return (parse_stamp(finished) - parse_stamp(started)).total_seconds()
    except (TypeError, ValueError):
        return None


def trial_effort(trial: dict) -> dict:
    """Time and money for one trial, from Harbor's own per-trial accounting.

    The job-level `stats` block carries no cost or token totals, so reading it
    silently reports "cost not reported" for every arm — which is how a comparison
    ends up stating a reward delta with no idea what the reward cost. These come from
    `agent_result`, which is the provider's own accounting rather than an estimate.

    Time is `agent_execution` only. Environment build and agent installation are
    excluded deliberately: they are identical work in both arms and dominated by
    Docker and apt, so including them would bury the difference the skill makes.
    """
    agent = trial.get("agent_result") or {}
    return {
        "seconds": phase_seconds(trial.get("agent_execution") or {}),
        "output_tokens": agent.get("n_output_tokens"),
        "input_tokens": agent.get("n_input_tokens"),
        "cache_tokens": agent.get("n_cache_tokens"),
        "cost_usd": agent.get("cost_usd"),
    }


def add_effort(into: dict, effort: dict) -> None:
    for field in EFFORT_FIELDS:
        value = effort.get(field)
        if isinstance(value, (int, float)):
            into[field] = into.get(field, 0) + value
            into[f"{field}_n"] = into.get(f"{field}_n", 0) + 1


def summarize(job_dir: Path) -> dict:
    """Per-task mean reward, what it cost, and what would make the mean untrustworthy."""
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return {
            "missing": True,
            "per_task": {},
            "errors": ["no result.json"],
            "mean": None,
            "effort": {},
            "per_task_effort": {},
            "trials": 0,
        }

    per_task: dict[str, list[float]] = {}
    errors: list[str] = []
    effort: dict = {}
    per_task_effort: dict[str, dict] = {}
    scored = 0
    for trial in collect_trials(result_path):
        name = trial.get("task_name") or "<unnamed>"
        if trial.get("exception_info"):
            kind = (trial["exception_info"] or {}).get("exception_type") or "unknown"
            errors.append(f"{name}: {kind}")
            continue
        reward = reward_of(trial)
        if reward is None:
            errors.append(f"{name}: no reward emitted")
            continue
        per_task.setdefault(name, []).append(reward)
        # Only trials that produced a reward are counted, so the cost table and the
        # reward table describe the same set of trials.
        scored += 1
        measurements = trial_effort(trial)
        add_effort(effort, measurements)
        add_effort(per_task_effort.setdefault(name, {}), measurements)

    means = {name: sum(values) / len(values) for name, values in per_task.items()}
    return {
        "missing": False,
        "per_task": means,
        "attempts": {name: len(values) for name, values in per_task.items()},
        "errors": errors,
        "mean": (sum(means.values()) / len(means)) if means else None,
        "effort": effort,
        "per_task_effort": per_task_effort,
        "trials": scored,
    }


def format_number(field: str, value) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    if field == "cost_usd":
        return f"${value:.3f}"
    if field == "seconds":
        return f"{value:.1f} s"
    return f"{value:,.0f}"


def ratio(candidate, baseline) -> str:
    """How much more the candidate spent, as a multiple of the baseline."""
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)):
        return "—"
    if baseline == 0:
        return "—" if candidate == 0 else "n/a (baseline 0)"
    return f"{candidate / baseline:.2f}x"


def cost_section(present: list[str], results: dict, tasks: list[str]) -> list[str]:
    """What the reward above cost, per arm and per task.

    A skill is not free even when it does not help. The first real run of this suite
    tied at reward 1.000 while the candidate arm spent 1.7x the wall-clock and 1.9x
    the output tokens, and a report that prints only the reward delta hides that
    entirely — which is the argument for the gate being a threshold rather than "did
    not regress".
    """
    labels = {
        "seconds": "agent wall-clock",
        "output_tokens": "output tokens",
        "input_tokens": "input tokens",
        "cache_tokens": "of which cache reads",
        "cost_usd": "cost",
    }
    lines = [
        "",
        "## Cost and effort",
        "",
        "Totals over the scored trials of each arm — trials that errored are excluded, so",
        "this describes the same trials as the reward table. Wall-clock is",
        "`agent_execution` only: environment build and agent installation are identical",
        "work in both arms and would bury the difference. Every figure is the provider's",
        "own accounting from each trial's `agent_result`, not an estimate.",
        "",
        "| | " + " | ".join(f"`{arm}`" for arm in present) + " | candidate ÷ no_skill |",
        "|---" * (len(present) + 2) + "|",
        "| scored trials | "
        + " | ".join(str(results[arm].get("trials") or 0) for arm in present)
        + " | |",
    ]
    for field in EFFORT_FIELDS:
        cells = [format_number(field, results[arm]["effort"].get(field)) for arm in present]
        against = (
            ratio(results["candidate"]["effort"].get(field), results[BASELINE]["effort"].get(field))
            if "candidate" in results and BASELINE in results
            else "—"
        )
        lines.append(f"| {labels[field]} | " + " | ".join(cells) + f" | {against} |")

    if "candidate" in results and BASELINE in results:
        rows = []
        for task in sorted(tasks):
            baseline = results[BASELINE]["per_task_effort"].get(task) or {}
            candidate = results["candidate"]["per_task_effort"].get(task) or {}
            if not baseline or not candidate:
                continue
            cells = []
            for field in ("seconds", "output_tokens", "cost_usd"):
                # Per attempt, so a task with fewer attempts stays comparable.
                left = baseline.get(field)
                right = candidate.get(field)
                left_n = baseline.get(f"{field}_n") or 0
                right_n = candidate.get(f"{field}_n") or 0
                if not (left_n and right_n):
                    cells.append("—")
                    continue
                left, right = left / left_n, right / right_n
                cells.append(
                    f"{format_number(field, left)} → {format_number(field, right)}"
                    + (f" ({(right - left) / left:+.0%})" if left else "")
                )
            rows.append(f"| `{task}` | " + " | ".join(cells) + " |")
        if rows:
            lines += [
                "",
                "Mean per attempt, `no_skill` → `candidate`:",
                "",
                "| Task | agent wall-clock | output tokens | cost |",
                "|---|---|---|---|",
                *rows,
            ]
    return lines


def format_report(args: argparse.Namespace, tasks: list[str], results: dict, provenance: dict) -> str:
    present = [arm for arm in ARMS if arm in results]
    lines = [
        f"# {args.skill} — Level 2 differential",
        "",
        f"- agent `{args.agent}`, model `{args.model}`, {args.attempts} attempt(s) per task",
        f"- tasks from `evaluation/harbor/suites.json`: {len(tasks)}",
        f"- repository at `{provenance['head']}`"
        + (", **working tree dirty**" if provenance["dirty"] else ""),
        f"- previous arm: {provenance['previous']}",
        "",
        "## Per-task mean reward",
        "",
        "| Task | " + " | ".join(present) + " | candidate − no_skill |",
        "|---" * (len(present) + 2) + "|",
    ]

    ceilings, deltas = [], []
    for task in sorted(tasks):
        cells = []
        for arm in present:
            value = results[arm]["per_task"].get(task)
            cells.append("—" if value is None else f"{value:.3f}")
        baseline = results[BASELINE]["per_task"].get(task)
        candidate = results["candidate"]["per_task"].get(task)
        if baseline is None or candidate is None:
            delta = "—"
        else:
            delta = f"{candidate - baseline:+.3f}"
            deltas.append(candidate - baseline)
        scores = [results[arm]["per_task"].get(task) for arm in present]
        if scores and all(score == 1.0 for score in scores if score is not None):
            ceilings.append(task)
        lines.append(f"| `{task}` | " + " | ".join(cells) + f" | {delta} |")

    suite_means = {arm: results[arm]["mean"] for arm in present}
    lines += ["", "## Suite mean", ""]
    for arm in present:
        value = suite_means[arm]
        lines.append(f"- `{arm}`: " + ("—" if value is None else f"{value:.3f}"))

    gate = None
    if suite_means.get(BASELINE) is not None and suite_means.get("candidate") is not None:
        gate = suite_means["candidate"] - suite_means[BASELINE]
        lines += ["", f"**candidate − no_skill = {gate:+.3f}** (gate: >= {args.min_delta:+.3f})"]

    all_errors = {arm: results[arm]["errors"] for arm in present if results[arm]["errors"]}
    if all_errors:
        lines += [
            "",
            "## The delta above is not usable",
            "",
            "An arm lost trials to harness errors, so the arms no longer differ only by",
            "the skill. Fix these and re-run rather than interpreting the number.",
            "",
        ]
        for arm, errors in all_errors.items():
            for error in errors:
                lines.append(f"- `{arm}`: {error}")

    if ceilings:
        lines += [
            "",
            "## Tasks at a ceiling",
            "",
            "Every arm scored 1.0, so these say nothing about the skill. They remain",
            "useful as smoke and regression coverage, but they inflate the suite mean",
            "toward zero delta and must not be counted as evidence either way.",
            "",
        ]
        for task in ceilings:
            lines.append(f"- `{task}`")

    lines += cost_section(present, results, tasks)

    lines += [
        "",
        "## What this does not measure",
        "",
        "Both skill arms run with an explicit treatment instruction, so the agent is",
        "told the skill is present. This isolates the skill's body and deliberately",
        "excludes whether the `description` would have led the agent to it unprompted.",
        "A green delta here is not evidence that the skill is discoverable.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skill", required=True, help="skill name, as in skills/<name>/")
    parser.add_argument("--model", required=True, help="model identifier passed to harbor")
    parser.add_argument("--agent", default="claude-code", help="harbor agent (default claude-code)")
    parser.add_argument("--attempts", type=int, default=3, help="attempts per task, per arm")
    parser.add_argument(
        "--previous-ref",
        default=None,
        help="git ref for the 'previous' arm. Omit on a skill's first evaluation.",
    )
    parser.add_argument("--jobs-dir", default="harbor-jobs")
    parser.add_argument("--job-prefix", default=None, help="default: the skill name")
    parser.add_argument("--n-concurrent", type=int, default=2)
    parser.add_argument("--harbor", default="harbor", help="path to the harbor executable")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.10,
        help="gate on candidate - no_skill (default 0.10, matching suites.json)",
    )
    parser.add_argument(
        "--allow-agent-host",
        action="append",
        default=[],
        metavar="HOST",
        help="host the agent may reach while it works — its inference endpoint. "
        "Repeatable. Applied identically to every arm.",
    )
    parser.add_argument(
        "--allow-environment-host",
        action="append",
        default=[],
        metavar="HOST",
        help="host reachable from environment start, which is where the agent is "
        "installed. Repeatable, and added to the built-in claude-code list.",
    )
    parser.add_argument(
        "--no-setup-hosts",
        action="store_true",
        help="drop the built-in claude-code installation allowlist "
        f"({', '.join(CLAUDE_CODE_SETUP_HOSTS)}) — for an image that already carries the agent",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="re-read the job directories of a finished comparison and rewrite the "
        "report, running no arms. For adding a section to a report without paying "
        "for the run again — the jobs are the record, the report is derived from them.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit nonzero when the gate is not met or an arm errored",
    )
    args = parser.parse_args()

    skill_dir = REPO_ROOT / "skills" / args.skill
    if not (skill_dir / "SKILL.md").is_file():
        die(f"no skills/{args.skill}/SKILL.md")
    treatment = INSTRUCTIONS_DIR / f"use-{args.skill}.md"
    if not treatment.is_file():
        die(f"no treatment instruction at {rel(treatment)}")

    tasks = suite_tasks(args.skill)
    prefix = args.job_prefix or args.skill
    jobs_dir = REPO_ROOT / args.jobs_dir
    arms_dir = jobs_dir / f"{prefix}-arms"

    if args.report_only:
        results = {}
        for arm in ARMS:
            job_dir = jobs_dir / f"{prefix}-{arm.replace('_', '-')}"
            if (job_dir / "result.json").is_file():
                results[arm] = summarize(job_dir)
            elif arm == BASELINE:
                die(f"no {rel(job_dir)}/result.json — nothing to report on")
        provenance = {
            "head": git("rev-parse", "HEAD") or "unknown",
            "dirty": bool(git("status", "--porcelain", "--", f"skills/{args.skill}")),
            # The working tree has moved on since the run, so it cannot be read as the
            # provenance of these jobs. Say so rather than implying it was checked.
            "previous": "read from existing job directories; --report-only does not"
            " re-resolve --previous-ref",
        }
        tasks = sorted({task for arm in results for task in results[arm]["per_task"]}) or tasks
        report = format_report(args, tasks, results, provenance)
        report_path = args.report_path or jobs_dir / f"{prefix}-comparison.md"
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"report written to {report_path}")
        return 0

    # Snapshot both skill arms before any arm runs, so an edit mid-comparison
    # cannot change what was compared.
    remove_tree(arms_dir)
    (arms_dir / "candidate").mkdir(parents=True)
    snapshot_candidate(args.skill, arms_dir / "candidate")

    previous_note = "not run (no --previous-ref given; treat as a first evaluation)"
    have_previous = False
    if args.previous_ref:
        (arms_dir / "previous").mkdir(parents=True)
        have_previous = snapshot_previous(args.skill, args.previous_ref, arms_dir / "previous")
        if have_previous:
            resolved = git("rev-parse", args.previous_ref) or args.previous_ref
            previous_note = f"`{args.previous_ref}` ({resolved[:12]})"
        else:
            remove_tree(arms_dir / "previous")
            previous_note = (
                f"not run — skills/{args.skill}/ does not exist at `{args.previous_ref}`, "
                "so this is the skill's first evaluation"
            )
            print(f"NOTE {previous_note}")

    provenance = {
        "head": git("rev-parse", "HEAD") or "unknown",
        "dirty": bool(git("status", "--porcelain", "--", f"skills/{args.skill}")),
        "previous": previous_note,
    }

    plan: list[tuple[str, str, list[str]]] = []
    for arm in ARMS:
        if arm == "previous" and not have_previous:
            continue
        directory = None if arm == BASELINE else arms_dir / arm
        job = f"{prefix}-{arm.replace('_', '-')}"
        plan.append((arm, job, build_command(args, arm, tasks, directory, job)))

    for arm, job, command in plan:
        print(f"\n### {arm} -> {args.jobs_dir}/{job}")
        print("  " + " ".join(command))
    if args.dry_run:
        print("\n--dry-run: nothing started. Snapshots left in", rel(arms_dir))
        return 0

    results: dict[str, dict] = {}
    for arm, job, command in plan:
        print(f"\n=== running {arm} ===", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            print(f"WARN harbor exited {completed.returncode} for arm {arm}")
        results[arm] = summarize(jobs_dir / job)

    report = format_report(args, tasks, results, provenance)
    report_path = args.report_path or jobs_dir / f"{prefix}-comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"report written to {report_path}")

    errored = any(results[arm]["errors"] for arm in results)
    baseline_mean = results.get(BASELINE, {}).get("mean")
    candidate_mean = results.get("candidate", {}).get("mean")
    if args.fail_on_regression:
        if errored:
            print("FAIL an arm errored; the comparison is not usable", file=sys.stderr)
            return 1
        if baseline_mean is None or candidate_mean is None:
            print("FAIL an arm produced no rewards", file=sys.stderr)
            return 1
        if candidate_mean - baseline_mean < args.min_delta:
            print(
                f"FAIL candidate - no_skill = {candidate_mean - baseline_mean:+.3f}, "
                f"below the {args.min_delta:+.3f} gate",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

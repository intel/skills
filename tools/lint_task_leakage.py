#!/usr/bin/env python3
"""Report how much of a skill's answer its own task instructions give away.

    python3 tools/lint_task_leakage.py                 # every task, ranked
    python3 tools/lint_task_leakage.py --task dpnp-linalg-matmul --show
    python3 tools/lint_task_leakage.py --fail-on-leak 5

A Level 2 task exists to separate an agent that has the skill from one that does
not. It cannot do that if its own `instruction.md` already contains the thing the
skill teaches: both arms then read the answer out of the prompt and both score 1.0.
That is not a weak skill, it is a task that measures nothing — and it costs the same
money as a task that measures something.

The first scored run of `dpnp-quickstart` (2026-08-26) put all five of its tasks at
exactly that ceiling, so this check exists to catch the next one before it is run
rather than after. It is static: no container, no model, no credentials.

What counts as leakage
----------------------

Every API symbol the skill *teaches* — a dotted call, or a keyword argument — taken
from the skill's fenced code blocks and inline code spans. If the same symbol turns
up in the instruction, the instruction is handing over that part of the answer.

Bare module names are not leakage: a task is allowed, and usually required, to say
which library to use. `import dpnp` is the premise; `dpnp.std(M, axis=0)` is the
answer. The line between them is exactly the line this tool draws, and
--allow extends it when a suite decides a symbol is part of its premise.

A verbatim shared code line is reported separately and weighted harder, because a
line an agent can copy out of the prompt removes the task rather than easing it.

What it cannot tell you
-----------------------

Leakage is necessary, not sufficient. A task can leak nothing and still sit at a
ceiling because the model already knows the answer from pre-training — that is what
the no-skill screening arm is for. Read this as "this task cannot possibly
discriminate", never as "this task will discriminate".
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Same guard as validate_skills.py, and for the same reason: on 3.10 `import tomllib`
# fails as a missing module rather than as a version floor.
if sys.version_info < (3, 11):
    sys.exit(
        "tools/lint_task_leakage.py needs Python 3.11 or newer (tomllib); "
        f"this is {sys.version.split()[0]}"
    )

import tomllib  # noqa: E402  -- after the version guard, deliberately
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "evaluation" / "harbor" / "tasks"
SKILLS_DIR = REPO_ROOT / "skills"
# --self-test reads the gate's budget out of the workflow rather than carrying a second
# copy of the number, so the two cannot drift apart.
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validate.yml"
GATE_BUDGET = re.compile(r"--fail-on-leak\s+(\d+)")

# A task must be able to name the library it is about, so a bare module or a common
# alias is premise rather than answer. Anything dotted onto them is the answer.
PREMISE = {"dpnp", "dpctl", "numpy", "np", "python", "python3", "pytest", "conda", "pip"}

FENCE = re.compile(r"^\s*```")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
# A qualified call, in either language this repository's tasks are written in:
# dpnp.std and tbb::parallel_for. Without the C++ separator the check is
# structurally blind to seven of the fifteen tasks, which is worse than no check —
# it would report a clean zero for a C++ instruction that hands over the algorithm.
SYMBOL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::)[A-Za-z_][A-Za-z0-9_]*)+)\b")
SEPARATOR = re.compile(r"::|\.")
CXX_ROOT_ALIAS = re.compile(r"^oneapi::")
KEYWORD = re.compile(r"\b([a-z_][a-z0-9_]*)\s*=\s*([A-Za-z0-9_.\-]+)")
IMPORT_AS = re.compile(r"\bimport\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)")


def aliases(text: str) -> dict[str, set[str]]:
    """alias -> every module this document binds it to.

    Without this the check silently passes the worst cases. `dpnp-quickstart`
    demonstrates its API as `import dpnp as np`, so it teaches `np.std`, while the
    task instruction spells the same call `dpnp.std` — two different strings for one
    symbol, and the intersection comes out empty.

    The mapping has to be one-to-many. That same skill also writes `import numpy as
    np` in its NumPy-fallback example, so `np` means both libraries in one file; a
    dict that keeps the last binding resolves `np.std` to `numpy.std` and misses the
    leak entirely. A symbol therefore expands to every module the alias could name,
    which over-reports rather than under-reports — the safe direction for a check
    whose whole job is to stop an unmeasurable task from being run.
    """
    out: dict[str, set[str]] = {}
    for module, alias in IMPORT_AS.findall(text):
        out.setdefault(alias, set()).add(module)
    return out


def code_regions(text: str) -> tuple[list[str], str]:
    """Fenced code lines, and the prose with fences removed.

    Both are needed: a symbol inside a fence is being demonstrated, and a symbol in
    an inline span in prose is being prescribed. Either one teaches it.
    """
    fenced: list[str] = []
    prose: list[str] = []
    inside = False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        (fenced if inside else prose).append(line)
    return fenced, "\n".join(prose)


def symbols(text: str) -> set[str]:
    """API symbols and keyword arguments taught or given away by this text.

    Aliases are resolved against the whole document, not just the code region, so a
    skill that demonstrates `import dpnp as np` in one section and the task that
    spells the same call `dpnp.std` in another compare equal.
    """
    alias_map = aliases(text)
    fenced, prose = code_regions(text)
    haystack = "\n".join(fenced + INLINE_CODE.findall(prose))
    found = set()
    for match in SYMBOL.finditer(haystack):
        # `oneapi::tbb::parallel_for` and `tbb::parallel_for` are the same symbol —
        # oneTBB ships `tbb` as a namespace alias for `oneapi::tbb`, and the skill and
        # the task instructions do not agree on which spelling to use. This is the C++
        # equivalent of the `import dpnp as np` problem below.
        raw = CXX_ROOT_ALIAS.sub("", match.group(1))
        # Split on whichever separator this symbol used. Splitting on "." only would
        # leave `tbb::parallel_for` as one unqualified head, which then fails the
        # "is it qualified?" test below and is dropped — the check would see the
        # symbol and still score the C++ tasks zero.
        cut = SEPARATOR.search(raw)
        if cut:
            raw_head, separator, tail = raw[: cut.start()], cut.group(0), raw[cut.end() :]
        else:
            raw_head, separator, tail = raw, "", ""
        for head in {raw_head} | alias_map.get(raw_head, set()):
            name = f"{head}{separator}{tail}" if tail else head
            if name in PREMISE:
                continue
            # dpnp.sum and tbb::parallel_for count; a bare module does not.
            # my_var.shape on an unknown head is noise, so require the head to be a
            # library we care about or the symbol to be qualified at all.
            if head in PREMISE or separator:
                found.add(name)
    for match in KEYWORD.finditer(haystack):
        found.add(f"{match.group(1)}=")
    return found


def code_lines(text: str) -> set[str]:
    fenced, _ = code_regions(text)
    out = set()
    for line in fenced:
        stripped = line.strip()
        # Short lines are shared by accident (`import dpnp`, `)`); long ones are not.
        if len(stripped) >= 20 and not stripped.startswith("#"):
            out.add(stripped)
    return out


def skill_text(skill: str) -> str | None:
    directory = SKILLS_DIR / skill
    main = directory / "SKILL.md"
    if not main.is_file():
        return None
    parts = [main.read_text(encoding="utf-8")]
    references = directory / "references"
    if references.is_dir():
        for path in sorted(references.rglob("*.md")):
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def task_skill(task_dir: Path) -> str | None:
    config = task_dir / "task.toml"
    if not config.is_file():
        return None
    try:
        document = tomllib.loads(config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return None
    return (document.get("metadata") or {}).get("skill")


def audit(task_dir: Path, allow: set[str]) -> dict | None:
    skill = task_skill(task_dir)
    if not skill:
        return None
    text = skill_text(skill)
    if text is None:
        return {"task": task_dir.name, "skill": skill, "error": f"no skills/{skill}/SKILL.md"}
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.is_file():
        return {"task": task_dir.name, "skill": skill, "error": "no instruction.md"}
    instruction = instruction_path.read_text(encoding="utf-8")

    taught = symbols(text)
    given = symbols(instruction)
    leaked = sorted((taught & given) - allow)
    shared_lines = sorted(code_lines(text) & code_lines(instruction))
    return {
        "task": task_dir.name,
        "skill": skill,
        "taught_symbols": len(taught),
        "leaked_symbols": leaked,
        "shared_code_lines": shared_lines,
        # A copyable line removes the task, so it is worth more than a named symbol.
        "score": len(leaked) + 3 * len(shared_lines),
    }


def _score(taught: set[str], taught_lines: set[str], instruction: str) -> int:
    """audit()'s score, for an instruction held in memory rather than read from disk."""
    return len(taught & symbols(instruction)) + 3 * len(taught_lines & code_lines(instruction))


def self_test() -> int:
    """Assert the detector still detects, against the skills and tasks in this tree.

    Every way this check can break is silent. A regex that stops matching, a PREMISE
    entry that swallows a real symbol, an alias map that resolves the wrong way: each of
    them prints a clean zero for every task and a green gate, which reads exactly like a
    repository whose tasks are all sound. A leakage budget is worth what the detector
    behind it is worth, so the properties it rests on are asserted rather than assumed —
    the same reason validate.yml asserts that `verify` refuses an installed skill that
    was altered.

    Real content, because a fixture proves the fixture: the assertions below run against
    `skills/` and `evaluation/harbor/tasks/` as committed. Nothing is written — the two
    injections are made to a copy of the instruction text in memory.
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name} - {detail}")
        if not ok:
            failures.append(name)

    python_skill, cxx_skill = skill_text("dpnp-quickstart"), skill_text("onetbb-quickstart")
    if python_skill is None or cxx_skill is None:
        sys.exit("FAIL --self-test needs skills/dpnp-quickstart and skills/onetbb-quickstart")

    taught = symbols(python_skill)
    check("a Python skill teaches a dotted symbol", "dpnp.std" in taught,
          f"dpnp-quickstart teaches {len(taught)} symbols")

    qualified = {symbol for symbol in symbols(cxx_skill) if "::" in symbol}
    check("a C++ skill teaches a :: symbol", "tbb::parallel_for" in qualified,
          f"onetbb-quickstart teaches {len(qualified)} of them")

    check("a keyword argument is taught, not only a call", "device=" in taught,
          "dpnp-quickstart teaches `device=`, which is what dpnp-device-fallback gives away")

    check("an alias bound twice keeps both meanings", aliases(python_skill).get("np") == {"dpnp", "numpy"},
          "dpnp-quickstart writes `import dpnp as np` and `import numpy as np` in one document")

    reports = [
        report
        for report in (audit(path, set()) for path in sorted(TASKS_DIR.iterdir()) if path.is_dir())
        if report is not None and "error" not in report
    ]
    scored = [report for report in reports if report["score"] > 0]
    worst_score = max(report["score"] for report in reports)

    # The budget is only a ratchet if it tracks the tree. Read it out of the workflow and
    # require it to be the worst task's score exactly: too high and the gate has slack
    # nobody voted for, too low and CI is red on content that was already merged. Cleaning
    # up the worst task therefore turns this red, which is the check asking for the number
    # to come down with it.
    budgets = (
        {int(found) for found in GATE_BUDGET.findall(WORKFLOW.read_text(encoding="utf-8"))}
        if WORKFLOW.is_file()
        else set()
    )
    check("the CI budget is the worst task in the tree", budgets == {worst_score},
          f"validate.yml runs --fail-on-leak {sorted(budgets) or 'nothing'}, worst of "
          f"{len(scored)} leaking tasks scores {worst_score}"
          + ("" if budgets == {worst_score} else " - move the budget in validate.yml to match"))

    given_away = {symbol for report in reports for symbol in report["leaked_symbols"]}
    unqualified = sorted(s for s in given_away if not SEPARATOR.search(s) and not s.endswith("="))
    check("no bare module is reported as an answer", not unqualified,
          ", ".join(unqualified) or f"all {len(given_away)} are qualified calls or keyword arguments")

    # The bite, on the task that is already worst: the budget is only a budget if one
    # more given-away symbol crosses it.
    worst = max(reports, key=lambda report: report["score"])
    instruction = (TASKS_DIR / worst["task"] / "instruction.md").read_text(encoding="utf-8")
    skill = skill_text(worst["skill"]) or ""
    taught, taught_lines = symbols(skill), code_lines(skill)
    base = _score(taught, taught_lines, instruction)
    check(f"{worst['task']} scores the same in memory", base == worst["score"],
          f"on disk {worst['score']}, in memory {base}")

    untold = sorted(taught - symbols(instruction))
    unshared = sorted(taught_lines - code_lines(instruction))
    if not untold or not unshared:
        sys.exit(f"FAIL --self-test needs a symbol and a line {worst['task']} does not already give away")

    # A library call rather than whatever sorts first: the regex also matches a filename
    # like `SKILL.md`, and injecting one of those would assert the arithmetic while
    # proving nothing about the thing being detected.
    call = next((s for s in untold if SEPARATOR.split(s)[0] in PREMISE), untold[0])
    with_symbol = _score(taught, taught_lines, f"{instruction}\n\nUse `{call}` for this.\n")
    check("one more given-away symbol costs one point", with_symbol == base + 1,
          f"`{call}` added to {worst['task']}: {base} -> {with_symbol}, so a budget of {base} fails")

    pasted = f"{instruction}\n\n```python\n{unshared[0]}\n```\n"
    shared_before, shared_after = (
        len(taught_lines & code_lines(instruction)),
        len(taught_lines & code_lines(pasted)),
    )
    with_line = _score(taught, taught_lines, pasted)
    check("a copyable line is caught and costs at least three",
          shared_after == shared_before + 1 and with_line >= base + 3,
          f"one line of the skill pasted into {worst['task']}: score {base} -> {with_line}, "
          f"shared lines {shared_before} -> {shared_after}")

    if failures:
        print(f"\nFAIL {len(failures)} self-test(s): " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nOK the detector detects. A regex that stopped matching would fail here rather")
    print("than report a clean zero for every task and leave the budget passing anything.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", action="append", default=[], help="limit to these task names")
    parser.add_argument("--show", action="store_true", help="list every leaked symbol")
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="SYMBOL",
        help="symbol that is this suite's premise rather than its answer. Repeatable.",
    )
    parser.add_argument(
        "--fail-on-leak",
        type=int,
        default=None,
        metavar="N",
        help="exit nonzero for any task scoring above N. Omit to report only. CI runs "
        "with 5, the score of the worst task here today, so the gate is a ratchet: it "
        "stops a task arriving worse than the worst one already in the tree.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="assert the detector still detects, against the skills and tasks in this "
        "tree. Writes nothing. Run it before the gate: a broken detector reports a clean "
        "zero for every task, which is indistinguishable from a clean repository.",
    )
    args = parser.parse_args()

    if not TASKS_DIR.is_dir():
        sys.exit(f"FAIL no {TASKS_DIR.relative_to(REPO_ROOT).as_posix()}")

    if args.self_test:
        return self_test()

    wanted = set(args.task)
    reports = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        if wanted and task_dir.name not in wanted:
            continue
        report = audit(task_dir, set(args.allow))
        if report is not None:
            reports.append(report)

    if not reports:
        sys.exit("FAIL no tasks matched")

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        reports.sort(key=lambda r: -r.get("score", 0))
        print(f"{'task':34} {'skill':22} {'score':>5} {'symbols':>8} {'lines':>6}")
        for report in reports:
            if "error" in report:
                print(f"{report['task']:34} {report['skill']:22}   ERR  {report['error']}")
                continue
            print(
                f"{report['task']:34} {report['skill']:22} {report['score']:5} "
                f"{len(report['leaked_symbols']):8} {len(report['shared_code_lines']):6}"
            )
            if args.show:
                for symbol in report["leaked_symbols"]:
                    print(f"    symbol  {symbol}")
                for line in report["shared_code_lines"]:
                    print(f"    line    {line}")
        print()
        print("score = leaked symbols + 3x verbatim shared code lines. A high score means")
        print("the instruction hands over what the skill teaches, so both arms can read the")
        print("answer out of the prompt and the task cannot discriminate. A zero score is")
        print("necessary but not sufficient: screen with a no-skill arm before trusting it.")

    if args.fail_on_leak is not None:
        over = [r for r in reports if r.get("score", 0) > args.fail_on_leak]
        if over:
            print(
                f"\nFAIL {len(over)} task(s) above the {args.fail_on_leak} leakage budget: "
                + ", ".join(r["task"] for r in over),
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

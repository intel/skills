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
        help="exit nonzero for any task scoring above N. Omit to report only, which is "
        "the default because the fifteen tasks in this repository all leak today.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if not TASKS_DIR.is_dir():
        sys.exit(f"FAIL no {TASKS_DIR.relative_to(REPO_ROOT).as_posix()}")

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

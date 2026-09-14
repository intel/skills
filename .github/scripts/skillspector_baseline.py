#!/usr/bin/env -S uv run --quiet
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///

"""Emit the SkillSpector baseline rules that apply to one skill.

SkillSpector suppresses findings natively: `skillspector scan --baseline <file>` drops
every finding a rule in that file matches, keeps the rule's `reason` with it, and
recomputes the risk score over what is left. That is the mechanism this repository uses,
so the suppression data lives in the scanner's own format (`.skillspector-baseline.yaml`)
rather than in a bespoke one, and no code here has to re-implement matching.

One thing the format cannot express is *which skill* an entry is for. Its rule schema is
`id`, `path`, `message` and `reason`, and `path` is matched against a finding's file
relative to the skill it was pointed at -- `SKILL.md`, `scripts/foo.py`. So a single
shared file reaches every skill in the catalog: the entry that accepts `exec()` in
xpu-port's verifier would also accept it in a skill nobody has written yet. Measured on
this tree, that is not hypothetical -- a probe skill shipping one `subprocess` call scores
0 with nothing active under a blanket `AST4` entry, and 9 with the finding active once the
entry names the two skills it was written for.

This script closes that hole with a `skills:` list the repository defines and the scanner
ignores. Each scan gets a baseline holding only its own skill's rules:

    uv run .github/scripts/skillspector_baseline.py --skill linux-perf --output base.yaml
    skillspector scan skills/linux-perf --no-llm --format json --baseline base.yaml

An entry with no `skills:` list applies everywhere, which is a deliberate statement that
the rule is inapplicable to this catalog as a whole -- `RP1`, `PE5` and `LP3` are the
three. A `skills:` pattern matching no skill on disk fails the run rather than reading as
cover for a typo nobody will revisit.

Exits 0 on success, 1 if the baseline is unreadable or a scope matches nothing.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASELINE = REPO_ROOT / ".skillspector-baseline.yaml"
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"

# The one field in the baseline that SkillSpector does not define, so it has to be
# stripped before the file reaches the CLI.
SCOPE_FIELD = "skills"


def known_skills(skills_dir: Path) -> set[str]:
    """Skill names on disk. A directory without a SKILL.md is not a skill."""
    if not skills_dir.is_dir():
        return set()
    return {p.name for p in skills_dir.iterdir() if (p / "SKILL.md").is_file()}


def in_scope(rule: dict, skill: str) -> bool:
    """Does this rule apply to this skill? A rule naming no skills applies to all."""
    scope = rule.get(SCOPE_FIELD)
    if scope is None:
        return True
    return any(fnmatch.fnmatch(skill, pattern) for pattern in scope)


def scoped_baseline(data: dict, skill: str, known: set[str]) -> dict:
    """The baseline document as the scanner should see it for one skill."""
    rules = data.get("rules") or []

    for position, rule in enumerate(rules, start=1):
        scope = rule.get(SCOPE_FIELD)
        if scope is None:
            continue
        if not isinstance(scope, list) or not scope:
            raise SystemExit(
                f"rule {position} ({rule.get('id', '?')}): {SCOPE_FIELD!r} must be a "
                "non-empty list of skill names or glob patterns"
            )
        for pattern in scope:
            if not any(fnmatch.fnmatch(name, pattern) for name in sorted(known)):
                raise SystemExit(
                    f"rule {position} ({rule.get('id', '?')}) is scoped to {pattern!r}, "
                    "which matches no skill on disk"
                )

    out = {key: value for key, value in data.items() if key != "rules"}
    out["rules"] = [
        {key: value for key, value in rule.items() if key != SCOPE_FIELD}
        for rule in rules
        if in_scope(rule, skill)
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skill", required=True, help="Skill the baseline is for.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Shared baseline to read (default: {DEFAULT_BASELINE}).",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=f"Directory containing skill folders (default: {DEFAULT_SKILLS_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the scoped baseline (default: stdout).",
    )
    args = parser.parse_args(argv)

    if not args.baseline.is_file():
        print(f"Baseline not found: {args.baseline}", file=sys.stderr)
        return 1
    try:
        data = yaml.safe_load(args.baseline.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print(f"{args.baseline}: not valid YAML ({exc})", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print(f"{args.baseline}: expected a mapping at the top level", file=sys.stderr)
        return 1

    known = known_skills(args.skills_dir.resolve())
    if args.skill not in known:
        print(
            f"No skill named {args.skill!r} under {args.skills_dir}", file=sys.stderr
        )
        return 1

    scoped = scoped_baseline(data, args.skill, known)
    text = yaml.safe_dump(scoped, sort_keys=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(
            f"{args.skill}: {len(scoped['rules'])} of "
            f"{len(data.get('rules') or [])} baseline rule(s) -> {args.output}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

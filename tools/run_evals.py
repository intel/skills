#!/usr/bin/env python3
"""Validate and score skill eval files.

Two modes:

  --validate      Schema gate. Checks every skills/*/evals/evals.json against
                  schemas/evals.schema.json. No model, no network, no key —
                  this is what runs on every pull request.

  --answers FILE  Scoring. Applies each eval's checks to a recorded agent
                  answer and reports a pass rate. Requires answers produced
                  outside this repository.

Matching is literal. Both the check term and the answer are lowercased and
have runs of whitespace collapsed to a single space, then plain substring
containment is tested. There is no regex anywhere in this file: a regex written
into must_include can never match, and one written into must_not_include
silently matches nothing. A condition that genuinely needs a pattern belongs in
a Harbor task's tests/rubric.json (evaluation/harbor/), where criteria are
evaluated as regular expressions; it does not belong in a check term.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
SCHEMA_PATH = REPO_ROOT / "schemas" / "evals.schema.json"

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Signals that a check term was written as a pattern: a backslash escape, an
# explicit quantifier on a wildcard, a character class, or alternation in a group.
REGEX_SMELL_RE = re.compile(r"\\[dsSwWb.(){}\[\]]|\.\*|\.\+|\[[^\]]+\]|\([^)]*\|[^)]*\)")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace. Applied to both sides of a check."""
    return re.sub(r"\s+", " ", text.casefold())


def eval_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/evals/evals.json"))


def validate_file(path: Path) -> list[str]:
    """Validate one evals.json. Returns a list of error strings."""
    errors: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    skill_dir = path.parent.parent.name

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{rel}: invalid JSON: {exc}"]

    if not isinstance(data, dict):
        return [f"{rel}: top level must be an object"]

    unknown = set(data) - {"skill_name", "evals"}
    if unknown:
        errors.append(f"{rel}: unknown top-level keys: {sorted(unknown)}")

    skill_name = data.get("skill_name")
    if not isinstance(skill_name, str) or not NAME_RE.match(skill_name):
        errors.append(f"{rel}: skill_name must be kebab-case, got {skill_name!r}")
    elif skill_name != skill_dir:
        errors.append(
            f"{rel}: skill_name {skill_name!r} does not match directory {skill_dir!r}"
        )

    evals = data.get("evals")
    if not isinstance(evals, list) or not evals:
        return errors + [f"{rel}: evals must be a non-empty array"]

    seen_ids: set[str] = set()
    for index, case in enumerate(evals):
        where = f"{rel}: evals[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{where}: must be an object")
            continue

        unknown = set(case) - {"id", "prompt", "expected_output", "checks"}
        if unknown:
            errors.append(f"{where}: unknown keys: {sorted(unknown)}")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not NAME_RE.match(case_id):
            errors.append(f"{where}: id must be kebab-case, got {case_id!r}")
        elif case_id in seen_ids:
            errors.append(f"{where}: duplicate id {case_id!r}")
        else:
            seen_ids.add(case_id)

        for field in ("prompt", "expected_output"):
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}: {field} must be a non-empty string")

        checks = case.get("checks")
        if not isinstance(checks, dict):
            errors.append(f"{where}: checks must be an object")
            continue

        unknown = set(checks) - {"must_include", "must_not_include"}
        if unknown:
            errors.append(f"{where}.checks: unknown keys: {sorted(unknown)}")

        must_include = checks.get("must_include")
        if not isinstance(must_include, list) or not must_include:
            errors.append(f"{where}.checks: must_include must be a non-empty array")
            must_include = []

        for key in ("must_include", "must_not_include"):
            for term in checks.get(key) or []:
                if not isinstance(term, str) or not term.strip():
                    errors.append(f"{where}.checks.{key}: terms must be non-empty strings")
                    continue
                # Not a schema rule, but the most common authoring mistake: the
                # matcher never compiles a pattern, so a regex is dead weight.
                # Only unambiguous signals are flagged — a literal may legitimately
                # contain "(" (dpnp.asnumpy() or "+" (2024.1+).
                if REGEX_SMELL_RE.search(term):
                    errors.append(
                        f"{where}.checks.{key}: {term!r} looks like a regex. "
                        "Checks are literal substrings — use a bare literal, or move "
                        "the condition into a Harbor task's tests/rubric.json."
                    )

    return errors


def cmd_validate() -> int:
    if not SCHEMA_PATH.is_file():
        print(f"FAIL missing schema: {SCHEMA_PATH}", file=sys.stderr)
        return 1

    files = eval_files()
    if not files:
        print("FAIL no skills/*/evals/evals.json found", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for path in files:
        all_errors.extend(validate_file(path))

    for error in all_errors:
        print(f"FAIL {error}", file=sys.stderr)

    if all_errors:
        print(f"\n{len(all_errors)} error(s) in {len(files)} eval file(s)", file=sys.stderr)
        return 1

    total_cases = sum(
        len(json.loads(p.read_text(encoding="utf-8"))["evals"]) for p in files
    )
    print(f"OK {len(files)} eval file(s), {total_cases} case(s)")
    return 0


def cmd_score(answers_path: Path, fail_under: float) -> int:
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    if not isinstance(answers, dict):
        print("FAIL answers file must be an object of {eval-id: answer}", file=sys.stderr)
        return 1

    passed = 0
    total = 0
    missing: list[str] = []

    for path in eval_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data["evals"]:
            total += 1
            case_id = case["id"]
            if case_id not in answers:
                missing.append(case_id)
                print(f"MISS {case_id}: no answer recorded")
                continue

            answer = normalize(str(answers[case_id]))
            checks = case["checks"]
            failures = [
                f"missing {term!r}"
                for term in checks["must_include"]
                if normalize(term) not in answer
            ]
            failures += [
                f"forbidden {term!r} present"
                for term in checks.get("must_not_include") or []
                if normalize(term) in answer
            ]

            if failures:
                print(f"FAIL {case_id}: " + "; ".join(failures))
            else:
                passed += 1
                print(f"PASS {case_id}")

    if total == 0:
        print("FAIL no eval cases found", file=sys.stderr)
        return 1

    rate = passed / total
    print(f"\n{passed}/{total} passed ({rate:.0%}); threshold {fail_under:.0%}")
    if missing:
        print(f"{len(missing)} case(s) had no recorded answer: {', '.join(missing)}")
    return 0 if rate >= fail_under else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate", action="store_true", help="schema gate only; no scoring"
    )
    parser.add_argument(
        "--answers", type=Path, help="JSON file of {eval-id: agent answer text}"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=1.0,
        help="minimum pass rate when scoring (default: 1.0)",
    )
    args = parser.parse_args()

    if args.validate:
        return cmd_validate()
    if args.answers:
        return cmd_score(args.answers, args.fail_under)

    parser.error("pass --validate or --answers FILE")
    return 2


if __name__ == "__main__":
    sys.exit(main())

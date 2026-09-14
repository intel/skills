#!/usr/bin/env -S uv run --quiet
#
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Merge per-skill SkillSpector SARIF into one upload for code scanning.

Three fixes are needed on the per-skill files before GitHub can use them.

SkillSpector reports a path relative to the skill it was pointed at (`SKILL.md`), and code
scanning resolves paths from the repository root, so every URI gains its `skills/<name>/`
prefix -- without it the alert lands on nothing and renders nowhere.

The matrix produces one file per skill and GitHub accepts a limited number of SARIF
uploads per commit, so the runs are merged into one: same tool driver, rules unioned by
id, results concatenated.

And a suppressed finding is dropped rather than carried. SkillSpector writes each one out
with the `reason` from `.skillspector-baseline.yaml` in `suppressions[].justification`,
which reads like it should arrive as a dismissed alert holding the sentence that accepted
it. Measured against the API, it does not: code scanning ignores `suppressions` on an
uploaded SARIF whether the kind is `external` or `inSource`, and every accepted finding
becomes an open alert. On this catalog that is 197 alerts nobody is going to act on
burying the 17 that want reading, which is how a security tab stops being read at all.
The reasons stay in the baseline file, which is the audit trail for them.

`--expect` is how the run refuses to claim coverage it does not have. Dropping the
suppressed findings means a skill whose findings were all accepted contributes nothing to
the merged file either way, so a missing or unreadable per-skill SARIF looks exactly like
a clean skill. Uploading fewer alerts than the scan found, silently, is the one failure
this stage must not have.

Usage:

    uv run .github/scripts/skillspector_sarif.py \
        --input-dir sarif --output skillspector.sarif --expect linux-perf,xpu-port

Exits 0 on success, 1 when an expected skill is missing from the merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/"
    "sarif-2.1/schema/sarif-schema-2.1.0.json"
)


def merge(input_dir: Path) -> tuple[dict, list[dict], set[str]]:
    """(tool driver, active results, skills whose SARIF parsed)."""
    driver: dict = {}
    rules: dict[str, dict] = {}
    results: list[dict] = []
    parsed: set[str] = set()

    for path in sorted(input_dir.rglob("*.sarif")):
        skill = path.stem
        try:
            run = json.loads(path.read_text(encoding="utf-8"))["runs"][0]
        except (json.JSONDecodeError, KeyError, IndexError, OSError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        parsed.add(skill)
        driver = driver or run.get("tool", {}).get("driver", {})
        for rule in run.get("tool", {}).get("driver", {}).get("rules", []):
            rules.setdefault(rule.get("id", ""), rule)
        for result in run.get("results", []):
            if result.get("suppressions"):
                continue
            for location in result.get("locations", []):
                artifact = (
                    location.get("physicalLocation", {}).get("artifactLocation", {})
                )
                if "uri" in artifact:
                    artifact["uri"] = f"skills/{skill}/{artifact['uri']}"
            results.append(result)

    driver = dict(driver)
    driver["rules"] = list(rules.values())
    return driver, results, parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path, help="Directory of <skill>.sarif files."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Where to write the merged SARIF."
    )
    parser.add_argument(
        "--expect",
        default="",
        help="Comma- or whitespace-separated skills that must appear in the merge.",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"Not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    driver, results, parsed = merge(args.input_dir)
    args.output.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "$schema": SCHEMA,
                "runs": [{"tool": {"driver": driver}, "results": results}],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"merged {len(results)} active finding(s) from {len(parsed)} skill(s) "
        f"-> {args.output}"
    )

    expected = {name for name in args.expect.replace(",", " ").split() if name}
    missing = sorted(expected - parsed)
    if missing:
        print(
            f"FAIL the merged SARIF is missing {len(missing)} scanned skill(s): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Merge the dokimi JSON reports in one folder into skills-check.json and a Markdown summary."""

import argparse
import json
from pathlib import Path


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def skills_check(d):
    if d is None:
        return {"status": "missing"}, {}
    rows = {}
    for s in d.get("skills", []):
        rows[s["name"]] = {
            "path": s.get("relpath"),
            "state": s.get("state"),
            "error": s.get("error"),
            "findings": [
                {"level": f.get("level"), "code": f.get("code"), "message": f.get("message")}
                for f in s.get("findings", [])
            ],
        }
    tasks = d.get("tasks") or []
    states = [r["state"] for r in rows.values()]
    broken = [{"name": t["name"], "reason": t.get("reason")} for t in tasks if t.get("state") == "error"]
    check = {
        "status": "fail" if "error" in states or broken else "pass",
        "skills": len(rows),
        "skills_ok": states.count("ok"),
        "skills_warn": states.count("warn"),
        "skills_error": states.count("error"),
        "tasks": len(tasks),
        "tasks_ok": sum(t.get("state") == "ok" for t in tasks),
        "tasks_skipped": sum(t.get("state") == "skipped" for t in tasks),
        "tasks_error": broken,
    }
    return check, rows


def routing(folder, rows):
    scored = 0
    for f in sorted(folder.glob("*.json")):
        lex = (load(f) or {}).get("lexical")
        if not lex or f.stem not in rows:
            continue
        rows[f.stem]["routing"] = {
            "recall_at_1": lex.get("recall_at_1"),
            "recall_at_3": lex.get("recall_at_3"),
            "mrr": lex.get("mrr"),
            "queries": lex.get("n_queries_scored"),
            "most_picked_instead": sorted(
                {q["picked"] for q in lex.get("per_query", []) if q.get("rank") != 1}
            ),
        }
        scored += 1
    return {"status": "report", "skills_scored": scored, "queries": "skill headings (triage only)"}


def plan(folder):
    result, steps = load(folder / "result.json"), load(folder / "plan.json")
    if result is None or steps is None:
        return {"status": "not run"}
    cells = result.get("cells", [])
    return {
        "status": "report",
        "skills_selected": len(steps.get("subjects", [])),
        "cells": len(cells),
        "cells_failed": sum(c.get("status") != "pass" for c in cells),
    }


def counts(d, key):
    if d is None:
        return {"status": "missing"}
    return {"status": "report", **(d.get(key) or {})}


def text(v):
    if isinstance(v, list):
        return ", ".join(x["name"] if isinstance(x, dict) else str(x) for x in v)
    return v


def markdown(summary):
    out = [f"## Skills check: {summary['verdict']}", ""]
    out += ["| Check | Status | Result |", "|---|---|---|"]
    for name, c in summary["checks"].items():
        rest = ", ".join(f"{k} {text(v)}" for k, v in c.items() if k != "status" and v not in ([], None))
        out.append(f"| {name} | {c['status']} | {rest} |")
    out += ["", "| Skill | State | Findings | Routing R@1 |", "|---|---|---|---|"]
    for name, s in sorted(summary["skills"].items()):
        codes = ", ".join(f["code"] for f in s["findings"]) or "-"
        r1 = s.get("routing", {}).get("recall_at_1")
        out.append(f"| {name} | {s['state']} | {codes} | {'-' if r1 is None else r1} |")
    out += ["", "Full data: `skills-check.json` in the `skills-check-reports` artifact."]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--dokimi-ref", default="")
    a = p.parse_args()
    o = a.folder
    check, rows = skills_check(load(o / "check.json"))
    inspect = load(o / "inspect.json")
    checks = {
        "skills and tasks load": check,
        "static plan": plan(o / "catalog-eval"),
        "static plan, changed skills": plan(o / "catalog-eval-changed"),
        "instruction references": counts(load(o / "scan.json"), "counts"),
        "instruction size": counts(load(o / "budget.json"), "totals"),
        "agent setup": counts(inspect, "counts"),
        "skill routing (lexical)": routing(o / "routing", rows),
    }
    if inspect:
        checks["agent setup"]["findings"] = ", ".join(f["rule"] for f in inspect.get("findings", []))
    summary = {
        "schema_version": 1,
        "dokimi_ref": a.dokimi_ref,
        "verdict": "fail" if check["status"] != "pass" else "pass",
        "checks": checks,
        "skills": rows,
    }
    (o / "skills-check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (o / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(summary["verdict"])


if __name__ == "__main__":
    main()

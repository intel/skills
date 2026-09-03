#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT
"""
gccbranchprob.py — report GCC static branch-probability estimates.

Parses a GCC profile_estimate dump file and reports per-branch probability
estimates as a Markdown table on stdout for direct agent ingestion.

Usage:
    python3 gccbranchprob.py <source.c> <dump_dir> [function ...] [options]

Arguments:
    source.c        C source file that was compiled with the dump flags
    dump_dir        Directory containing the GCC profile_estimate dump
                    (produced by: gcc -fdump-tree-profile_estimate-lineno
                                       -dumpdir <dump_dir> ...)
    function ...    Optional: one or more function names to filter output.
                    When provided only branches inside these functions are shown.

Options:
    --fuzzy         Match function names as substrings rather than exact names

The output format uses the same columns as branchprob.py (the perf-based
tool) where applicable, so the two outputs can be compared side-by-side.
The key difference: 'GCC Est%' is a static compiler estimate; 'Taken%' from
branchprob.py is a measured runtime probability.  Significant divergence
between the two is a signal worth investigating.

Requirements:
    - GCC dump file in dump_dir, produced with:
        -g -fdump-tree-profile_estimate-lineno -dumpdir <dump_dir>
    - The dump file is named <source_basename>.<NNN>t.profile_estimate
      (the pass number NNN is version-dependent; the script finds it automatically)
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# When a body block has multiple incoming probability edges, average them
# only if they agree within this many percentage points; otherwise skip.
_CONFLICT_THRESHOLD = 20.0

# ---------------------------------------------------------------------------
# Dump file discovery
# ---------------------------------------------------------------------------

def find_dump_file(dump_dir, src_basename):
    """Return the path to the profile_estimate dump file for src_basename
    inside dump_dir, or None if not found.

    GCC names the file: <source>.<NNN>t.profile_estimate
    where NNN is a pass number that varies across GCC versions.
    """
    try:
        candidates = [
            f for f in os.listdir(dump_dir)
            if f.endswith('.profile_estimate')
            and f.startswith(src_basename + '.')
        ]
    except OSError as e:
        print(f'Error reading dump directory: {e}', file=sys.stderr)
        return None
    if not candidates:
        return None
    return os.path.join(dump_dir, sorted(candidates)[0])

# ---------------------------------------------------------------------------
# Dump parsing
# ---------------------------------------------------------------------------

def parse_dump(dump_path, src_basename):
    """Parse a GCC profile_estimate dump and extract branch probabilities.

    Returns a list of:
        (func_name, condition_line, target_line, prob_pct)
    sorted by (func_name, condition_line, target_line).

    condition_line: source line of the 'if (...)' expression (the branch point)
    target_line:    first source line of the target basic block (where execution
                    goes when the branch is taken along this edge)
    prob_pct:       GCC's estimated probability (0.0–100.0) of this edge firing

    Only edges from condition blocks (blocks with ≥2 outgoing probability edges)
    to body blocks in the same source file are reported.
    """
    # Match full-path source references: [/path/to/foo.c:11:8]
    # We match on the basename to work regardless of compile directory.
    src_ref_re = re.compile(
        r'\[(?:[^:\]]+/)?' + re.escape(src_basename) +
        r':(\d+):\d+(?:\s+discrim\s+\d+)?\]'
    )
    func_name_re = re.compile(r'^;; Function (\S+) \(')

    with open(dump_path) as f:
        dump = f.read()

    # Split into per-function chapters.
    chapters = re.split(r'(?=^;; Function )', dump, flags=re.MULTILINE)

    results = []

    for chapter in chapters:
        if not chapter.startswith(';; Function'):
            continue
        m = func_name_re.match(chapter)
        func_name = m.group(1) if m else '(unknown)'

        # Split the chapter into per-basic-block sections.
        sections = re.split(r'\n(?=  <bb \d+>)', chapter)

        # First pass: collect source lines and outgoing edges per block.
        bb_source_lines = {}   # bb_num → set of source line numbers
        bb_outgoing     = {}   # bb_num → list of (to_bb, prob)

        for section in sections:
            hdr = re.match(r'\s*<bb (\d+)>', section)
            if not hdr:
                continue
            bb_num = int(hdr.group(1))
            bb_source_lines[bb_num] = set()
            bb_outgoing[bb_num]     = []

            for sm in src_ref_re.finditer(section):
                bb_source_lines[bb_num].add(int(sm.group(1)))

            for edge_m in re.finditer(r'goto <bb (\d+)>; \[(\d+\.\d+)%\]', section):
                bb_outgoing[bb_num].append((int(edge_m.group(1)), float(edge_m.group(2))))

        # Condition blocks have 2+ outgoing probability edges.
        condition_blocks = {
            bb for bb, edges in bb_outgoing.items() if len(edges) >= 2
        }

        # Second pass: emit one row per outgoing edge from a condition block.
        for from_bb in sorted(condition_blocks):
            src_lines = bb_source_lines.get(from_bb, set())
            if not src_lines:
                continue
            # The condition line is the highest source line in the block —
            # GCC places the 'if (...)' expression as the last statement.
            condition_line = max(src_lines)

            for to_bb, prob in bb_outgoing[from_bb]:
                target_lines = bb_source_lines.get(to_bb, set())
                if not target_lines:
                    continue
                target_line = min(target_lines)
                results.append((func_name, condition_line, target_line, prob))

    results.sort(key=lambda r: (r[0], r[1], r[2]))
    return results

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_markdown(rows, src_basename, func_filter, fuzzy):
    """Build and return the markdown report from parsed rows.

    Groups rows by function and emits one section per function.
    """
    if not rows:
        return '*(no branch probability data found in dump)*\n'

    # Apply function filter.
    if func_filter:
        if fuzzy:
            rows = [r for r in rows
                    if any(needle in r[0] for needle in func_filter)]
        else:
            wanted = set(func_filter)
            rows = [r for r in rows if r[0] in wanted]

    if not rows:
        names = ', '.join(f'`{f}`' for f in func_filter)
        return f'*(no data for {names} — check function names in the dump)*\n'

    # Group by function.
    from collections import OrderedDict
    by_func = OrderedDict()
    for func_name, cond_line, tgt_line, prob in rows:
        by_func.setdefault(func_name, []).append((cond_line, tgt_line, prob))

    lines = []
    for func_name, func_rows in by_func.items():
        lines.append(f'## `{func_name}`\n')
        lines.append('| GCC Est% | Source line | Target line |\n')
        lines.append('|---------:|------------:|------------:|\n')
        for cond_line, tgt_line, prob in func_rows:
            lines.append(
                f'| {prob:8.2f}% | :{cond_line:<10} | :{tgt_line:<11} |\n'
            )
        lines.append('\n')

    return ''.join(lines)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Report GCC static branch-probability estimates.')
    parser.add_argument('source',
                        help='C source file (e.g. foo.c)')
    parser.add_argument('dump_dir',
                        help='Directory containing the GCC profile_estimate dump')
    parser.add_argument('functions', nargs='*', metavar='function',
                        help='Function name(s) to report (default: all)')
    parser.add_argument('--fuzzy', action='store_true',
                        help='Match function names as substrings')
    args = parser.parse_args()

    src_basename = Path(args.source).name
    dump_file = find_dump_file(args.dump_dir, src_basename)

    if dump_file is None:
        print(
            f'Error: no profile_estimate dump found for {src_basename!r} '
            f'in {args.dump_dir!r}.\n'
            f'Compile with: gcc -g -fdump-tree-profile_estimate-lineno '
            f'-dumpdir {args.dump_dir} ...',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f'[dump] {dump_file}', file=sys.stderr)
    rows = parse_dump(dump_file, src_basename)

    header = (
        f'# Branch probabilities — `{src_basename}`\n\n'
        f'Source: GCC static estimate (`-fdump-tree-profile_estimate-lineno`).\n'
        f'These are compiler heuristics, not measured runtime values.\n'
        f'Compare with `branchprob.py` output to find divergences.\n\n'
    )
    if args.functions:
        header += f'Functions: {", ".join(f"`{f}`" for f in args.functions)}\n\n'

    print(header + build_markdown(rows, src_basename, args.functions, args.fuzzy))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT
"""
branchprob.py — measure branch probabilities using Linux perf.

Runs the binary under perf, collects Intel PMU branch-retirement events,
and reports per-branch taken-probabilities as a Markdown table on stdout
for direct agent ingestion.

Usage:
    python3 branchprob.py <source.c> <binary> [function ...] [options]

Arguments:
    source.c        C source file to analyse
    binary          Binary compiled with -g (debug symbols required)
    function ...    Optional: one or more function names to filter output.
                    When provided only branches inside these functions are shown.
                    Without them all sampled branches are reported.

Options:
    --period N      perf sampling period — one sample per N branch events
                    (default: 1000; lower = more samples, larger perf.data)
    --min-samples N Suppress branches with fewer than N total samples when no
                    function filter is active (default: 5)
    --fuzzy         Match function names as substrings rather than exact names
    --profile FILE  Also write an annotated copy of source.c to FILE, with
                    /* PERF PROB: X.XX% */ comments before cold-path lines
    --debug         Write raw perf annotate output to <source>.annotate.out

Requirements:
    - Intel CPU with BR_INST_RETIRED.NEAR_TAKEN / NOT_TAKEN PMU events
    - perf and addr2line in PATH
    - Binary compiled with -g
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERF_EVENTS = '{BR_INST_RETIRED.NEAR_TAKEN:upp,BR_INST_RETIRED.NOT_TAKEN:upp,cycles:u}'

# All x86 conditional-jump mnemonics (AT&T syntax, lowercase).
COND_JUMPS = frozenset([
    'ja', 'jae', 'jb', 'jbe', 'jc', 'je', 'jg', 'jge', 'jl', 'jle',
    'jna', 'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl',
    'jnle', 'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp', 'jpe', 'jpo', 'js', 'jz',
    'jrcxz', 'jecxz', 'jcxz',
    'loop', 'loope', 'loopz', 'loopne', 'loopnz',
])

UNCOND_JUMPS = frozenset(['jmp', 'jmpq', 'ljmp'])
RET_INSTRS   = frozenset(['ret', 'retq', 'retl', 'retn', 'retw',
                           'iret', 'iretq', 'iretl'])

# perf annotate --stdio -l -n produces lines like:
#   "  NEAR_TAKEN   NOT_TAKEN   CYCLES :   ADDR:   MNEMONIC  ..."
_LINE_RE = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s*:\s+([0-9a-f]+):\s+(\S+)(.*)')

# Function-header lines, e.g. "  0000000000001170 <process_data>:"
_FUNC_RE = re.compile(r'[0-9a-f]+\s+<([^>+]+)>:\s*$')

# Extracts target address from a jump operand, e.g. "ja  11a4 <fn+0x34>" → "11a4"
_JUMP_TARGET_RE = re.compile(r'^\S+\s+([0-9a-f]+)\b')

# ---------------------------------------------------------------------------
# Parsing perf annotate output
# ---------------------------------------------------------------------------

def parse_annotate(text):
    """Parse perf annotate --stdio output.

    Returns:
        func_groups: list of (func_name, instrs) where each instr is
            [taken, not_taken, addr, mnemonic, rest]
        cycles_by_addr: dict {addr: total_cycles_samples}
    """
    func_groups = []
    current_func = None
    current_instrs = []
    cycles_by_addr = {}

    for line in text.splitlines():
        fm = _FUNC_RE.search(line)
        if fm:
            if current_func is not None and current_instrs:
                func_groups.append((current_func, current_instrs))
            current_func = fm.group(1)
            current_instrs = []
            continue
        m = _LINE_RE.match(line)
        if m and current_func is not None:
            taken, not_taken, cycles, addr, mn, rest = m.groups()
            current_instrs.append([int(taken), int(not_taken), addr, mn.lower(), rest])
            c = int(cycles)
            if c:
                cycles_by_addr[addr] = cycles_by_addr.get(addr, 0) + c

    if current_func is not None and current_instrs:
        func_groups.append((current_func, current_instrs))

    return func_groups, cycles_by_addr


def fix_macrofusion(instrs):
    """Move branch counts from a fused cmp/test instruction to the following
    conditional jump.  x86 CPUs can macrofuse a compare with the next
    conditional jump into one micro-op; the PMU then attributes the retirement
    event to the compare, not the jump.  Modifies instrs in-place."""
    for i in range(len(instrs) - 1):
        taken, not_taken, _addr, mn, _rest = instrs[i]
        if mn in COND_JUMPS:
            continue
        if taken == 0 and not_taken == 0:
            continue
        if instrs[i + 1][3] in COND_JUMPS:
            instrs[i + 1][0] += taken
            instrs[i + 1][1] += not_taken
            instrs[i][0] = 0
            instrs[i][1] = 0


def mark_fallthrough(func_instrs):
    """Return the set of addresses on the fall-through (dominant) path of
    func_instrs.  Fall-through = the sequence of instructions reached if no
    conditional branches are ever taken.

    Rules:
      1. Normal instruction → next instruction is fall-through
      3. Unconditional jmp  → jump target is fall-through
      4. ret / already seen → stop propagation
      5. First instruction  → starts as fall-through
      6. Conditional jump target → NOT automatically fall-through
    """
    n = len(func_instrs)
    if n == 0:
        return set()

    addr_to_idx = {instr[2]: i for i, instr in enumerate(func_instrs)}
    fallthrough = set()
    stack = [0]

    while stack:
        idx = stack.pop()
        if idx < 0 or idx >= n:
            continue
        addr = func_instrs[idx][2]
        if addr in fallthrough:
            continue
        fallthrough.add(addr)

        mn   = func_instrs[idx][3]
        rest = func_instrs[idx][4]
        code = (mn + rest).strip()

        if mn in RET_INSTRS:
            pass
        elif mn in UNCOND_JUMPS:
            m = _JUMP_TARGET_RE.match(code)
            if m:
                target_idx = addr_to_idx.get(m.group(1))
                if target_idx is not None:
                    stack.append(target_idx)
        elif mn in COND_JUMPS:
            pass  # Rule 6: do not propagate to target; Rule 2 omitted
        else:
            if idx + 1 < n:
                stack.append(idx + 1)

    return fallthrough


def expand_instructions(func_instrs, fallthrough_addrs):
    """Return list of (taken, not_taken, addr, code_line, branch_target_addr,
    is_fallthrough) for every instruction in the function."""
    result = []
    for taken, not_taken, addr, mn, rest in func_instrs:
        code_line = (mn + rest).strip()
        branch_target = None
        if mn in COND_JUMPS:
            m = _JUMP_TARGET_RE.match(code_line)
            if m:
                branch_target = m.group(1)
        result.append((taken, not_taken, addr, code_line, branch_target,
                        addr in fallthrough_addrs))
    return result

# ---------------------------------------------------------------------------
# addr2line
# ---------------------------------------------------------------------------

def resolve_addresses(binary, addresses):
    """Batch addr2line call for all addresses.  Returns a list of raw output
    lines parallel to addresses."""
    if not addresses:
        return []
    out = subprocess.check_output(
        ['addr2line', '-e', binary] + list(addresses),
        text=True, stderr=subprocess.DEVNULL,
    )
    return out.strip().splitlines()


def parse_addr2line(raw):
    """Parse one addr2line output line → (basename, lineno) or (None, None)."""
    raw = raw.strip()
    if raw.startswith('??'):
        return None, None
    m = re.match(r'^(.*):(\d+)', raw)
    if not m:
        return None, None
    return os.path.basename(m.group(1)), int(m.group(2))

# ---------------------------------------------------------------------------
# Branch probability computation
# ---------------------------------------------------------------------------

def compute_target_probabilities(all_instrs, addr_to_src, src_basename):
    """Return {target_lineno: taken_pct} for all sampled conditional jumps
    in src_basename.  Multiple jumps to the same target line are summed."""
    taken_by_target  = {}
    total_by_target  = {}

    for taken, not_taken, addr, _code, branch_target, _ft in all_instrs:
        if branch_target is None:
            continue
        total = taken + not_taken
        if total == 0:
            continue
        file_base, _ln = addr_to_src.get(addr, (None, None))
        if file_base != src_basename:
            continue
        bt_file, bt_lineno = addr_to_src.get(branch_target, (None, None))
        if bt_file != src_basename or bt_lineno is None:
            continue
        taken_by_target[bt_lineno]  = taken_by_target.get(bt_lineno, 0) + taken
        total_by_target[bt_lineno]  = total_by_target.get(bt_lineno, 0) + total

    return {
        ln: 100.0 * taken_by_target[ln] / total_by_target[ln]
        for ln in taken_by_target
        if total_by_target[ln] > 0
    }

# ---------------------------------------------------------------------------
# Output: markdown table (stdout) and optional .profile annotation
# ---------------------------------------------------------------------------

def build_markdown(func_groups_filtered, addr_to_src, cycles_by_addr,
                   src_basename, func_pct, min_samples):
    """Build and return the markdown branch-probability report."""
    lines = []
    any_data = False

    for func_name, instrs in func_groups_filtered:
        fallthrough_addrs = mark_fallthrough(instrs)
        expanded = expand_instructions(instrs, fallthrough_addrs)

        rows = []
        for taken, not_taken, addr, code_line, branch_target, is_ft in expanded:
            total = taken + not_taken
            if total == 0:
                continue
            file_base, lineno = addr_to_src.get(addr, (None, None))
            if file_base != src_basename:
                continue
            bt_file, bt_lineno = addr_to_src.get(branch_target, (None, None)) \
                if branch_target else (None, None)
            taken_pct = 100.0 * taken / total
            rows.append((total, taken_pct, lineno, bt_lineno, code_line))

        # Apply min-samples filter only when there is no function filter
        # (caller passes min_samples=0 when functions were explicitly named).
        rows = [r for r in rows if r[0] >= min_samples]
        if not rows:
            continue

        any_data = True
        pct_str = f" — {func_pct[func_name]:.1f}% cycles" \
            if func_name in func_pct else ""
        lines.append(f"## `{func_name}`{pct_str}\n")
        lines.append("| Taken% | Samples | Source line | Branch target | Assembly |\n")
        lines.append("|-------:|--------:|------------:|--------------:|----------|\n")

        rows.sort(key=lambda r: -r[0])  # hottest branches first
        for total, taken_pct, lineno, bt_lineno, code_line in rows:
            src_col = f":{lineno}" if lineno else "—"
            tgt_col = f":{bt_lineno}" if bt_lineno else "—"
            lines.append(
                f"| {taken_pct:6.1f}% | {total:>7} | {src_col:>11} |"
                f" {tgt_col:>13} | `{code_line}` |\n"
            )
        lines.append("\n")

    if not any_data:
        return "*(no sampled branches found — try a longer run or lower --period)*\n"
    return "".join(lines)


def write_profile(src_path, lineno_probs):
    """Write a copy of src_path annotated with /* PERF PROB: X.XX% */ comments
    before each branch-target line not on the fall-through path."""
    src_lines = Path(src_path).read_text().splitlines(keepends=True)
    output = []
    offset = 0
    for ln_probs in sorted(lineno_probs.items()):
        lineno, prob = ln_probs
        idx = lineno - 1 + offset
        if 0 <= idx < len(src_lines):
            indent = len(src_lines[idx]) - len(src_lines[idx].lstrip())
            comment = ' ' * indent + f'/* PERF PROB: {prob:.2f}% */\n'
            output.insert(idx, comment)
            offset += 1

    # Build final annotated list: original lines with inserts applied
    result = list(src_lines)
    for lineno in sorted(lineno_probs, reverse=True):
        idx = lineno - 1
        if 0 <= idx < len(result):
            indent = len(result[idx]) - len(result[idx].lstrip())
            comment = ' ' * indent + f'/* PERF PROB: {lineno_probs[lineno]:.2f}% */\n'
            result.insert(idx, comment)

    return result

# ---------------------------------------------------------------------------
# parse perf report for function hotness %
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(r'(\d+\.\d+)%')

def parse_perf_report(report_out):
    """Return {func_name: cycles_pct} for user-space symbols from perf report."""
    result = {}
    for line in report_out.splitlines():
        if not line or line.startswith('#'):
            continue
        pcts = _PCT_RE.findall(line)
        if not pcts or '[.]' not in line:
            continue
        after = line.split('[.]', 1)[1].strip()
        symbol = after.split()[0] if after else None
        if symbol:
            result[symbol] = float(pcts[-1])
    return result

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Measure branch probabilities with Linux perf (Intel PMU).')
    parser.add_argument('source',
                        help='C source file to analyse (e.g. foo.c)')
    parser.add_argument('binary',
                        help='Binary compiled with -g debug symbols')
    parser.add_argument('functions', nargs='*', metavar='function',
                        help='Function name(s) to report (default: all)')
    parser.add_argument('--period', type=int, default=1000,
                        help='perf sampling period (default: 1000)')
    parser.add_argument('--min-samples', type=int, default=5, metavar='N',
                        help='Suppress branches with fewer than N samples '
                             'when no function filter is active (default: 5)')
    parser.add_argument('--fuzzy', action='store_true',
                        help='Match function names as substrings')
    parser.add_argument('--profile', metavar='FILE',
                        help='Write annotated source copy to FILE')
    parser.add_argument('--debug', action='store_true',
                        help='Write raw perf annotate output to <source>.annotate.out')
    args = parser.parse_args()

    src_path = Path(args.source)
    src_basename = src_path.name
    binary = args.binary
    if not os.path.isabs(binary) and '/' not in binary:
        binary = './' + binary

    # --- perf record ---
    with tempfile.NamedTemporaryFile(suffix='.data', delete=False) as tf:
        perf_data = tf.name

    try:
        print('[1/3] Recording with perf...', file=sys.stderr, flush=True)
        subprocess.run(
            ['perf', 'record', '-c', str(args.period), '-e', PERF_EVENTS,
             '-o', perf_data, binary],
            check=True, stderr=subprocess.DEVNULL,
        )

        print('[2/3] Running perf annotate...', file=sys.stderr, flush=True)
        annotate_out = subprocess.check_output(
            ['perf', 'annotate', '--source', '--no-vmlinux',
             '-l', '-n', '--stdio', '-i', perf_data],
            text=True, stderr=subprocess.DEVNULL,
        )

        report_out = subprocess.check_output(
            ['perf', 'report', '--stdio', '-i', perf_data],
            text=True, stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(perf_data)

    if args.debug:
        debug_path = src_path.with_suffix(src_path.suffix + '.annotate.out')
        debug_path.write_text(annotate_out)
        print(f'[debug] Raw annotate output → {debug_path}', file=sys.stderr)

    print('[3/3] Processing samples...', file=sys.stderr, flush=True)

    func_pct = parse_perf_report(report_out)
    func_groups, cycles_by_addr = parse_annotate(annotate_out)

    # Apply macrofusion correction per function.
    for _func_name, func_instrs in func_groups:
        fix_macrofusion(func_instrs)

    # Filter to requested functions.
    if args.functions:
        if args.fuzzy:
            func_groups_filtered = [
                (fn, instrs) for fn, instrs in func_groups
                if any(needle in fn for needle in args.functions)
            ]
        else:
            requested = set(args.functions)
            func_groups_filtered = [
                (fn, instrs) for fn, instrs in func_groups
                if fn in requested
            ]
        if not func_groups_filtered:
            names = ', '.join(f'`{f}`' for f in args.functions)
            print(f'Warning: none of {names} found in perf annotate output.',
                  file=sys.stderr)
            print('Available functions with samples:',
                  ', '.join(fn for fn, _ in func_groups), file=sys.stderr)
        min_samples = 0  # don't filter by sample count when caller asked for specific functions
    else:
        func_groups_filtered = func_groups
        min_samples = args.min_samples

    # Collect all instruction addresses + branch targets for a single
    # batch addr2line call.
    all_instrs_flat = []
    fallthrough_by_func = {}
    for func_name, func_instrs in func_groups_filtered:
        ft = mark_fallthrough(func_instrs)
        fallthrough_by_func[func_name] = ft
        all_instrs_flat.extend(expand_instructions(func_instrs, ft))

    addrs = [i[2] for i in all_instrs_flat]
    target_addrs_extra = sorted(
        {i[4] for i in all_instrs_flat if i[4] is not None} - set(addrs)
    )
    all_addrs = addrs + target_addrs_extra
    raw_lines = resolve_addresses(binary, all_addrs)
    addr_to_src = {addr: parse_addr2line(raw)
                   for addr, raw in zip(all_addrs, raw_lines)}

    # --- stdout: markdown report ---
    header = (
        f'# Branch probabilities — `{src_basename}`\n\n'
        f'Measured with `perf` using Intel PMU events '
        f'`BR_INST_RETIRED.NEAR_TAKEN` / `NOT_TAKEN`.\n'
        f'Sampling period: {args.period} branch events per sample.\n\n'
    )
    if args.functions:
        header += f'Functions: {", ".join(f"`{f}`" for f in args.functions)}\n\n'

    table = build_markdown(func_groups_filtered, addr_to_src, cycles_by_addr,
                           src_basename, func_pct, min_samples)
    print(header + table)

    # --- optional annotated source ---
    if args.profile:
        lineno_probs = compute_target_probabilities(
            all_instrs_flat, addr_to_src, src_basename)
        # Suppress annotations for fall-through lines.
        ft_lines = set()
        for func_name, func_instrs in func_groups_filtered:
            ft = fallthrough_by_func[func_name]
            for _tk, _nt, addr, _mn, _rest in func_instrs:
                if addr in ft:
                    fb, ln = addr_to_src.get(addr, (None, None))
                    if fb == src_basename and ln is not None:
                        ft_lines.add(ln)
        lineno_probs = {ln: p for ln, p in lineno_probs.items()
                        if ln not in ft_lines}
        annotated = write_profile(src_path, lineno_probs)
        Path(args.profile).write_text(''.join(annotated))
        print(f'[profile] Annotated source → {args.profile}', file=sys.stderr)


if __name__ == '__main__':
    main()

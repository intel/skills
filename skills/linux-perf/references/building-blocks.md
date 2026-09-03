<!-- (C) 2026 Intel Corporation, MIT license -->
# Building blocks

Read this file when executing any building block referenced from Part 2 flows.

## Table of contents
- [Check CPU capabilities](#building-block-check-cpu-capabilities)
- [Ensure debug symbols](#building-block-ensure-debug-symbols)
- [perf stat](#building-block-perf-stat)
- [perf record](#building-block-perf-record)
- [perf annotate (assembly view)](#building-block-perf-annotate-assembly-view)
- [resolve address to source](#building-block-resolve-address-to-source)
- [c2c hot cache lines](#building-block-c2c-hot-cache-lines)
- [c2c access map for a cache line](#building-block-c2c-access-map-for-a-cache-line)
- [Top-N functions](#building-block-top-n-functions)
- [Top-N lines within a function](#building-block-top-n-lines-within-a-function)
- [Dual-profile comparison](#building-block-dual-profile-comparison)
- [Annotate pattern scan](#building-block-annotate-pattern-scan)
- [Branch probability measurement](#building-block-branch-probability-measurement)
- [GCC static branch probability](#building-block-gcc-static-branch-probability)

---


These are focused, composable primitives that flows and other skills can invoke directly. They collect and format one specific type of data with no interpretation overhead.

---

## Building block: Check CPU capabilities

Report which SIMD instruction sets and feature extensions the current CPU supports.
This result is used by the Annotate pattern scan and any other block that needs to
know the CPU's vector width tier or other capabilities.

**Check once per session and cache the result** — the CPU does not change between steps.

### Step 1 — Extract flags from `/proc/cpuinfo`

```bash
grep -m1 'flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse4_2|avx|avx2|fma|avx512f|avx512dq|avx512bw|avx512vl|avx512vnni|bmi|bmi2|popcnt|aes)$' | sort
```

Note: `/proc/cpuinfo` uses underscores (`sse4_2`, `avx512f`) rather than GCC
dot-notation (`sse4.2`). Also report online CPU count:

```bash
nproc
```

### Step 2 — Interpret for perf analysis

From the output, determine the **vector width tier** — used by the Annotate pattern scan:

| Tier | Condition | Max packed register |
|------|-----------|---------------------|
| AVX-512 | `avx512f` present | ZMM (512-bit) |
| AVX2 | `avx2` present, no avx512f | YMM (256-bit) |
| AVX | `avx` present, no avx2 | YMM (256-bit, packed FP only) |
| SSE | none of the above | XMM (128-bit) |

Report the tier as a one-liner: *"CPU vector width tier: AVX-512 (avx512f + avx512bw/dq/vl)"*

---

## Building block: Ensure debug symbols

Check whether the binary has DWARF debug info and, if it doesn't, offer to recompile with `-g` before any expensive perf collection begins. Running this check first avoids wasted time: a `perf annotate` pass on a stripped binary yields only assembly addresses, not source line numbers, making the results harder to act on.

**Run this building block before `perf record` in any flow that will reach `perf annotate`.**

### Step 1 — Identify the binary

Determine the full path of the binary to be profiled from the user's command or from the installed test directory (e.g., for phoronix tests: `~/.phoronix-test-suite/installed-tests/pts/<test-name>/`).

### Step 2 — Check for DWARF debug info

```bash
objdump --dwarf=info <binary> 2>/dev/null | grep -c 'DW_TAG' || echo 0
```

- **Count > 0** → DWARF present; source-line annotation will work. Proceed without recompiling.
- **Count = 0** → No DWARF. Note the result and proceed to Step 3.

Also check whether the binary is stripped (no symbol names at all):
```bash
nm <binary> 2>/dev/null | head -3 || echo "stripped"
```

### Step 3 — Offer to recompile (if DWARF absent)

> ⛔ **If the workload is `pts/<name>`, STOP here.** Do not attempt any manual recompilation. The `phoronix-test-suite` skill MUST handle the rebuild — invoke it now if not already done. It owns source extraction, `-g` injection, and binary deployment for all PTS tests.

For all other workloads, **check for a build skill first** (see priority table below). If
no skill is available, ask the user:
> *"The binary has no debug info (`-g` was not used at compile time), so `perf annotate` will show assembly addresses only — no source line numbers. Would you like me to recompile with `-g` for better annotation? (I'll re-run the profiling after.)"*

If the user declines, note in the final report that source line mapping is unavailable and proceed.

**If the user agrees, recompile — use a skill if one is available:**

| Priority | Workload type | Recompile approach |
|---|---|---|
| 1 | **Skill available** | Use the relevant skill's compile/rebuild step — it knows the source location, Makefile, install paths, and any special environment required. |
| 2 | **User-built binary** — no skill | Re-run the exact original build command with `-g` appended. Do **not** change any other flags. |
| 3 | **System binary** (e.g., `nginx`, `python3`) | Source not available for recompile. If the distro ships a `*-dbgsym` package, suggest installing it instead (`apt install <pkg>-dbgsym`). |

**Critical: only add `-g`, change nothing else.** Changing optimization flags (`-O3` → `-O2`) or removing defines (`-DUSE_X`) will produce a different binary that may exhibit different hotspots — the resulting profile will not represent the workload the user cares about.

After recompiling, verify the DWARF is now present by re-running Step 2.

### Step 4 — Output

If DWARF was present or has been added: *"Binary has debug symbols — source line annotation is available."*

If DWARF is absent and user declined recompile: *"Proceeding without debug symbols. `perf annotate` will show assembly only; source line numbers will not be available in the report."*

**Note**: after recompiling with `-g`, any existing `perf.data` is stale (the addresses no longer match). The calling flow must re-record before annotating.

---

## Building block: perf stat

Collect hardware performance counters for a command and return the raw numbers as a formatted table. No interpretation — just the counts and derived ratios.

### Invocation

Basic (default counters):
```bash
perf stat -- <command>
```

Comprehensive baseline set (recommended when in doubt):
```bash
perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses,branches -- <command>
```

### Useful flags

| Flag | Effect |
|------|--------|
| `-r <N>` | Repeat N times; report mean ± variance (reduces noise on short workloads) |
| `-e <event-list>` | Specific events, e.g. `cycles,instructions,cache-misses` |
| `--all-user` | Count only user-space events (ignore kernel time) |

### Output columns

`perf stat` reports each event with its raw count, a computed rate (e.g. IPC, miss rate), and a time-based rate. The key derived metrics are:

| Column / line | What it is |
|---|---|
| `insns per cycle` | IPC — instructions retired per clock |
| `cache-misses / cache-references` | Last-level cache miss rate |
| `branch-misses / branches` | Branch misprediction rate |
| `% of time in kernel` | Fraction spent in kernel (task-clock line) |

---

## Building block: perf record

Capture a performance recording. All flows and blocks that need a `perf.data` file start here.

**Parameters:**
- **command / workload** — what to profile
- **call graphs** — whether to capture caller chains (needed for Top-N functions, Flow B interpretation); ask the user if not specified

### Invocation

Basic (no call graphs):
```bash
perf record -- <command>
```

With call graphs (recommended unless recording is very long):
```bash
perf record --call-graph dwarf -- <command>
```

Quick call-graph alternative (lower overhead, less accurate for deep stacks):
```bash
perf record -g -- <command>
```

### Useful flags

| Flag | Effect |
|------|--------|
| `-e <event>` | Profile on a specific event (default: `cycles`) |
| `-a` | Record all CPUs (whole-system snapshot) |
| `-F <Hz>` | Sampling frequency (default: 4000 Hz; lower for long workloads) |
| `--all-user` | Omit kernel samples |

System-wide snapshot for a fixed duration:
```bash
perf record -a -- sleep 10
```

Reuse existing `perf.data` instead of re-recording whenever the user prefers or the workload is hard to reproduce.

---

## Building block: perf annotate (assembly view)

Annotate a function's assembly with per-instruction overhead percentages. Use this when you need to understand *what the CPU is executing* inside a hot function — not just which source line, but which instruction and why. Distinct from **Top-N lines within a function** which filters to source lines only.

**Prerequisite**: binary compiled with `-g`. If stripped, use **Building block: resolve address to source** instead.

**N is a parameter:**
- Specified (e.g. "top 5 instructions") → hard limit on instructions reported
- Omitted → accumulate until **cumulative instruction overhead exceeds 99%**; include the tipping entry

### Step 1 — Annotate

```bash
perf annotate --stdio <function_name> 2>/dev/null
```

For C++ names with templates or namespaces, quote them. If the symbol is ambiguous across DSOs, add `--dsos <binary_name>`.

### Step 2 — Extract hot instructions

Scan the annotate output for lines with a non-zero percentage in the leftmost column. Each such line is an assembly instruction with measured overhead. Apply the N rule on instructions sorted by overhead descending.

### Step 3 — Format as Markdown

| Overhead | Instruction | Source hint |
|----------|-------------|-------------|
| 35.00% | `mulps %ymm1, %ymm0` | `dot.c:42` |
| 20.00% | `addss %xmm1, %xmm0` | `dot.c:42` |
| 12.50% | `movaps (%rax), %ymm1` | `dot.c:38` |

---

## Building block: resolve address to source

Given a code address, find the corresponding source file and line. Use as a fallback when `perf annotate` or `perf report` does not show source information (stripped binary, kernel code, or inline-expanded code).

### For user-space binaries

```bash
addr2line -e <binary> -f -i <code_address>
```

- `-f` — print function name above each result
- `-i` — follow inlining chains (gives the full inline call stack, not just the outermost frame)

### For stripped binaries (no `-g`)

```bash
objdump -d --no-show-raw-insn <binary> | grep -A 60 "<function_name>:"
```

Correlate the target address against the instruction offsets in the disassembly manually. Offer to recompile with `-g` if the source is available — it's faster and more reliable.

### For kernel code (`[kernel.kallsyms]`)

Use the matching `vmlinux` debug image:
```bash
addr2line -e <vmlinux> -f -i <code_address>
```

If the debug image is unavailable, the `Symbol` + `Source:Line` columns from `perf report`/c2c Pareto output are usually sufficient to identify the problematic code without addr2line.

### Prefer other tools when available

If the agent has access to a language server, debugger, or symbol lookup tool, prefer those over `addr2line` — they handle inlining, templates, and multiple compilation units better.

---



## Building block: c2c hot cache lines

Surface the cache lines with the most HITM (Hit Modified) events from a c2c recording. Returns a Markdown table of hot cache lines — no Pareto analysis, no fix suggestions.

**N is a parameter:**
- Specified (e.g. "top 5 cache lines") → hard limit
- Omitted → accumulate `Tot Hitm` until the cumulative total exceeds 99%; include the entry that tips it over

**Threshold**: ignore entries below **5% Tot Hitm** by default; user may override.

### Step 1 — Record

```bash
perf c2c record -g -- <application>
```

`-g` captures call graphs needed for Pareto analysis later. Reuse existing `perf.data` if the user prefers.

### Step 2 — Report to file

```bash
perf c2c report --full-symbols --stdio --double-cl > c2c_report.txt
```

Always redirect — c2c output is thousands of lines. `--double-cl` groups by 128-byte double cache lines (matches hardware prefetcher granularity); `--full-symbols` prevents symbol truncation.

### Step 3 — Extract and format

```bash
grep -A 20 "Shared Data Cache Line Table" c2c_report.txt | grep -v "^#" | grep -v "^=" | grep -E "^\s+[0-9]"
```

Key columns:
- **Index** — 0-based; links to the Pareto section
- **Address** — virtual address of the double cache line
- **Tot Hitm** — fraction of total HITM events; primary importance score
- **LclHitm / RmtHitm** — local vs remote NUMA HITM

Apply the N / threshold rule, then report as a Markdown table:

| # | Address | Tot Hitm | LclHitm | RmtHitm |
|---|---------|----------|---------|---------|
| 0 | `0xffff8ede402e0800` | 15.96% | 34 | 0 |
| 1 | `0xffff8ede400e8800` | 6.10% | 13 | 0 |

---

## Building block: c2c access map for a cache line

Given a c2c report and a target cache line, extract which code locations are accessing it and classify the sharing pattern (false vs true). No fix suggestions — pure data extraction.

**Input**: `c2c_report.txt` + a cache line **index** (from the summary table above).

### Step 1 — Locate the Pareto block

In the **"Shared Cache Line Distribution Pareto"** section, find the block for the target index — each block is delimited by a dashed separator line containing the index and address:

```
  ----------------------------------------------------------------------
      0        0       34        0        0        0  0xffff8ede402e0800
  ----------------------------------------------------------------------
```

### Step 2 — Extract access rows

Each data row within the block is a code location accessing this cache line. Extract:
- **LclHitm %** — fraction of HITMs from this location
- **Offset** — byte offset within the cache line (e.g. `0x18`, `0x3c`)
- **Code address** — virtual address of the instruction
- **Symbol** — function name with `+offset`
- **Source:Line** — file and source line if available

### Step 3 — Classify: false vs true sharing

**The Offset column is the definitive discriminator — check this before reading source code:**

- **All significant accesses share the same Offset** → same field → **True sharing** (genuine contention on the same datum)
- **Significant accesses use different Offsets** → different fields landing on the same cache line → **False sharing**

### Step 4 — Format as Markdown

| LclHitm | Offset | Function | Source location |
|---------|--------|----------|----------------|
| 76.47% | `0x18` | `pty_write_room` | `pty.c:123` |
| 23.53% | `0x20` | `pty_write` | `pty.c:87` |

State the classification explicitly:
- `Different offsets (0x18 vs 0x20) → False sharing`
- `Same offset (0x00) → True sharing`

---

## Building block: Top-N functions

Surface the hottest functions from a recording as a Markdown table. No interpretation, no drill-down.

**N is a parameter:**
- Specified (e.g. "top 10 functions") → hard limit
- Omitted (e.g. "give me the top functions") → accumulate until **cumulative overhead exceeds 99%**; include the entry that tips it over

### Step 1 — Record

Use **Building block: perf record** (Part 4) with `-g` (call graphs are required for accurate per-function attribution).

Reuse existing `perf.data` if the user prefers.

### Step 2 — Report and extract

```bash
perf report --stdio -g none 2>/dev/null | grep -E '^\s+[0-9]+\.[0-9]+%'
```

`-g none` collapses call graphs so each function appears exactly once.

- **N specified**: pipe through `head -N`
- **N omitted**: accumulate percentages; stop after the entry that pushes the total past 99%

### Step 3 — Format as Markdown

| Rank | Overhead | Cumulative | Function | Binary / Object |
|------|----------|------------|----------|-----------------|
| 1 | 35.12% | 35.12% | `some_function` | `myprogram` |
| 2 | 12.34% | 47.46% | `sys_read` | `[kernel]` |

- **Cumulative** — running total; shows how dominant the top entries are
- Strip the `[.]`/`[k]` marker from Function but note userspace vs kernel in a footnote

Include a one-line header: profiled command + sample count (from "Samples: N of event '...'" in `perf report --stdio` output).

---

## Building block: Top-N lines within a function

Given a known-hot function, pinpoint which source lines consume cycles. Returns source file + line numbers — no assembly analysis, no cross-skill calls.

**Prerequisite**: binary compiled with `-g`. If stripped, offer to recompile first.

**N is a parameter:**
- Specified (e.g. "top 5 lines") → hard limit
- Omitted → accumulate until **cumulative source-line overhead exceeds 99%**; include the tipping entry. If fewer than 3 lines have attributed overhead, include all of them.

### Step 1 — Annotate

```bash
perf annotate --stdio <function_name> 2>/dev/null
```

For C++ names with templates or namespaces, quote them. If the symbol is ambiguous across DSOs, add `--dsos <binary_name>`.

### Step 2 — Extract hot source lines

Scan for lines where **both** hold:
1. A non-zero percentage in the leftmost column (e.g. `  12.50  :`)
2. The line is **source code** (plain C/C++), not an assembly mnemonic

The annotate header shows the canonical source file path. Sort by overhead descending, then apply the N rule.

### Step 3 — Format as Markdown

```
## Hot lines in `<function_name>`

Source file: /absolute/path/to/source.cpp
Profiled binary: <binary>  |  Event: cycles

| Line | Overhead | Cumulative | Source snippet                    |
|------|----------|------------|-----------------------------------|
|   42 | 35.00%   | 35.00%     | x = a[i] * b[i];                 |
|   45 | 20.00%   | 55.00%     | sum += partial;                   |
|   38 | 12.50%   | 67.50%     | for (size_t i = 0; i < n; ++i) { |
```

If no source lines can be attributed (binary stripped), say so and suggest recompiling with `-g`.

---

## Building block: Dual-profile comparison

Run the workload under two conditions — single-core and all-cores — and produce a
side-by-side delta table of the top-15 hottest functions. The functions that "jump up"
between the single-core and all-cores profiles are the targets for scaling investigation.

### Preparation

Determine the core count:
```bash
nproc          # number of usable cores on this system
```

Ensure the workload can be pinned to a single core via `taskset`. If the workload has
its own thread-count flag (e.g., `--threads 1`), use that instead of `taskset`.

### Step 1 — Single-core profile

```bash
taskset -c 0 perf record -F 99 -a -g -o perf_1core.data -- <command>
perf report -i perf_1core.data --stdio --no-children -n \
    --sort=comm,dso,sym 2>/dev/null \
    | grep -E '^\s+[0-9]+\.[0-9]+%' | head -15 > /tmp/top15_1core.txt
```

If `taskset` is not usable (privileged environment, container), use the workload's own
single-thread flag and note the difference in the report.

### Step 2 — All-cores profile

```bash
perf record -F 99 -a -g -o perf_ncore.data -- <command>
perf report -i perf_ncore.data --stdio --no-children -n \
    --sort=comm,dso,sym 2>/dev/null \
    | grep -E '^\s+[0-9]+\.[0-9]+%' | head -15 > /tmp/top15_ncore.txt
```

### Step 3 — Midpoint profile (optional)

Only collect this when the delta in Step 4 is overwhelming (more than ~7 functions
jump significantly, making it hard to prioritize). The problem is often that the
1-core baseline is noisy — contention effects appear even at 2–3 cores, so the
1-vs-N delta catches everything at once. Comparing **N/2-core vs N-core** instead
filters out that low-count noise and surfaces only the bottlenecks that grow in the
high-core-count regime, which is usually the actionable one.

If Phase 0 was run, `perf_halfcore.data` is already available — skip straight to
the report step. Otherwise record now:

```bash
HALF=$(( $(nproc) / 2 ))
taskset -c 0-$((HALF-1)) perf record -F 99 -a -g -o perf_mid.data -- <command>
perf report -i perf_mid.data --stdio --no-children -n \
    --sort=comm,dso,sym 2>/dev/null \
    | grep -E '^\s+[0-9]+\.[0-9]+%' | head -15 > /tmp/top15_mid.txt
```

Use the midpoint list as the new baseline: compare **N-core vs N/2-core** to get a
shorter, more actionable jumper list focused on the high-core-count scaling wall.

### Step 4 — Produce the delta table

Present the two top-15 lists side by side. The "baseline" profile is 1-core normally,
or N/2-core when using the midpoint narrowing from Step 3. Label the header accordingly:

```markdown
## Dual-profile comparison — <workload> — <N> cores vs 1 core

| N-core rank | Function | N-core % | 1-core rank | 1-core % | Rank Δ | % Δ      |
|-------------|----------|----------|-------------|----------|--------|----------|
| 1           | `spin_lock` | 12.3% | 8        | 2.1%     | ▲7     | +10.2%   |
| 2           | `some_func` | 8.5%  | 5        | 6.2%     | ▲3     | +2.3%    |
| 3           | `new_func`  | 7.1%  | —        | —        | new    | +7.1%    |
```

When using N/2-core as the baseline (Step 3 midpoint):

```markdown
## Dual-profile comparison — <workload> — <N> cores vs <N/2> cores

| N-core rank | Function | N-core % | N/2-core rank | N/2-core % | Rank Δ | % Δ   |
|-------------|----------|----------|---------------|------------|--------|-------|
```

Functions not in the baseline top-15 get rank `—` and % `—`; flag them as **new
entrants** in the Rank Δ column.

### Step 5 — Identify "jumper" functions

Flag a function as a **jumper** (scaling bottleneck candidate) if any of these
thresholds are met:

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Rank rise | ≥ 3 places | A function at rank 12 in 1-core but rank 3 in N-core is a strong signal |
| Percentage ratio | ≥ 2× increase | From 2% at 1-core to ≥ 4% at N-core |
| Absolute percentage gain | ≥ 3% | From 1% to 4% is a 4× ratio but absolute gain matters too |
| New entrant | any % ≥ 2% | Function not in top-15 at 1-core but visible at N-core |

These thresholds are defaults — use judgment for borderline cases where several
criteria are close to the threshold simultaneously.

Bold the jumper rows in the delta table. These functions are the input to the
next phase of Flow D.

---

## Building block: Annotate pattern scan

Scan the assembly output of `perf annotate` for a single function and return a
structured table of detected performance anti-patterns. Each entry names the pattern,
cites the assembly evidence, and references the applicable resolution strategy.

Flows use this output differently: diagnostic flows (Flow B) apply the suggested
resolution strategy; reporting flows (Flow E) surface the pattern name and evidence
as observation bullets and omit the RS column.

### Prerequisites

**CPU capability flags** — some patterns depend on knowing what the CPU supports.
Use **Building block: Check CPU capabilities** (Part 4) once per session and cache
the result. From its output, note the vector width tier (AVX-512 / AVX2 / AVX / SSE).

**Perf stat numbers** (optional) — IPC and cache-miss rate from a prior `perf stat`
run improve confidence for the serial-accumulator pattern. Pass them in if available.

### Step 1 — Run annotate

```bash
perf annotate --stdio -l -s <function_name> 2>/dev/null | tee /tmp/annotate_<function>.txt
```

The output contains:
- Source lines with their percentage (relative within the function, sums to ~100%)
- Assembly instructions with per-instruction percentages
- Source ↔ assembly interleaving

### Step 2 — Apply pattern checklist

Work through these checks in order. A function may match multiple patterns.

This list is mandatory but not exhaustive: in addition to these patterns, the agent
should do an analysis step on the source code to see if any other known problem
patterns are in play.

#### Pattern 1: Scalar FP — no vectorization

**Signal**: the hot instructions are exclusively scalar floating-point variants —
`vmovsd`, `vaddsd`, `vmulsd`, `vfmadd*sd`, `vcvtsd*` — and **no** packed
(`vaddpd`, `vmulps`, `vfmadd*ps`, etc.) instructions appear anywhere in the output.

**Confidence**: high if scalar instructions account for > 50% of the function's
samples and zero packed instructions appear.

**Suggested RS**: SIMD vector width upconversion (after diagnosing the vectorization
blocker — most commonly a strided inner loop or missing `restrict`).

---

#### Pattern 2: Narrow SIMD — register width opportunity

**Signal**: packed instructions use `xmm` registers (128-bit SSE) on a CPU with
AVX2 or AVX-512 support, or `ymm` registers (256-bit AVX2) on a CPU with AVX-512.

**Confidence**: high if the CPU capability check confirms the wider width is
available and the packed instructions are in the hot path (> 20% of function samples).

**Suggested RS**: SIMD vector width upconversion.

---

#### Pattern 3: Serial accumulator

**Signal**: a **single** FP instruction (e.g., `vaddss`, `vaddpd`, `vfmadd213ps`,
`vfmadd213pd`) accounts for a disproportionate share (> 40%) of the function's
samples, **and** either:
- IPC (from perf stat) is ≪ 1 (< 0.5), **or**
- Cache-miss rate (from perf stat) is low (< 2%), **or**
- The instruction's source operand and destination operand are the same register
  (e.g., `vaddss %xmm0, %xmm1, %xmm0`) forming an obvious dependency chain

**Confidence**: high when two or more of the above sub-signals are present.
Medium when only the dominant-instruction signal is present without perf stat data.

**Suggested RS**: Parallel accumulator rewrite.

---

#### Pattern 4: Horizontal-reduction anti-pattern

**Signal**: `shufps`, `addss`, or `unpckhps` instructions appear **immediately
after** (within 3–5 instructions of) a `mulps`, `mulss`, or similar packed multiply.
These are the signature of a compiler-generated horizontal reduction of a SIMD
multiply result.

**Confidence**: high if this cluster accounts for > 10% of function samples combined.

**Suggested RS**: Parallel accumulator rewrite (the horizontal-reduction variant —
the `performance-patterns` skill will recognize this specific pattern).

---

#### Pattern 5: Test-and-Set spin (`lock cmpxchg`)

**Signal**: `lock cmpxchg`, `lock xchg`, or `lock cmpxchg8b` appears in a tight
loop (the annotate output shows the same or adjacent source line repeated, or the
instruction is inside a loop body) and accounts for > 10% of the function's samples.

**Confidence**: high if the hot `lock cmpxchg` is clearly inside a loop structure.
Medium if it appears once with no visible loop context.

**Suggested RS**: Test-and-Test-and-Set (TTAS).

---

#### Pattern 6: Memory load pressure

**Signal**: load instructions (`vmovsd`, `vmovups`, `movq`, `vmovdqu`, `movaps`)
account for the largest share of samples (> 30% combined) with no obvious FP
computation at comparable weight.

**Confidence**: medium — may indicate cache misses, but could also be a pipeline
stall from data dependency. Correlate with high cache-miss rate from `perf stat`
to increase confidence.

**Suggested RS**: none (structural — consider cache locality, working-set reduction,
or prefetching; not covered by the current resolution strategies).

---

#### Pattern 7: Atomic counter contention

**Signal**: `lock add`, `lock inc`, `lock xadd`, or `lock sub` instructions appear in
the hot path and account for a meaningful share (> 10%) of the function's samples.
These are the signature of an atomic increment/decrement on a **shared** counter — as
opposed to Pattern 5's `lock cmpxchg` which indicates a spin-wait loop.

Distinguish from Pattern 5:
- `lock cmpxchg` / `lock xchg` → spinlock / CAS loop → apply TTAS (Pattern 5)
- `lock add` / `lock inc` / `lock xadd` → could be either; check what follows:
  - **Loop on the result** (conditional branch `jz`/`jnz`/`je` immediately after, or
    the flags are tested before the next iteration) → this is a lock implemented with
    an add → apply TTAS (Pattern 5)
  - **Result completely ignored** (no branch, no test, execution falls straight
    through) → this is a statistics counter → apply Per-CPU statistics aggregation
    (this pattern)

**Confidence**: high if the instruction is clearly not inside a CAS retry loop and the
function name or field name suggests accounting (`count`, `hits`, `stat`, `total`, etc.).
Correlate with c2c HITM on the same address to confirm cross-thread contention.

**Suggested RS**: Per-CPU statistics aggregation.

---

### Step 3 — Format the output

Present matched patterns as a table. Omit patterns with no match.

```markdown
### Pattern scan — `<function_name>`

| Pattern | Evidence | Suggested RS |
|---------|----------|-------------|
| Scalar FP — no vectorization | `vaddsd` at 95% of samples; no packed instructions found | SIMD vector width upconversion |
| Serial accumulator | `vaddsd` at 95%, accumulates into `xmm0` each iteration; IPC 0.4 | Parallel accumulator rewrite |
```

If no patterns match: output a single line — *"No anti-patterns detected in
`<function_name>`."*

**When invoked from a reporting flow (Flow E)**: suppress the Suggested RS column
and present only the Pattern and Evidence columns as observation bullets.


---

## Building block: Branch probability measurement

Measure the real runtime probability of each conditional branch in a hot
function using Intel PMU branch-retirement events.  Answers: *"which of the
branches inside this function are almost never taken?"* — the key input for
deciding which callees to annotate `[[gnu::cold]]`.

**Intel CPUs only.**  The events `BR_INST_RETIRED.NEAR_TAKEN` and
`BR_INST_RETIRED.NOT_TAKEN` are Intel-specific.  Check before using:

```bash
grep -q 'vendor_id.*GenuineIntel' /proc/cpuinfo || echo "NOT an Intel CPU"
```

### Manual procedure

Use this when `tools/branchprob.py` is not available.

**Step 1 — Record**

```bash
perf record -c 1000 \
  -e '{BR_INST_RETIRED.NEAR_TAKEN:upp,BR_INST_RETIRED.NOT_TAKEN:upp,cycles:u}' \
  -o /tmp/branch.data \
  ./binary
```

Lower `--period` (e.g. `-c 100`) for more samples on short-lived workloads.
The `:upp` modifier requests user-space, precise, precise-IP sampling — reduces
skid so counts land on the right instruction.

**Step 2 — Annotate**

```bash
perf annotate --source --no-vmlinux -l -n --stdio -i /tmp/branch.data \
  > /tmp/branch.annotate
```

Output format per data line:
```
  NEAR_TAKEN   NOT_TAKEN   CYCLES :   ADDR:   MNEMONIC  [operands]
```
The event group order matches the recording order: col1 = taken, col2 = not-taken.

**Step 3 — Macrofusion correction**

x86 CPUs can fuse a `cmp`/`test` with the following conditional jump into one
micro-op.  When this happens the PMU attributes both counts to the *compare*
instruction, not the jump.  Detect and correct:

- Find a non-jump instruction with nonzero col1 or col2, immediately followed
  by a conditional jump.
- Move both counts from the compare to the jump; zero the compare.

**Step 4 — Calculate probabilities**

For each conditional jump instruction:

```
taken%  = 100 × col1 / (col1 + col2)
```

Resolve instruction addresses to source lines with `addr2line`:

```bash
addr2line -e ./binary ADDR1 ADDR2 ...
```

Group results by source line.  A branch with `taken% < 0.1%` is a strong cold
candidate; the callee it reaches should be reviewed for `[[gnu::cold]]`.

**Step 5 — Interpret**

Present results as a table sorted by sample count (hottest branches first):

| Taken% | Samples | Source line | Branch target | Assembly |
|-------:|--------:|------------:|--------------:|----------|
|  91.2% |  607094 | :11         | :18           | `ja ...` |
|   0.3% |     412 | :27         | :42           | `je ...` |

A `Taken%` near 0% or near 100% means the branch is highly predictable.
Near 0% taken means the *target* line is almost never reached — that line's
callee is a `[[gnu::cold]]` candidate.

---

### Automated procedure

`tools/branchprob.py` automates Steps 1–5.  Its output goes to **stdout** for
direct agent ingestion.  Progress messages go to stderr.

**Basic usage:**

```bash
python3 tools/branchprob.py <source.c> <binary>
```

**Focused on hot functions** (recommended when you already know which functions
are hot from Flow B):

```bash
python3 tools/branchprob.py foo.c foo process_data compute_hash
```

Only branches inside `process_data` and `compute_hash` are reported.
This keeps the output small and directly actionable.

**Substring matching** (for overloaded or versioned names):

```bash
python3 tools/branchprob.py foo.c foo --fuzzy process
# matches: process_data, process_request, process_v2, ...
```

**Write an annotated source copy** (for the human to read):

```bash
python3 tools/branchprob.py foo.c foo process_data --profile foo.c.profile
```

**Options summary:**

| Option | Default | Effect |
|--------|---------|--------|
| `[function ...]` | (all) | Restrict output to named functions |
| `--period N` | 1000 | Sampling period (lower = more samples) |
| `--min-samples N` | 5 | Hide branches with fewer samples (ignored when functions are named) |
| `--fuzzy` | off | Substring function-name matching |
| `--profile FILE` | none | Write `/* PERF PROB: X.XX% */` annotated source |
| `--debug` | off | Save raw `perf annotate` output for inspection |

**Interpreting the output:**

The script reports one section per function.  Within each section, branches
are sorted by sample count (hottest first).  `Taken%` is the fraction of
executions where the branch was taken.  `Branch target` is the source line
the jump lands on when taken.

A branch with low `Taken%` (< ~0.1%) means execution almost never follows
that path — the callee at the target line is a candidate for
`[[gnu::cold]]`.  See `patterns/cold-path-annotation.md` in the
`performance-patterns` skill.

---

## Building block: GCC static branch probability

Report GCC's compile-time branch-probability estimates for a C/C++ source file. Use this when:
- **No workload is available** to run `branchprob.py` — GCC's heuristics are a useful proxy
- **Calibrating perf data** — divergences between GCC estimates and measured `Taken%` highlight optimization opportunities
- **Cross-platform** — unlike the perf-based building block this works on any architecture GCC targets

**Platform note:** GCC static estimates are available on any architecture. The Intel-specific PMU events used by `branchprob.py` are NOT required.

### Manual procedure

**Step 1 — Recompile with profile-estimate dump enabled**

The agent should inject these extra flags into the project's build command. The exact mechanism is project-specific:

| Build system | How to add flags |
|---|---|
| Plain `gcc`/`g++` | Append to the compile invocation |
| `make` | `make CFLAGS="$CFLAGS -fdump-tree-profile_estimate-lineno -dumpdir dump/" ...` |
| CMake | `cmake -DCMAKE_C_FLAGS="... -fdump-tree-profile_estimate-lineno -dumpdir dump/" ...` |
| Meson | Add to `c_args` in `meson.build` or pass `--cflags-override` depending on project |
| Custom build | Ask the user for the correct flag injection method |

The flags to inject:
```
-g -fdump-tree-profile_estimate-lineno -dumpdir dump/
```

Create the dump directory first: `mkdir -p dump/`

**Step 2 — Locate the dump file**

After compilation:
```bash
ls dump/*.profile_estimate
```

The file is named `<source_basename>.<NNN>t.profile_estimate` where `NNN` is a GCC version-specific pass number. If multiple files match, the correct one has a name like `foo.c.053t.profile_estimate`.

**Step 3 — Inspect a function**

Open the dump file and search for the function. Each function section starts with:
```
;; Function function_name (function_name, ...)
```

Within each section, condition blocks show their outgoing edges with probabilities:
```
  [foo.c:11:8] if (_2 == 0)
    goto <bb 3>; [50.00%]
  else
    goto <bb 5>; [50.00%]
```

The `[X.XX%]` is GCC's estimated probability for that edge. Near-zero percentages (< ~5%) indicate cold paths.

**Step 4 — Interpret**

Present results as a table (one row per conditional branch edge), sorted by probability ascending (coldest first):

| GCC Est% | Source line | Target line |
|---------:|------------:|------------:|
|   0.00%  | :27         | :30         |
|  33.00%  | :13         | :14         |
|  50.00%  | :11         | :12         |

A `GCC Est%` near 0% means the compiler predicts the target line is almost never reached. The callee at that line is a `[[gnu::cold]]` candidate.

If GCC estimates disagree significantly with `branchprob.py`'s `Taken%`:
- **GCC hot / perf cold** → GCC was wrong; annotating `[[gnu::cold]]` would be a meaningful win
- **GCC cold / perf hot** → may need `[[gnu::hot]]`; profile first
- **GCC ~50-50 / perf strongly skewed** → data-dependent behaviour GCC cannot see statically

---

### Automated procedure

`tools/gccbranchprob.py` automates Steps 2–4. Its output goes to **stdout** for
direct agent ingestion. Progress messages go to stderr.

**Basic usage:**

```bash
python3 tools/gccbranchprob.py <source.c> <dump_dir>
```

**Focused on specific functions** (recommended when you already know which functions are hot):

```bash
python3 tools/gccbranchprob.py foo.c dump/ process_data compute_hash
```

Only branches inside `process_data` and `compute_hash` are reported.

**Substring matching** (for overloaded or versioned names):

```bash
python3 tools/gccbranchprob.py foo.c dump/ --fuzzy process
# matches: process_data, process_request, process_v2, ...
```

**Options summary:**

| Option | Default | Effect |
|--------|---------|--------|
| `[function ...]` | (all) | Restrict output to named functions |
| `--fuzzy` | off | Substring function-name matching |

**Interpreting the output:**

The script reports one section per function. Within each section, branches are
sorted by `(condition_line, target_line)`. `GCC Est%` is the compiler's estimate
of how often execution follows that edge. `Source line` is the `if (...)` line;
`Target line` is the first line of the basic block reached when the branch fires.

A `GCC Est%` of 0% or near-0% means the compiler expects the path is almost
never taken — the callee at the target line is a candidate for `[[gnu::cold]]`.
See `patterns/cold-path-annotation.md` in the `performance-patterns` skill.

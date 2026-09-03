---
name: linux-perf
description: >-
  Profile and fix Linux performance problems using `perf`. Workflows: (A) hardware counters -- IPC, cache-miss, branch mispredictions; (B) hotspot profiling -- which functions and source lines consume CPU, with SIMD and accumulator detection; (C) cache-line contention -- false sharing, HITM, `perf c2c`; (D) core-count scaling -- dual-profile comparison, bottleneck categorization; (E) structured hotspot report with annotated source and pattern observations. Resolution strategies: TTAS spinlock, SIMD upconversion, parallel accumulator, structured false-sharing fix, per-CPU stats. Trigger on: perf, profiling, profile, hotspot, hotspots, cache miss, IPC, false sharing, HITM, scaling, core count, thread scaling, bottleneck, slow code, CPU bound, why is this slow, where does time go, does not scale. When in doubt, invoke this skill -- better to use it unnecessarily than to miss a performance opportunity.
---

<!-- (C) 2026 Intel Corporation, MIT license -->


# Linux perf profiling skill

Guide the user through profiling with Linux `perf` — from setup and data collection through reporting and interpretation.

The skill is organized into five parts:
- **Part 1: Setup** — permissions and build flags (always check these first)
- **Part 2: Flows** — end-to-end workflows (A: quick stats, B: hotspot profiling, C: cache contention)
- **Part 3: Cross-skill integrations** and **Reference** — when to delegate, and quick lookup tables
- **Part 4: Building blocks** — focused data collection primitives that flows and other skills can call directly
- **Part 5: Resolution strategies** — common fix patterns that flows can reference by name

---

# Part 1: setup

## Check permissions (always do this first)

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

- **≤ 1**: no `sudo` needed for most data collection
- **2 or higher**: hardware events (cycles, cache-misses, etc.) require root

To lower the limit (ask the user first):
```bash
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

**Sudo protocol**: if `sudo` is needed, present these options and wait for the user's answer before proceeding:
1. Run with `sudo`
2. Lower paranoia level with the command above
3. Skip perf collection

## Ensure debug symbols

**Always compile with `-g`** alongside existing optimization flags (e.g. `-O2 -g`). This adds DWARF debug info so `perf report` and `perf annotate` show function names and source lines — it does not weaken optimization.

If given a pre-built binary without `-g`, offer to recompile. If you cannot, use **Building block: resolve address to source** (Part 4) for address correlation.

```bash
gcc -O2 -g -o <output> <source>
```

---

# Part 2: flows

Choose a flow based on what the user needs:

| Flow | Best for | Time needed |
|------|----------|-------------|
| **Flow A** — `perf stat` | Quick counters: IPC, cache-miss rate, branch-miss rate | Seconds |
| **Flow B** — `perf record` + `perf report` | Which functions/lines are hot | Minutes |
| **Flow C** — `perf c2c` | Cache-line contention in multi-threaded code | Minutes |
| **Flow D** — dual-profile + focused c2c | Core-count scaling problems: workload does not get faster (or gets slower) as threads increase | 30+ min |
| **Flow E** — hotspot analysis report | Structured formatted report: top functions table + annotated source with % column + pattern observations | Minutes |

When in doubt, start with Flow A — it's fast and often answers the question without a full recording.

---

## Flow A: Quick statistics with `perf stat`

Best for workloads like `openssl speed ...` or any case where you want a fast CPU efficiency answer.

Use **Building block: perf stat** (Part 4) to collect the counters.


Read `references/flow-a.md` for the execution steps.

---

## Flow B: Hotspot profiling with `perf record` + `perf report`

Use this when the user wants to know *which functions* are consuming time, not just aggregate stats.


Read `references/flow-b.md` for the execution steps.

---

## Flow C: Cache-line contention with `perf c2c`

Use to diagnose **false sharing** or **true sharing** of cache lines in multi-threaded code.

**Trigger this flow when:**
- User mentions multithreaded scalability problems
- Flow B shows significant non-streaming loads/stores (repeated access to the same small structures, not iterating over large arrays/trees)
- `lock` or `mutex` appears prominently in function names, variable names, or hotspot call chains
- User explicitly mentions cache line contention, false sharing, or HITM events

**Threshold**: ignore entries below **5% Tot Hitm** by default. User may override (e.g. "show me everything above 2%").


Read `references/flow-c.md` for the execution steps.

---

## Flow D: Core-count scaling analysis

Use when a workload does not scale with core count — throughput plateaus or regresses
as threads increase.

**Trigger this flow when:**
- User says "adding more threads doesn't help" or "performance gets worse with more cores"
- A scaling sweep (throughput vs core count) shows a flat or declining curve
- Flamegraph or profile shows lock-related functions dominating at high core counts

Read `references/flow-d.md` for the execution steps.

---

## Flow E: Hotspot analysis report

Use when the user wants a structured, formatted deliverable rather than an exploratory
profiling session — a document they can read, share, or file.

**Trigger this flow when:**
- User asks for a "hotspot report", "profiling report", or "analysis report"
- User wants to know the top functions with annotated source and percentages
- User wants a formatted, shareable summary of where time is spent

Read `references/flow-e.md` for the execution steps.

---


# Part 3: cross-skill integrations

## `phoronix-test-suite`: PTS benchmarks

**Invoke `phoronix-test-suite` immediately** — the moment the profiling target is identified as a `pts/<name>` benchmark, before any perf steps begin.

### Trigger

- **Workload is `pts/<name>`** (any Phoronix Test Suite benchmark) → invoke `phoronix-test-suite` skill before Phase 0 of any flow.

### What it handles

The `phoronix-test-suite` skill owns the full lifecycle for PTS benchmarks: install, source extraction, rebuild (with `-g`), binary deployment, and result recording. It knows the correct `install.sh` build flags, source layout, and where to copy the resulting binary — things a manual `gcc` or `make` invocation will get wrong.

**Do not attempt to install, build, or rebuild a PTS test manually.** Always delegate to this skill.

When the trigger fires, say: *"I'll invoke the `phoronix-test-suite` skill to handle install and rebuild before profiling."* then invoke it — before running any `perf` commands.

---

## SIMD optimization → `performance-patterns`

**Invoke `performance-patterns` as early as possible** — trigger it the moment you see any of these patterns, even from reading source code or looking at `perf stat` output. Do not wait for `perf annotate`.

### Early triggers (source code or perf stat — no annotate needed)

- **Serial accumulator in source** (`s += a[i] * b[i]`, `sum += x[i]`, running max/min) → always a SIMD opportunity; invoke `performance-patterns` immediately
- **IPC < 2.0 + CPU-bound** (low kernel%, low cache-miss rate) → dependency-chain stall; serial accumulator is the most common cause

### Later triggers (from Flow B report or annotate output)

- **Scalar instructions in a hot loop** (`addsd`, `mulss`, `movsd` without `ymm`/`zmm`)
- **Narrow SIMD** (`xmm`) on a CPU that supports `ymm` (AVX2) or `zmm` (AVX-512)
- **Horizontal-reduction anti-pattern** (`shufps`/`addss`/`unpckhps` after `mulps`) — see Flow B Phase 4
- **Extreme cost on first SSE instruction after an AVX function** — AVX↔SSE transition penalty; see below

### AVX↔SSE transition penalty (missing vzeroupper)

Detect with:
```bash
perf stat -e other_assists.avx_to_sse,other_assists.sse_to_avx ./program
```

A non-zero `other_assists.avx_to_sse` count confirms the penalty. In
`perf annotate`, it appears as an extreme cycle count on the first SSE instruction
following a function that used `ymm` or `zmm0`–`zmm15` registers.

When detected, invoke `performance-patterns` with the `missing-vzeroupper` pattern:
read `patterns/missing-vzeroupper.md` for the fix.

When any trigger fires, say: *"I'll now invoke the `performance-patterns` skill to analyze and optimize the hot loop."* then invoke it — before doing further manual assembly analysis yourself.

---

# Part 4: building blocks

Detailed command syntax, flags, and output formats for each primitive. Read
`references/building-blocks.md` when you need to execute one.

| Building block | Purpose |
|----------------|---------|
| `Check CPU capabilities` | Read `/proc/cpuinfo` flags and `nproc` to determine supported ISA levels, vector width tier, and feature extensions; used by Annotate pattern scan and other blocks |
| `Ensure debug symbols` | Check binary for DWARF info; offer to recompile with `-g` before expensive perf collection |
| `perf stat` | Collect hardware counters (IPC, cache-miss rate, branch-miss rate) |
| `perf record` | Sample a workload and write a perf.data file |
| `perf annotate (assembly view)` | Per-instruction cycle attribution for a named function |
| `resolve address to source` | Map raw addresses → file:line via addr2line / objdump |
| `c2c hot cache lines` | Record + report cache-line HITM summary table |
| `c2c access map for a cache line` | Per-offset accessor table for one cache line |
| `Top-N functions` | Ranked list of hottest functions from a perf.data recording |
| `Top-N lines within a function` | Ranked source lines inside one function |
| `Dual-profile comparison` | Run top-15 at 1 core and at N cores; produce delta table of rank/% changes to identify scaling bottleneck candidates |
| `Annotate pattern scan` | Scan `perf annotate` output for a function and return a structured table of detected anti-patterns (scalar FP, narrow SIMD, serial accumulator, horizontal reduction, lock CAS, memory load pressure) with suggested resolution strategies |
| `Branch probability measurement` | Measure per-branch taken-probabilities in hot functions using Intel PMU events; identify near-zero-probability branches as `[[gnu::cold]]` candidates |
| `GCC static branch probability` | Parse GCC's compile-time profile_estimate dump to obtain static branch-probability estimates; works on any platform; use as a proxy when no workload is available, or compare against perf data to find divergences worth optimizing |

# Part 5: resolution strategies

The resolution strategies are owned by the **`performance-patterns` skill**.
When a flow identifies a named pattern, invoke `performance-patterns` — it will
load `triggers/from-profile.md` to match the signal, then the appropriate
`patterns/<name>.md` for the full fix.

| Resolution strategy | `performance-patterns` pattern file |
|---------------------|-------------------------------------|
| Test-and-Test-and-Set (TTAS) | `patterns/ttas.md` |
| SIMD vector width upconversion | `patterns/simd-upconversion.md` |
| Parallel accumulator rewrite | `patterns/parallel-accumulator.md` |
| Structured false-sharing fix | `patterns/false-sharing.md` |
| Per-CPU statistics aggregation | `patterns/per-cpu-stats.md` |
| Missing vzeroupper (AVX↔SSE penalty) | `patterns/missing-vzeroupper.md` |

# Reference

## Useful commands

```bash
perf list                    # list all available event types
sudo perf list               # full list including hardware events (requires root)
```

Always check `perf list` before using `-e <event>` — event names vary between kernel versions and hardware.
Some perf events require full root (sudo) privileges; the updated paranoia level is not sufficient for these.

## Common event names

| Event | What it measures |
|-------|-----------------|
| `cycles` | CPU clock cycles (default) |
| `instructions` | Instructions retired |
| `cache-misses` | Last-level cache misses |
| `cache-references` | Last-level cache accesses |
| `branch-misses` | Branch mispredictions |
| `branches` | Branch instructions retired |
| `page-faults` | Page faults (useful for I/O-heavy workloads) |
| `context-switches` | OS context switches |
| `cpu-migrations` | Process migrations between CPUs |

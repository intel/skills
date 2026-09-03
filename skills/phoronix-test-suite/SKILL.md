---
name: phoronix-test-suite
description: Install, run, parse, and optimize benchmarks from the Phoronix Test Suite (PTS). Use this skill whenever the user mentions "phoronix", "pts/", or "phoronix-test-suite", or asks to run, measure, improve, or optimize a PTS test — e.g., "run pts/mt-dgemm", "optimize pts/compress-zstd", "what score does pts/x265 get". Trigger immediately on any `pts/<testname>` reference, even if the user doesn't explicitly say "phoronix". Also trigger when the user asks to find or edit the source code of a PTS test.
---

<!-- (C) 2026 Intel Corporation, MIT license -->

# Phoronix Test Suite skill

End-to-end support for running and optimizing benchmarks with the Phoronix Test Suite CLI (`phoronix-test-suite`). **PTS** is used as shorthand throughout.

This skill is organized into two parts:
- **Part 1: flows** — end-to-end workflows (running, optimizing)
- **Part 2: primitives** — focused building blocks that flows compose

---

# Part 1: flows

## Flow A: running a test

Use when the user wants to run or benchmark a PTS test.

1. **Resolve the test version** — if not specified, run [Refresh test list](#refresh-test-list) and then [Resolve test version](#resolve-test-version)
2. **Install** — use [Install a test](#install-a-test)
3. **Run** — use [Run a test](#run-a-test)
4. **Parse output** — use [Parse test output](#parse-test-output)
5. **Save and report** — save to the [results store](#results-persistence) and report to the user:
   - State the test name, version, score, and unit
   - If a prior result for this test exists in the store, report the delta using [Evaluate test scores](#evaluate-test-scores)

---

## Flow B: optimizing a test

Use when the user wants to improve the performance of a PTS test.

**Step 1 — Establish a baseline**
Run Flow A and label the result `"baseline"` in the results store.

**Step 2 — Analyze the code**
- Use [Prepare source code for editing](#prepare-source-code-for-editing) to locate the source.
- Read the test's entry-point shell script in the install directory to find the compiled binary.
- **Invoke the `linux-perf` skill** to profile the binary and identify hotspots.
- **Invoke the `performance-patterns` skill** to identify SIMD, vectorization, and other recognized code-pattern opportunities in hot paths.
- Check `generated.json` → `honors_cflags`: if `"1"`, compiler flag changes (for example, `-march=native`, `-O3`, `-mavx512f`) are respected by the test runner and are worth trying.
- Check `pts-install.json` in the install directory for the original compiler flags as a starting point.

**Step 3 — Make improvements**
Apply optimizations such as SIMD rewrites, compiler-flag changes, algorithmic changes, or data-layout changes. Edit the source directly in the prepared source directory.

**Step 4 — Rebuild and measure**
- Use [Rebuild a test](#rebuild-a-test)
- Run Flow A again and label the result with a short description of the change (for example, `"after-avx512"`)
- Compare using [Evaluate test scores](#evaluate-test-scores)

**Step 5 — Report using this template**

```
## Optimization Report: pts/<test-name>-<version>

### Baseline
- Score: <value> <unit>

### Changes Made
1. <Change description>
2. <Change description>
...

### Result
- Score: <value> <unit>
- Delta: <+/- value> (<+/- percentage>%)
- Verdict: **Improved** / **Regressed** / **No change**

### Pending Ideas
<Any ideas not yet tried, or "None">
```

**Step 6 — Offer another round**
If "Pending Ideas" is non-empty, ask: "Would you like me to try another round of optimization?"

---

# Part 2: primitives

## Resolve PTS root directory

PTS stores data in different locations:
- Running as **root**: `/var/lib/phoronix-test-suite`
- Running as a **regular user**: `~/.phoronix-test-suite`

Check which path exists. This is `$ROOT` throughout the rest of this skill.

---

## Refresh test list

Run this before resolving versions to ensure the local test-profile cache is up to date with OpenBenchmarking.org. This downloads the latest test metadata and may reveal newer versions than what is locally cached.

```bash
phoronix-test-suite list-available-tests
```

Run this whenever:
- The user asks for the latest version and you have not refreshed yet in this session
- A test is not found in the local test-profiles directory
- The user suspects a newer version exists

---

## Resolve test version

When no version is specified, find the latest version available:

1. **Refresh first** — run [Refresh test list](#refresh-test-list) to pull the latest profiles from OpenBenchmarking.org
2. **Check the local cache** — find the highest version in the test-profiles directory:

```bash
ls $ROOT/test-profiles/pts/ | grep "^<test-name>-" | sort -V | tail -1
```

Use the version suffix from the resulting directory name (for example, `mt-dgemm-1.2.0` → version `1.2.0`).

---

## Install a test

**Check whether it is already installed** — skip the install if this path exists:
```
$ROOT/installed-tests/pts/<test-name>-<version>/<test-name>
```

If it is not installed:
```bash
phoronix-test-suite batch-install <test-name>-<version>
```

---

## Run a test

Ensure the test is installed first. Running a test can take a long time — 10 minutes is common.

**Auto-detect `FORCE_TIMES_TO_RUN`** based on `run_time_avg` from `generated.json`:
- If `run_time_avg < 10` seconds → set `FORCE_TIMES_TO_RUN=5`
- Otherwise → set `FORCE_TIMES_TO_RUN=1`

```bash
export FORCE_TIMES_TO_RUN=<value>
phoronix-test-suite batch-run <test-name>-<version>
```

---

## Parse test output

PTS run output ends with a summary block. Example:

```
ACES DGEMM 1.0:
    pts/mt-dgemm-1.2.0
    Test 1 of 1
    ...
        Started Run 1 @ 16:00:50

    Sustained Floating-Point Rate:
        2.361582

    Average: 2.361582 GFLOP/s
    Samples: 1
```

Extract:
- **Score**: the numeric value on the `Average:` line
- **Unit**: the unit string on the same line (for example, `GFLOP/s`)

---

## Evaluate test scores

Read `hib` from `$ROOT/test-profiles/pts/<test-name>-<version>/generated.json`:

- `hib = 1` (**Higher Is Better**): score A is better if `A > B`
- `hib = 0` (**Lower Is Better**): score A is better if `A < B`

---

## Prepare source code for editing

First, read `install.sh` in the test-definition directory:
```
$ROOT/test-profiles/pts/<test-name>-<version>/install.sh
```

If there is no compilation step (pre-built binary), source editing is not possible — tell the user.

**Scenario 1 — Source directory is present**
`install.sh` does not delete the extracted directory after building. The source is already in the install directory. Find its name by looking at the `tar -x` (or equivalent) extraction step in `install.sh`.

**Scenario 2 — Source directory was deleted**
`install.sh` removes the source directory after building, often because it conflicts with the test entry-point name. To recreate it safely:

```bash
mkdir -p $ROOT/installed-tests/pts/<test-name>-<version>/src
cd $ROOT/installed-tests/pts/<test-name>-<version>/src
# then follow the extraction steps from install.sh, e.g.:
tar -xf ../<archive>
```

> **Important**: Always extract into `src/` rather than directly into the install directory. Extracting in place risks overwriting the test's main entry-point script if the archive contains a directory or file of the same name.

---

## Rebuild a test

Read `install.sh` to understand the build process. The build commands generally start after the extraction step.

**Scenario 1 — Source is in place**: navigate to the source directory and re-run the build commands from `install.sh`.

**Scenario 2 — Source is in `src/`**: navigate to the extracted source inside `src/` and run the same build commands.

After building, `install.sh` typically copies the resulting binary to the install directory. Repeat that copy:

```bash
cp <binary> $ROOT/installed-tests/pts/<test-name>-<version>/
```

---

## Results persistence

After each test run, append the result to the session's results store.

**Location**: `files/pts-results.json` inside the session folder (the path is listed in `<session_context>` at the top of each conversation). If no session context is available, use `~/.pts-results.json`.

**Format**:
```json
{
  "results": [
    {
      "test": "pts/mt-dgemm-1.2.0",
      "score": 2.361582,
      "unit": "GFLOP/s",
      "hib": 1,
      "timestamp": "2025-04-11T16:00:50Z",
      "label": "baseline",
      "notes": ""
    }
  ]
}
```

Use `label` to tag runs (for example, `"baseline"`, `"after-avx512"`, `"round-2"`). When reporting results, look up prior entries for the same test name and report a delta if found.

---

## Key file reference

| File | Location | Purpose |
|------|----------|---------|
| `generated.json` | `test-profiles/pts/<test>-<ver>/` | unit, hib, run_time_avg, honors_cflags, scales_cpu_cores |
| `install.sh` | `test-profiles/pts/<test>-<ver>/` | build and install steps; use as a guide for rebuild and source preparation |
| `test-definition.xml` | `test-profiles/pts/<test>-<ver>/` | human-readable test description |
| `pts-install.json` | `installed-tests/pts/<test>-<ver>/` | compiler flags used at install time |
| `<test-name>` | `installed-tests/pts/<test>-<ver>/` | entry-point script; parse to find the compiled binary path |

For full PTS CLI reference, see the [upstream documentation](https://raw.githubusercontent.com/phoronix-test-suite/phoronix-test-suite/refs/heads/master/documentation/phoronix-test-suite.md).

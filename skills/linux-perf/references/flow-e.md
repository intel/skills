<!-- (C) 2026 Intel Corporation, MIT license -->
# Flow E: execution steps — Hotspot analysis report

Produce a structured, formatted hotspot report: a top-functions table followed by
per-function annotated source with a percentage column. The output is a self-contained
Markdown document.

---

## Phase 0 — Ensure debug symbols

Run **Building block: Ensure debug symbols** (Part 4) on the binary before recording.
If the binary lacks DWARF and the user agrees to recompile, do so now — before Phase 1 —
so that the recording and annotation use the same binary. If the user declines, note the
limitation; source-unavailable fallback format applies in Phase 4a.

---

## Phase 1 — Record

Use **Building block: perf record** (Part 4). Note the wall-clock duration of the
recording — it determines whether to run `perf stat` in Phase 2.

If a `perf.data` file already exists, do not ask if you know it is valid based on the session:
> *"I found an existing perf.data file. Should I use it, or re-record?"*

Call graphs are not required for this flow (percentages come from annotate, not call
chains), but if the user has already recorded with `--call-graph dwarf`, that data can
be reused.

---

## Phase 2 — Perf stat (conditional)

| Condition | Action |
|-----------|--------|
| `perf stat` was already collected in this session | Reuse those numbers — do not re-run |
| Recording wall-clock duration < 60 s | Run `perf stat` now using **Building block: perf stat** (Part 4) |
| Recording duration ≥ 60 s | Skip — note its absence in the report header |

**If perf stat data is available**, format it as a compact summary table followed by
one sentence describing the performance regime:

```markdown
## System-level summary

| Metric              | Value  |
|---------------------|--------|
| IPC                 | 0.62   |
| Cache-miss rate     | 67.3%  |
| Branch-miss rate    | 1.2%   |
| CPU time            | 4.21 s |

*The workload is **memory-bound** (low IPC with high LLC miss rate).*
```

Regime sentence rules:
- IPC < 1.0 **and** cache-miss rate > 3% → "memory-bound (low IPC with high LLC miss rate)"
- IPC ≥ 2.0 **and** cache-miss rate < 1% → "compute-bound (high IPC, few cache misses)"
- kernel% > 30% → "I/O- or syscall-bound (dominant kernel time)"
- Otherwise → "moderately CPU-bound (IPC within normal range)"

---

## Phase 3 — Top functions table

Use **Building block: Top-N functions** (Part 4) to get the ranked function list.

**Selection cutoffs** (apply the first rule that fires):
- Default maximum: **5 functions**; user may override (e.g., "show me 7" → use 7)
- Stop when the next function would be **below 5%**
- Stop when the **cumulative percentage has already reached or exceeded 95%** —
  but always include the function that tips it over 95%

**Location column**: show the function's **definition line** (not the hottest line).
Obtain from `perf report` output or `addr2line`. If unavailable, show `<binary>` or
`<stripped>`.

**Percentage**: round to nearest whole percent.

```markdown
## Top functions

| Rank | Function     | Location      | % |
|------|--------------|---------------|---|
|    1 | `foo(int)`   | src/foo.c:91  | 79% |
|    2 | `bar(void)`  | src/bar.c:21  | 19% |
```

---

## Phase 4 — Per-function detailed sections

For each function in the top table, produce a named section with annotated source.

### Step 4a — Get annotated source

Run `perf annotate` in source mode for each function:

```bash
perf annotate --stdio -l -s <function_name> 2>/dev/null
```

This outputs source lines interleaved with assembly, with per-line percentages that
are **relative within the function** (they sum to approximately 100%). Use the
source-line percentages for the report; use the assembly lines for Observations (4d).

If source lines are not available (stripped binary or missing `-g`), show the annotated
**assembly** instead — using the same 6-char % prefix format. Add a note above the block:

> ⚠️ *Binary has no debug symbols — assembly view shown. Source is available at `<path>` if known.*

Assembly fallback format: show the hot inner loop(s) with a few instructions of context
before and after. Omit cold setup/teardown preamble entirely. Use `...` (6-space prefix)
to represent omitted sections. Add a brief `;` comment per instruction to name what it
does (load, FMA, store, loop back, etc.) — this replaces the source-line context that
would otherwise be visible.

Example assembly fallback:

```asm
      ; <function_name> — inner loop (source unavailable)
      ...
   36%  :   3b30:   vmovupd (%rbx,%rdi,1),%ymm0    ; load C[j]
   44%  :   3b35:   vfmadd213pd (%rax,%rdi,1),%ymm2,%ymm0  ; C += a_val * A
   14%  :   3b3b:   vfmadd231pd (%r9,%rdi,1),%ymm1,%ymm0   ; C += a_val2 * B
    5%  :   3b41:   vmovupd %ymm0,(%rax,%rdi,1)    ; store C[j]
         :   3b46:   add    $0x20,%rdi
         :   3b4d:   jne    3b30                   ; loop back
      ...
```

### Step 4b — Short vs long function

Count the number of **source lines** in the function body (excluding blank lines and
comments if ambiguous).

- **≤ 20 source lines**: show the **entire function** — no omissions, no `...`
- **> 20 source lines**: apply the summarization strategy in Step 4c

### Step 4c — Summarization strategy (functions > 20 lines)

Show a condensed view that keeps the most relevant context and hides the rest.

**Always show:**
1. **Function signature and opening brace** (no percentage needed even if % > 0 there)
2. **Variable definitions** for any variable that appears in a hot line — helps
   the reader understand the hot expression without needing to scroll elsewhere
3. **Closing `}`**

**For each hot line (≥ 5% within-function), show:**
- 1–2 lines of context **before** the hot line
- The hot line itself
- 1–2 lines of context **after** the hot line
- If the hot line is inside a loop, show the **loop header** as well — loops
  amplify the per-iteration cost and are essential context

**Merging**: if the context windows of two adjacent hot regions overlap, merge them
into a single continuous block (no `...` between them).

**Omitted code**: represent with a single `...` line (6-space prefix, no percentage).

### Step 4d — Percentage column format

Apply this prefix to **every line shown** (hot lines, context lines, and the function
header and footer):

```
[pos 1–4: right-aligned integer][pos 5: % sign][pos 6: space][source line as-is]
```

Rules:
- Percentage is the **within-function** share of cycles for that source line
- Round to the **nearest whole number** (0.5% rounds up to 1%)
- Lines with **< 0.5%** (would round to 0) show **6 spaces** — no number, no `%`
- The `%` sign is always at position 5; numbers are right-aligned against it:
  - 1% → `    1% ` (3 spaces + digit + % + space)
  - 12% → `   12% ` (2 spaces + digits + % + space)
  - 99% → `   99% ` (1 space + digits + % + space)
- `...` omission lines get 6 spaces (`      ...`)
- After the 6-char prefix, the source line appears **exactly as in the source file**,
  preserving its original indentation

Example (long function, summarized):

```c

      void foo(int a)
      {
          int loop;
          double counter = 0;
          ...
          for (loop = 1; loop < a; loop++) {
              ...
   4%         do_something_else();
  95%         counter += exp(memory[loop]);
              ...
          }
          ...
      }

```

Example (short function, ≤ 20 lines, shown in full):

```c

      void bar(void)
      {
          int loop;
          double counter = 0;

          for (loop = 1; loop < 5000; loop++) {
 100%          counter += exp(memory[loop]);
          }
      }

```

### Step 4e — Observations

Run **Building block: Annotate pattern scan** (Part 4) on the assembly output from Step 4a.
Present each detected pattern as a bullet — include the **Pattern** name and **Evidence**;
omit the Suggested RS column (this is a reporting flow, not a prescriptive one).

If no patterns are detected, omit this section.

**Do not prescribe fixes.** Name the pattern only. If the user wants to act on an
observation, they can invoke the appropriate flow or resolution strategy separately.

---

## Phase 5 — Output and save

Print the complete report to the terminal in Markdown.

**Save prompt** — only if this flow was invoked directly by a user (not called from
another flow or agent). After printing, ask:
> *"Would you like to save this report to `hotspot-report-<appname>.md`?"*

If yes, write the file. If the session is non-interactive (invoked programmatically),
skip this prompt entirely.

---

## Report template

````markdown
# Hotspot report for `<appname>`

Hotspot analysis was performed using the command: `<full command line>`

## System-level summary

| Metric              | Value  |
|---------------------|--------|
| IPC                 | X.XX   |
| Cache-miss rate     | XX.X%  |
| Branch-miss rate    | X.X%   |
| CPU time            | X.XX s |

*<one-sentence regime description>*

---

## Top functions

| Rank | Function     | Location      | %   |
|------|--------------|---------------|-----|
|    1 | `<function>` | `<file>:<line>` | XX% |
|    2 | `<function>` | `<file>:<line>` | XX% |

---

### Function 1: `<function>` detailed report

```c

      <annotated source with % column>

```

**Observations:**
- <pattern, if any>

---

### Function 2: `<function>` detailed report

```c

      <annotated source with % column>

```

---
````

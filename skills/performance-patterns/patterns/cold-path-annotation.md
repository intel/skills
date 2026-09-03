<!-- Modified by intel/skills: upstream repository-relative paths rewritten to resolve where this skill installs. Provenance: .source.json -->
<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: cold-path annotation

## When to apply

Apply when a **hot function** calls other functions that are only reached on
rarely-taken branches. Marking those callees `[[gnu::cold]]` (or
`__attribute__((cold))`) tells the compiler that the branch leading to them is
almost never taken. The compiler responds by reordering code so that the cold
callees are emitted far from the hot path — improving instruction-cache density
and generating branch-predictor-friendly constructs for the caller.

The benefit is **indirect**: the cold function itself does not speed up. The
caller gets faster because its hot-path instructions are no longer interleaved
with rarely-executed cold-path instructions.

### When is a callee cold?

A callee qualifies as cold when it is reached on a **rarely-taken branch** at
its call site — a branch probability below roughly 0.1%:

```c
if (likely_condition) {         /* taken ~99.9%+ of the time */
    hot_path();
} else {
    cold_callee();              /* ← cold: rarely reached */
}
```

### Source code signals

Scan the callees of any function identified as hot. A callee is a cold candidate
when it matches any of these patterns:

**By purpose:**
- Error reporters: functions whose entire job is to log or surface a failure
  (`report_error`, `log_warn`, `fatal`, `die`, `panic`, `oom_handler`, …)
- Impossible-state handlers: assertions, contract violations, `abort()` wrappers
- Setup/teardown paths called once per process or once per connection
  (`init_once`, `cleanup`, `teardown`, …)
- Very rare corner-case handlers with a clear "this should almost never happen"
  comment

**By body:**
- Body contains only (or primarily): `fprintf`, `fputs`, `perror`, `syslog`,
  `exit`, `abort`, `throw`, or a combination of these
- Body constructs an error message string (string formatting, `snprintf`)
- Body is a single call to another cold candidate

**By call site:**
- Called only inside `if (ret < 0)`, `if (errno != 0)`, `if (!ptr)`,
  `if (n > max_allowed)`, or similar guard conditions
- Call is inside a `default:` case of a `switch` that handles unexpected values
- Call is guarded by an assertion macro or a bounds check

---

## Why the hot path suffers without it

Without the annotation the compiler has no way to know how likely each branch
is. It interleaves the cold-path instructions with the hot-path instructions in
program order. This has two costs for the caller:

1. **Instruction-cache pollution.** The cold-path instructions occupy cache
   lines. Every time the hot path runs, it potentially evicts useful hot-path
   instructions to make room for code that almost never executes.

2. **Branch predictor pressure.** The compiler generates generic branch
   sequences. With the annotation it can emit the branch in a form the CPU's
   static branch predictor recognizes as "almost never taken," saving a
   mis-prediction penalty.

---

## The fix

### Syntax

Prefer `[[gnu::cold]]` in modern C++ (C++11 and later); use
`__attribute__((cold))` for older standards or plain C:

```cpp
// Modern C++
[[gnu::cold]]
void report_error(std::string_view message)
{
    std::cerr << "error: " << message << '\n';
}

// Older C or C++ standards
__attribute__((cold))
static void report_error(const char *message)
{
    fprintf(stderr, "error: %s\n", message);
}
```

Place the annotation on the **function definition**. Adding it to a declaration
is allowed but has no effect unless the definition is also annotated.

### Applying to a hot function's callees

When you have identified a hot function, scan each of its direct callees:

1. Identify all callees that match the cold-candidate signals above.
2. Annotate each cold callee at its definition.
3. If a cold callee itself calls further error-only helpers, annotate those too
   (the annotation does not automatically propagate to callees of callees).

**Example — before:**

```cpp
void process(const Record *records, int n) {
    for (int i = 0; i < n; i++) {
        if (records[i].magic != MAGIC) {
            report_corruption(records[i]);   // cold callee, not annotated
            continue;
        }
        do_work(records[i]);                 // hot path
    }
}

void report_corruption(const Record &r) {
    fprintf(stderr, "bad magic 0x%x at record %d\n", r.magic, r.id);
}
```

**After:**

```cpp
[[gnu::cold]]
void report_corruption(const Record &r) {
    fprintf(stderr, "bad magic 0x%x at record %d\n", r.magic, r.id);
}

/* process() is unchanged; the benefit comes from annotating the callee */
```

---

## Verification

### Confirm candidate branches are truly cold (data-driven)

If you have an Intel CPU and the binary is compiled with `-g`, use the
**Branch probability measurement** building block from the `linux-perf` skill
to confirm the branch probability before annotating:

```bash
python3 linux-perf/tools/branchprob.py foo.c foo process
```

Look at the `Taken%` column for branches whose target is the candidate callee.
A `Taken%` below ~0.1% is strong evidence the callee is cold and the annotation
is appropriate.  This step is optional but removes guesswork for borderline cases.

**No workload available? Use GCC's static estimate as a proxy.**

Recompile once with extra dump flags, then run `gccbranchprob.py`:

```bash
# recompile — inject these flags via your build system
gcc -g -fdump-tree-profile_estimate-lineno -dumpdir dump/ ... foo.c

python3 linux-perf/tools/gccbranchprob.py foo.c dump/ process
```

Look at `GCC Est%`. A value near 0% means the compiler itself expects this
branch to be almost never taken — a strong cold-path signal even without runtime
data.  See the **GCC static branch probability** building block in `linux-perf`
for full details.

### Confirm the annotation moved code out of the hot path

Build with `-O2` (or `-O3`) and compare the generated assembly:

```bash
g++ -O2 -S -o before.s before.cpp
g++ -O2 -S -o after.s after.cpp
```

In `after.s`, the hot path of `process()` should be a tight loop with the
cold branch jumping forward (out of the main loop body), and `report_corruption`
should appear after the function's main return sequence — not interleaved within
the loop.

A quick check:
```bash
# Confirm the cold function is placed after the return in the object:
objdump -d after.o | grep -A 20 '<process>'
```

The cold callee's instructions should appear well after the `ret` of the hot
function, not inside the hot loop.

---

## Presenting this to the user

1. Show the hot function with its callees identified and the cold candidates
   highlighted.
2. Explain that the annotation benefits the **caller**, not the cold function
   itself — the compiler will move the cold code out of the hot loop's
   instruction stream.
3. Show the annotated version of each cold callee (the hot function body is
   unchanged).
4. Optionally show the before/after assembly diff for the hot function to make
   the code-motion effect visible.

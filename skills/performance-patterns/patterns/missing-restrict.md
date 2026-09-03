<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: missing `restrict` on pointer parameters

## When to apply

Apply to **C functions** (not C++) that take two or more pointer parameters
where at least one is written, and those pointers are guaranteed by the caller
to refer to non-overlapping memory. Without `restrict`, the compiler must assume
that a write through one pointer might alias a read through another, which forces
it to either:

- emit a **runtime alias check** (versioned loop: overlap test → scalar fallback
  or vectorized path), adding preamble code and i-cache pressure, or
- **give up on auto-vectorization entirely** for that loop.

`restrict` is a contract: you promise the compiler the pointers do not overlap.
In return it eliminates the alias check and may produce a tighter, fully
vectorized loop.

This pattern applies **only to C**. C++ does not have a standard `restrict`
keyword; `__restrict__` (GCC/Clang extension) can be used in C++ but is not
portable.

### Source code signals

- A C function takes two or more pointer parameters, at least one is written
- The function processes independent input and output buffers:
  `void process(float *in, float *out, int n)`,
  `void add(const float *a, const float *b, float *dst, int n)`
- No `restrict` qualifier on any of the pointer parameters
- The loop body reads from one pointer and writes to a different one
- Caller documentation or usage clearly shows the buffers never overlap

### Profiling signals

- `perf annotate` shows a **scalar preamble** before the main loop body —
  typically a pointer subtraction, comparison, and conditional branch to a
  scalar fallback — even though the loop body is otherwise simple and
  vectorizable
- Two versions of the same loop appear in the assembly: one vectorized
  (`ymm`/`zmm` instructions), one scalar; the scalar version is the aliasing
  fallback and should be unreachable in normal use
- A hot loop is fully scalar (`addss`, `movss`, `mulss`) even though there are
  no dependencies, branches, or function calls in the loop body — the compiler
  abandoned vectorization due to aliasing uncertainty

---

## Why this is slow

The C standard allows any two pointers of compatible type to alias each other.
When the compiler sees:

```c
void add(float *a, float *b, float *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = a[i] + b[i];
}
```

It cannot prove that `dst` doesn't overlap with `a` or `b`. A write to
`dst[i]` could change the value that `a[i+1]` or `b[i+1]` reads on the next
iteration. To handle this safely the compiler either:

1. Generates **two loop versions** and a runtime overlap check — the vectorized
   path is taken only when the check confirms no aliasing. This adds ~10–20
   instructions of preamble before every call and increases i-cache footprint.
2. Stays **fully scalar** if the compiler's cost model decides the versioned
   approach is not worth it.

Either outcome wastes cycles on alias bookkeeping that the programmer knows is
unnecessary.

---

## The fix

Add `restrict` to each pointer parameter that the caller guarantees will not
overlap any other pointer parameter:

```c
/* Before */
void add(float *a, float *b, float *dst, int n);

/* After — input buffers marked const restrict, output marked restrict */
void add(const float * restrict a,
         const float * restrict b,
         float       * restrict dst,
         int n);
```

With `restrict`:
- The compiler eliminates the runtime overlap check entirely.
- The loop becomes a single straight-line vectorized body — no scalar fallback,
  no branching preamble.
- `const` on input pointers is a separate but complementary improvement: it
  signals read-only access and prevents accidental writes.

### Guidelines for applying `restrict`

| Pointer role | Qualifier |
|---|---|
| Read-only input | `const float * restrict` |
| Write-only output | `float * restrict` |
| Read-write in-place (e.g. `scale(float *buf, ...)`) | `float * restrict` |
| Pointer to a scalar parameter (e.g. `int *count`) — never aliased | `restrict` rarely needed; omit unless profiler confirms overhead |

### Confirming the alias check is gone

Check the generated assembly before and after:

```bash
# Before: look for pointer subtraction + conditional branch before the loop
gcc -O2 -S -o before.s add.c
grep -A20 "add:" before.s   # look for sub/cmp/jb preamble

# After: the preamble should be gone; loop starts immediately
gcc -O2 -S -o after.s add.c
grep -A20 "add:" after.s
```

Or with a more targeted check:

```bash
gcc -O2 -fopt-info-vec=vec.log add.c
grep "restrict\|alias\|version" vec.log
```

---

## Caveats and when not to apply

- **`restrict` is a correctness contract.** If you add it and the caller ever
  passes overlapping pointers, the behavior is undefined — not just slow.
  Confirm the no-overlap invariant with code review, documentation, or an
  assert before applying.
- **Do not apply to aliased parameters by design.** An in-place filter
  (`filter(float *buf, float *buf, int n)`) is intentionally aliased; adding
  `restrict` would be incorrect.
- **C++ only:** use `__restrict__` (GCC/Clang) as a non-standard extension if
  needed. It has the same semantics but is not guaranteed portable across all
  compilers.
- **Inline functions and LTO**: with link-time optimization (`-flto`), the
  compiler may already prove non-aliasing at call sites. `restrict` is still
  useful for the header/declaration alone without LTO.

---

## Verification

1. **Assembly check** — compare `gcc -O2 -S` output before and after; confirm
   the scalar aliasing fallback and preamble overhead are gone.
2. **Correctness** — run the full test suite. If any test fails after adding
   `restrict`, the caller was passing overlapping pointers — remove `restrict`
   and fix the caller instead.
3. **Profiling** — rerun `perf annotate`; the scalar preamble instructions
   should disappear from the hot path.

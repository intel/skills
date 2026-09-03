<!-- (C) 2026 Intel Corporation, MIT license -->
# Library: runtime CPU dispatch

A reusable implementation guide for selecting the fastest available variant of a
function at runtime, based on what the CPU supports. This technique applies
whenever a pattern fix produces multiple performance-level implementations of the
same function.

**Used by:** `patterns/simd-upconversion.md`, `patterns/parallel-accumulator.md`,
and any future pattern that generates multiple CPU-specific variants.

---

## Decision guide: which mechanism to use?

| Situation | Use |
|-----------|-----|
| Plain C/C++ loop, no manual intrinsics, compiler already auto-vectorizes it | **Mechanism A: `target_clones`** |
| Code uses `_mm_*` / `_mm256_*` intrinsics, or inline asm, or you just wrote new hand-tuned variants | **Mechanism B: `__builtin_cpu_supports`** |
| You need `target_clones` on existing code that already has the attribute but at a lower ISA level | **Mechanism A — extend the clone list** |
| Cross-compiler or non-x86 target | **Mechanism B — more portable** |

When in doubt, use **Mechanism B** — it works for everything, requires no compiler magic, and makes the dispatch logic explicit and auditable.

---

## Mechanism A: `target_clones` (compiler-driven, plain C/C++ only)

`target_clones` instructs the compiler to emit multiple clones of a function —
one per listed ISA level — and wire them to an IFUNC resolver that picks the best
available variant at load time.

**Hard limits:**
- `target_clones` does **not** upgrade manually written intrinsics to a different
  width. If the code contains `_mm256_*` calls, the compiler cannot promote them
  to `_mm512_*` automatically. Use Mechanism B for those.
- The function must be **non-static** (the IFUNC resolver requires an external
  symbol) and effectively **non-inlined** — if the compiler inlines it everywhere,
  no resolver is emitted and the dispatch silently disappears. Always pair with
  `__attribute__((noinline))`.

### ISA target strings

Use **feature strings**, not microarchitecture level names. The `x86-64-v3` /
`x86-64-v4` names work with `-march=` on the command line but are **not** accepted
by the `target_clones` attribute in GCC:

| Clone target string | What it enables |
|---------------------|-----------------|
| `"default"` | Baseline (SSE2 on x86-64) |
| `"avx2,fma"` | AVX2 + FMA (Haswell and later) |
| `"avx512f"` | AVX-512F (Skylake-X, Ice Lake, and later) |

GCC decomposes comma-separated strings into individual clones. Always include
`"default"` as the last entry — it is the fallback for CPUs that match nothing else.

### Project-wide macro template

Define this in a shared header so it degrades gracefully on non-x86 targets and
older compilers:

```c
#ifndef __target_clones
#  ifdef __x86_64__
#    define __target_clones \
         __attribute__((noinline, target_clones("default", "avx2,fma", "avx512f")))
#  else
#    define __target_clones   /* no-op on non-x86 */
#  endif
#endif
```

### Usage

```c
__target_clones
void process(float *src, float *dst, int n)
{
    /* plain C loop — compiler generates scalar/AVX2/AVX-512 clones */
    for (int i = 0; i < n; i++)
        dst[i] = src[i] * 2.0f;
}
```

### Verifying the resolver was emitted

```bash
objdump -t <binary> | grep process    # should show a .resolver symbol
```

If no `.resolver` appears, the function was inlined. Add `__attribute__((noinline))`
explicitly rather than relying on `target_clones` to do it.

### Extending an existing `target_clones` attribute

If the code already has `target_clones` but at a lower ISA level, add the missing
targets. Replace `"x86-64-v3"` with `"avx2,fma"` and `"x86-64-v4"` with
`"avx512f"` if present — these string forms are not accepted by `target_clones`.

---

## Mechanism B: `__builtin_cpu_supports` (hand-written variants)

Use when you have (or are about to write) multiple hand-tuned variants and need to
wire them together. The dispatch function runs once at the first call, stores the
result in a `static` function pointer, and every subsequent call is just an
indirect branch — essentially zero overhead on any modern branch predictor.

### Core pattern

```c
typedef float (*dot_fn_t)(const float * restrict, const float * restrict, int);

static dot_fn_t dot_dispatch(void)
{
    if (__builtin_cpu_supports("x86-64-v4"))  return dot_avx512;
    if (__builtin_cpu_supports("x86-64-v3"))  return dot_avx2;
    if (__builtin_cpu_supports("x86-64-v2"))  return dot_sse2;
    return dot_c;
}

float dot(const float * restrict a, const float * restrict b, int n)
{
    static dot_fn_t fn = NULL;
    if (!fn)
        fn = dot_dispatch();
    return fn(a, b, n);
}
```

The `static` pointer is written once and never changes, making it safe in
multi-threaded code without a lock (worst case: two threads race on first call
and both write the same pointer value).

### Subarchitecture level strings

Prefer subarchitecture-level checks over individual feature strings — they are
more future-proof and easier to read:

| Argument | Guarantees |
|----------|-----------|
| `"x86-64-v1"` | SSE, SSE2 (baseline for all x86-64) |
| `"x86-64-v2"` | + SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT |
| `"x86-64-v3"` | + AVX, AVX2, FMA, BMI1, BMI2, F16C, LZCNT, MOVBE |
| `"x86-64-v4"` | + AVX-512F, BW, CD, DQ, VL |

Use a fine-grained string (e.g., `"avx512f"`, `"avx"`) only when the code needs
exactly one feature not covered by any subarchitecture level, or when you need to
distinguish within a level.

If a guard chains multiple `__builtin_cpu_supports()` calls with `&&`, find the
covering subarchitecture level and replace them with a single call.

### `__attribute__((target(...)))` on each variant

Guard each variant's function definition with a matching `target` attribute. This
lets the compiler emit the correct instruction encoding even when the translation
unit is compiled at a lower `-march` baseline:

```c
__attribute__((target("avx512f")))
static float dot_avx512(const float * restrict a, const float * restrict b, int n)
{
    /* AVX-512 intrinsics or code that uses zmm registers */
}

__attribute__((target("avx2,fma")))
static float dot_avx2(const float * restrict a, const float * restrict b, int n)
{
    /* AVX2 + FMA intrinsics */
}
```

The combination of `__attribute__((target))` on the definition and
`__builtin_cpu_supports` in the dispatch function is the correct pattern. Without
`target` on the definition, the compiler may not encode the instructions correctly
when the file-level `-march` is lower than what the variant requires.

### Correctness rule: the guard must cover all required flags

The `__builtin_cpu_supports` argument must fully cover (equal or be a superset of)
all CPUID flags required by the guarded code. Under-specification causes
**illegal instruction faults** on CPUs that pass the guard but lack a required
feature.

After generating or modifying dispatch code, verify the CPUID annotations and
guard strings — any widening or rewriting operation changes the required flags.
If the `cpuid-check` skill is available, invoke it; otherwise verify manually
that each `__builtin_cpu_supports` argument fully covers all instructions used
in the guarded block.

---

## Adapting to non-x86 or non-GCC environments

| Environment | Adaptation |
|-------------|-----------|
| Clang | `__builtin_cpu_supports` and `target_clones` work identically |
| MSVC | Use `IsProcessorFeaturePresent()` with `PF_AVX_INSTRUCTIONS_AVAILABLE` etc. instead of `__builtin_cpu_supports` |
| ARM / AArch64 | Use `getauxval(AT_HWCAP)` / `AT_HWCAP2` for runtime feature detection; no `target_clones` equivalent |
| Shared library (.so / .dll) | Mechanism B is preferred — IFUNC resolvers in shared libraries can have init-ordering issues on some linkers |

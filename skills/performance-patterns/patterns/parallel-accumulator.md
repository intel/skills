<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: serial accumulator → parallel accumulator rewrite

## When to apply

Apply when a hot loop is **latency-bound** due to a sequential dependency chain
through a single accumulator variable. Each iteration must wait for the previous
iteration's result before it can begin, even though the underlying data is
independent.

### Source code signals

- A single floating-point variable is updated inside a loop: `sum += a[i] * b[i]`,
  `acc = fma(a[i], b[i], acc)`, `total += x[i]`, running max/min
- The reduction operation is the only loop-carried dependency
- Loop body is otherwise straight-line (no branches on the accumulator)

### Profiling signals

- The **accumulate instruction** (`vaddss`, `vaddpd`, `vmulss`, `vfmadd213ps`,
  `addsd`, etc.) carries a disproportionately high cycle count in `perf annotate`
- IPC is low (well below 1.0) despite **low cache-miss rates** and no branch
  mispredictions — the combination of low IPC + low misses is the key indicator
- Cycles-per-iteration ≥ the FP latency of the operation (typically 4–5 cycles
  for `vadd`/`vfma`) despite the loop body being short
- `perf stat` shows `instructions/cycle` ≪ 1 with low `cache-misses/instructions`

---

## Why this is slow

Modern CPUs can issue multiple FP operations per cycle, but only if those
operations are **independent**. A single accumulator forces strictly sequential
execution: the add at iteration `i+1` cannot begin until the add at iteration `i`
retires. The CPU's out-of-order engine stalls waiting for the dependency to
resolve, giving throughput limited by FP add **latency** (~4–5 cycles) rather than
FP add **throughput** (~0.5 cycles). This is an 8–10× gap on modern hardware.

Example serial pattern:
```c
float sum = 0.0f;
for (int i = 0; i < n; i++)
    sum += a[i] * b[i];   /* each += depends on previous sum */
```

### Associativity requirement

All fixes below reorder floating-point operations. This is mathematically valid
only when the reduction operation is **associative** — i.e., `(A op B) op C == A
op (B op C)`. Addition, multiplication, min, and max all satisfy this. Results
will differ from the serial version by floating-point rounding (ε), not by a
logical error. If exact reproducibility of FP results is required, these
optimizations cannot be applied.

---

## The fix: three levels of improvement

Apply the level appropriate for the context. Each level builds on the previous
and can be combined. Measured speedups are for a simple float array sum on a
modern x86 CPU.

### Level 1 — Pure C: multiple accumulators (no SIMD required)

Unroll the loop across independent accumulator variables so the out-of-order
engine can overlap their dependency chains. No SIMD, no intrinsics — the
compiler handles everything. **Measured: ~4× faster than the serial version.**

```c
float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
int i;
for (i = 0; i + 3 < n; i += 4) {
    s0 += a[i+0];
    s1 += a[i+1];
    s2 += a[i+2];
    s3 += a[i+3];
}
/* scalar tail */
for (; i < n; i++) s0 += a[i];
float sum = (s0 + s1) + (s2 + s3);
```

Use this when SIMD is unavailable, forbidden by policy, or when the improvement
from Level 1 alone is sufficient.

**Important**: when the function has a scalar fallback path (e.g. the `nr < 4`
remainder at the end of a SIMD function), that path should *also* use Level 1
accumulators — not a fresh single-accumulator loop. A single-accumulator fallback
re-introduces the exact serial dependency the fix is trying to eliminate for small
inputs.

### Level 2 — SSE2 SIMD accumulator (baseline, any x86-64 CPU)

Process 4 floats per iteration using `xmm` registers. **Measured: ~4–8× faster.**

```c
#include <immintrin.h>

float sum_sse2(const float *a, int n) {
    __m128 vsum = _mm_setzero_ps();
    int i;
    for (i = 0; i + 3 < n; i += 4)
        vsum = _mm_add_ps(vsum, _mm_loadu_ps(&a[i]));
    /* horizontal reduction: sum 4 lanes → scalar */
    __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(vsum, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    float result = _mm_cvtss_f32(sums);
    /* scalar tail */
    for (; i < n; i++) result += a[i];
    return result;
}
```

### Level 3 — SIMD + unrolled accumulators (recommended for performance-critical paths)

Combines Level 1 (multiple accumulators) and Level 2 (SIMD). **Measured: ~8×
faster with SSE2; ~15× with AVX2 (8 floats/register × unrolled).** Use an unroll
factor of at least 4 for arrays of 1024+ elements.

```c
/* AVX2: 8 floats/ymm × 4 accumulators = 32 elements per iteration */
#include <immintrin.h>

float dot_avx2(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i;
    for (i = 0; i + 31 < n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+0),  _mm256_loadu_ps(b+i+0),  s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), s3);
    }
    /* combine ymm accumulators */
    __m256 sum8 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    /* horizontal reduction: ymm → scalar */
    __m128 hi = _mm256_extractf128_ps(sum8, 1);
    __m128 lo = _mm256_castps256_ps128(sum8);
    __m128 r  = _mm_add_ps(hi, lo);
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float result = _mm_cvtss_f32(r);
    /* scalar tail */
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}
```

To wire SSE2 / AVX2 / AVX-512 variants together so the right one is chosen
at runtime, read `library/cpu-dispatch.md`. For the AVX-512 variant template,
read `patterns/simd-upconversion-impl.md` (Capability 2 section).

### Accumulator count guideline

A good rule of thumb is **FP latency ÷ throughput** accumulators to saturate the
pipeline. With FP latency 4 and throughput 0.5: 4 / 0.5 = 8 needed. In practice,
4 accumulators covers most cases; use 8 for maximum throughput on tight loops.

---

## Verification

After applying the fix:

1. **Correctness** — run on the same input as the original and confirm outputs
   match within floating-point rounding (ε). Result differences beyond ε indicate
   a logic error in the rewrite.
2. **Profiling** — rerun `perf stat` and confirm IPC rises (target ≥ 2× vs. serial).
3. **Benchmark** — measure wall-clock time; expect:
   - Level 1 (C unrolling): ~4× improvement
   - Level 2 (SSE2): ~4–8× improvement
   - Level 3 (AVX2 + unrolled): ~8–15× improvement

---

## Presenting this to the user

1. Show the current loop with `/* ◄ serial dependency */` on the accumulate line.
2. Explain the latency vs. throughput gap in one short paragraph.
3. **Always warn about floating-point associativity**, even when a quick
   verification shows zero diff on your test data — the difference only appears
   with certain inputs or compilers. Use phrasing like: *"Reordering FP additions
   requires associativity. Results may differ from the serial version by
   floating-point rounding (ε), not a logic error. If bit-exact reproducibility
   is required, these optimizations cannot be applied."*
4. Add `restrict` to pointer parameters when rewriting — it lets the compiler
   assume no aliasing and can eliminate a runtime overlap-check preamble. See
   `patterns/missing-restrict.md` for the full rule (C only; use `__restrict__`
   for C++).
5. Pick the appropriate level: Level 1 if SIMD is not available or not worth the
   complexity; Level 2 if SSE2 is the target baseline; Level 3 for maximum
   performance on AVX2/AVX-512 CPUs.
6. For Level 3, mention runtime dispatch via `library/cpu-dispatch.md` if the
   binary must run on multiple CPU generations. For the AVX-512 variant, read
   `patterns/simd-upconversion-impl.md` (Capability 2 section).

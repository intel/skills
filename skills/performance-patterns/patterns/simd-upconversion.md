<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: narrow SIMD → SIMD vector width upconversion

## When to apply

Apply when a hot loop uses narrower SIMD registers than the CPU supports, or uses
scalar floating-point entirely. Each doubling of vector width processes twice as
many elements per instruction, giving roughly **2× throughput** on compute-bound
loops. Two doublings (scalar → xmm → ymm → zmm) can yield up to 8×.

### Source code signals

- Intrinsics use `_mm_*` (128-bit SSE) on a CPU that supports `_mm256_*` (AVX2)
- Intrinsics use `_mm256_*` on a CPU that supports `_mm512_*` (AVX-512)
- A tight float/double loop with no SIMD at all (scalar `+=`, `*=`, etc.)
- Compiler auto-vectorized to `xmm` but did not widen to `ymm` or `zmm`

### Profiling signals

- `perf annotate` shows scalar FP instructions (`addsd`, `mulss`, `movsd`,
  `vaddss`, `vmulss`) in a tight hot loop — no vector registers visible
- `perf annotate` shows `xmm` register names (`xmm0`–`xmm15`) in packed
  operations on a CPU that supports `ymm` (AVX2) or `zmm` (AVX-512)
- `perf annotate` shows `ymm` in a hot loop on an AVX-512 capable CPU
- Low IPC on a compute-bound workload (check: low cache misses, CPU-bound)

---

## Caveats before proceeding

| Concern | Detail |
|---------|--------|
| **CPU feature check** | AVX2 required for `ymm`; AVX-512 required for `zmm`. Check `/proc/cpuinfo` flags (`avx2`, `avx512f`): `grep -m1 flags /proc/cpuinfo \| tr ' ' '\n' \| grep avx`. |
| **Memory-bandwidth-bound loops** | If the loop is already limited by DRAM bandwidth, wider vectors process more data per cycle but hit the same wall. The gain is real but smaller than the theoretical 2×. Check with `perf stat -e cycles,cache-misses,mem-loads`. |
| **Latency-bound loops (serial accumulator)** | Dependency chains between iterations are not helped by wider vectors alone. Apply the **parallel accumulator** pattern (`patterns/parallel-accumulator.md`) first, then consider width upconversion — `performance-patterns` can apply both together. |
| **AVX-512 frequency throttling** | On many Intel CPUs, executing ZMM instructions triggers a core frequency downclocking that can partially or fully offset the throughput gain from doubling lanes. Verify sustained frequency with `perf stat -e cpu-clock` or `turbostat` before and after widening. If frequency drops significantly on a workload that is not heavily compute-bound, AVX2 (YMM) may deliver better real-world throughput. This caveat applies only to the YMM -> ZMM step |
| **Numerical accuracy** | Widening changes the order in which floating-point operations are evaluated. FP results can differ from the original by up to 1 ULP; bit-exact test suites need their reference values regenerated after widening. Integer loops are unaffected by FP rounding but must be checked for sign/unsigned semantics — see [Step 7: consolidation step 2 (vector)](simd-upconversion-impl.md#step-7-consolidation-step-2-vector) in `patterns/simd-upconversion-impl.md`. |

### Checking for memory-bandwidth saturation

Before widening, determine whether the bottleneck is compute or memory bandwidth:

```bash
perf stat -e cycles,instructions,cache-misses,mem-loads,mem-stores -a -- <workload>
```

Key indicators of bandwidth saturation:
- Low IPC despite no branch mispredictions
- High `cache-misses` relative to `instructions`
- High LLC miss rates on the hot loop's working set

**Decision rule:**
- Clearly compute-bound (IPC near or above 1, few cache misses) → proceed
- Possibly bandwidth-bound → ask the user before proceeding
- Clearly bandwidth-saturated → advise that reducing working-set size or improving
  cache locality may yield more, then proceed if user agrees

### Numerical accuracy

Widening introduces two distinct sources of floating-point result differences:

**1. Operation reordering (associativity)**
Doubling the lane count changes which elements are grouped together per iteration.
Because FP addition is not associative — `(a + b) + c != a + (b + c)` in general —
the widened loop produces results that differ from the original by rounding (ε).
This is not a logic error, but it breaks bit-exact reproducibility against a scalar
or narrower reference.

**2. FMA fusion during consolidation**
[Step 7](simd-upconversion-impl.md#step-7-consolidation-step-2-vector) of the zipper algorithm merges paired vector operations. When a `vmulps` and
a `vaddps` are adjacent and paired, consolidate them into a `vfmadd*` instruction.
FMA computes `a×b + c` with a **single** rounding step instead of two — faster and
more accurate. The result differs from the original separate multiply-then-add by at
most 1 ULP.

**How to handle accuracy differences**

| Scenario | What to do |
|---|---|
| FP reordering (all loops) | Regenerate reference test vectors from the widened code; 1 ULP differences are not logic errors |
| Integer sign/unsigned semantics | See [Step 7: consolidation step 2 (vector)](simd-upconversion-impl.md#step-7-consolidation-step-2-vector) in `patterns/simd-upconversion-impl.md` |

---

## Why this is fast

Vector units on modern x86 CPUs issue one operation per cycle regardless of
register width. A `ymm` add processes 8 `float` values in the same time a scalar
`addss` processes one. Two doublings (128->256->512 bit) give up to 4x and 8x
speedup respectively on compute-bound code.

---

## The fix

Read `patterns/simd-upconversion-impl.md` for the step-by-step procedure.
That file contains:

- **Capability 1** — The systematic 10-step zipper algorithm for widening
  existing asm or intrinsics (SSE->AVX2 or AVX2->AVX-512), including loop
  replication, register reallocation, the zipper interleaving step, and
  vector/integer consolidation.
- **Capability 2** — AVX-512 extension of the parallel accumulator template;
  SSE2/AVX2 templates are in `patterns/parallel-accumulator.md`.
- **Post-transformation checklist** — mandatory CPUID annotation verification,
  `vzeroupper` handling, and clobber list update after any widening.

### Deciding which capability to apply

| Situation | Use |
|-----------|-----|
| Existing asm or intrinsics use narrow registers; want to double the width | Capability 1 (zipper) |
| Loop has a serial accumulator dependency AND the loop is compute-bound | Capability 2 (accumulator) — also read `parallel-accumulator.md` first |
| Compiler auto-vectorized to XMM on an AVX2/AVX-512 CPU, no manual intrinsics | `library/cpu-dispatch.md` Mechanism A (`target_clones`) |
| Multiple hand-written SIMD variants exist but no runtime selector wires them | `library/cpu-dispatch.md` Mechanism B (`__builtin_cpu_supports`) |

### Steps

1. From the annotate output or source, identify the narrow register width
   (`scalar`, `xmm`, or `ymm`) and the instruction(s) involved.
2. Confirm the target CPU supports the wider width.
3. Check for memory-bandwidth saturation (see above) and decide whether to
   proceed automatically or prompt the user.
4. For YMM→ZMM widening, note the AVX-512 frequency throttling caveat above and
   verify sustained frequency after the change.
5. Read `patterns/simd-upconversion-impl.md` and apply the appropriate capability.
   Pay particular attention to:
   - [Step 2: loop replication](simd-upconversion-impl.md#step-2-loop-replication): tail handling and out-of-bounds risk when doubling the minimum iteration count.
   - [Step 7: consolidation step 2 (vector)](simd-upconversion-impl.md#step-7-consolidation-step-2-vector): FMA fusion, sign/unsigned integer semantics, and lane-crossing shuffle behavior.
   - [Scalar tail handling](simd-upconversion-impl.md#scalar-tail-handling): whether to keep a scalar tail, promote an existing XMM tail to the fallback loop, or use AVX-512 masked loads/stores to eliminate the scalar tail entirely.
6. Add `restrict` to pointer parameters in C functions if not already present —
   this eliminates runtime alias checks that survive widening. See
   `patterns/missing-restrict.md` (C only; use `__restrict__` for C++).
7. Run the post-transformation checklist from that file before presenting results.

---

## Verification

After applying the fix:

1. **Correctness** — run on reference inputs and compare outputs. FP results may
   differ by rounding (ε) due to operation reordering and FMA fusion (see
   Numerical accuracy section above); this is expected. Differences larger than ε
   indicate a logic error in the transformation. For integer loops, verify that
   signed/unsigned semantics are preserved (see [Step 7: consolidation step 2 (vector)](simd-upconversion-impl.md#step-7-consolidation-step-2-vector) in `patterns/simd-upconversion-impl.md`).
2. **CPUID guards** — verify or generate the correct CPUID comment and
   `__builtin_cpu_supports` guard on the widened code. If the `cpuid-check`
   skill is available, invoke it; otherwise verify manually using the
   instruction list in `patterns/simd-upconversion-impl.md`.
3. **Profiling** — rerun `perf annotate` and confirm wider registers appear in the
   hot loop.
4. **Benchmark** — measure wall-clock time; expect near-2× improvement per
   register doubling on compute-bound workloads.

---

## Presenting this to the user

1. Show the current hot loop with `/* ◄ narrow: xmm */` (or `/* ◄ scalar */`)
   annotating the narrow operation.
2. Explain the width doubling opportunity in one sentence.
3. Note the target width and any caveats (bandwidth, latency dependency).
4. Apply the fix from `simd-upconversion-impl.md` and show the widened loop.
5. Show expected IPC improvement from profiling or a quick benchmark.

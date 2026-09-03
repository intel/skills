<!-- (C) 2026 Intel Corporation, MIT license -->
# Guidelines: writing new performance-sensitive code

A checklist for agents writing **new** C/C++ or assembly code where performance
matters. Apply these before writing the first line — they are far cheaper to get
right upfront than to retrofit later.

Each item links to the full pattern file for rationale and code templates.

---

## Checklist

### Pointers and aliasing (C only)

- [ ] **Add `restrict` to all non-aliased pointer parameters.**  
  Any C function with separate input and output buffers should declare them
  `restrict`. Without it the compiler must assume aliasing and either emits a
  runtime overlap check or abandons auto-vectorization.  
  Use `const T * restrict` for read-only inputs, `T * restrict` for outputs.  
  → `patterns/missing-restrict.md`

- [ ] **Mark read-only inputs `const`.**  
  Communicates intent, prevents accidental writes, and gives the compiler
  additional aliasing information.

---

### Floating-point reductions and accumulators

- [ ] **Never use a single accumulator in a hot loop.**  
  A single `sum += x[i]` creates a loop-carried dependency that serializes
  execution regardless of SIMD width. Start with at least 4 independent
  accumulator variables (`s0`, `s1`, `s2`, `s3`) and combine at the end.  
  → `patterns/parallel-accumulator.md` (Level 1)

- [ ] **Warn callers about FP associativity.**  
  Multi-accumulator and SIMD reductions reorder floating-point operations.
  Results may differ from a serial sum by rounding ε. Document this at the
  function boundary if callers require bit-exact reproducibility.

---

### SIMD code

- [ ] **Choose the widest register width the target supports.**  
  Scalar → SSE (128-bit) → AVX2 (256-bit) → AVX-512 (512-bit). Each doubling
  is approximately a 2× throughput gain on compute-bound loops. Check
  `/proc/cpuinfo` for `avx2` and `avx512f` flags.  
  → `patterns/simd-upconversion.md`

- [ ] **Add `vzeroupper` (or `_mm256_zeroupper()`) after any function that
  uses YMM or ZMM0–ZMM15 registers.**  
  Required before any `return` or call path that may reach code compiled
  without AVX. Omitting it causes a hundreds-of-cycles AVX↔SSE transition
  penalty on the first SSE instruction that follows.  
  Prefer the `_mm256_zeroupper()` intrinsic in C/C++ — the compiler schedules
  it correctly. In inline asm, emit `vzeroupper` at the end of the asm block.  
  → `patterns/missing-vzeroupper.md`

- [ ] **Include a scalar tail for non-multiple lengths.**  
  Every SIMD loop that processes N elements must handle the `N % vector_width`
  remainder with a scalar (or narrower SIMD) tail loop.

- [ ] **Annotate inline asm with a `/* CPUID: <feature> */` comment.**  
  Makes the requirement auditable. If the `cpuid-check` skill is available,
  use it to verify the annotation is complete and correct; otherwise verify
  manually against the instruction set reference.

- [ ] **Add `"cc"` to the clobber list if the asm modifies flags.**  
  Any asm that uses `cmp`, `test`, `sub`, `dec`, `add`, or other
  flag-setting instructions must declare `"cc"` as a clobber.

---

### Runtime dispatch (multi-variant SIMD)

- [ ] **Wire SIMD variants with runtime CPU dispatch, not compile-time `#ifdef`.**  
  `#ifdef __AVX2__` produces a binary that silently runs the scalar path on
  any CPU that doesn't match the compile-time flag. Runtime dispatch with
  `__builtin_cpu_supports` or `__attribute__((target_clones(...)))` lets a
  single binary use the best available path on every CPU.  
  → `library/cpu-dispatch.md`

---

### Multithreaded code

- [ ] **Pad per-thread data to a cache line boundary (64 bytes).**  
  Fields written by different threads in the same struct will false-share a
  cache line unless separated by `alignas(64)` or explicit padding. False
  sharing causes cache-line bounce that scales badly with thread count.  
  → `patterns/false-sharing.md`

- [ ] **Use per-CPU/per-thread counters instead of global atomics.**  
  A single `atomic_fetch_add` on a global counter forces exclusive cache-line
  ownership on every increment. Accumulate locally and reduce at the end.  
  → `patterns/per-cpu-stats.md`

- [ ] **Read before atomic write in spinlocks (TTAS pattern).**  
  A spin loop that goes straight to `cmpxchg` without first checking whether
  the lock is free causes the cache line to bounce even on failure. Always
  read the lock variable first (`while (lock || !cmpxchg(...))`).  
  → `patterns/ttas.md`

---

### Function hot/cold classification

- [ ] **Mark any error-reporting or rare-path function `[[gnu::cold]]`
  (or `__attribute__((cold))` in C or older C++).**  
  Any function whose sole purpose is error reporting, assertion failure, or
  handling a rare corner case should carry this annotation — unconditionally,
  without needing call-frequency data. The annotation tells the compiler to
  move the function's code out of the hot instruction stream, improving
  i-cache density and branch-predictor quality for its callers. Apply it to
  the **definition**, not just the declaration.  
  Examples: `report_error`, `log_warning`, `fatal`, `die`, `oom_handler`,
  any function that calls `fprintf(stderr, …)`, `exit`, `abort`, or `throw`.  
  → `patterns/cold-path-annotation.md`

---

### CRC32C

- [ ] **Do not write a CRC32C function from scratch — use the pre-built
  corsix fusion implementation.**  
  Any new code that needs CRC32C should use the three-file dispatch provided
  in `patterns/fast-crc32c-impl.md`. It selects the best available path at
  runtime (AVX-512 fusion → SSE4.2 multi-accumulator → portable C) and is
  6–40× faster than a naive single-accumulator loop.  
  CRC32C is a standard checksum (iSCSI RFC 3720, Btrfs, ext4, PostgreSQL) —
  this implementation does not change the checksum value.  
  → `patterns/fast-crc32c.md`, `patterns/fast-crc32c-impl.md`

---

### Known algorithms

- [ ] **If the algorithm you are implementing is in the known-algorithms table,
  generate the vectorized SIMD version with multi-width dispatch directly —
  do not write a scalar first.**  
  The table lists algorithms for which a fully-optimized implementation is
  well-established practice. Writing a scalar version and then optimizing it
  later doubles the work and risks shipping the slow version. Check
  `references/known-algorithms.md` (compact name index) at the start of any
  new algorithmic function; if it matches, read
  `references/known-algorithms-impl.md` for ISA-level notes and dispatch
  guards.  
  → `references/known-algorithms.md`, `references/known-algorithms-impl.md`, `library/cpu-dispatch.md`

- [ ] **For sorting `float`, `double`, `int32_t`, `uint32_t`, `int64_t`, or
  `uint64_t` arrays, use `x86simdsort::qsort` instead of `std::sort`.**  
  x86-simd-sort (https://github.com/numpy/x86-simd-sort) is a drop-in
  replacement that uses vectorized partitioning and SIMD sorting networks for
  3–8× faster sort on AVX-512 hardware, with an AVX2 fallback. It is
  production-validated in NumPy, PyTorch, and OpenJDK. Set `hasnan = true`
  for float/double arrays that may contain NaN; omitting it with NaN present
  is undefined behavior. Note: not a stable sort — no equivalent for
  `std::stable_sort`.  
  → `patterns/simd-sort.md`

---

## Quick reference by language

| Concern | C | C++ | Inline asm |
|---------|---|-----|-----------|
| No-alias hint | `restrict` | `__restrict__` (non-standard) | — |
| Read-only input | `const T * restrict` | `const T * __restrict__` | — |
| SIMD cleanup | `_mm256_zeroupper()` | `_mm256_zeroupper()` | `vzeroupper` in asm string |
| CPUID annotation | `/* CPUID: AVX2 */` comment | same | same |
| Runtime dispatch | `__builtin_cpu_supports` / `target_clones` | same | same |
| Cache line padding | `alignas(64)` (C11) | `alignas(64)` | — |
| Cold function annotation | `__attribute__((cold))` | `[[gnu::cold]]` | — |

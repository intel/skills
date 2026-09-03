<!-- (C) 2026 Intel Corporation, MIT license -->
# Known algorithms — implementation details

Per-algorithm ISA levels, dispatch guards, and implementation notes.
Load this file **only after confirming** that a function name from
`references/known-algorithms.md` is present in the code or profile.

For runtime dispatch wiring, see `library/cpu-dispatch.md`.

---

## Cosine Similarity

**What it computes:** The cosine of the angle between two vectors:
`cos(θ) = dot(A, B) / (|A| * |B|)`, where `dot(A,B) = Σ a[i]*b[i]` and
`|A| = sqrt(Σ a[i]²)`. Returns 1.0 for identical direction, 0.0 for
orthogonal, −1.0 for opposite. Widely used in ML embeddings, NLP, and
recommendation systems.

**Why scalar is slow:** Three serial accumulator loops (dot product, two norms)
each have loop-carried FP dependencies. A naive implementation makes three
passes over the data; a smart one makes a single pass but still serializes
on one accumulator. Modern CPUs support FMA (fused multiply-add) and can
process 8–16 floats per cycle with AVX2/AVX-512.

**Key insight — single-pass, multi-accumulator:** Compute `dot_ab`, `dot_aa`,
and `dot_bb` in one loop with independent SIMD accumulators for each. Combine
at the end: `result = dot_ab / sqrt(dot_aa * dot_bb)`. This reads each array
once and exploits instruction-level parallelism across the three FMA streams.

**ISA levels and approach:**

| ISA level | Technique |
|-----------|-----------|
| FMA (baseline) | 4 independent `float` accumulators per stream; `fmaf(a[i], b[i], dot_ab)` etc.; combine with `sqrtf` |
| AVX2 + FMA | `_mm256_fmadd_ps` (8 floats/iter), 4 accumulators per stream (12 YMM registers total); horizontal reduce with `_mm256_hadd_ps` at the end |
| AVX-512 + FMA | `_mm512_fmadd_ps` (16 floats/iter), 4 accumulators per stream; reduce with `_mm512_reduce_add_ps` |

**Dispatch guards:**
```c
/* CPUID: AVX512F */
if (__builtin_cpu_supports("avx512f"))  → AVX-512 path
/* CPUID: AVX2,FMA */
else if (__builtin_cpu_supports("avx2") &&
         __builtin_cpu_supports("fma")) → AVX2+FMA path
/* CPUID: FMA */
else if (__builtin_cpu_supports("fma")) → scalar FMA path
else                                    → scalar fallback
```

**Key implementation notes:**
- Pre-normalized inputs (`|A| = |B| = 1`) reduce to a plain dot product —
  detect this case and skip the norm computation.
- Guard against division by zero: if `dot_aa * dot_bb < epsilon`, return 0.
- FP associativity: multi-accumulator results may differ from a serial sum by
  rounding ε; document this at the function boundary.
- All paths need a scalar tail for `n % vector_width` remaining elements.
- The final `sqrt` and division are a negligible fraction of runtime for
  any array longer than ~16 elements; optimize the loop, not the epilogue.

---

## Hamming Distance

**What it computes:** The number of positions where two equal-length sequences
differ. For bit strings: `popcount(a XOR b)`. For byte arrays: the sum of
`popcount(a[i] ^ b[i])` over all byte positions.

**Why scalar is slow:** A single-accumulator loop has a loop-carried dependency
on the count variable, and processes one byte (or word) at a time. Modern CPUs
support POPCNT in hardware since Nehalem (2008) and can XOR 32–64 bytes per
instruction with AVX2/AVX-512.

**ISA levels and approach:**

| ISA level | Technique | Throughput |
|-----------|-----------|-----------|
| POPCNT (baseline) | 8-byte chunks: `a64 ^ b64` → `__builtin_popcountll`; 4 independent accumulators to hide latency | ~4–6 GB/s |
| AVX2 | `_mm256_xor_si256` (32 bytes/iter) + bit-sliced Harley-Seal popcount or 4-bit lookup table | ~20–30 GB/s |
| AVX-512VPOPCNTDQ | `_mm512_xor_si512` + `_mm512_popcnt_epi8` (64 bytes/iter); reduce with `_mm512_reduce_add_epi64` after widening | ~40–60 GB/s |

**Dispatch guards:**
```c
/* CPUID: AVX512VPOPCNTDQ */
if (__builtin_cpu_supports("avx512vpopcntdq")) → AVX-512 path
/* CPUID: AVX2 */
else if (__builtin_cpu_supports("avx2"))        → AVX2 path
/* CPUID: POPCNT */
else if (__builtin_cpu_supports("popcnt"))      → POPCNT path
else                                            → scalar fallback
```

**Key implementation notes:**
- All paths need a scalar tail for `n % vector_width` remaining bytes.
- For AVX2 bit-sliced popcount, the Harley-Seal algorithm (see Muła, Kurz,
  Lemire 2017) avoids the latency of a per-byte lookup by operating on 256-bit
  words; it is the standard approach for this ISA level.
- `__builtin_popcountll` compiles to a single `popcnt` instruction with `-mpopcnt`
  or `-march=native`; do not implement popcount manually.
- Accumulate partial 64-bit sums to avoid overflow when processing large arrays.

---

## Jaccard Distance

**What it computes:** For two bit vectors A and B:
`Jaccard similarity = popcount(A AND B) / popcount(A OR B)`;
`Jaccard distance = 1 - similarity`. Measures set overlap — 0 means identical,
1 means disjoint. Also known as intersection-over-union (`iou`) in image/ML
contexts.

**Why scalar is slow:** A naive loop computes `AND` and `OR` popcount with two
separate accumulators, one byte or word at a time, with the same loop-carried
dependency problem as Hamming Distance. The key insight is that `AND` and `OR`
are independent operations on the same two input words — they can be computed
in a single pass through the data with interleaved SIMD operations, costing no
extra memory bandwidth compared to Hamming Distance.

**Single-pass trick:** If `popcount(A)` and `popcount(B)` are already known
(e.g., stored alongside the vectors), the union can be derived without `OR`:
`popcount(A OR B) = popcount(A) + popcount(B) - popcount(A AND B)`. This
reduces the loop to a single `AND` + popcount, halving the work.

**ISA levels and approach:**

| ISA level | Technique |
|-----------|-----------|
| POPCNT (baseline) | 8-byte chunks: `__builtin_popcountll(a & b)` and `__builtin_popcountll(a \| b)`; 4 independent accumulators per stream |
| AVX2 | `_mm256_and_si256` + `_mm256_or_si256` (32 bytes/iter); Harley-Seal bit-sliced popcount on both results in the same loop |
| AVX-512VPOPCNTDQ | `_mm512_and_si512` + `_mm512_or_si512` + `_mm512_popcnt_epi8` on both (64 bytes/iter); reduce with `_mm512_reduce_add_epi64` |

**Dispatch guards:** identical to Hamming Distance (same ISA requirements):
```c
/* CPUID: AVX512VPOPCNTDQ */
if (__builtin_cpu_supports("avx512vpopcntdq")) → AVX-512 path
/* CPUID: AVX2 */
else if (__builtin_cpu_supports("avx2"))        → AVX2 path
/* CPUID: POPCNT */
else if (__builtin_cpu_supports("popcnt"))      → POPCNT path
else                                            → scalar fallback
```

**Key implementation notes:**
- Guard against division by zero: if `popcount(A OR B) == 0` both vectors are
  all-zero; return similarity 1.0 (or distance 0.0) by convention.
- The final division is one `float` operation on two accumulated `uint64_t`
  values — negligible cost; optimize the loop, not the epilogue.
- All paths need a scalar tail for `n % vector_width` remaining bytes.
- For the AVX2 path, Harley-Seal processes both AND and OR accumulators in the
  same loop body — the two streams share the same loop counter and scalar tail.
- Accumulate partial 64-bit sums to avoid overflow for arrays longer than ~4 GB.

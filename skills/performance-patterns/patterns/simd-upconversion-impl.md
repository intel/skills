<!-- (C) 2026 Intel Corporation, MIT license -->
# SIMD width upconversion — implementation guide

This file provides the full step-by-step procedures for widening SIMD code.
It is demand-loaded from `patterns/simd-upconversion.md` once a width
upconversion opportunity has been identified.

---

## Table of contents

1. [Capability 1: Zipper algorithm (asm/intrinsics widening)](#capability-1-zipper-algorithm)
2. [Scalar tail handling](#scalar-tail-handling)
3. [Capability 2: Vector-sequential optimization (serial accumulator)](#capability-2-vector-sequential-optimization)
4. [Post-transformation checklist](#post-transformation-checklist)
5. [Swap rules for the zipper step](#rules-for-swapping-two-assembly-instructions)

---

## Capability 1: zipper algorithm

Apply when existing vectorized code uses narrower registers than the target CPU
supports (XMM→YMM or YMM→ZMM).

Two notes before starting:

- **SSE→AVX preparation**: if the input uses non-VEX SSE instructions (e.g.
  `ADDPS`, `MOVUPS` without the `V` prefix), apply [Step 1: Preparation](#step-1-preparation) (VEX conversion) first.
  This is a prerequisite, not a separate first-class capability.
- **Abort condition**: if any step cannot be completed (a swap is illegal, a
  register cannot be merged), abandon the transformation and explain which step
  failed and why.

This algorithm uses GNU inline assembly (AT&T syntax), but applies equally to
standalone `.asm`/`.S` files. Use the zipper whenever there is any doubt about
whether a direct doubling is safe — simple loops can sometimes be doubled
directly, but the algorithm is the safe default.

### Running example

The steps below are illustrated with this AVX function (YMM→ZMM upscale):

```c
void vector_add(float *srcA, float *srcB, float *dst, int elementcount)
{
    /* CPUID: AVX */
    asm volatile(
        "    cmp     $8, %[cnt]\n\t"
        "    jl      2f\n\t"
        "1:\n\t"
        "    vmovups (%[srcA]), %%ymm0\n\t"
        "    vaddps  (%[srcB]), %%ymm0, %%ymm0\n\t"
        "    vmovups %%ymm0, (%[dst])\n\t"
        "    add     $32, %[srcA]\n\t"
        "    add     $32, %[srcB]\n\t"
        "    add     $32, %[dst]\n\t"
        "    sub     $8, %[cnt]\n\t"
        "    cmp     $8, %[cnt]\n\t"
        "    jge     1b\n\t"
        "2:\n\t"
        "    test    %[cnt], %[cnt]\n\t"
        "    jz      4f\n\t"
        "3:\n\t"
        "    vmovss  (%[srcA]), %%xmm0\n\t"
        "    vaddss  (%[srcB]), %%xmm0, %%xmm0\n\t"
        "    vmovss  %%xmm0, (%[dst])\n\t"
        "    add     $4, %[srcA]\n\t"
        "    add     $4, %[srcB]\n\t"
        "    add     $4, %[dst]\n\t"
        "    dec     %[cnt]\n\t"
        "    jnz     3b\n\t"
        "4:\n\t"
        "    vzeroupper\n\t"
        : [srcA] "+r"(srcA), [srcB] "+r"(srcB), [dst] "+r"(dst),
          [cnt]  "+r"(elementcount)
        :
        : "ymm0", ..., "ymm15", "cc", "memory"
    );
}
```

The four structural elements the algorithm operates on:

1. **Pre-loop condition check** — verifies enough elements exist for one vectorized iteration.
2. **Vectorized loop body** — the core computation.
3. **Loop end condition** — tests whether another iteration is possible and jumps back.
4. **Tail handling** — processes remaining elements when count is not a multiple of vector width.

The pre-loop condition and loop end condition are usually identical checks.

---

### Step 1: Preparation

#### SSE → AVX consideration

If converting from SSE to AVX(2), first convert each vector instruction from its
non-VEX SSE form (e.g., `ADDPS`) to the 3-operand VEX form (e.g., `VADDPS`). The
explicit source/destination separation is required by later steps.

#### Mark non-vector instructions for flags interaction

For every non-vector instruction, add an inline comment with one of:

- `; no flags impact`
- `; flags consumed`
- `; flags produced`
- `; flags clobbered` (writes flags but they are not read before being overwritten)

Example:
```
"    add     $4, %[srcA]   ; flags clobbered\n\t"
"    dec     %[cnt]        ; flags produced\n\t"
"    jnz     3b            ; flags consumed\n\t"
```

These comments will be removed at the end and are for internal reasoning only.

---

### Step 2: loop replication

Duplicate the entire loop (pre-loop condition, body, and end condition). The copy
is preserved unchanged through Steps 3–7 and becomes the narrower-width fallback
loop once the algorithm completes. Its pre-loop condition check will fail when fewer
elements remain than the original width, causing fall-through to tail handling.

**Out-of-bounds risk.** Each widening step doubles the minimum element count
required for one full vector iteration. SSE→AVX2 raises the minimum from 4 to 8
floats; AVX2->AVX-512 raises it from 8 to 16. If [Step 2: loop replication](#step-2-loop-replication) is skipped or the fallback
loop is removed, the widened loop will read and write past the end of the array for
any count that is not a multiple of the new width — silent memory corruption with no
fault. The fallback loop and scalar tail are not optional.

**No existing scalar tail.** If the original function has no scalar tail (the caller
guarantees count is an exact multiple of the original vector width), that guarantee
must now hold for the doubled width too. If it does not, a scalar tail must be added.
See the [Scalar tail handling](#scalar-tail-handling) section.

**Array alignment.** Widening increases the natural alignment requirement: 16-byte
aligned for XMM, 32-byte for YMM, 64-byte for ZMM. If the original used aligned
loads (`vmovaps`, `vmovdqa`), verify that the array is aligned to the new width
before promoting — a misaligned `vmovaps` causes a general-protection fault at
runtime. If alignment cannot be guaranteed, switch to unaligned loads (`vmovups`,
`vmovdqu`), which incur no penalty on modern Intel microarchitectures when the data
happens to be aligned, and only a small penalty when it is not.

#### Label handling

Labels must be unique within an `asm` block.

- **Numeric labels**: In the copy, replace all numeric labels with new unique numbers
  and update the loop-back jump accordingly. Jumps to tail processing are unchanged.
- **Named labels**: Append a suffix (e.g., `_avx2` when promoting to AVX2,
  `_avx512` when promoting to AVX-512).

The first loop's pre-loop condition must jump to the replicated loop's pre-loop
condition — not directly to tail processing. Add a new label if needed.

Example after replication (label `1` renamed to `5` in the copy; new label `6` as entry):
```
        "    cmp     $8, %[cnt]        ; flags produced\n\t"
        "    jl      6f                ; flags consumed\n\t"
        "1:\n\t"
        "    vmovups (%[srcA]), %%ymm0\n\t"
        ...
        "    jge     1b                ; flags consumed\n\t"

        "6:\n\t"
        "    cmp     $8, %[cnt]        ; flags produced\n\t"
        "    jl      2f                ; flags consumed\n\t"
        "5:\n\t"
        "    vmovups (%[srcA]), %%ymm0\n\t"
        ...
        "    jge     5b                ; flags consumed\n\t"
```

---

### Step 3: loop unrolling

Unroll the first (non-replicated) loop exactly once:

1. Double the element count in the pre-loop condition check.
2. Double the element count in the loop end condition check.
3. Append a second copy of the loop body after the first.
4. Remove the intermediate condition check so both halves always execute.

Track which instructions belong to the first half and which to the new second half —
this distinction is needed in Steps 4 and 5.

Example after unrolling:
```
        "    cmp     $16, %[cnt]   ; flags produced\n\t"
        "    jl      6f            ; flags consumed\n\t"
        "1:\n\t"
        /* first half */
        "    vmovups (%[srcA]), %%ymm0\n\t"
        "    vaddps  (%[srcB]), %%ymm0, %%ymm0\n\t"
        "    vmovups %%ymm0, (%[dst])\n\t"
        "    add     $32, %[srcA]  ; flags clobbered\n\t"
        "    add     $32, %[srcB]  ; flags clobbered\n\t"
        "    add     $32, %[dst]   ; flags clobbered\n\t"
        "    sub     $8, %[cnt]    ; flags clobbered\n\t"
        /* second half (copy of first, intermediate cmp/jge removed) */
        "    vmovups (%[srcA]), %%ymm0\n\t"
        "    vaddps  (%[srcB]), %%ymm0, %%ymm0\n\t"
        "    vmovups %%ymm0, (%[dst])\n\t"
        "    add     $32, %[srcA]  ; flags clobbered\n\t"
        "    add     $32, %[srcB]  ; flags clobbered\n\t"
        "    add     $32, %[dst]   ; flags clobbered\n\t"
        "    sub     $8, %[cnt]    ; flags clobbered\n\t"
        "    cmp     $16, %[cnt]   ; flags produced\n\t"
        "    jge     1b            ; flags consumed\n\t"
```

---

### Step 4: register reallocation

Operate only on the second half of the unrolled loop. For each register used as a
destination — and not yet renamed in this step — apply the following mapping:

```
ymm0  → ymm16    ymm4  → ymm20    ymm8  → ymm24    ymm12 → ymm28
ymm1  → ymm17    ymm5  → ymm21    ymm9  → ymm25    ymm13 → ymm29
ymm2  → ymm18    ymm6  → ymm22    ymm10 → ymm26    ymm14 → ymm30
ymm3  → ymm19    ymm7  → ymm23    ymm11 → ymm27    ymm15 → ymm31
```

After renaming a register at its definition, rename all subsequent uses of that
register within the second half as well.

Some low-numbered YMM registers may remain in the second half if they hold
loop-invariant values (e.g., a broadcast constant). These need special handling in
[Step 8: handle loop-invariant XMM/YMM registers](#step-8-handle-loop-invariant-xmmymm-registers).

> **Note**: This step temporarily introduces ymm16–ymm31, which requires AVX512VL.
> That requirement will likely be removed after Steps 6–7.

---

### Step 5: the zipper step

Move each instruction from the second half to immediately below its matching
instruction from the first half, working one instruction at a time. The loop end
condition instructions are not moved.

Use legal pairwise swaps (like a single pass of bubble sort): starting with the
topmost instruction of the second half, swap it upward one position at a time until
it sits immediately below its first-half counterpart. Then repeat for the next
instruction.

See [Rules for swapping](#rules-for-swapping-two-assembly-instructions) below.
For readability, leave a blank line after each paired group.

**If any instruction cannot reach its paired position** (all swap paths are
blocked), the algorithm has failed — abandon and report the blocking instruction pair.

Example after the zipper step (all second-half instructions interleaved):
```
        "1:\n\t"
        "    vmovups (%[srcA]), %%ymm0\n\t"
        "    vmovups 32(%[srcA]), %%ymm16\n\t"

        "    vaddps  (%[srcB]), %%ymm0, %%ymm0\n\t"
        "    vaddps  32(%[srcB]), %%ymm16, %%ymm16\n\t"

        "    vmovups %%ymm0, (%[dst])\n\t"
        "    vmovups %%ymm16, 32(%[dst])\n\t"

        "    add     $32, %[srcA]  ; flags clobbered\n\t"
        "    add     $32, %[srcA]  ; flags clobbered\n\t"
        ...
        "    sub     $8, %[cnt]    ; flags clobbered\n\t"
        "    sub     $8, %[cnt]    ; flags clobbered\n\t"

        "    cmp     $16, %[cnt]   ; flags produced\n\t"
        "    jge     1b            ; flags consumed\n\t"
```

---

### Step 6: consolidation step 1 (integer)

Fuse adjacent identical integer operations into a single operation.

```
        "    add     $32, %[srcA]  ; flags clobbered\n\t"
        "    add     $32, %[srcA]  ; flags clobbered\n\t"
```
Becomes:
```
        "    add     $64, %[srcA]  ; flags clobbered\n\t"
```

---

### Step 7: consolidation step 2 (vector)

For each paired vector instruction group, merge the two narrow operations into a
single wide operation.

Two YMM loads at offsets 0 and 32 cover exactly 512 bits → merge into a single
ZMM load at offset 0:
```
        "    vmovups (%[srcA]), %%ymm0\n\t"
        "    vmovups 32(%[srcA]), %%ymm16\n\t"
```
Becomes:
```
        "    vmovups (%[srcA]), %%zmm0\n\t"
```

Apply the same merge to all other paired vector instruction groups (arithmetic,
stores, etc.).

**Sign and unsigned integer semantics.** When the loop operates on integers,
verify that each merged instruction preserves signed vs. unsigned intent:

- *Widening loads*: `vpmovsxbd` sign-extends bytes to dwords; `vpmovzxbd`
  zero-extends. Choosing the wrong one gives silently incorrect results on
  negative values.
- *Comparisons*: `vpcmpgtd` is a signed comparison. There is no direct unsigned
  greater-than in SSE/AVX2; unsigned comparisons require XOR-ing with the sign
  bit before comparing (`vpxor` with `0x80000000`). If the original loop used a
  signed comparison where unsigned was intended (or vice versa), widening
  preserves that bug. Verify intent before consolidating.
- *Arithmetic shifts*: `vpsrad` (arithmetic, sign-fills) vs. `vpsrld` (logical,
  zero-fills). Confirm the shift direction is correct for the data type.

**Lane-crossing shuffle and permute instructions.** Some instructions operate only
within a 128-bit lane and do not cross lane boundaries: `vpermilps`, `vshufps`,
`vpunpcklbw`, and similar. When you widen from XMM to YMM, a 128-bit intra-lane
shuffle operates identically on each 128-bit half of the YMM register — the upper
lane is shuffled independently of the lower lane. When you widen from YMM to ZMM,
the same applies to each 128-bit quarter. This is correct if the original logic
intended independent per-lane shuffles. If the intent was to shuffle across the full
register width (e.g., to bring the highest element to the lowest position), a
cross-lane instruction is needed instead: `vperm2f128` (YMM), `vpermd`, or
`vpermps` (ZMM). Inspect each shuffle instruction at the consolidation step and
confirm whether intra-lane or cross-lane behavior is intended.

After Steps 6 and 7, the running example becomes:
```
        "    cmp     $16, %[cnt]   ; flags produced\n\t"
        "    jl      6f            ; flags consumed\n\t"
        "1:\n\t"
        "    vmovups (%[srcA]), %%zmm0\n\t"
        "    vaddps  (%[srcB]), %%zmm0, %%zmm0\n\t"
        "    vmovups %%zmm0, (%[dst])\n\t"
        "    add     $64, %[srcA]  ; flags clobbered\n\t"
        "    add     $64, %[srcB]  ; flags clobbered\n\t"
        "    add     $64, %[dst]   ; flags clobbered\n\t"
        "    sub     $16, %[cnt]   ; flags clobbered\n\t"
        "    cmp     $16, %[cnt]   ; flags produced\n\t"
        "    jge     1b            ; flags consumed\n\t"
```

---

### Step 8: handle loop-invariant XMM/YMM registers

If any XMM registers remain after an SSE→AVX2 promotion, or YMM registers remain
after an AVX2→AVX-512 promotion, check whether they hold loop-invariant values
(e.g., a register initialized before the loop and read throughout without modification).
These must also be promoted to the wider width.

#### Direct promotion

If the register is initialized inside the ASM block with an instruction that
naturally extends to the wider width, promote directly. Examples:

```
    vxorpd  %%ymm3, %%ymm3, %%ymm3      ; zeroing idiom
```
→
```
    vxorpd  %%zmm3, %%zmm3, %%zmm3
```

Other promotable instructions: `VBROADCASTSS`, `VBROADCASTSD`.

#### Duplication

If direct promotion is not applicable, copy the YMM value into both halves of the
ZMM register using `VINSERTF64X4`:

```
    vinsertf64x4   $1, %%ymm3, %%zmm3, %%zmm3
```

At the end of [Step 8](#step-8-handle-loop-invariant-xmmymm-registers), no pre-promotion (narrow) registers should remain in the loop.

---

### Step 9: handle the accumulator case

When one or more YMM registers accumulate values across loop iterations (e.g., in a
dot-product reduction), extract the upper half after the widened loop completes:

```
    vextractf64x4   $1, %%zmm0, %%ymm1
```

`ymm0` now holds the lower 256 bits; `ymm1` holds the upper 256 bits. Apply the
original reduction operation between them to combine the two halves, restoring the
algorithm to the same state it would have reached with the narrower loop.

---

### Step 10: cleanup

1. Remove all flag-interaction comments added in [Step 1: Preparation](#step-1-preparation).
2. Update the clobber list:
   - Replace `ymm0`–`ymm15` with the ZMM registers actually used (`zmm0`, etc.).
   - Add any new high registers (zmm16–zmm31) if they survived consolidation
     (they normally don't, but check).
3. Update the CPUID comment: the new code requires AVX-512F (and possibly AVX-512VL
   if any ymm16–ymm31 survived). If the `cpuid-check` skill is available, invoke
   it to verify; otherwise confirm manually against the instruction list.
4. Apply the vzeroupper rule from the post-transformation checklist below.

---

## Scalar tail handling

After the zipper algorithm completes, the structure is:

```
ZMM wide loop -> YMM/XMM fallback loop -> scalar tail
```

The scalar tail is the loop body from the original code that processes one element
at a time (e.g., `vmovss`/`vaddss` in the running example). It handles the 0 to
(width−1) remaining elements after both vector loops have run.

### The scalar tail is preserved unchanged

Do not widen the scalar tail. It exists precisely because there are fewer remaining
elements than one vector iteration requires. Its job is correctness for the
remainder — not throughput. Leave it exactly as it was in the original.

### If the original has no scalar tail

Some functions require the caller to guarantee that the element count is an exact
multiple of the vector width. After widening, that guarantee must hold for the new
width (doubled). If it does not, the remainder elements will be silently skipped.
In this case, add a scalar tail:

```c
/* scalar tail — processes remaining count % new_vector_width elements */
for (; i < n; i++) dst[i] = srcA[i] + srcB[i];
```

Or, for an asm block, replicate the scalar loop from the running example (labels
3–4, `vmovss`/`vaddss`/`dec`/`jnz`).

### If the original has an XMM tail (SSE -> AVX2 case)

When widening SSE (4-wide) to AVX2 (8-wide), the original may have a 4-wide XMM
tail loop rather than a purely scalar tail. In that case:

- The XMM tail loop becomes the **fallback loop** (the [Step 2](#step-2-loop-replication) copy of the original),
  handling 4–7 remaining elements.
- A new scalar tail (0–3 elements) must be preserved or added below it.

The three-tier structure is: `YMM loop → XMM fallback → scalar tail`.

### AVX-512 masked tail (preferred over scalar tail for ZMM targets)

For AVX-512 targets, mask registers (`k0`–`k7`) can eliminate the scalar tail
entirely, replacing it with a single masked ZMM operation. This reduces three code
paths to two and removes the scalar loop overhead.

**Building the tail mask from the remaining element count:**

```asm
/* remaining = count % 16 (0 to 15) */
mov     $1, %eax
shlx    %[remaining], %eax, %eax    /* eax = 1 << remaining */
dec     %eax                        /* eax = (1 << remaining) - 1 = mask */
kmovw   %eax, %k1
```

**Masked load, compute, masked store:**

```asm
vmovups (%[srcA]){%k1}{z}, %zmm0   /* load remaining lanes; zero-fill rest */
vaddps  (%[srcB]){%k1}{z}, %zmm0, %zmm0
vmovups %zmm0, (%[dst]){%k1}       /* store only remaining lanes */
```

The `{z}` (zeroing masking) on the loads ensures unwritten lanes are zero rather
than holding stale register contents, which prevents the arithmetic from
incorporating garbage values in the zero-filled lanes.

**In intrinsics:**

```c
__mmask16 mask = (1u << remaining) - 1;
__m512 va = _mm512_maskz_loadu_ps(mask, srcA);
__m512 vb = _mm512_maskz_loadu_ps(mask, srcB);
_mm512_mask_storeu_ps(dst, mask, _mm512_add_ps(va, vb));
```

**When to use masked tail vs. scalar tail:**

| Situation | Prefer |
|---|---|
| Target is AVX-512 and remaining count is known at runtime | Masked tail — fewer code paths, no branch to scalar |
| Target is AVX2 (no mask registers) | Scalar tail or XMM fallback loop |
| Remaining count is always zero (exact-multiple guarantee) | Neither — no tail needed |
| Mixed AVX-512 / non-AVX-512 dispatch path | Masked tail in the AVX-512 variant; scalar tail in the AVX2 variant |

---

## Capability 2: vector-sequential optimization

Apply when a loop accumulates into a single scalar value through dependent operations.
For the full diagnosis, detection signals, and C-level (Level 1) fix, read
`patterns/parallel-accumulator.md`. That file also contains the SSE2 and AVX2
templates with concrete speedup measurements.

### AVX-512 variant template

When the target CPU supports AVX-512, extend the Level 3 template from
`parallel-accumulator.md` to 16-wide accumulators:

```c
/* AVX-512: 4 accumulators × 16 floats = 64 floats per iteration */
#include <immintrin.h>

__attribute__((target("avx512f")))
static float dot_avx512(const float * restrict src1,
                        const float * restrict src2, int nr)
{
    int i = 0, nr64 = nr & ~63;
    __m512 vs0 = _mm512_setzero_ps(), vs1 = _mm512_setzero_ps();
    __m512 vs2 = _mm512_setzero_ps(), vs3 = _mm512_setzero_ps();

    for (; i < nr64; i += 64) {
        vs0 = _mm512_fmadd_ps(_mm512_loadu_ps(&src1[i]),
                               _mm512_loadu_ps(&src2[i]), vs0);
        vs1 = _mm512_fmadd_ps(_mm512_loadu_ps(&src1[i+16]),
                               _mm512_loadu_ps(&src2[i+16]), vs1);
        vs2 = _mm512_fmadd_ps(_mm512_loadu_ps(&src1[i+32]),
                               _mm512_loadu_ps(&src2[i+32]), vs2);
        vs3 = _mm512_fmadd_ps(_mm512_loadu_ps(&src1[i+48]),
                               _mm512_loadu_ps(&src2[i+48]), vs3);
    }
    vs0 = _mm512_add_ps(_mm512_add_ps(vs0, vs1), _mm512_add_ps(vs2, vs3));
    float sum = _mm512_reduce_add_ps(vs0);   /* built-in horizontal reduction */

    for (; i < nr; i++) sum += src1[i] * src2[i];
    return sum;
}
```

Note: `_mm512_reduce_add_ps` performs the full horizontal reduction internally —
no manual shuffle-and-add sequence needed, unlike the SSE2/AVX2 equivalents.

Wire all three variants together using `library/cpu-dispatch.md`.

### Adapting the template to other operations

| Operation | SSE2 | AVX2 | AVX-512 |
|-----------|------|------|---------|
| Add floats | `_mm_add_ps` | `_mm256_add_ps` | `_mm512_add_ps` |
| Multiply-add floats | `_mm_add_ps(_mm_mul_ps(...))` | `_mm256_fmadd_ps` | `_mm512_fmadd_ps` |
| Add doubles | `_mm_add_pd` | `_mm256_add_pd` | `_mm512_add_pd` |
| Multiply-add doubles | `_mm_add_pd(_mm_mul_pd(...))` | `_mm256_fmadd_pd` | `_mm512_fmadd_pd` |
| Max floats | `_mm_max_ps` | `_mm256_max_ps` | `_mm512_max_ps` |
| Add int32 | `_mm_add_epi32` | `_mm256_add_epi32` | `_mm512_add_epi32` |

**Horizontal reduction patterns:**
- SSE2/AVX2: shuffle-and-add (as shown in `parallel-accumulator.md`)
- AVX-512: use `_mm512_reduce_add_ps` / `_mm512_reduce_add_pd` (simpler, no manual shuffle)

**Unroll factor guidance:**
- Default: 4 accumulators (hides ~4-cycle FP latency)
- Large arrays (1024+ elements): consider 8 accumulators for higher throughput
- Integer operations: 4 accumulators often sufficient (lower latency)

---

## Post-transformation checklist

Run this after any code change produced by the capabilities above:

1. **CPUID annotations and guards** — verify or generate the CPUID annotations
   on the transformed code. This is mandatory: any transformation (widening,
   rewriting, or adding dispatch) changes which CPUID flags are required, and
   getting this wrong causes illegal-instruction faults on older CPUs. If the
   `cpuid-check` skill is available, invoke it; otherwise verify manually:
   - Compute the correct minimal CPUID flag set for each asm block or intrinsic section.
   - Verify or generate the `/* CPUID: <flags> */` comment above each block.
   - Verify or generate the `__builtin_cpu_supports` guard with the correct
     subarchitecture string (e.g. `"x86-64-v4"` for AVX-512F code).

2. **`__builtin_cpu_supports` guard** — if a guard is missing,
   add one using the dispatch pattern in `library/cpu-dispatch.md`.

3. **`vzeroupper` handling**:
   - Required after any use of **YMM registers** (ymm0–ymm15) or **ZMM0–ZMM15**
     (which alias ymm0–ymm15 and can leave the same dirty upper-bit state that
     causes AVX↔SSE transition penalties).
   - Can be omitted only if the asm block exclusively uses **ZMM16–ZMM31** (and/or
     XMM registers) — these have no YMM alias that legacy SSE code can observe.

4. **Clobber list** — update to reflect new register usage. ZMM registers alias
   YMM/XMM: if the clobber list listed `"ymm0"`, replace with `"zmm0"` after
   widening. Add any new high-numbered registers (zmm16–zmm31) used.

---

## Rules for swapping two assembly instructions

### No interference

If two instructions have no overlap between their inputs and outputs, swap freely.

When both instructions have a `; flags clobbered` annotation, the flags register
does not count as an overlap.

### Overlap on an integer register

If one instruction adds or subtracts a constant from an integer register, and the
other uses that same register as a memory base address, the swap is legal — adjust
the memory offset by the same amount.

Before swap:
```
        "    add     $32, %[srcA]       ; flags clobbered\n\t"
        "    vmovups (%[srcA]), %%ymm16\n\t"
```
After swap:
```
        "    vmovups 32(%[srcA]), %%ymm16\n\t"
        "    add     $32, %[srcA]       ; flags clobbered\n\t"
```

If there is already an existing offset, add the adjustment to it:
```
        "    add     $32, %[srcA]       ; flags clobbered\n\t"
        "    vmovups 64(%[srcA]), %%ymm16\n\t"
```
→
```
        "    vmovups 96(%[srcA]), %%ymm16\n\t"
        "    add     $32, %[srcA]       ; flags clobbered\n\t"
```

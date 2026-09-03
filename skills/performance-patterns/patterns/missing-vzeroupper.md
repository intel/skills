<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: missing vzeroupper

## When to apply

Apply when an x86 program mixes AVX/AVX-512 code (using YMM or ZMM0–15 registers)
with legacy SSE code, and the AVX section does not emit `vzeroupper` before
returning to SSE code. This causes the CPU to pay an AVX↔SSE transition penalty
on the first SSE instruction after the AVX section.

### Profiling signals

- `perf stat -e other_assists.avx_to_sse,other_assists.sse_to_avx` reports a
  non-zero (especially large) count for `other_assists.avx_to_sse`
- `perf annotate` shows the **first SSE instruction** after an AVX function carries
  an extreme cycle count relative to its instruction cost — the penalty amortizes
  there
- The function immediately before the expensive SSE instruction uses `ymm` or `zmm`
  registers in its assembly but has no `vzeroupper` before returning or calling SSE
  code
- Slowdown appears only after AVX code runs; a cold-start profile (before any AVX
  code has executed) does not show the same SSE cost

### Source code signals

- Inline assembly block uses `ymm` or `zmm0–zmm15` registers but has no
  `vzeroupper` in the asm string or cleanup block
- A function uses AVX intrinsics (`_mm256_*`, `_mm512_*` with zmm0–15) and then
  calls, or is followed by, a function that uses SSE intrinsics (`_mm_*`) or
  compiles to SSE instructions
- A shared library entry point or callback that uses AVX internally but has no
  `vzeroupper` before returning to the caller (the caller may be compiled without
  AVX)
- Any use of `ymm` registers or `zmm0`–`zmm15` without a matching `vzeroupper`
  before the next function call or return

---

## Why this is slow

When the CPU sees a write to a YMM (or ZMM0–15, which aliases the same state)
register, it marks those registers' upper bits as potentially "dirty". Legacy SSE
instructions treat the upper 128 bits of XMM registers as undefined and do not
zero them. When the CPU transitions from a code section that has dirty upper bits
to a section that uses legacy SSE, it must save and restore the full 256-bit (or
512-bit) register state, even though the SSE instruction only needs 128 bits.

The penalty is paid on the **first SSE instruction** after the dirty-upper-bit
state is set, and costs hundreds of cycles on some microarchitectures.

`vzeroupper` explicitly zeroes the upper 128 bits of all YMM registers (clearing
the dirty state) at negligible cost (~1 cycle). It must be executed before any
return or call path that may reach SSE code.

---

## The fix

### Preferred: `_mm256_zeroupper()` intrinsic

For C/C++ code using AVX intrinsics, prefer the intrinsic form — the compiler
understands it and can schedule it correctly:

```c
#include <immintrin.h>

void my_avx_function(float *a, float *b, int n)
{
    /* ... AVX2/AVX-512 work using ymm/zmm registers ... */
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
    }

    _mm256_zeroupper();   /* ← clear dirty upper bits before returning */
}
```

Place `_mm256_zeroupper()` at every exit point of the function (before `return`,
before calls to SSE functions, or before callbacks that may dispatch to SSE code).

### Inline assembly: `vzeroupper` instruction

For inline asm blocks that use `ymm` or `zmm0–zmm15` registers, emit `vzeroupper`
at the end of the asm block:

```c
asm volatile(
    "vmovups (%[src]), %%ymm0\n\t"
    "vaddps  (%[b]),   %%ymm0, %%ymm0\n\t"
    "vmovups %%ymm0,   (%[dst])\n\t"
    "vzeroupper\n\t"            /* ← required: clears dirty upper bits */
    :
    : [src] "r"(src), [b] "r"(b), [dst] "r"(dst)
    : "ymm0", "memory"
);
```

**Clobber list when adding `vzeroupper`:** `vzeroupper` zeroes the upper
128 bits of *all* ymm0–ymm15 registers (equivalently, zmm0–zmm15 upper
bits). This means adding `vzeroupper` to an asm block implicitly clobbers
every ymm/zmm register that was not already clobbered. Add all of them to
the clobber list:

```c
asm volatile(
    /* ... uses ymm0 and ymm1 ... */
    "vzeroupper\n\t"
    :
    : [src] "r"(src), [dst] "r"(dst)
    : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
      "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
      "memory"
);
```

If the block uses ZMM registers (zmm0–zmm15), use `"zmm0"` through
`"zmm15"` in the clobbers instead — the compiler treats zmm and ymm
clobbers as distinct even though they alias the same hardware state.
```

### When `vzeroupper` is required vs. optional

| Register usage in the block | `vzeroupper` needed? |
|-----------------------------|----------------------|
| `ymm0`–`ymm15` used | **Yes** |
| `zmm0`–`zmm15` used (alias ymm0–15) | **Yes** |
| `zmm16`–`zmm31` only (no ymm alias) | No |
| XMM only (`xmm0`–`xmm15`) | No |

Note: even if you only write ZMM registers (zmm0–zmm15), the underlying YMM alias
is dirtied and `vzeroupper` is still required before any SSE code.

### Compiler-managed functions (`__attribute__((target("avx2")))`)

When using GCC/Clang target attributes or `target_clones`, the compiler inserts
`vzeroupper` automatically at function boundaries. You do not need to add it
manually unless you are using inline assembly within such a function that bypasses
the compiler's register tracking.

---

## Verification

```bash
# Before fix: should show non-zero avx_to_sse assists
perf stat -e other_assists.avx_to_sse,other_assists.sse_to_avx ./your_program

# After fix: avx_to_sse should drop to zero (or near zero)
perf stat -e other_assists.avx_to_sse,other_assists.sse_to_avx ./your_program
```

A successful fix reduces `other_assists.avx_to_sse` to zero and eliminates
the extreme cycle count on the first SSE instruction after the AVX section.

If `other_assists.avx_to_sse` remains non-zero after adding `vzeroupper`,
check for additional AVX callsites that also lack `vzeroupper`:

```bash
# Disassemble and search for ymm/zmm use without vzeroupper
objdump -d your_binary | grep -E "ymm|zmm" | grep -v vzeroupper
```

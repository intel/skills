---
name: xe2-dpas-patterns
description: "Use this skill when writing, loading operands for, or storing results from XMX DPAS instructions on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU using SYCL ESIMD. Xe2 is the GPU architecture; LNL and BMG are product names. Covers all four DPAS operand load patterns (lsc_load_2d, lsc_gather, VNNI packing, transposed u32 load), scatter/store write-back, Usage 1 vs Usage 2 orientation, and the SOA property of lsc_gather. Applicable to any kernel using DPAS: GEMM, attention, convolution, etc. Not for joint_matrix or library GEMM calls (oneMKL, oneDNN), or for general 2D block load limits (see esimd-lsc-2d-gather-scatter)."
compatibility: "Requires the oneAPI DPC++ compiler (icpx -fsycl). Code checked with DPC++ 2025.3.3 and 2026.1.1 on an Intel Arc B580 (Xe2)."
---

# Xe2 DPAS Patterns Skill

Reference for loading and storing DPAS operands on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) via SYCL ESIMD.
All patterns below are exercised by `assets/fp16_dpas_ult.cpp` (4 test cases).

---

## Setup and Notation

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>   // lsc_load_2d, lsc_gather, lsc_scatter
using namespace sycl::ext::intel::esimd;
namespace xmx    = sycl::ext::intel::esimd::xmx;
namespace xesimd = sycl::ext::intel::experimental::esimd;
constexpr auto cached     = xesimd::cache_hint::cached;
constexpr auto write_back = xesimd::cache_hint::write_back;
```

Patterns 1 and 2 and the Usage 1 write-back use the experimental `lsc_load_2d`/`lsc_store_2d`, as existing kernels and the asset do. For new code, prefer the non-experimental `load_2d`/`store_2d` in `sycl::ext::intel::esimd` (same block shapes and surface parameters, passed as arguments instead of a `config_2d_mem_access` payload; Pattern 4 uses it).

All matrices are row-major. For `C[M,N] = A[M,K] * B^T`:
- `B[N,K]` — B stored N-major (row `n` holds K values).
- `B_T[K,N]` — the same matrix stored K-major. This is the standard GEMM "B" layout of `C = A * B`.

---

## DPAS Register Contract (FP16, XE2)

```cpp
// xmx::dpas<SystolicDepth, RepeatCount, T, CT, BT, AT>(acc, b_tile, a_tile)  // C = acc + A*B
// xmx::dpas<SystolicDepth, RepeatCount, T, BT, AT>(b_tile, a_tile)           // C = A*B (acc = 0)
// T = result type, CT = accumulator type, BT/AT = operand types
// XE2: SystolicDepth = 8 (fixed); RepeatCount = 1..8 (rows of a_tile and acc); N = 16 (fixed)
// FP16/BF16: 8 systolic steps x 2 elements/step = 16 K-elements per call
simd<float, 8*16> acc(0.f);
acc = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(acc, b_tile, a_tile);
```

| Register | Type | Size | Layout | Role |
|----------|------|------|--------|------|
| `a_tile` | `simd<half, RC*16>` | 128 half (RC=8) | `[m*16+k]` row-major | RC M-rows × 16 K-cols |
| `b_tile` | `simd<half, 16*16>` | 256 half | `uint32[k_pair*16+n]` VNNI | 16 K-elems × 16 N-cols |
| `acc`    | `simd<float or half, RC*16>` | 128 (RC=8) | `[m*16+n]` row-major | RC M-rows × 16 N-cols |

Result: `acc[m*16+n] += sum_k  a_tile[m*16+k] * b_tile_fp16(n,k)`

Use an fp32 accumulator (`T = CT = float`) for long K reductions; a half accumulator (`T = CT = sycl::half`) also works and is what the asset uses.

> **Arg order**: `dpas(acc, b_tile, a_tile)` — b_tile is 2nd, a_tile is 3rd. Easy to mix up.

---

## Two Usage Orientations

### Usage 1 — Standard (M=8, N=16, K=16)

```
A[M×K] → a_tile     B[N×K] or B_T[K×N] → b_tile
dpas(acc, b_tile, a_tile) → acc[m*16+n] = C[m,n]   (direct store)
```

### Usage 2 — Swapped (M=16, N=8, K=16)

Swap roles: A→b_tile (gather VNNI), B→a_tile (load_2d).
Useful when N < 16 at the DPAS call level.

```
A[M×K] → b_tile (gather VNNI, 16 M-lanes)
B[N×K] or B_T[K×N] → a_tile (load_2d)
dpas(acc, b_tile=A_vnni, a_tile=B) → acc[n*16+m] = C[m,n]  (TRANSPOSED — needs scatter write-back)
```

---

## VNNI Layout (b_tile)

```
b_tile uint32[k_pair * 16 + n_col] = packed { fp16[n_col, k_pair*2], fp16[n_col, k_pair*2+1] }
  k_pair = 0..7  (outer, stride 16)
  n_col  = 0..15 (inner, stride 1)
```

---

## lsc_gather SOA Property (key insight)

`lsc_gather<T, NElts, DS, L1H, L2H, N>(ptr, offsets)` returns `simd<T, N*NElts>` in **SOA** layout:

```
result[elem * N + lane]  =  T at  ptr[ byte_offset[lane] + elem * sizeof(T) ]
```

This SOA property is what makes gather produce DPAS-ready layouts without register repack:
- For **b_tile** (B[N,K]): use N rows as lanes with `uint32_t` → `result[k_pair*16+n]` = VNNI

**NElts > 1 needs a 4- or 8-byte type.** `lsc_gather`/`lsc_scatter` of `sycl::half` with NElts > 1 compiles but fails at vISA finalization ("only D32 and D64 support vector load"). Gather/scatter fp16 data as `uint32_t` pairs instead.

---

## Four Load Patterns

### Pattern 1: a_tile from A[M×K] — lsc_load_2d, no transform

```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=K*/, 8/*BH=M*/, 1> payA(
    A, K*2u-1u, M-1u, K*2u-1u, 0u, 0u);
simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 8, 1, false/*T*/, false/*VNNI*/, cached, cached>(payA);
// a_tile[m*16+k] = A[m, k]  ✓
```

### Pattern 2: b_tile from B_T[K×N] — lsc_load_2d VNNI=true

```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=N*/, 16/*BH=K*/, 1> payB(
    B_T, N*2u-1u, K-1u, N*2u-1u, 0u, 0u);
simd<sycl::half, 16*16> b_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 16, 1, false, true/*VNNI*/, cached, cached>(payB);
// b_tile uint32[k_pair*16+n] = {B_T[k*2,n], B_T[k*2+1,n]}  VNNI ✓
```

### Pattern 3: b_tile from B[N×K] — lsc_gather u32/NElts=8, N-lanes

```cpp
// N=16 lanes (one per output N-column), NElts=8 k-pairs per lane (= 16 fp16 = K tile)
const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B);
simd<uint32_t, 16> b_off;
for (int n = 0; n < 16; n++)
    b_off[n] = ((uint32_t)(n_base + n) * K + k_start) * 2u;  // byte offset to B[n, k_start]
simd<sycl::half, 16*16> b_tile;
b_tile.template bit_cast_view<uint32_t>() =
    xesimd::lsc_gather<uint32_t, 8,
        xesimd::lsc_data_size::u32, cached, cached,
        16, uint32_t>(B_u32, b_off);
// SOA: result_u32[k_pair*16+n] = {B[n,k_start+k_pair*2], B[n,k_start+k_pair*2+1]}  VNNI ✓
```

`K` is the row length of B; `k_start` must be even (4-byte aligned u32 addresses).

### Pattern 4: a_tile from B_T[K×N] (Usage 2) — transposed u32 load_2d + deinterleave

`Transposed=true` is **rejected for fp16** at compile time, and an fp16 gather with NElts > 1 fails at vISA. Load fp16 pairs as `uint32_t` with a transposed 2D load, then split each pair into two a_tile rows:

```cpp
// Block: 4 u32 (= 8 fp16 along N) wide x 16 K-rows high, transposed
// p[j*16+k] = {B_T[k][n_base+2j], B_T[k][n_base+2j+1]} = {B[2j,k], B[2j+1,k]}
simd<uint32_t, 4*16> p = load_2d<uint32_t, 4, 16, 1, /*Transposed=*/true, false>(
    reinterpret_cast<const uint32_t*>(B_T),
    N*2u-1u, K-1u, N*2u-1u,          // surface width/height/pitch of B_T (bytes-1, rows-1, bytes-1)
    n_base / 2, k_start,             // x in u32 elements, y in rows
    properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>});
auto p_h = p.template bit_cast_view<sycl::half>();
simd<sycl::half, 8*16> a_tile;
#pragma unroll
for (int j = 0; j < 4; j++) {
    a_tile.template select<16, 1>((2*j)   * 16) = p_h.template select<16, 2>(j*32);
    a_tile.template select<16, 1>((2*j+1) * 16) = p_h.template select<16, 2>(j*32 + 1);
}
// a_tile[n*16+k] = B[n, k]  ✓  (8 strided register moves)
```

Requirements: the pitch `N*2` bytes must be a multiple of 16 B (N multiple of 8), and `n_base` must be even. `load_2d` is the non-experimental API in `sycl::ext::intel::esimd`.

---

## Store / Write-back Patterns

### Usage 1 write-back — lsc_store_2d (acc is direct C[m,n])

```cpp
xesimd::lsc_store_2d<sycl::half, 16, 8, write_back, write_back>(
    C, N*2u-1u, M-1u, N*2u-1u, (uint32_t)n_start, (uint32_t)m_start, acc);
```

### Usage 2 write-back — pack to u32 + lsc_scatter<uint32_t,4> (acc is transposed C[n,m])

`acc[ni*16+mj] = C[mj, ni]`. Each of the 16 M-lanes writes one output row of 8 fp16 = 4 u32. Pack column pairs `(2j, 2j+1)` into one u32, then scatter: `packed[j*16+mj] = C[mj, 2j..2j+1]`.

```cpp
simd<uint32_t, 4*16> packed;
auto packed_h = packed.template bit_cast_view<sycl::half>();
#pragma unroll
for (int j = 0; j < 4; j++) {
    packed_h.template select<16, 2>(j*32)     = acc.template select<16, 1>((2*j)   * 16);
    packed_h.template select<16, 2>(j*32 + 1) = acc.template select<16, 1>((2*j+1) * 16);
}
simd<uint32_t, 16> sc_off;
for (int mj = 0; mj < 16; mj++)
    sc_off[mj] = ((uint32_t)(m_base + mj) * N_total + (uint32_t)n_base) * 2u;  // byte offset to C[m, n_base]
xesimd::lsc_scatter<uint32_t, 4,
    xesimd::lsc_data_size::u32, write_back, write_back,
    16, uint32_t>(reinterpret_cast<uint32_t*>(C), sc_off, packed);
```

Requirements: `n_base` and `N_total` even (4-byte aligned row starts). With an fp32 accumulator and fp32 C, no packing is needed: `lsc_scatter<float, 8, lsc_data_size::u32, write_back, write_back, 16, uint32_t>(C, sc_off, acc)` writes `acc` directly.

---

## Summary: Message and Shuffle Cost

| Operation | Method | Register work |
|-----------|--------|---------------|
| a_tile ← A[M,K] | `lsc_load_2d` no-transform | none |
| b_tile ← B_T[K,N] | `lsc_load_2d` VNNI=true | none (hardware) |
| b_tile ← B[N,K] | `lsc_gather<u32,8,u32,N=16>` N-lanes | none (SOA=VNNI) |
| a_tile ← B_T[K,N] | transposed `load_2d<uint32_t,4,16>` | 8 strided moves (deinterleave) |
| store C (Usage 1) | `lsc_store_2d` | none |
| store C (Usage 2, fp16) | pack + `lsc_scatter<u32,4,u32,N=16>` | 8 strided moves (pack) |
| store C (Usage 2, fp32) | `lsc_scatter<float,8,u32,N=16>` | none |

Every operation is a single memory message. Avoid the alternatives that look shuffle-free but are worse: fp16 gather/scatter with NElts > 1 does not compile, eight NElts=1 fp16 messages cost 8 messages plus per-message address math, and a full register transpose of `acc` compiles to many moves.

---

## Asset

| File | Purpose |
|------|---------|
| `assets/fp16_dpas_ult.cpp` | Unit test for all 4 load patterns + both write-backs. All cases PASS. |

Compile and run:
```bash
icpx fp16_dpas_ult.cpp -o fp16_dpas_ult \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg"
./fp16_dpas_ult
```

---

## Related Skills

| Skill | Use for |
|-------|---------|
| `esimd-base` | ESIMD kernel structure, vectorization, reductions, alignment |
| `esimd-lsc-2d-gather-scatter` | 2D block load/store limits, surface parameters, cache hints |
| `esimd-lsc-slm` | Shared local memory and barriers |

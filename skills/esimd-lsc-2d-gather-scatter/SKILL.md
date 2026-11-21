---
name: esimd-lsc-2d-gather-scatter
description: >
  2D block load/store/prefetch, 1D block load/store, and gather/scatter in Intel
  ESIMD (SYCL explicit SIMD) kernels for Intel GPUs such as Arc (Xe2, Battlemage)
  and Lunar Lake. Use this skill when writing or debugging load_2d, store_2d, prefetch_2d
  (or the older experimental lsc_load_2d, lsc_store_2d, lsc_prefetch_2d,
  config_2d_mem_access), block_load, block_store, gather, or scatter in ESIMD
  kernels. Covers surface parameters, VNNI and transposed loads for DPAS
  operands, tile size limits, cache hints, out-of-bounds behavior, and pitfalls
  such as unaligned pitch and the fp16 transpose limitation. Not for shared
  local memory: see esimd-lsc-slm.
metadata:
  version: 2.0.0
compatibility: "Requires the oneAPI DPC++ compiler (icpx -fsycl). Code checked with DPC++ 2025.3.3 and 2026.1.1 on an Intel Arc B580 (Xe2)."
---

# ESIMD 2D Block Operations, Gather, and Scatter

Memory access patterns in Intel ESIMD: 2D block operations for tiled matrix access, 1D block operations for contiguous access, and gather/scatter for non-contiguous access. All APIs below are in `sycl::ext::intel::esimd` and come from `<sycl/ext/intel/esimd.hpp>`. The older experimental `lsc_*_2d` API is covered at the end for reading existing code.

---

## Table of Contents

1. [2D Block Operations](#2d-block-operations)
2. [Surface Parameters](#surface-parameters)
3. [VNNI and Transposed Loads](#vnni-and-transposed-loads)
4. [2D Prefetch](#2d-prefetch)
5. [Tile Size Limits](#tile-size-limits)
6. [1D Block Load / Store](#1d-block-load--store)
7. [Gather / Scatter](#gather--scatter)
8. [Cache Hints](#cache-hints)
9. [Common Pitfalls](#common-pitfalls)
10. [Legacy Experimental API](#legacy-experimental-api)
11. [Choosing the Right Access Pattern](#choosing-the-right-access-pattern)

---

## 2D Block Operations

`load_2d`, `store_2d` and `prefetch_2d` access a rectangular tile of a 2D surface in one message. They are the primary way to load matrix tiles for DPAS/XMX.

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
using namespace sycl::ext::intel::esimd;

// Template: T, BlockWidth (elements), BlockHeight (rows), NBlocks, Transposed, Transformed
simd<T, N> tile = load_2d<T, W, H, NB, Transposed, Transformed>(
    ptr, width_bytes - 1, height_rows - 1, pitch_bytes - 1, x, y [, props]);

// Store: no NBlocks, no Transposed/Transformed
store_2d<T, W, H>(ptr, width_bytes - 1, height_rows - 1, pitch_bytes - 1, x, y, data [, props]);

// Prefetch: no Transposed/Transformed; cache-hint props are REQUIRED
prefetch_2d<T, W, H, NB>(ptr, width_bytes - 1, height_rows - 1, pitch_bytes - 1, x, y, props);
```

`props` is a compile-time property list of cache hints, e.g. `properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>}`.

**Returned size**: `W * H * NB` elements for plain and transposed loads. With `Transformed=true` the rows are padded: `roundUp(H, 4 / sizeof(T)) * nextPow2(W) * NB` (e.g. a 12x4 fp16 VNNI load returns 64 elements, not 48). Use `auto` or the power-of-two widths you will use anyway.

**Availability**: gate on `sycl::ext::intel::esimd::info::device::has_2d_block_io_support` (true on Xe2/Xe3). Write `sycl::info::...` for other device queries (`info` is ambiguous with both namespaces open, see `esimd-base`).

---

## Surface Parameters

```cpp
// Load a 32x8 fp16 tile at column col0, row row0 of a [rows x cols] row-major matrix
const uint32_t w = cols * sizeof(sycl::half) - 1;   // surface width in BYTES, minus 1
const uint32_t h = rows - 1;                         // surface height in ROWS, minus 1
const uint32_t p = cols * sizeof(sycl::half) - 1;   // surface pitch in BYTES, minus 1
simd<sycl::half, 256> tile = load_2d<sycl::half, 32, 8>(ptr, w, h, p, col0, row0);
```

- Width, height and pitch are **value minus 1**; width and pitch are in **bytes**.
- `x` is in **elements**, `y` is in **rows**.
- Move between tiles by changing `x` and `y`; keep `ptr` at the surface base. For a sub-matrix or a paged buffer, put the page/sub-matrix start into `ptr` and keep the geometry unchanged.
- **Pitch must be a multiple of 16 bytes**, and `pitch >= width`. A non-multiple compiles and runs but returns wrong data silently (an fp16 matrix with 1001 columns has a 2002 B pitch: use `block_load` instead).
- The base address should be 64-byte aligned (hardware spec). Misaligned bases happened to load correctly in tests; do not rely on it.
- **Out of bounds**: elements outside the surface read as 0, and stores outside it are dropped. Edge tiles of a matrix whose size is not a multiple of the tile need no masking as long as `width`/`height` describe the real matrix.

---

## VNNI and Transposed Loads

DPAS B operands (fp16/bf16) need VNNI layout: each 32-bit lane holds two consecutive-K values of the same column n.

### Transformed=true: VNNI from a K-major matrix B[K][N]

```cpp
// b_tile[ (kp*16 + n)*2 + j ] = B[k0 + 2*kp + j][n0 + n], 16 K x 16 N
simd<sycl::half, 256> b_tile =
    load_2d<sycl::half, 16, 16, 1, /*Transposed=*/false, /*Transformed=*/true>(
        B, N * 2 - 1, K - 1, N * 2 - 1, n0, k0);
```

Only for 1- and 2-byte types.

### Transposed=true: only for 4- and 8-byte types

`Transposed=true` turns `[H rows x W cols]` in memory into `[W rows x H cols]` in registers. It is rejected at compile time for `half`. For an N-major fp16 matrix `B[N][K]`, load pairs of fp16 as `uint32_t` and transpose: the result is exactly the VNNI tile.

```cpp
// B is [N][K] fp16, row-major. 16 N rows x 8 dwords (= 16 fp16 along K)
simd<uint32_t, 128> u = load_2d<uint32_t, 8, 16, 1, /*Transposed=*/true, /*Transformed=*/false>(
    reinterpret_cast<const uint32_t *>(B), K * 2 - 1, N - 1, K * 2 - 1, k0 / 2, n0);
simd<sycl::half, 256> b_tile = u.bit_cast_view<sycl::half>();   // u must be a named variable
```

Transposed widths: up to 8 dwords (16 fp16 along K) or 4 qwords; `NB = 1`.

---

## 2D Prefetch

Prefetch the next tile while computing the current one. Prefetch requires explicit hints: L1 in {cached, uncached, streaming}, L2 in {cached, uncached}, not both uncached.

```cpp
constexpr auto pf = properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
prefetch_2d<sycl::half, 16, 16, 1>(B, N * 2 - 1, K - 1, N * 2 - 1, n0, k0 + 16, pf);  // next K tile
```

---

## Tile Size Limits

Checked at compile time (static assertions), except the surface rules above.

| | load / prefetch | store |
|---|---|---|
| row width | `W * NB * sizeof(T) <= 64` B, `W * sizeof(T)` multiple of 4 | `4 <= W * sizeof(T) <= 64` B |
| height | `H <= 32` | `H <= 8` for every type |
| blocks | `NB` in {1,2,4} for 1-/2-byte T, {1,2} for 4-byte, 1 for 8-byte | 1 only |
| total | `W * H * NB * sizeof(T) <= 2048` B | `<= 512` B |
| Transformed | 1-/2-byte T, `NB` in {1,2,4}, H 4..32 B / 2..32 words, W 4..16 B / 2..16 words | n/a |
| Transposed | 4-/8-byte T, `NB = 1`, H 1..32 (dword) / 8 (qword), W 1..8 dwords / 1,2,4 qwords | n/a |

Transposed and Transformed cannot both be true.

| Type | Largest plain load | Largest store |
|---|---|---|
| fp16 / bf16 | 32 x 32 | 32 x 8 |
| float / uint32_t | 16 x 32 | 16 x 8 |
| uint8_t | 64 x 32 | 64 x 8 |

A 16-row fp16 tile is stored as two `store_2d` calls at `y` and `y + 8`.

---

## 1D Block Load / Store

For contiguous access (full rows, cooperative loads into SLM) `block_load`/`block_store` are simpler and often faster than 2D, for example for cooperative K+V loads into SLM.

```cpp
simd<sycl::half, 128> data = block_load<sycl::half, 128>(ptr + offset);
block_store<sycl::half, 128>(ptr + offset, data);

// With alignment and cache hints in one property list
simd<sycl::half, 128> d2 = block_load<sycl::half, 128>(ptr + offset,
    properties{alignment<16>, cache_hint_L1<cache_hint::streaming>, cache_hint_L2<cache_hint::cached>});
```

- With cache hints, the alignment must be at least 4 bytes and N must be one of the hardware sizes (for 2-byte T: 2, 4, 6, 8, 16, 32, 64, 128).
- Alignment rules for unaligned fp16 offsets (`alignment<2>`): see `esimd-base`.

---

## Gather / Scatter

One element per lane from per-lane **byte** offsets.

```cpp
simd<uint32_t, 16> offsets = simd<uint32_t, 16>(0, 1) * (row_stride * sizeof(sycl::half));  // bytes
simd<sycl::half, 16> col = gather<sycl::half, 16>(base_ptr, offsets);
scatter<sycl::half, 16>(base_ptr, offsets, col);

simd_mask<16> pred = simd<uint32_t, 16>(0, 1) < valid;   // predicated
simd<sycl::half, 16> c2 = gather<sycl::half, 16>(base_ptr, offsets, pred);
scatter<sycl::half, 16>(base_ptr, offsets, c2, pred);
```

- Use for strided columns, index lookups (embeddings), scattered outputs.
- Several consecutive elements per lane (`gather<T, N, VS>` with VS > 1) need 4- or 8-byte T: for fp16 pairs, gather `uint32_t` and bit-cast.
- Much slower than block or 2D operations when the data is actually contiguous or tiled.

---

## Cache Hints

Properties `cache_hint_L1<...>` and `cache_hint_L2<...>` (L2 is the last-level cache; older code calls it L3).

| Operation | Legal combinations (else static assertion) |
|---|---|
| load | none/none; L1 in {uncached, cached, streaming} with L2 in {uncached, cached}; read_invalidate/cached |
| prefetch | as load, hints required, not uncached/uncached |
| store | none/none; write_back/write_back; L1 in {uncached, write_through, streaming} with L2 in {uncached, write_back}. **L2 `cached` is illegal for stores** |

- Tiles reused across iterations (GEMM A/B, attention K/V): `cached/cached` loads.
- `block_store`: default (no hints) was fastest; `(streaming, write_back)` was slower and `(streaming, uncached)` much slower.

---

## Common Pitfalls

1. **Pitch not a multiple of 16 B**: silently wrong data. Pad the row stride or use `block_load`.
2. **x in bytes**: `x` is in elements (`col`, not `col * sizeof(T)`).
3. **Raw sizes**: width, height and pitch must be value minus 1.
4. **Advancing the base pointer** per tile instead of changing `x`/`y`: breaks the bounds check (and the zero fill at edges).
5. **Transposed fp16**: compile error; load `uint32_t` transposed and bit-cast (see above).
6. **Store too large**: height > 8 or > 512 B is a compile error; split into 8-row stores.
7. **`bit_cast_view` on a load result**: `load_2d<...>(...).bit_cast_view<half>()` does not compile (rvalue). Assign to a named `simd` first.
8. **Prefetch without hints**: static assertion; pass L1/L2 hints.
9. **VNNI size mismatch**: a non-power-of-two width with `Transformed=true` returns a padded vector; declare with `auto` or use widths 2, 4, 8, 16.

---

## Legacy Experimental API

Older code uses `lsc_load_2d`, `lsc_store_2d`, `lsc_prefetch_2d` and the descriptor class `config_2d_mem_access` from `sycl::ext::intel::experimental::esimd` (`#include <sycl/ext/intel/experimental/esimd/memory.hpp>`). 

| Legacy | Current |
|---|---|
| `lsc_load_2d<T, W, H, NB, Tr, Tf, L1, L2>(ptr, w, h, p, x, y)` | `load_2d<T, W, H, NB, Tr, Tf>(ptr, w, h, p, x, y, properties{cache_hint_L1<L1>, cache_hint_L2<L2>})` |
| `lsc_store_2d<T, W, H, L1, L2>(ptr, w, h, p, x, y, v)` | `store_2d<T, W, H>(ptr, w, h, p, x, y, v, props)` |
| `lsc_prefetch_2d<T, W, H, NB, L1, L2>(ptr, w, h, p, x, y)` | `prefetch_2d<T, W, H, NB>(ptr, w, h, p, x, y, props)` |
| `config_2d_mem_access<T, W, H, NB> pay(ptr, w, h, p, x, y)` + `lsc_load_2d<T, W, H, NB, Tr, Tf, L1, L2>(pay)`, `pay.set_x()/set_y()` | no equivalent: pass the six surface arguments on each call |

Only the legacy payload overload of `lsc_prefetch_2d` takes the Transposed/Transformed bools; its pointer overload does not.

---

## Choosing the Right Access Pattern

| Access pattern | Operation | When to use |
|---|---|---|
| `load_2d` / `store_2d` | 2D tiles | Matrix tiles for DPAS, 2D regions with 16 B-multiple pitch |
| `block_load` / `block_store` | 1D contiguous | Full rows, SLM cooperative loads, any pitch |
| `gather` / `scatter` | Non-contiguous | Indexed access, strided columns, irregular patterns |
| `prefetch_2d` | 2D prefetch | Hide latency of the next iteration's tiles |

Decision flow: contiguous data -> `block_load`/`block_store`; rectangular tile with a 16 B-multiple pitch -> `load_2d`/`store_2d`; otherwise `gather`/`scatter`.

---

## Related Skills

| Skill | Relevance |
|-------|-----------|
| `esimd-base` | Foundational ESIMD programming, alignment of 1D loads |
| `esimd-lsc-slm` | SLM-specific operations (slm_block_load, lsc_slm_gather/scatter) |

---
name: esimd-base
description: >
  Foundational Intel ESIMD GPU programming skill. Use this skill proactively
  whenever the user is writing, optimizing, or debugging any SYCL/ESIMD kernel
  for Intel GPUs — including Intel Arc, Iris Xe, or Data Center GPU Max. Covers
  kernel design, memory access patterns (block_load, gather, SLM), data types,
  vectorization, workgroup patterns, hardware characteristics, performance
  analysis, and troubleshooting. Trigger this even when the user does not
  explicitly say "ESIMD" — invoke it for any Intel GPU kernel development,
  performance bottleneck questions, or SYCL optimization tasks targeting Intel
  hardware.
---

# Intel ESIMD Base Skill

Foundational guidance for writing, optimizing, and debugging Intel GPU ESIMD kernels — applicable to any kernel regardless of algorithm. Covers kernel design, all memory access patterns, data types, vectorization, workgroup patterns, hardware characteristics, performance analysis, and troubleshooting.

---

## Table of Contents

1. [What Is ESIMD](#what-is-esimd)
2. [Hardware Characteristics](#hardware-characteristics)
3. [Optimization Priority Checklist](#optimization-priority-checklist)
4. [Kernel Design](#kernel-design)
5. [Memory Access Patterns](#memory-access-patterns)
6. [Data Types](#data-types)
7. [Vectorization Techniques](#vectorization-techniques)
8. [Correctness Traps](#correctness-traps)
9. [Further Reading](#further-reading)

---

## What Is ESIMD

ESIMD (Explicit SIMD) is a C++ programming model for Intel GPU Execution Units (EUs). Each ESIMD work-item maps to one EU thread and operates on explicit `simd<T, N>` register vectors rather than the implicit SIMD of standard SYCL. This gives direct control over:
- Register allocation and vector widths
- Memory access patterns (`block_load`, `gather`, SLM)
- Hardware intrinsics (`hmax`, `hmin`, `pack_mask`, `fbl`, `xmx::dpas`)

**Required headers and annotation**:
```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

// Kernel lambda must be annotated:
[=](sycl::id<1> idx) SYCL_ESIMD_KERNEL { ... }
// or equivalently:
[=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] { ... }
```

**Namespace**: bring in ESIMD intrinsics with:
```cpp
using namespace sycl::ext::intel::esimd;
```
ESIMD has its own `info` namespace: together with `using namespace sycl;`, a bare `info::device::name` is ambiguous. Write `sycl::info::...` (and `sycl::ext::intel::esimd::info::...` for ESIMD device queries).

---

## Hardware Characteristics

### Intel Arc GPU / iGPU (Xe architecture)

| Resource | Spec |
|----------|------|
| EUs (Execution Units) | 128–512 (model dependent) |
| Threads per EU | 8 |
| Total threads | EUs × 8 |
| SLM per sub-slice | 128 KB |
| Recommended SLM | < 64 KB |

**Register file budget**:

| Architecture | GRF per thread | FP16 capacity | FP32 capacity |
|---|---|---|---|
| **Xe1** (Arc Alchemist, iGPU Gen12) | 128 × 32B = **4 KB** | up to ~2048 elements | up to ~1024 elements |
| **Xe2+** (Arc Battlemage, Lunar Lake+) | 128 × 64B = **8 KB** (16 KB in large GRF mode) | up to ~4096 elements | up to ~2048 elements |

- 512 FP16 values = 1 KB — 25% of Xe1 budget (safe), 12.5% of Xe2 budget (very safe)
- Register spilling to L1 cache incurs significant latency
- If a kernel exceeds the register budget, enable **large GRF mode** (see Compilation in `references/quick-reference.md`)

**SIMD instruction width**:

| Architecture | SIMD width | FP16 elements/instruction |
|---|---|---|
| **Xe1** (Arc Alchemist, iGPU Gen12) | 256-bit | **16 FP16** per instruction |
| **Xe2+** (Arc Battlemage, Lunar Lake+) | 512-bit | **32 FP16** per instruction |

- For `simd<half, N>` with N > the native width, the compiler **automatically splits** the operation into multiple native-width instructions — no manual tiling needed.
- Optimal `simd` widths: multiples of 16 (Xe1) or 32 (Xe2+). E.g., `simd<half, 128>` = 8 instructions on Xe1, 4 on Xe2+.
- This means Xe2+ issues **2× fewer instructions** for the same vector width, contributing to its higher throughput alongside the larger GRF.

**Query thread count at runtime** (instead of hard-coding):
```cpp
const int ts = (int)q.get_device()
    .get_info<sycl::info::device::max_compute_units>() * 8;
```

---

## Optimization Priority Checklist

Apply in this order — each step can dwarf the next:

| Priority | Technique | Typical Impact |
|----------|-----------|---------------|
| 0 | **Kernel design** (range dim, work/thread, template params) | **10–35×** |
| 1 | Memory access (block_load, alignment, coalescing) | 2–5× |
| 2 | Data types (FP16 > FP32, uint4/int8 for weights) | 2–8× |
| 3 | Vectorization (hmax/hmin, SIMD ops, avoid scalar loops) | 1.5–2× |
| 4 | Parallelization (workgroup size, SLM reduction) | 10–30% |
| 5 | Register file vs SLM (~1 cycle vs ~30 cycles) | up to 2× |
| 6 | Algorithm complexity (O(N) vs O(K×N), unroll, hoist) | varies |

---

## Kernel Design

This is the single most impactful decision. A bad design can be **10–35× slower** before any micro-optimization.

### Rule 1: Match Kernel Range Dimensionality to Data Shape

Use multi-dimensional `range<D>` so indices map directly to data dimensions. Never decode a 1D linear index into multi-dimensional coordinates at runtime.

```cpp
// BAD: 1D range + manual 4D decode → 1% bandwidth
q.parallel_for(range<1>(bsz * heads * seq * blocks),
    [=](id<1> idx) SYCL_ESIMD_KERNEL {
        int d4 =  idx[0] % blocks;
        int d3 = (idx[0] / blocks) % seq;
        int d2 = (idx[0] / (blocks * seq)) % heads;
        int d1 =  idx[0] / (blocks * seq * heads);
        output[idx[0]] = process(input[idx[0]]);  // 1 element/thread
    });

// GOOD: 3D range → ~73% bandwidth (21.7× faster before any other change)
q.parallel_for(range<3>(bsz, heads, seq),
    [=](id<3> idx) SYCL_ESIMD_KERNEL {
        int b = idx[0], h = idx[1], s = idx[2];
        // each thread processes many output elements for this (b, h, s)
    });
```

### Rule 2: Each Thread Must Do Meaningful Work

Processing 1 element per thread creates massive scheduling overhead. Target **32–128+ elements per thread** (or one complete row/column/tile).

```cpp
// GOOD: thread processes 128 elements
constexpr int ELEMS = 128;
const int base = idx[0] * ELEMS;
simd<sycl::half, ELEMS> data = block_load<sycl::half, ELEMS>(input + base);
// ... process data ...
block_store<sycl::half, ELEMS>(output + base, data);
```

### Rule 3: Use Template Parameters for All Compile-Time Constants

Runtime variables prevent loop unrolling and constant propagation. Make all array sizes and loop bounds `constexpr` or template parameters.

```cpp
// BAD: runtime N prevents unrolling
void kernel(int N) { for (int i = 0; i < N; i++) ... }

// GOOD: compile-time N enables full unroll and SIMD vectorization
template<int N>
void kernel() {
    #pragma unroll
    for (int i = 0; i < N; i++) ...
}
```

### Rule 4: Separate Load and Compute Loops

When computing multiple dot products or reductions, issue all `block_load`s before any compute. This lets the hardware pipeline memory fetches:

```cpp
// BAD: each load stalls its own compute
for (int i = 0; i < K; i++) {
    auto v = block_load<half, D>(ptr + i * D);
    results[i] = detail::sum<half, half, D>(v * w);
}

// GOOD: all loads in-flight before compute begins (+29% measured)
simd<half, D> vecs[K];
for (int i = 0; i < K; i++) vecs[i] = block_load<half, D>(ptr + i * D);
for (int i = 0; i < K; i++) results[i] = detail::sum<half, half, D>(vecs[i] * w);
```

---

## Memory Access Patterns

### Block Load / Store (Contiguous Access)

The primary memory primitive. Maps to one or a few hardware memory transactions.

```cpp
// Load N elements starting at ptr (ptr must be N*sizeof(T) aligned by default)
simd<T, N> data = block_load<T, N>(ptr);
simd<T, N> data = block_load<T, N>(ptr + offset);  // offset in elements

// Store
block_store<T, N>(ptr + offset, data);
```

**Efficiency rules**:
- Prefer loads of **128B or 256B** (e.g., `block_load<half, 128>` = 256B, `block_load<float, 64>` = 256B)
- All threads should access consecutive memory addresses (coalescing)
- Larger individual loads are more efficient than many small ones

**CRITICAL — address alignment**: `block_load` and `block_store` require the **byte address to be a multiple of 4** by default. For 4-byte types (`float`, `int`) this is automatically satisfied. For 2-byte types (`half`, `bfloat16`) an odd element index produces a non-4-byte-aligned byte address and gives wrong results. **When you cannot guarantee the address is 4-byte aligned, you must pass an `alignment<>` property.**

```cpp
// ── half / bfloat16 (2 bytes each) ───────────────────────────────────────
// Element index 3 → byte address +6 → NOT a multiple of 4 → wrong results!
simd<half, 8> data = block_load<half, 8>(ptr + 3);                          // WRONG
simd<half, 8> data = block_load<half, 8>(ptr + 3, properties{alignment<2>}); // OK

// Safe rule: always use alignment<2> for half* / bfloat16* block_load/store
// unless you can statically prove the element offset is even (byte offset % 4 == 0).
block_store<half, 8>(ptr + 3, data, properties{alignment<2>});               // OK

// ── float / int (4 bytes each) ────────────────────────────────────────────
// 4-byte elements: any element index gives a 4-byte-aligned byte address → safe.
simd<float, 8> data = block_load<float, 8>(ptr + 3);   // OK (byte addr = +12)

// ── uint8_t (1 byte each) ────────────────────────────────────────────────
// 1-byte elements: any byte address → always need alignment<1>.
simd<uint8_t, 8> data = block_load<uint8_t, 8>(ptr + 3, properties{alignment<1>});
```

**Summary by type**:
| Type | Size | Default required alignment | When to annotate |
|------|------|---------------------------|-----------------|
| `float`, `int`, `uint32_t` | 4 B | 4 B | never needed (4B × any index = multiple of 4) |
| `half`, `bfloat16` | 2 B | 4 B | whenever element offset may be odd |
| `uint8_t`, `int8_t` | 1 B | 4 B | always (`alignment<1>`) |

**Symptom of missing annotation**: incorrect values loaded/stored, sporadic failures depending on offset parity, no compile error.

### Gather / Scatter (Non-Contiguous Access)

Use when elements are at irregular or strided offsets — e.g., sliding windows, strided output.

```cpp
// Gather: load N elements from per-lane byte offsets
simd<T, N> data = gather<T, N>(base_ptr, simd<uint32_t, N>(byte_offsets));

// Scatter: store N elements to per-lane byte offsets
scatter<T, N>(base_ptr, simd<uint32_t, N>(byte_offsets), data);
```

**Key rules**:
- Offsets are in **bytes**, not element indices. Convert: `byte_offsets = elem_offsets * sizeof(T)`
- Offset type must be `simd<uint32_t, N>` — unsigned 32-bit
- Negative element indices cast to `uint32_t` wrap to huge values → clamp before cast
- Optimal batch size: **N=32** for FP16 (matches hardware gather width)
- Slightly slower than `block_load` for contiguous data, much faster for strided patterns

**Batch gather pattern** (process 32 outputs in parallel):
```cpp
constexpr int BS = 32;
simd<int, BS> rel = simd<int, BS>(0, 1) * STRIDE;  // [0, S, 2S, ..., 31S]

for (int out = 0; out < num_outputs; out += BS) {
    // Clamp offsets to valid range (branchless — safe for idempotent ops like max)
    simd<int, BS> offs = max(out * STRIDE + rel + OFFSET, 0);
    offs = min(offs, max_elem);

    simd<T, BS> vals = gather<T, BS>(
        input + base, simd<uint32_t, BS>(offs) * sizeof(T));

    // ... process vals ...

    int n = std::min(BS, num_outputs - out);
    if (n == BS) block_store<T, BS>(output + out_base + out, result);
    else         for (int i = 0; i < n; i++) output[out_base + out + i] = result[i];
}
```

**Performance reference**: gather achieved 72.8% bandwidth on a strided sliding-window pooling kernel (82.9× total speedup over naive scalar 1D design).

### SLM — Shared Local Memory

Use for inter-thread communication within a workgroup. Requires `nd_range` and `barrier()`.

```cpp
// Initialize SLM (compile-time size, inside kernel)
constexpr int SLM_BYTES = GROUP_SIZE * sizeof(float);
slm_init<SLM_BYTES>();

// Per-thread write (byte offset)
slm_block_store<float, 1>(local_id * sizeof(float), simd<float, 1>(partial));

// Synchronize all threads in workgroup
barrier();

// Read all GROUP_SIZE values (thread 0 only, or all threads for broadcast)
simd<float, GROUP_SIZE> parts = slm_block_load<float, GROUP_SIZE>(0);
```

**SLM limits and cost**:
- Max 128 KB per sub-slice; keep usage < 64 KB
- Access latency ~30 cycles vs ~1 cycle for register file
- `slm_init` must be called inside the kernel, unconditionally
- `barrier()` must be reached by **all** threads in the workgroup — never inside a branch that some threads skip

---

## Data Types

### FP16 vs FP32

**Use FP16** for: weights, activations, intermediate results in neural network ops, any case where error < 0.1 is acceptable.

**Use FP32** for: accumulators summing >1000 terms (use `simd<float, N> acc` even when inputs are FP16), numerical stability-critical paths, softmax exponents.

```cpp
// Pattern: FP16 inputs → FP32 accumulation → FP16 output
simd<sycl::half, 128> a = block_load<sycl::half, 128>(ptr_a);
simd<sycl::half, 128> b = block_load<sycl::half, 128>(ptr_b);
simd<float, 128> acc(0.f);
acc += convert<float>(a) * convert<float>(b);
float result = sycl::ext::intel::esimd::detail::sum<float, float, 128>(acc);
block_store<sycl::half, 1>(out, simd<sycl::half, 1>(sycl::half(result)));
```

---

## Vectorization Techniques

### Arithmetic

```cpp
// Element-wise ops (all broadcast scalars automatically)
simd<T, N> c = a + b;
simd<T, N> c = a * b + scalar;   // FMA when T=float
simd_mask<N> m = a > b;
simd<T, N> r = merge(a, b, m);   // r[i] = m[i] ? a[i] : b[i]
```

### Scalar × Vector (No Explicit Broadcast Needed)

Intel GPU hardware natively supports multiplying a scalar by a `simd` vector — the scalar is broadcast implicitly by the FMA unit. **No `replicate_w` or `simd<T,N>(scalar)` constructor needed.**

```cpp
// GOOD: extract scalar from simd, multiply directly — one instruction
half k_val = k_tile[tt * SUB_HD + ii];   // simd_view → half (implicit conversion)
acc[ii] += k_val * v_row;                // scalar * simd<half, N> → simd<half, N>

// BAD: explicit broadcast via replicate_w — wastes a GRF shuffle instruction
simd<half, N> k_bcast = k_tile.template replicate_w<N, 1>(tt * SUB_HD + ii);
acc[ii] += k_bcast * v_row;
```

**Why**: `replicate_w<N, 1>(i)` emits a hardware GRF shuffle (region select) to fill an N-wide register. When used only as a multiplier, the GPU's scalar-broadcast path in the EU instruction set handles this for free inside the FMA — eliminating the shuffle saves one instruction per (tt, ii) pair.

**Extraction note**: `simd<T,N>::operator[]` returns a `simd_view`, not a `T`. Assign it to a scalar first: `half k_val = simd_vec[i]` gives a true scalar `half` register. Using the element directly in arithmetic with a `simd` (`acc * v[i]`, `acc -= v[i]`) does not compile ("invalid operands" / "no viable overloaded operator").

### Reductions

```cpp
// Horizontal sum: use detail::sum, not reduce<T>(vec, std::plus<T>()) (see Correctness Traps)
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
sycl::half s = sycl::ext::intel::esimd::detail::sum<sycl::half, sycl::half, N>(vec);

// Horizontal max / min (direct hardware instruction, works on FP16)
sycl::half mx = hmax<sycl::half>(vec);
sycl::half mn = hmin<sycl::half>(vec);
float      mx = hmax<float>(vec);

// For FP16 reductions that need FP32 precision:
float mx = hmax<float>(convert<float>(vec));  // upcast first
```

### Sub-Range Access (`select`)

```cpp
// Read a contiguous sub-range: elements [offset, offset+COUNT)
simd<T, COUNT> sub = vec.template select<COUNT, 1>(offset);

// Read with stride: elements [offset, offset+STRIDE, offset+2*STRIDE, ...]
simd<T, COUNT> sub = vec.template select<COUNT, STRIDE>(offset);

// Write a sub-range
vec.template select<COUNT, 1>(offset) = other_vec;
```

`select` returns a `simd_view` (not `simd`) — assign to a named `simd<T,N>` when you need to pass it to a function.

### Type Conversion

```cpp
simd<float, N>      fp32 = convert<float>(fp16_vec);
simd<sycl::half, N> fp16 = convert<sycl::half>(fp32_vec);
simd<float, N>      fp32 = convert<float>(uint8_vec);

// Bit-cast reinterpret (zero cost, requires lvalue)
auto u32_vec = some_expression();  // store to named variable first
auto i32_vec = u32_vec.template bit_cast_view<int32_t>();  // reinterpret in place
```

### `simd` Constructor Patterns

```cpp
// Broadcast scalar
simd<float, N> v(0.f);            // all zeros
simd<sycl::half, N> v(-65504.f); // FP16 -inf

// Arithmetic sequence: [start, start+step, start+2*step, ...]
simd<int, 32> iota(0, 1);         // [0, 1, 2, ..., 31]
simd<int, 32> strided(0, 4);      // [0, 4, 8, ..., 124]
```

### pack_mask and fbl

Used to find the first lane matching a condition, without scalar loops:

```cpp
// pack_mask: simd_mask<N> → uint32_t bitmask (N = 8, 16, or 32 only)
uint32_t m = pack_mask(heap == min_val);   // bit i = 1 if heap[i] == min_val

// fbl (find bit lowest): index of least significant set bit; -1 if all zero
uint32_t pos = fbl(m);                     // scalar
simd<uint32_t, K> pos_vec = fbl(mask_vec); // vectorized per-element

// For N > 32: split into 32-element chunks, vectorized fbl, reconstruct global index
constexpr int MASK_N = N / 32;
simd<uint32_t, MASK_N> masks;
#pragma unroll
for (int mi = 0; mi < MASK_N; mi++) {
    simd<sycl::half, 32> chunk = vec.template select<32, 1>(mi * 32);
    masks[mi] = pack_mask(chunk == target);
}
simd<uint32_t, MASK_N> fbl_r   = fbl(masks);
simd<int32_t,  MASK_N> bit_idx = fbl_r.template bit_cast_view<int32_t>();
simd<int32_t,  MASK_N> glob    = bit_idx + simd<int32_t, MASK_N>(0, 32);
glob = merge(simd<int32_t, MASK_N>(INT_MAX), glob, bit_idx == -1);  // no match -> INT_MAX (<climits>)
int pos = (int)hmin<int32_t>(glob);          // smallest index = first match
if (pos == INT_MAX) pos = -1;                // -1 if none
// Do not use hmax here: it returns the match in the LAST matching chunk.
```

---

## Correctness Traps

Silent or confusing failures worth checking first (details and examples in `references/pitfalls.md`):

- `reduce<float>(vec, std::plus<float>())` with `using namespace sycl::ext::intel::esimd` compiles and returns 0; use `detail::sum<float, float, N>(vec)` (see Reductions).
- `block_load<half, N>` at an odd element offset reads wrong data unless it passes `properties{alignment<2>}` (see Memory Access Patterns).
- `properties{alignment<2>{}}` is a parse error; write `properties{alignment<2>}`.
- Lambdas inside a `SYCL_ESIMD_KERNEL` that call ESIMD intrinsics need `[[intel::sycl_explicit_simd]]`.
- `sycl::min` / `sycl::max` do not compile in ESIMD kernels; use `std::min` / `std::max`.
- Reading an uninitialized `simd<T, N>` is undefined behavior; broadcast-construct or load it first.

---

## Further Reading

Load these on demand:

| File | Contents |
|------|----------|
| `references/patterns.md` | Workgroup reductions via SLM, register file vs SLM, loop optimization, hybrid scalar/vector boundary handling |
| `references/testing.md` | Accuracy and performance test patterns for every new kernel |
| `references/pitfalls.md` | Common performance and correctness traps, with BAD/GOOD examples |
| `references/quick-reference.md` | Headers, API cheat sheet, kernel skeletons, compilation and large GRF mode |

Related skills: `esimd-lsc-2d-gather-scatter` (2D block loads, gather/scatter), `esimd-lsc-slm` (shared local memory, barriers), `xe2-dpas-patterns` (XMX DPAS operands).

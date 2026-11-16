# ESIMD Quick Reference

Part of the `esimd-base` skill.

## Quick Reference

### Headers

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
// For xmx::dpas:
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
// For lsc_load_2d:
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
```

### Memory

```cpp
// Block load/store — address must be a multiple of 4 bytes by default
simd<T, N> v = block_load<T, N>(ptr + offset);                              // float/int: always OK
simd<T, N> v = block_load<T, N>(ptr + offset, properties{alignment<2>});    // half/bf16: odd offset
simd<T, N> v = block_load<T, N>(ptr + offset, properties{alignment<1>});    // uint8_t: any offset
block_store<T, N>(ptr + offset, v);
block_store<T, N>(ptr + offset, v, properties{alignment<2>});               // half/bf16: odd offset

// Gather/scatter (non-contiguous, byte offsets)
simd<T, N> v = gather<T, N>(base, simd<uint32_t, N>(byte_offs));
scatter<T, N>(base, simd<uint32_t, N>(byte_offs), v);

// SLM
slm_init<BYTES>();                                    // inside kernel, unconditional
simd<T, N> v = slm_block_load<T, N>(byte_offset);
slm_block_store<T, N>(byte_offset, v);
barrier();
```

### Vector Ops

```cpp
// Arithmetic (scalar broadcast automatic)
simd<T,N> c = a + b;   simd<T,N> c = a * scalar;   simd<T,N> fma = a * b + c;

// Comparison and merge
simd_mask<N> m = a > b;
simd<T, N> r = merge(a, b, m);   // r[i] = m[i] ? a[i] : b[i]

// Sub-range
simd<T, C> sub = vec.template select<C, S>(start);   // read
vec.template select<C, 1>(off) = other;               // write

// Convert
simd<T2, N> v2 = convert<T2>(v);
auto r = lvalue.template bit_cast_view<T2>().read();  // reinterpret (lvalue only!)

// Arithmetic sequence
simd<int, N> iota(0, 1);    // [0,1,...,N-1]
simd<int, N> stride(0, S);  // [0,S,2S,...]

// Reductions (CORRECT)
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
T mx = hmax<T>(vec);   T mn = hmin<T>(vec);   // horizontal max/min

// pack_mask + fbl (find first matching lane, N ≤ 32)
uint32_t pos = fbl(pack_mask(vec == val));
```

### Kernel Skeletons

```cpp
// Independent threads (no SLM)
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class K>(sycl::range<D>(...),
        [=](sycl::id<D> idx) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            // ...
        });
});

// Workgroup reduction (SLM)
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class K>(sycl::nd_range<1>(N * G, G),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            const int gid = item.get_group(0);
            const int lid = item.get_local_id(0);
            slm_init<G * sizeof(float)>();
            float partial = compute(gid, lid);
            slm_block_store<float,1>(lid*4, simd<float,1>(partial));
            barrier();
            if (lid == 0) {
                auto parts = slm_block_load<float, G>(0);
                output[gid] = sycl::half(
                    sycl::ext::intel::esimd::detail::sum<float,float,G>(parts));
            }
        });
});
```

### Compilation

```bash
# Release
icpx -fsycl -O3 kernel.cpp -o kernel

# Debug
icpx -fsycl -g -O0 kernel.cpp -o kernel_debug

# List available SYCL devices
sycl-ls
```

**Register file and large GRF mode**:

Intel GPU threads have a fixed register file per thread:
- **Xe1** (Arc Alchemist, Gen12 iGPU): 128 GRF × 32B = **4 KB** per thread
- **Xe2+** (Arc Battlemage, Lunar Lake+): 128 GRF × 64B = **8 KB** per thread (256 × 64B = 16 KB in large GRF mode)

If a kernel uses too many `simd<T, N>` variables and exceeds this budget, the compiler spills registers to L1 cache, causing severe performance degradation. To double the register file size per thread, enable **large GRF mode**:

```bash
# Enable large GRF mode (doubles register file: 4 KB → 8 KB on Xe1, 8 KB → 16 KB on Xe2+)
# Note: doubling GRF halves the number of threads that can run simultaneously per EU,
# so use only when register pressure is the bottleneck.
icpx -fsycl -O3 kernel.cpp -Xs "-doubleGRF" -o kernel
```

**When to use `-doubleGRF`**:
- Kernel has many large `simd<T, N>` live simultaneously (e.g., `q_tiles[MS][KS]` arrays in XMX kernels)
- Compiler warns about register spilling, or profiler shows high L1 traffic
- Performance improves after enabling it (verify — not always beneficial due to reduced thread occupancy)

**Rule of thumb**: on Xe1, if total live FP16 elements across all `simd` variables in the hot loop exceeds ~1500, consider `-doubleGRF`.

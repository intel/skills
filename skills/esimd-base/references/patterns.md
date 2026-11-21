# Workgroup, Register and Loop Patterns

Part of the `esimd-base` skill.

## Workgroup and SLM Patterns

### Pattern 1: Independent Threads (No SLM)

Each thread works on a private slice of data. Use `range<D>` (not `nd_range`).

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class MyKernel>(
        sycl::range<1>(num_rows),
        [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            const int row = (int)idx[0];
            simd<sycl::half, ROW_DIM> data =
                block_load<sycl::half, ROW_DIM>(input + row * ROW_DIM);
            // ... process ...
            block_store<sycl::half, ROW_DIM>(output + row * ROW_DIM, result);
        });
});
```

### Pattern 2: Workgroup Reduction via SLM

Each workgroup computes one output element; GROUP_SIZE threads split the K dimension.

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class ReduceKernel>(
        sycl::nd_range<1>(num_outputs * GROUP_SIZE, GROUP_SIZE),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;

            const int group_id = (int)item.get_group(0);
            const int local_id = (int)item.get_local_id(0);

            slm_init<GROUP_SIZE * sizeof(float)>();

            // Each thread computes partial result over its K slice
            float partial = compute_partial(group_id, local_id);

            // Store partial to SLM
            slm_block_store<float, 1>(
                local_id * sizeof(float), simd<float, 1>(partial));
            barrier();

            // Thread 0 reduces and writes output
            if (local_id == 0) {
                simd<float, GROUP_SIZE> parts =
                    slm_block_load<float, GROUP_SIZE>(0);
                float result = sycl::ext::intel::esimd::detail::sum<
                    float, float, GROUP_SIZE>(parts);
                output[group_id] = sycl::half(result);
            }
        });
});
```

**Guidelines**:
- `GROUP_SIZE` 4–8 is optimal; larger sizes increase SLM pressure and barrier overhead
- `slm_init` must be called unconditionally inside the kernel body
- `barrier()` must be reached by all GROUP_SIZE threads — never inside a conditional

## Register File vs SLM

When working data fits in registers (≤ ~2 KB), prefer registers over SLM.

| | Register file | SLM |
|--|--------------|-----|
| Latency | ~1 cycle | ~30 cycles |
| Requires `nd_range` | No | Yes |
| Requires `barrier()` | No | Yes |
| Per-thread budget | 4 KB (Xe1), 8 KB (Xe2+) | 128 KB / sub-slice (shared) |
| GROUP_SIZE overhead | None | Yes |

```cpp
// BAD: SLM for per-thread data (30× slower than needed)
slm_init<1024>();
slm_block_store<sycl::half, 32>(offset, chunk);
simd<sycl::half, 32> v = slm_block_load<sycl::half, 32>(offset);

// GOOD: register file (1 cycle, simpler kernel)
simd<sycl::half, 512> data;                         // 1 KB, 25% of 4 KB budget
data.template select<32, 1>(chunk_idx * 32) = chunk;
simd<sycl::half, 32> v = data.template select<32, 1>(chunk_idx * 32);
```

**SLM is still the right choice** for shared data across threads in a workgroup (e.g., K-dimension reduction where GROUP_SIZE > 1). The rule is: SLM for **inter-thread** sharing, registers for **intra-thread** working data.

## Loop Optimization

### #pragma unroll

Add to every loop with a compile-time iteration count:

```cpp
#pragma unroll
for (int i = 0; i < STATIC_N; i++) { ... }  // fully unrolled

#pragma unroll 4
for (int i = 0; i < large_static_n; i++) { ... }  // unrolled 4 at a time
```

Without `#pragma unroll`, the compiler may generate a loop counter and branch, preventing vectorization and adding overhead.

### Hoist Loop-Invariant Work

```cpp
// BAD: recompute per iteration
for (int i = 0; i < N; i++) {
    float scale = 1.f / std::sqrt((float)HEAD_DIM);  // recomputed each time
    results[i] = dot(q[i], k) * scale;
}

// GOOD: hoist
const float scale = 1.f / std::sqrt((float)HEAD_DIM);
for (int i = 0; i < N; i++) results[i] = dot(q[i], k) * scale;
```

## Hybrid Scalar/Vector Strategy

For kernels with boundary cases (first/last element, partial tiles), use vectorized code for the hot path and simple scalar code for edges:

```cpp
for (int i = 0; i < num_outputs; i++) {
    int start = i * STRIDE - 1;
    int end   = start + WINDOW;

    if (start < 0 || end > num_elems) {
        // Boundary: correct scalar path
        start = std::max(0, start);
        end   = std::min(num_elems, end);
        float mx = -INFINITY;
        for (int j = start; j < end; j++) mx = std::max(mx, (float)input[j]);
        output[i] = sycl::half(mx);
    } else {
        // Hot path: vectorized (handles majority of iterations)
        simd<sycl::half, WINDOW> v =
            block_load<sycl::half, WINDOW>(input + start,
                                            properties{alignment<2>});
        output[i] = sycl::half(hmax<float>(convert<float>(v)));
    }
}
```

**Benefit**: avoids complex vectorized boundary logic, keeps the hot path clean, easy to debug.
**Measured**: 1.31 ms (scalar only) → 0.808 ms (hybrid) = 1.62× speedup.

For kernels where offsets can be clamped safely (e.g., max pooling where reading duplicate elements is harmless), prefer **branchless offset clamping** over hybrid:

```cpp
// Fully branchless — no hot/cold path split needed
simd<int, BS> offs = min(max(computed_offs, 0), max_elem);
simd<T, BS> vals   = gather<T, BS>(ptr, simd<uint32_t, BS>(offs) * sizeof(T));
// max(vals, running_max) — duplicate reads are harmless for max
```

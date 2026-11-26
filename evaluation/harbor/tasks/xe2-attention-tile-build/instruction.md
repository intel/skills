# Attention for one query block on an Intel Arc B-series GPU

Write `/app/attention.hpp` providing:

```cpp
sycl::event attention_tile(sycl::queue &q, const sycl::half *Q, const sycl::half *K,
                           const sycl::half *V, sycl::half *O);
```

It computes `O = softmax(Q · Kᵀ / sqrt(64)) · V`, softmax taken over each row, for

- `Q`: 8 × 64, `K`: 256 × 64, `V`: 256 × 64 — fp16, row-major, device memory
- `O`: 8 × 64 — fp16, row-major, device memory

Requirements:

- A single explicit-SIMD (ESIMD) SYCL kernel for Intel Xe2 (Battlemage), launched as one
  work-group of 4 work-items. Each work-item processes 64 of the 256 keys; the four
  partial results are combined through work-group local memory.
- The matrix products run on the XMX matrix engine.
- Treat it as performance code: matrix-engine operands must not be reshuffled in
  registers to get them into the order the engine expects, and the kernel must not
  spill registers.
- Numerically sound for inputs in [-2, 2]: the result matches a double-precision
  computation to fp16 accuracy.

There is no GPU in this environment. `/app/build.sh` builds `/app/check.cpp`, which
calls your function, ahead of time for Battlemage. Build with `/app/build.sh` as
provided; do not change it.

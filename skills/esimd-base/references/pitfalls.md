# Common Performance and Correctness Traps

Part of the `esimd-base` skill.

## Common Performance Traps

### 1D Range with Manual Index Decode (10–35× slowdown)

See Kernel Design in `SKILL.md`. Most impactful mistake.

### Element-Wise Memory Access

```cpp
// BAD: 128 separate scalar loads
for (int i = 0; i < 128; i++) val[i] = ptr[offset + i];

// GOOD: one 256B block load
simd<sycl::half, 128> val = block_load<sycl::half, 128>(ptr + offset);
```

### Non-Aligned Offset

```cpp
// BAD: offset 7 is not aligned
simd<float, 32> v = block_load<float, 32>(ptr, 7 * sizeof(float));

// GOOD: offset must be a multiple of load width (or use alignment<> property)
simd<float, 32> v = block_load<float, 32>(ptr, 8 * sizeof(float));
```

### Excessive Synchronization

```cpp
// BAD: barrier per iteration
for (int i = 0; i < N; i++) { compute(i); barrier(); }

// GOOD: barrier once after all writes
for (int i = 0; i < N; i++) slm_block_store<float, 1>(i*4, partial[i]);
barrier();
// ... reduce ...
```

### Register Spilling

```cpp
// BAD: many large vectors live simultaneously → register spill to L1
simd<float, 256> a, b, c, d, e, f, g, h;  // 8 KB — the whole Xe2 register file, twice Xe1's

// GOOD: reuse temporaries
simd<float, 128> tmp;
for (...) { tmp = load(); acc += process(tmp); }
```

### Branch Divergence

```cpp
// BAD: threads in same EU take different paths → serialized execution
if (local_id < 2) { path_a(); } else { path_b(); }

// GOOD: mask-based select (both paths execute, result selected per lane)
simd_mask<GROUP_SIZE> m = (simd<int,GROUP_SIZE>(0,1) < 2);
simd<T, N> result = merge(path_a_result, path_b_result, m);
```

### Uninitialized `simd<T, N>` Gives Garbage

`simd<T, N> v;` without initialization before element reads is undefined behavior. Always initialize before use — either broadcast-construct or seed with real data:

```cpp
simd<half, K> heap(-65504.f);           // broadcast init
simd<half, K> heap = block_load<...>(); // or initialize from load
```

### `alignment<2>{}` Parse Error in Template Context

```cpp
block_load<half, N>(ptr, properties{alignment<2>});   // GOOD
block_load<half, N>(ptr, properties{alignment<2>{}});  // BAD — parse error
```

### Inner Lambda Must Use `[[intel::sycl_explicit_simd]]`

Inner lambdas calling ESIMD intrinsics inside a `SYCL_ESIMD_KERNEL` must be marked `[[intel::sycl_explicit_simd]]`. Without it, the compiler errors: "SYCL kernel without 'sycl_explicit_simd' attribute can't call a function with this attribute". A `#define` macro is a valid fallback if the compiler doesn't support the attribute on lambdas.

### `sycl::min` / `sycl::max` Illegal in ESIMD Kernels

```cpp
// BAD — compile error in ESIMD
int n = sycl::min(a, b);

// GOOD
int n = std::min(a, b);
int n = (a < b) ? a : b;
```

### `reduce<T>(vec, std::plus<T>())` — Silent Zero Return

With `using namespace sycl::ext::intel::esimd` in scope, `reduce<float>(vec, std::plus<float>())` resolves to C++17 `std::reduce` (iterator overload) instead of the ESIMD intrinsic. It compiles without error and silently returns 0.

```cpp
// BAD — silently returns 0 (std::reduce, not esimd::reduce)
float s = reduce<float>(vec, std::plus<float>());

// GOOD — fully qualified, unambiguous
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
sycl::half s = sycl::ext::intel::esimd::detail::sum<sycl::half, sycl::half, N>(vec);
```

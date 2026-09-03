<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: SIMD Sort

## When to apply

**From source code** — any of the following on arrays or `std::vector` of
`float`, `double`, `int32_t`, `uint32_t`, `int64_t`, or `uint64_t`:

- `std::sort(arr, arr + n)` or `std::sort(vec.begin(), vec.end())`
- `std::nth_element` (partial partition)
- `std::partial_sort`
- `qsort()` with a numeric comparator
- A hand-written quicksort, merge sort, or introsort over numeric primitives

The data type is the key signal. This pattern applies to the six numeric
primitive types listed above. It does not apply to sorts of strings, pointers,
or complex objects without a numeric key.

**From a profile** — `std::sort`, `_introsort_loop`, `__gnu_cxx::__ops`,
`std::__sort`, or `std::__introsort` appearing prominently in `perf report`
or `perf annotate` on a workload that sorts numeric data. The sort is often
visible as a cluster of comparison + swap instructions with high branch
misprediction rates (`perf stat` shows elevated `branch-misses`).

---

## Why this is slow

`std::sort` uses introsort — a hybrid of quicksort, heapsort, and insertion
sort — operating on one element at a time. Every comparison is a branch, and
partitioning an array of n elements requires O(n log n) comparisons with no
data parallelism. Modern CPUs support AVX-512 operations that touch 16 floats
(or 8 doubles) simultaneously; `std::sort` uses none of this capacity.

**x86-simd-sort** (https://github.com/numpy/x86-simd-sort) vectorizes the
quicksort partitioning step using AVX-512 `vcompressps`/`vpcompressd`
instructions, and uses SIMD sorting networks for small sub-arrays (below
~64–512 elements depending on type). The result is 3–8× faster than
`std::sort` for numeric primitive arrays on AVX-512 hardware, with an AVX2
fallback (~2–4×). The library auto-selects the best available path at
runtime. It is used in production by NumPy, PyTorch, and OpenJDK.

---

## The fix

### Step 1 — Install the library

**Option A: installed library (recommended for projects with a build system)**

```bash
git clone https://github.com/numpy/x86-simd-sort
cd x86-simd-sort
meson setup --buildtype release builddir && cd builddir
meson compile
sudo meson install
# then in your build: pkg-config --cflags --libs x86simdsortcpp
```

As a Meson subproject, add to `meson.build`:
```meson
xss = subproject('x86-simd-sort')
xss_dep = xss.get_variable('x86simdsortcpp_dep')
```

**Option B: header-only (no install, compile-time ISA selection)**

Copy `src/x86simdsort-static-incl.h` into your source tree and include it.
Compile with `-mavx512f -mavx512dq -mavx512vl -O3` for AVX-512, or
`-mavx2 -O3` for AVX2. No runtime dispatch — the ISA is fixed at compile time.

### Step 2 — Replace call sites

```cpp
#include "x86simdsort.h"   // installed library
// or: #include "x86simdsort-static-incl.h"  // header-only
```

| C++ standard | x86simdsort equivalent | Notes |
|---|---|---|
| `std::sort(p, p+n)` | `x86simdsort::qsort(p, n, hasnan)` | Not stable |
| `std::nth_element(p, p+k, p+n)` | `x86simdsort::qselect(p, k, n, hasnan)` | Equivalent |
| `std::partial_sort(p, p+k, p+n)` | `x86simdsort::partial_qsort(p, k, n, hasnan)` | Equivalent |
| `std::stable_sort` | *(no equivalent — see caveats)* | |
| sort two arrays together | `x86simdsort::keyvalue_qsort(keys, vals, n, hasnan)` | 32/64-bit only |
| get sort indices | `x86simdsort::argsort(p, n, hasnan)` | Returns `std::vector<size_t>` |

**Example — replacing std::sort on a float array:**

```cpp
// Before
std::sort(arr.data(), arr.data() + n);

// After
x86simdsort::qsort(arr.data(), n, /*hasnan=*/false);
```

**Sorting custom objects by a numeric key:**

```cpp
// Sorts by the x member; O(N) extra space, benchmark before using
x86simdsort::object_qsort(points.data(), points.size(),
    [](const Point& p) { return p.x; });
```

### Step 3 — Set the `hasnan` flag correctly

The `hasnan` parameter (default `false`) must be set to `true` if the
floating-point array may contain NaN values. With `hasnan = false` and NaN
present, behavior is **undefined**.

When `hasnan = true`: NaNs are moved to the end of the array. For `qsort`,
bit-exact NaN values are **not preserved** — all NaNs become
`std::numeric_limits<T>::quiet_NaN()`. For `qselect` and `partial_qsort`,
the original bit-exact NaNs are preserved. For `argsort`/`argselect` with
NaN detected, the algorithm silently falls back to scalar `std::sort` (no
SIMD speedup).

Integer types (`int32_t`, `uint32_t`, `int64_t`, `uint64_t`) have no NaN;
always pass `hasnan = false`.

---

## Verification

```cpp
// Correctness check: compare against std::sort on a copy
std::vector<float> ref = arr;
std::sort(ref.begin(), ref.end());
x86simdsort::qsort(arr.data(), arr.size(), false);
assert(arr == ref);

// Timing (rough): time both sorts over the same data
auto t0 = std::chrono::high_resolution_clock::now();
x86simdsort::qsort(arr.data(), n, false);
auto t1 = std::chrono::high_resolution_clock::now();
// compare against std::sort on identical input
```

Compile-time check for C++17:
```bash
g++ -std=c++17 -O3 ... your_file.cpp
```

---

## Caveats summary

| Caveat | Detail |
|--------|--------|
| **C++17 required** | GCC ≥ 8, Clang ≥ 8, MSVC ≥ 2019 (16.8) |
| **Not a stable sort** | Relative order of equal elements is not preserved. No drop-in for `std::stable_sort` |
| **NaN undefined without `hasnan`** | Always set `hasnan = true` for float/double if NaN is possible |
| **NaN bit-exactness** | `qsort` with `hasnan=true` replaces all NaN with quiet_NaN; `qselect`/`partial_qsort` preserve the original NaN bits |
| **argsort + NaN → scalar fallback** | SIMD argsort/argselect silently degrades to `std::sort` when NaN is detected |
| **`object_qsort` O(N) space** | Allocates `n * sizeof(key_t) + n * sizeof(uint32_t)` bytes; profile the key function and measure before using |
| **AVX-512 ISA for 16-bit types** | `uint16_t`/`int16_t`/`_Float16` require AVX-512BW/VBMI2/FP16; 32/64-bit types work with AVX2 |
| **Header-only = compile-time ISA** | `x86simdsort-static-incl.h` requires explicit `-mavx512f` or `-mavx2`; no runtime dispatch |

---

## Presenting this to the user

Lead with the replacement table and a concrete before/after for their call site.
Then highlight:

1. **Why it's faster** (vectorized partitioning + sorting networks — not just "SIMD")
2. **The `hasnan` flag** — this is the most common correctness pitfall; always address it
3. **Stability** — only mention if the existing code uses `std::stable_sort`
4. **Production validation** — NumPy, PyTorch, OpenJDK all ship this; it is not experimental

Do not suggest benchmarking before adopting for straightforward array sorts of
the supported types — the speedup is well-established. Do recommend benchmarking
for `object_qsort` with expensive key functions before committing to it.

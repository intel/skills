# Parallel Sort with oneTBB

You have a C++ program at `/app/sort_benchmark.cpp` that sorts a large array of integers
using `std::sort` (single-threaded). Your task is to rewrite it to use
**oneTBB's `tbb::parallel_sort`** so it runs faster on multi-core hardware.

## Requirements

1. **Modify `/app/sort_benchmark.cpp`** (or create `/app/sort_parallel.cpp` — your choice)
   to sort a `std::vector<int>` of **10,000,000** elements using `tbb::parallel_sort`.
2. The sort must produce a **correctly sorted** array (ascending order).
3. Build the binary as **`/app/sort_parallel`** using the system `g++` compiler.
   oneTBB is already installed at `/usr/local` (headers + `libtbb.so`).
   Link with `-ltbb`.
4. The parallel version must complete in **under 5 seconds** on this machine.

## Hints

- Include `<oneapi/tbb/parallel_sort.h>` (or the older `<tbb/parallel_sort.h>`).
- `tbb::parallel_sort(vec.begin(), vec.end())` is the drop-in replacement.
- Compile example:
  ```bash
  g++ -O2 -std=c++17 sort_parallel.cpp -o /app/sort_parallel -ltbb
  ```

## What success looks like

Running `/app/sort_parallel` should print a single line:
```text
SORTED
```
and exit with code 0.

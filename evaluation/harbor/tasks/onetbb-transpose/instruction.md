# oneTBB tiled matrix transpose

You are given `/app/transpose.cpp`, a serial C++17 implementation of a matrix transpose inspired by the Parallel Research Kernels transpose workload.

Create a oneTBB implementation that:

1. Initializes an `n x n` matrix deterministically.
2. Repeatedly computes `B[j,i] = A[i,j] + iter`.
3. Validates probe coordinates `(0,0)`, `(n/3,n/5)`, `(n-1,n-1)`, and `(n/2,n/2)`. Each final value must match `A[i,j] + (iterations - 1)` after the last iteration.
4. Writes an executable binary at `/app/transpose_tbb`.

Requirements:

- Use oneTBB parallelism, preferably `tbb::parallel_for` / `oneapi::tbb::parallel_for` with `blocked_range2d` or another tiled approach.
- Accept optional CLI arguments: `<matrix_order> <iterations> [tile_size]`.
- Print a line containing `VALID` and a non-zero numeric signature derived from the validated probe values when validation passes.

A typical compile command is:

```bash
g++ -O3 -std=c++17 /app/transpose_tbb.cpp -ltbb -o /app/transpose_tbb
```

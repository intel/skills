# oneTBB 2D stencil

You are given `/app/stencil.cpp`, a serial C++17 implementation of a simple 2D four-neighbor stencil inspired by the Parallel Research Kernels stencil workload.

Create a oneTBB implementation that:

1. Initializes grid entries exactly as `grid[i,j] = (i * 17 + j * 13) % 101`.
2. Applies the update for the interior cells: `out[i,j] = 0.25 * (up + down + left + right)`.
3. Repeats the stencil for the requested number of iterations.
4. Computes a norm and prints a line containing `VALID`.
5. Writes an executable binary at `/app/stencil_tbb`.

Requirements:

- Use oneTBB parallelism with `tbb::parallel_for` / `oneapi::tbb::parallel_for` and `blocked_range2d` for the tiled stencil update.
- Do not update boundary cells in the stencil loop.
- Accept optional CLI arguments: `<grid_size> <iterations>`.

A typical compile command is:

```bash
g++ -O3 -std=c++17 /app/stencil_tbb.cpp -ltbb -o /app/stencil_tbb
```

# oneTBB parallel_reduce aggregate

You are given `/app/reduce.cpp`, a serial C++17 program that generates deterministic integer data and computes a validation signature from the sum, sum of squares, minimum, and maximum.

Create a oneTBB implementation that:

1. Generates the same data as the serial reference: `((i * 17 + 13) % 1009) - 504`.
2. Uses `oneapi::tbb::parallel_reduce` or `tbb::parallel_reduce` to compute the aggregate values.
3. Prints a line containing `VALID` and the same numeric signature as the serial reference.
4. Writes an executable binary at `/app/reduce_tbb`.

Requirements:

- Use a real oneTBB reduction body, not a serial loop hidden behind TBB includes.
- Accept optional CLI argument: `<length>`.
- Reject zero-length inputs with a non-zero exit code.

A typical compile command is:

```bash
g++ -O3 -std=c++17 /app/reduce_tbb.cpp -ltbb -o /app/reduce_tbb
```

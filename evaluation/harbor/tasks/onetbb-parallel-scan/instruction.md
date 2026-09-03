# oneTBB parallel_scan prefix sum

You are given `/app/scan.cpp`, a serial C++17 program that computes an inclusive prefix sum over deterministic integer data and prints a validation signature.

Create a oneTBB implementation that:

1. Generates the same input values as the serial reference: `(i % 11) + 1`.
2. Uses `oneapi::tbb::parallel_scan` or `tbb::parallel_scan` to compute the inclusive prefix sum.
3. Computes and prints the same signature as the serial reference.
4. Writes an executable binary at `/app/scan_tbb`.

Requirements:

- Use the real two-phase scan semantics (`pre_scan` / `final_scan` or equivalent oneTBB scan body).
- Accept optional CLI argument: `<length>`.
- Reject zero-length inputs with a non-zero exit code.
- Print a line containing `VALID` when validation passes.

A typical compile command is:

```bash
g++ -O3 -std=c++17 /app/scan_tbb.cpp -ltbb -o /app/scan_tbb
```

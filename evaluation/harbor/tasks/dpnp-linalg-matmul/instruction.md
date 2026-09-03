# dpnp matrix multiplication with linalg

You are given `/app/reference.py`, a serial NumPy program that generates two
deterministic `N × N` float64 matrices and computes their product, printing a
validation signature equal to the sum of absolute values of all elements in the
result matrix.

Create `/app/solution.py` that reproduces the **same** computation using **dpnp**:

1. Generate the same matrices as the reference:
   - `A[i, j] = ((i * 7 + j * 3) % 13) - 6` (float64)
   - `B[i, j] = ((i * 5 + j * 11) % 17) - 8` (float64)
   - Default size `N = 512`; accept optional CLI argument `<N>`.
2. Compute `C = A @ B` (matrix multiply) using **dpnp** — use either
   `dpnp.matmul(A, B)` or the `@` operator on dpnp arrays.
3. Print a line containing `VALID` and `sig=<value>` where the signature is
   `float(dpnp.sum(dpnp.abs(C)))`, formatted to 6 decimal places.
4. Reject `N < 1` with a non-zero exit code.

Requirements:

- The solution must use `dpnp.matmul` or the `@` operator on dpnp arrays —
  not a plain Python loop or NumPy.
- Results must match the NumPy reference within `1e-6` relative tolerance
  (oneMKL backend may differ from NumPy's BLAS in the last few ULP).
- The solution must run successfully on CPU (no GPU required).

```bash
python3 /app/reference.py
python3 /app/solution.py
python3 /app/solution.py 256
```

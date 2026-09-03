# dpnp batch statistical reductions

You are given `/app/reference.py`, a NumPy program that generates a deterministic
batch of vectors and computes per-column statistics, printing a validation
signature.

Create `/app/solution.py` that reproduces the **same** computation using **dpnp**:

1. Generate the same deterministic data matrix of shape `(ROWS, COLS)` where
   `ROWS = 10_000`, `COLS = 256`:
   - `M[i, j] = float(((i * 31 + j * 17) % 997) - 498)`
2. Compute **per-column** (axis=0) statistics using dpnp:
   - `col_sum  = dpnp.sum(M, axis=0)`     — shape `(COLS,)`
   - `col_mean = dpnp.mean(M, axis=0)`    — shape `(COLS,)`
   - `col_std  = dpnp.std(M, axis=0)`     — shape `(COLS,)`
3. Combine into a validation signature:
   `sig = float(dpnp.sum(col_sum)) * 1e6 + float(dpnp.sum(col_mean)) * 1e9 + float(dpnp.sum(col_std)) * 1e11`
4. Print a line containing `VALID` and `sig=<value>` formatted to 6 decimal places.

Requirements:

- Use `dpnp` for all array creation and reduction operations (not NumPy).
- Use the `axis=0` parameter for all reductions — do not reshape or transpose.
- Results must match the NumPy reference within `1e-7` relative tolerance.

```bash
python3 /app/reference.py
python3 /app/solution.py
```

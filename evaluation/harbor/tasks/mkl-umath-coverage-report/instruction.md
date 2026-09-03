# Which ufuncs does mkl_umath actually accelerate?

`/app/reference.py` runs the `arc_distance` kernel: a great-circle distance over
two sets of coordinate pairs, built from `np.sin`, `np.cos`, `np.sqrt` and
`np.arctan2`. It runs on stock NumPy, with `mkl_umath` installed but dormant.

Write `/app/solution.py` that activates `mkl_umath` for the same kernel and then
reports, accurately, which parts of it actually dispatch to oneMKL.

Requirements:

1. Activate `mkl_umath` so the existing `np.*` calls are accelerated. Do not
   rewrite the kernel: import `arc_distance`, `initialize` and `bit_signature`
   from `reference.py` and call them unchanged.
2. Print one line containing `VALID`, plus `n=<n>`, `patched=True`, and
   `sig=<value>` with the integer signature from `bit_signature`.
3. The signature must equal what the patched kernel produces and must **differ**
   from the unpatched reference. The VML kernels are a different implementation,
   so tens of thousands of elements change even though a plain float sum of the
   output would stay identical — that is why the signature is a bit-pattern sum.
4. Write a coverage report to `/app/coverage.json`:
   ```json
   {
     "patched": true,
     "covered": ["<ufunc>", ...],
     "not_covered": ["<ufunc>", ...],
     "complex_capable": {"<ufunc>": true|false, ...}
   }
   ```
   For each of the four ufuncs the kernel uses (`sin`, `cos`, `sqrt`,
   `arctan2`), decide whether `mkl_umath` registers it at all, and whether its
   registered loops include any complex dtype.
5. **Determine this from the installed package, not from memory.** The registry
   is introspectable at runtime; a hardcoded list will be wrong the moment the
   package changes. The verifier recomputes both the coverage report and the
   expected signature in its own process, so a hand-written report or a
   remembered number will not pass.
6. Accept an optional CLI argument `<n>` (at least 1024). Reject smaller values
   with a non-zero exit code.

Notes:

- Patching is signature-matched: a dtype absent from a ufunc's registered loops
  cannot be accelerated, whatever the array size.
- Do not report a speedup. This task is about which code paths change, not by
  how much, and nothing here is timed.

```bash
python3 /app/reference.py
python3 /app/solution.py
python3 /app/solution.py 2000000
```

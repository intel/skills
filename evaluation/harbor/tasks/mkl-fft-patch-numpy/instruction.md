# Route existing np.fft call sites through oneMKL

`/app/reference.py` builds a deterministic two-tone signal, transforms it with
`np.fft.fft`, and prints a bit-exact signature of the resulting spectrum.

This numpy build is **stock**: `mkl_fft` is installed but FFT is dormant, so
`np.fft.fft` still resolves to `numpy.fft`.

Create `/app/solution.py` that computes the same signature with the transform
dispatched to oneMKL instead.

Requirements:

1. **Do not rewrite the transform call site.** The body must still call
   `np.fft.fft(signal)`. Activate `mkl_fft` so that call routes to oneMKL rather
   than replacing it with a direct `mkl_fft.fft(...)` call.
2. Reuse the reference's own helpers: import `make_signal` and
   `spectrum_signature` from `reference.py` rather than reimplementing them.
3. Use the real top-level API. Do not import from private submodules and do not
   invent function names.
4. Print one line containing `VALID`, plus `n=<n>`, `fft_module=<value>` where
   the value is `np.fft.fft.__module__` observed at call time, and `sig=<value>`
   with the integer signature.
5. The signature must equal what oneMKL produces and must **differ** from the
   stock-NumPy reference. MKL's DFTI and NumPy's pocketfft are different
   implementations, so the raw spectra differ in the last bits; the signature is
   bit-exact and will not match the reference. That is expected.
6. Accept an optional CLI argument `<N>` (a power of 2, at least 8). Reject a
   non-power-of-2 or `N < 8` with a non-zero exit code and a message on stderr.

Notes:

- `mkl_fft.is_patched()` is not on its own a reliable proof that FFT is active:
  on a build-wired numpy it reads False while FFT is live. The authoritative
  check is `np.fft.fft.__module__`.
- Patch and restore are reference-counted: N patch calls need N matching restore
  calls to fully unpatch.
- The verifier recomputes both the stock and the oneMKL signature in its own
  process and compares. Printing a remembered or hand-computed number will not
  pass.

```bash
python3 /app/reference.py
python3 /app/solution.py
python3 /app/solution.py 8192
```

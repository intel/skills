# dpnp FFT power spectrum

You are given `/app/reference.py`, a NumPy program that generates a deterministic
1D signal (sum of two sine waves), computes its FFT, and prints a validation
signature derived from the power spectrum.

Create `/app/solution.py` that reproduces the **same** computation using **dpnp**:

1. Generate the same deterministic signal of length `N = 65536` (power of 2):
   - `t = dpnp.arange(N, dtype=dpnp.float64) / N`
   - `signal = dpnp.sin(2 * dpnp.pi * 50 * t) + 0.5 * dpnp.sin(2 * dpnp.pi * 120 * t)`
2. Compute the FFT using `dpnp.fft.fft(signal)`.
3. Compute the power spectrum: `power = dpnp.abs(spectrum) ** 2`.
4. Print a line containing `VALID` and `sig=<value>` where the signature is
   `float(dpnp.sum(power[:N//2]))` formatted to 4 decimal places.
5. Accept an optional CLI argument `<N>` (must be a power of 2 ≥ 8); reject
   non-power-of-2 or `N < 8` with a non-zero exit code.

Requirements:

- Use `dpnp.fft.fft` (not `numpy.fft` or `scipy.fft`).
- Results must match the NumPy reference within `1e-6` relative tolerance.
- The solution must run successfully on CPU.

```bash
python3 /app/reference.py
python3 /app/solution.py
python3 /app/solution.py 8192
```

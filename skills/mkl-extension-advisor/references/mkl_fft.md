# mkl_fft reference

Load when the code uses `np.fft.*`, `scipy.fft.*`, or `scipy.fftpack.*`.

A Python interface over oneMKL DFTI. Covered transforms: `fft`/`ifft`/`fft2`/`ifft2`/`fftn`/`ifftn`, `rfft`/`rfft2`/`rfftn`, `irfft`/`irfft2`/`irfftn`, plus hermitian `hfft`/`ihfft`. Handles strided arrays without a forced copy, and preserves input precision (`float32 -> complex64`).

Do not claim "stock NumPy upcasts float32 to double" - that was true of older NumPy only. Current NumPy computes natively in float, double or long double, so on a modern stack there is no dtype change to flag. Check rather than assert:

```python
np.fft.rfft(np.ones(8, dtype=np.float32)).dtype   # complex64 => native
```

## What it does NOT accelerate

- FFT helpers (`fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`). These are a vendored copy of NumPy's, built from `np.arange`/`np.roll`, kept only to avoid a circular import when patching. Patching still changes their `__module__` to `mkl_fft.interfaces._numpy_helper` - that is a rebind, not acceleration.
- Tiny transforms where dispatch overhead dominates.
- Extended precision, and **the two interfaces disagree**: the numpy interface silently downcasts `longdouble -> float64` / `clongdouble -> complex128`, while the scipy interface raises `NotImplementedError`. Since current NumPy supports longdouble FFT natively, routing such arrays through mkl_fft is a precision *regression*. Flag the silent path especially - it loses precision with no error.

## Decide fit

YES on large transforms, batched or N-D transforms along axes, repeated FFTs in a loop, real-input `rfft*`, multi-core machines. Dtypes: float32/float64/complex64/complex128.

## Dependency probe

```python
import mkl_fft
hasattr(mkl_fft.interfaces, "scipy_fft")   # needs scipy AND mkl-service
```

Safe after a bare import: the package imports `interfaces` at top level, and `interfaces/__init__.py` guards `import scipy` / `import mkl` in try/except, binding `scipy_fft` only on success. So a missing dependency makes the backend **silently absent** rather than raising. That is an environment check, not an API check - take the patch functions as available.

## Activation

NumPy, process-wide:
```python
import mkl_fft
mkl_fft.patch_numpy_fft()
# ... np.fft.* now routes to mkl_fft ...
mkl_fft.restore_numpy_fft()
```
Scoped: `with mkl_fft.mkl_fft():`

SciPy is a **backend, not a numpy patch** - `patch_numpy_fft()` does nothing for `scipy.fft.*`:
```python
with scipy.fft.set_backend(mkl_fft.interfaces.scipy_fft, only=True):
    result = scipy.fft.fft(x)
```
The scipy interface also carries hermitian 2-D/N-D transforms the numpy one lacks (`hfft2`, `ihfft2`, `hfftn`, `ihfftn`) and is the only one with `get_workers`/`set_workers`. It defaults workers to max MKL threads, not SciPy's default of 1.

Patch/restore are reference-counted: N patches need N restores. Two patches then one restore leaves numpy **still patched** - a real hazard when a library and the application both patch. Verify with `__module__`, do not assume. An over-restore warns, but the guard is thread-local, so the two-patch/one-restore case is silent.

A redundant patch on already-wired numpy is harmless and idempotent. Noise worth removing, not a bug.

Multi-threading: upstream says only "prefer the `mkl_fft` context manager". The patch is a GIL-atomic `setattr`, so the hazard is which implementation a concurrent thread gets, not memory unsafety (unlike mkl_umath).

## Proof it is active

`np.fft.fft.__module__ == "mkl_fft.interfaces._numpy_fft"`. This is upstream's own recommended check.

`mkl_fft.is_patched()` alone is **not** sufficient: it tracks only the explicit patch counter, so a build-wired numpy reads False while FFT is live. Read together they tell you *how* it got wired:

| `__module__` | `is_patched()` | Meaning |
|---|---|---|
| MKL | False | the numpy build itself is wired |
| MKL | True | something in-process patched it |
| `numpy.fft` | False | dormant |

Wiring has several causes: an Intel-built numpy, a persistent `.pth` patch from the package CLI (runs at every interpreter start, invisible to `pip list`), a `sitecustomize.py` injection, or an earlier patch in the same process. `python -m mkl_fft --patch status` reports the persistent case.

For an Intel conda numpy, the wiring lives in the numpy **recipe**, which rebinds `numpy/fft/__init__.py` globals directly and never calls `patch_numpy_fft()`. That is why the counter stays 0.

## Numerical effect

DFTI and pocketfft are different implementations, so bitwise identity is not guaranteed (it happens to hold at trivial sizes). Round-trips return to the input at the same noise floor.

No tolerance is published; the project's own cross-implementation test uses `64 * eps` and its README asserts only `np.allclose`. Measured agreement sits at rounding level but drifts with transform size, so treat it as an observation, not a bound.

If the code has golden files, flag it - and warn about the trap: **`atol` must be scaled to signal magnitude.** On a pure tone the relative L2 error can sit near epsilon while the max element-wise relative error is enormous, because near-zero spectral bins blow up `rtol`. Mitigation: MKL output is bitwise stable run-to-run and across thread counts, so regenerating goldens once is viable.

## Caveats

- `mkl_fft.*` direct functions take first arg `x`; `interfaces.numpy_fft.*` take `a`.
- `irfft*` may modify its input; do not rely on it being untouched.

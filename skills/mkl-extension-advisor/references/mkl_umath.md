# mkl_umath reference

Load when the code uses element-wise math ufuncs on arrays (`np.sin`, `np.exp`, `np.log`, ...).

Swaps oneMKL VML kernels in as the C-level inner loops of NumPy's ufuncs, so existing `np.*` call sites accelerate with no rewrite.

## Read coverage from the installed build

```python
import mkl_umath._ufuncs as u
"sin" in dir(u)        # is this ufunc registered at all?
u.sin.types            # which dtype loops exist -> ['f->f', 'd->d']
```

`types` answers every "does it cover complex / integer / this dtype" question. Patching is signature-matched, so a dtype absent from `types` cannot be accelerated at any array size. Do not recite a coverage list from memory; the build is the authority.

Covered families: trig, hyperbolic, exp/log, `sqrt`/`cbrt`/`square`/`reciprocal`, basic arithmetic, rounding, comparisons, logical ops, `isfinite`/`isinf`/`isnan`/`fmax`/`fmin`.

## What it does NOT cover

- **Complex transcendentals.** `exp`, `log`, `sin`, `cos`, `sqrt`, `tanh` are float-only (`f->f`, `d->d`). Patching installs nothing for complex input, so results stay bit-identical to stock NumPy. Complex-heavy DSP or FFT post-processing is a NO.
  Complex loops do exist for ~two dozen non-transcendental ufuncs (arithmetic, comparisons, predicates, logical ops). Check `types` rather than assuming either way.
- `maximum` / `minimum` - not covered. But `fmax` / `fmin` **are**. Easy to get backwards.
- `power` with an arbitrary exponent. NumPy rewrites only `**2`->`square`, `**0.5`->`sqrt`, `**-1`->`reciprocal`. `**(1/3)` and `**2.5` stay on `power`, which is not registered. `cbrt` is covered only as `np.cbrt(a)`.
- `arctan2`, `hypot`, `logaddexp`, `heaviside`, `remainder`, `floor_divide`, `gcd`, `lcm`, bitwise ops.
- Integer-only work. The VML paths are float, double and complex.

## Eligibility gates

A patched ufunc reaches VML only when all three hold:

1. **Contiguous** input and output. Tested against `steps[]` versus `sizeof(type)` - **strides, not flags**.
2. **Non-overlapping** buffers.
3. **Coalesced inner-loop length above a per-operation threshold.** Three tiers: transcendentals lowest, `divide` just below that, `add`/`subtract`/`multiply` an order of magnitude higher.

The threshold values are private `#define`s in `mkl_umath/src/mkl_umath_loops.c.src` (names starting `VML_`), not API, and can change between releases. Read them from the installed build before advising on a borderline size, and say which build they came from. If you cannot read them, answer with the gate rather than a number: "worth patching only if the inner dimension is well above the transcendental threshold".

**A transpose does not defeat the contiguity test.** `.T` of a 2-D C-contiguous array is F-contiguous, so the axes coalesce into one run and the test passes; only the size test can fail. `np.ones((1000, 4)).T` gives an inner length of 4000, not 4. Scale it up and it dispatches. Check directly:

```python
a.flags["C_CONTIGUOUS"], a.flags["F_CONTIGUOUS"], a.strides, a.itemsize
```

Either contiguity flag means the run coalesces. A genuinely strided view (`a[::2]`, `a[:, ::2]`) does not reach VML: NumPy buffers it into chunks capped at the buffer size, which equals the transcendental threshold, and the gate is a strict `>`, so a buffered run is always one element short. Copy to contiguous first if you need dispatch. This is buffer-size dependent, so confirm it rather than assuming.

Below the threshold the call does **not** revert to NumPy's loop - the patch already replaced it, so mkl_umath's own fallback runs. Whether that beats stock NumPy depends on how both were built, so do not assert a sub-threshold regression. Say "the VML kernel never runs, so expect no speedup".

## Numerical effect

Not a bitwise drop-in. `sin`, `cos`, `exp`, `log`, `tanh`, `sqrt` and `divide` can differ in the last bit: measured on float64 `sqrt`, 166 of 20000 elements differ, max relative difference 2.2e-16.

`add` and `multiply` (array-array) do come back bitwise identical.

No accuracy bound is published and no VML accuracy mode is set, so oneMKL's default high-accuracy mode applies. Say "close, no specific bound claimed" rather than quoting a ULP figure. If the code has exact-output tests or golden files, flag it; the fix is a tolerance comparison, not abandoning the patch.

## Decide fit

YES on large, contiguous, real float32/float64 transcendental math in hot code, above the thresholds.

NO on sub-threshold arrays, genuinely strided views, complex transcendentals, uncovered functions, integer work, bandwidth-limited cheap ops. Not automatically NO for a transposed array.

## Activation

```python
import mkl_umath
mkl_umath.patch_numpy_umath()
# ... np.sin / np.exp now use VML ...
mkl_umath.restore_numpy_umath()
```
Scoped: `with mkl_umath.mkl_umath():`

## Proof it is active

`mkl_umath.is_patched()`. Read it rather than predicting it: importing does not patch, but a `.pth` or `sitecustomize.py` hook may already have. Those hooks swallow exceptions, so an incompatible NumPy leaves the process silently unpatched - prefer an explicit call, then verify.

## Thread safety

The `patch_numpy_umath` docstring is explicit: running NumPy calls in one thread while another patches or unpatches "will lead to undefined behavior at best, and segmentation faults at worst", and recommends the context manager. The reason is that patching mutates C function pointers inside live ufunc objects; the internal lock serialises patch-vs-patch, not patch-vs-execution. Patch before worker threads start, or scope it.

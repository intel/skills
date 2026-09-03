---
name: dpnp-quickstart
description: >-
  NumPy-compatible array operations optimized for Intel hardware. Use when the user
  wants to migrate or port NumPy code to dpnp, asks whether a NumPy hot path can run
  on an Intel CPU or GPU, needs to check dpnp installation or SYCL device selection
  with dpctl, hits a NumPy API dpnp does not implement, or wants to compare dpnp
  against NumPy. Covers install, device control, fallback patterns, and profiling.
license: Apache-2.0
compatibility: "Requires dpnp and dpctl. GPU execution requires the Intel GPU driver stack and a SYCL-visible device."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp quickstart

## Purpose

Runs NumPy-style array operations on Intel CPUs and GPUs through `dpnp`, which
mirrors the NumPy API over SYCL device memory. Covers installation, SYCL device
selection with `dpctl`, migrating existing NumPy code, falling back to NumPy for
unimplemented APIs, and measuring whether the move actually paid off.

Prefer this over plain NumPy when the arrays are large and the work is math-heavy
on Intel hardware. Prefer plain NumPy when the arrays are small — `dpnp` has
dispatch overhead that a small array cannot amortize.

## When to Use This Skill

Use this skill when:

- The user is migrating or porting NumPy code to Intel CPU or GPU execution.
- The user asks whether a NumPy hot path can run on an Intel GPU.
- The user needs to check a `dpnp` install, or which SYCL device will be used.
- The user hit a NumPy API that `dpnp` does not implement.
- The user wants to compare `dpnp` against NumPy.

Use `dpnp` for:

- Large arrays (>10,000 elements)
- Math-heavy operations (linear algebra, FFT, reductions)
- Intel CPU/GPU acceleration

Do **not** use this skill — stick with NumPy — for:

- Small arrays (<1,000 elements)
- I/O operations
- APIs not yet implemented in dpnp

## Quick Start

Run this first to confirm the environment before changing any user code.

```bash
# Conda (officially recommended)
conda install -c https://software.repos.intel.com/python/conda -c conda-forge --override-channels dpnp

# Alternative: pip (may have native dependency issues)
pip install dpnp

# verify installation
python -c "import dpnp; print(dpnp.__version__)"
```

Then confirm which device the arrays will actually land on:

```python
import dpctl
import dpnp as np

print(dpctl.select_default_device())

x = np.arange(100_000)
print(x.sycl_device)
```

Report what this prints. Do not claim GPU execution if the output shows a CPU
device — the default SYCL device is the GPU only when one is visible.

## Implementation Guide

1. **Confirm the environment** — install and print the device, as in Quick Start.
2. **Swap the import.** `dpnp` is a drop-in NumPy replacement for the covered API:

   ```python
   # Drop-in NumPy replacement
   import dpnp as np

   # Create arrays
   x = np.array([1, 2, 3, 4])
   y = np.arange(1000000)

   # Operations work like NumPy
   result = np.sum(y)
   dot_product = np.dot(x, x)
   ```

3. **Port the hot path.** Array creation, reductions, and linear algebra keep
   their NumPy spelling:

   ```python
   import dpnp as np

   # Array creation
   a = np.zeros((100, 100))
   b = np.ones(1000)
   c = np.linspace(0, 10, 100)

   # Math operations
   sum_val = np.sum(a)
   mean_val = np.mean(b)
   std_val = np.std(c)

   # Linear algebra
   mat = np.random.randn(100, 100)
   result = np.dot(mat, mat.T)
   ```

4. **Add fallbacks for uncovered APIs.** `dpnp` implements a subset of NumPy, and
   coverage is per-parameter, not just per-function. Check with `dir(dpnp)` or the
   documentation, and guard the call:

   ```python
   import dpnp
   import numpy as np

   def safe_unique(x):
       try:
           return dpnp.unique(x)
       except (NotImplementedError, TypeError):
           host_x = dpnp.asnumpy(x) if isinstance(x, dpnp.ndarray) else x
           return np.unique(host_x)
   ```

5. **Convert at API boundaries, not inside loops.** Use `dpnp.asnumpy()` when a
   downstream library needs a NumPy array. Pandas, scikit-learn, and many
   NumPy-based libraries usually expect NumPy arrays: use `dpnp` for the numeric
   hot path, then convert once with `asnumpy()` before calling host-oriented
   libraries. Repeated device-to-host copies inside tight loops can erase
   acceleration gains.
6. **Constrain the device when the code must be portable.** There is no ambient
   current-device setting and no context manager to enter — placement is an argument
   at construction. Inspect what is visible with `dpctl`, then pin allocation with
   `device=` or by reusing an existing array's queue, when the same code runs across
   workstations, containers, and cloud VMs with different SYCL devices:

   ```python
   import dpctl
   import dpnp

   print([d.filter_string for d in dpctl.get_devices()])   # what is actually there

   x = dpnp.arange(100_000, device="cpu")                  # pin this array
   print(x.sycl_device)

   y = dpnp.zeros(x.size, sycl_queue=x.sycl_queue)         # keep the next one beside it
   ```

7. **Validate, then measure.** Compare against NumPy with
   `numpy.testing.assert_allclose()` for critical math before making any
   performance claim.

## Performance

No verified benchmark numbers ship with this skill yet. Do not state a speedup
that measurement in the user's own environment does not support.

To measure:

- Warm up once before timing, to avoid measuring first-run compilation.
- Compare against NumPy with the same inputs and dtype.
- Profile end-to-end pipelines, including conversions and host-library calls.

## Gotchas & Limitations

- **First run is slow**: JIT compilation happens on first execution. Time the second run.
- **Not all NumPy APIs available**: Check compatibility with `dir(dpnp)` or documentation.
- **Data transfer cost**: Converting between dpnp and NumPy arrays has overhead. Avoid in tight loops.
- **Small arrays slower**: dpnp has dispatch overhead. Use NumPy for small arrays (<1,000 elements).
- **Parameter-level gaps**: a function existing in `dpnp` does not mean every
  NumPy keyword argument is accepted — `TypeError` on an unexpected keyword is the
  common symptom.
- **Device assumptions**: the default SYCL device is whatever is visible. Code that
  works on a GPU workstation can land on CPU in CI without erroring.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need to check a claim against upstream documentation — API coverage for a given release, install prerequisites, or which `dpnp` version implements a function |

Two questions in this skill should not be answered from memory, and this is where
they get answered: **is this NumPy API covered** (coverage is per keyword argument
and changes between releases) and **is this package available for the user's
platform**.


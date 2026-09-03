---
name: dpnp-linalg-fft
description: >-
  Linear algebra and FFT with dpnp on Intel CPUs and GPUs, backed by oneMKL. Use
  when a matrix multiply, solve, decomposition, eigenvalue problem, or Fourier
  transform is the hot part of NumPy code, when the user asks whether dpnp covers
  a linalg or FFT call, when an FFT result differs slightly from NumPy's, or when
  a large transform runs out of device memory. Covers the supported surface,
  transform sizing and plan reuse, fallbacks for what is missing, and how to check
  conditioning before solving.
license: Apache-2.0
compatibility: "Requires dpnp with its oneMKL backend. Fallback examples use SciPy."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp linear algebra and FFT

## Purpose

Routes linear algebra and Fourier transforms through `dpnp`, which dispatches
them to oneMKL on Intel CPUs and GPUs. Covers what the API includes, how to size
and warm a transform, what is not implemented and how to fall back to SciPy for
it, and which numerical differences from NumPy are expected rather than bugs.

Prefer this skill when matrix or transform work is the expensive part of a NumPy
program. Prefer plain NumPy for small operands, where the dispatch cost is not
amortized, and SciPy for sparse problems, which have no `dpnp` equivalent.

## When to Use This Skill

Use this skill when:

- A matmul, `solve`, decomposition, or eigenvalue problem is the hot path.
- An FFT or a convolution through FFT is the hot path.
- The user needs to know whether `dpnp` implements a specific linalg or FFT call.
- An FFT or a decomposition disagrees with NumPy in the last digits.
- A large transform fails on device memory.

Do **not** use this skill for sparse linear algebra (`scipy.sparse`, on host
arrays), for small operands, or as a source of speedup figures — there are none
here on purpose.

## Quick Start

```python
import dpnp as np

A = np.random.randn(1000, 1000)
B = np.random.randn(1000, 1000)
C = A @ B                      # matmul, dispatched to oneMKL
x = np.linalg.solve(A, B)      # linear system

_ = np.fft.fft(np.zeros(1024))       # warm the plan for this size first
spectrum = np.fft.fft(np.random.randn(1024))

real_spectrum = np.fft.rfft(np.random.randn(2048))   # output length 1025
```

## Implementation Guide

1. **Use the surface that exists.** Matrix product as `@`, `matmul()`, or
   `dot()`; `svd()`, `qr()`, `cholesky()`; `eig()`, `eigvals()`, `eigh()`,
   `eigvalsh()`; `solve()`, `lstsq()`; `inv()`, `det()`, `norm()`, `cond()`,
   `matrix_rank()`; `inner()`, `outer()`, `cross()`. Prefer `@` for readability
   and the function form when it has to be passed around. `dot()` and `matmul()`
   agree on 2-D and differ for higher rank.

2. **Batch instead of looping.** The linalg functions take leading batch
   dimensions, so `np.linalg.solve(A, B)` with `A.shape == (10, 100, 100)` solves
   ten systems in one dispatch rather than ten.

3. **Pick the right transform.** `fft`/`ifft` for complex input, `rfft`/`irfft`
   for real input (output length `n // 2 + 1`, exploiting conjugate symmetry),
   `hfft`/`ihfft` for Hermitian data, `fft2`/`rfft2` for images, `fftn`/`rfftn`
   for higher rank, with `fftfreq`, `rfftfreq`, `fftshift`, and `ifftshift` as
   helpers. Using the real variants on real data halves the output.

4. **Size and warm transforms.** A power-of-two length is the friendliest case;
   pad up to one when the trailing samples do not matter:

   ```python
   n_padded = 2 ** int(np.ceil(np.log2(n)))
   spectrum = np.fft.fft(np.pad(signal, (0, n_padded - n)))
   ```

   The first call on a new size pays for plan creation and compilation;
   subsequent calls on that size reuse it, with no explicit plan management.

5. **Check conditioning before solving**, so a singular matrix produces an
   answer rather than an exception:

   ```python
   if np.linalg.cond(A) < 1e15:
       x = np.linalg.solve(A, b)
   else:
       x = np.linalg.lstsq(A, b, rcond=None)[0]
   ```

6. **Fall back deliberately for what is missing.** Sparse problems, generalized
   eigenvalue problems, matrix functions such as `expm`, `logm`, `sqrtm`, and
   short-time transforms have no `dpnp` implementation. Convert, call SciPy, and
   convert back — once, at the boundary:

   ```python
   import scipy.linalg

   result = np.asarray(scipy.linalg.expm(np.asnumpy(A)))
   ```

   Verify coverage against the installed release rather than a remembered list:
   the set has grown between versions, and coverage can be per keyword argument.

## Performance

No measured numbers ship with this skill, and none belong in it. What holds
regardless of hardware:

- There is a size below which dispatch dominates and NumPy is the better answer.
  Matrices in the hundreds of rows and transforms of a few hundred points are in
  that region; measure the actual shapes rather than adopting a threshold.
- Compare a warmed-up run. The first call on a new shape includes compilation and
  plan creation, so timing it measures the wrong thing.
- Real-input transforms do less work than complex ones on the same data.
- Batched calls amortize dispatch that a Python loop pays per iteration.
- A fallback to SciPy costs two transfers and host compute; count it in the
  end-to-end number, not just the call.

## Gotchas & Limitations

- **Results are not bit-identical to NumPy.** `dpnp` goes through oneMKL and
  NumPy through pocketfft or its own LAPACK; agreement is to within roundoff, and
  a comparison must use `numpy.testing.assert_allclose`, not equality.
- **Integer operands are not the fast path.** Keep operands in `float64` or
  `complex128` unless there is a reason not to.
- **No sparse support at all.** `scipy.sparse` on host arrays is the answer, not
  a `dpnp` equivalent.
- **Coverage claims expire.** Any list of unimplemented functions is true of one
  release. Check `dir(dpnp.linalg)`, `dir(dpnp.fft)`, and the documentation for
  the installed version before telling a user something is missing.
- **Large transforms fail on device memory.** Chunk the signal, or switch to a
  real-input transform; see `dpnp-memory` for capacity and chunking.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the linalg or FFT coverage of a specific dpnp release, the oneMKL routine behind a call, or the SciPy function to fall back to |

Two questions here should never be answered from memory: **which linalg and FFT
functions the installed release implements** (the list grows, and coverage can be
per keyword argument) and **which oneMKL routine backs a call**, which is what
explains a numerical difference from NumPy.

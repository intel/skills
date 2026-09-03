# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the question is whether a
linalg or FFT call is implemented in the user's release, which oneMKL routine backs
it, or why a result disagrees with NumPy in the last digits.

## dpnp — the covered surface

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point; the release this skill's guidance is written against |
| [Linear algebra reference](https://intelpython.github.io/dpnp/reference/linalg.html) | the implemented `dpnp.linalg` functions and their signatures |
| [FFT reference](https://intelpython.github.io/dpnp/reference/fft.html) | the implemented transforms, including which real and Hermitian variants exist |
| [NumPy comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) | **the authoritative coverage answer.** Coverage is per keyword argument, so a function that exists can still reject the call being written |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | source and release notes; where a newly added decomposition shows up first |

## oneMKL — what the call dispatches to

| Source | Use it for |
|---|---|
| [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | which routine families back BLAS, LAPACK, and DFT calls, and the accuracy modes that explain a last-digit difference from NumPy |
| [oneAPI specification](https://uxlfoundation.github.io/oneAPI-spec/) | the specified behaviour of the math library interfaces, when a question is about the contract rather than one implementation |

## The comparison targets

| Source | Use it for |
|---|---|
| [NumPy linalg](https://numpy.org/doc/stable/reference/routines.linalg.html) | the reference semantics being reproduced, including `dot` versus `matmul` on higher-rank operands |
| [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html) | normalization conventions and the output length of the real transforms |
| [SciPy linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html) | the host fallback for matrix functions such as `expm`, `logm`, `sqrtm` |
| [SciPy sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) | the answer for sparse problems, which have no device equivalent here |

## How to use these

- **Coverage questions are answered by the comparison table, not from memory.** Any
  list of unimplemented functions describes a single release, and the set grows.
- **A numerical difference is explained by the backend, not apologised for.** oneMKL
  and NumPy's own kernels agree to within roundoff, which is why comparisons must use
  `assert_allclose`. Cite the accuracy behaviour rather than calling it a bug.
- **Do not paraphrase a performance figure out of these pages.** They describe
  someone else's hardware; `SKILL.md` says how to measure, and the user's own number
  is the only one worth quoting.

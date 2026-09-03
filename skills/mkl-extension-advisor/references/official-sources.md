# Official sources for this skill

Where the claims in `SKILL.md` and the three per-surface references come from. Load
this when the question is what an installed build covers, what the activation API is
called now, or which channel serves a package — the things this skill deliberately
refuses to answer from memory.

## The extension packages

| Source | Use it for |
|---|---|
| [IntelPython/mkl_fft](https://github.com/IntelPython/mkl_fft) | the activation API, the SciPy backend, and the CLI that installs a persistent `.pth` patch — the mechanism behind a process that is already wired |
| [IntelPython/mkl_random](https://github.com/IntelPython/mkl_random) | **which parts of the NumPy random API are covered.** The `Generator`/`default_rng` question is answered here, and the answer decides whether this skill has anything to offer |
| [IntelPython/mkl_umath](https://github.com/IntelPython/mkl_umath) | the covered ufuncs and dtypes, and the source the dispatch thresholds are private `#define`s in |
| [IntelPython/mkl-service](https://github.com/IntelPython/mkl-service) | what the `mkl` module provides — the import the SciPy FFT backend needs at module level |
| [mkl-fft on PyPI](https://pypi.org/project/mkl-fft/) | released versions and wheel availability |
| [mkl-random on PyPI](https://pypi.org/project/mkl-random/) | the same for `mkl_random` |
| [mkl-umath on PyPI](https://pypi.org/project/mkl-umath/) | the same for `mkl_umath` |
| [mkl-service on PyPI](https://pypi.org/project/mkl-service/) | the same for `mkl-service`, including whether the platform is covered at all |

## The surfaces being patched

| Source | Use it for |
|---|---|
| [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html) | the unpatched semantics and normalization the patched path has to reproduce |
| [NumPy random sampling](https://numpy.org/doc/stable/reference/random/index.html) | the difference between the legacy `RandomState` API and the `Generator` API, which is the line `mkl_random` coverage falls on |
| [NumPy ufunc reference](https://numpy.org/doc/stable/reference/ufuncs.html) | what `types` reports and why it is the coverage probe rather than a documented list |
| [SciPy FFT](https://docs.scipy.org/doc/scipy/reference/fft.html) | the backend registration mechanism the `mkl_fft` SciPy interface plugs into |

## Environment

| Source | Use it for |
|---|---|
| [threadpoolctl](https://github.com/joblib/threadpoolctl) | what `threadpool_info()` reports and, more importantly, what it does not prove about which BLAS numpy links against |
| [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | the routine families behind the three extensions, and the accuracy modes that explain a numerical difference after patching |
| [NumPy build configuration](https://numpy.org/doc/stable/reference/generated/numpy.show_config.html) | reading the BLAS a numpy was built against, which is the authoritative answer the threadpool cannot give |

## How to use these

- **A fact with a probe attached stays true; a fact with a version attached
  expires.** Prefer `u.sin.types`, `np.fft.fft.__module__`, and `is_patched()` over
  anything read here. These pages explain what the probe means.
- **Coverage grows and docs lag source.** When a page and the installed build
  disagree, the build wins and the answer is the build's.
- **No page here supplies a speedup number for an answer.** Eligibility is static and
  answerable; benefit is a measurement on the user's own hardware.
- **Install channels are read, not remembered.** The commands in `SKILL.md` are a
  snapshot, and Intel's index is a partial mirror whose coverage is not uniform across
  this family.

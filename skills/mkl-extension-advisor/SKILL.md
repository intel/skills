---
name: mkl-extension-advisor
description: >-
  Deciding whether Intel's MKL extension packages apply to NumPy or SciPy code on
  Intel CPUs. Use when a user asks whether mkl_fft, mkl_random, or mkl_umath help
  their code, or points at a snippet, function, file, or codebase using np.fft,
  scipy.fft, np.random, or element-wise math ufuncs. Also use to check whether these
  extensions are already active in an environment, to fix an install so they and the
  SciPy FFT backend actually work, or to judge whether patching would change results
  and break exact-output tests. DO NOT use for GPU work, non-Intel CPUs, or
  mkl-service thread tuning.
license: Apache-2.0
compatibility: "Intel CPUs. Requires numpy; per-surface features require mkl_fft, mkl_random, or mkl_umath. The SciPy FFT backend additionally requires scipy and mkl-service. The environment probe uses threadpoolctl when available."
allowed-tools: Read, Grep, Glob, Bash, Edit
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# MKL extension advisor

## Purpose

Given NumPy or SciPy code, decides whether `mkl_fft`, `mkl_random`, or `mkl_umath`
apply to it, and presents verified changes with short reasons. A fit check, not a
profiler.

The judgement runs on two independent axes:

- **Install** — which numpy binary is present and what it links against.
- **Runtime** — whether call sites dispatch to oneMKL, which is set either by an
  activation call or by build-level pre-wiring.

They are genuinely independent. An MKL-backed BLAS says nothing about FFT: numpy's
FFT is bundled pocketfft with no BLAS linkage, so `threadpool_info()` can report
`mkl` while `np.fft.fft.__module__` is still `numpy.fft`.

The reverse inference is just as wrong. An `mkl` entry in the threadpool does not
prove numpy's BLAS is MKL — importing any `mkl_*` extension loads oneMKL into the
process, so a stock OpenBLAS numpy reports `['mkl', 'openblas', 'openmp']` after
`import mkl_fft`. For the BLAS question read `np.__config__.show()`, not the pool.

## When to Use This Skill

Use this skill when:

- The user asks whether the MKL extensions apply to a snippet, file, or codebase.
- Code calls `np.fft.*`, `scipy.fft.*`, `scipy.fftpack.*`, `np.random.*`, or
  element-wise math ufuncs on arrays.
- An environment has to be checked for whether these extensions are already active.
- An install has to be repaired so the extensions or the SciPy FFT backend work.
- The numerical effect of patching matters, because the code has exact-output tests
  or golden files.

Do **not** use this skill for GPU work, for non-Intel CPUs, for `mkl-service`
thread tuning, or as a source of speedup figures.

## Quick Start

Read the environment before advising anything:

```bash
python scripts/probe_env.py
```

It prints one JSON object: whether numpy is MKL-backed, whether each surface is
patched, how FFT got wired, which packages are importable, and any blocking gaps.
It is read-only — it never patches, never times, never writes. If it cannot be run,
each reference carries an equivalent probe snippet.

Then read the state, not the installer: `np.fft.fft.__module__` for whether FFT is
live, `is_patched()` for how it got that way, `u.sin.types` for umath coverage.

## Implementation Guide

1. **Take the input** — a snippet, function, file, or codebase — and scan every
   call site, categorizing each one:

   | Call site | Surface |
   |---|---|
   | `np.fft.*`, `scipy.fft.*`, `scipy.fftpack.*` | `mkl_fft` |
   | `np.random.*`, `RandomState`, `default_rng`/`Generator` | `mkl_random` |
   | element-wise math ufuncs on arrays | `mkl_umath` |

2. **Load only the references the code implicates.** One per distinct surface
   found, and none for a surface the code does not use:
   [`references/mkl_fft.md`](references/mkl_fft.md) for transforms, the SciPy
   backend, activation and proof;
   [`references/mkl_random.md`](references/mkl_random.md) for the covered API, the
   reproducibility hazard and parallel streams;
   [`references/mkl_umath.md`](references/mkl_umath.md) for the coverage probe, the
   eligibility gates and thread safety.

3. **Establish the install axis once** with `scripts/probe_env.py`, shared by all
   surfaces. Do this before judging any call site — the answer for a build-wired
   numpy is different from the answer for a stock one.

4. **Judge each call site against the loaded reference.** Does that extension cover
   this call and this usage? Decide fit, including an honest no. When shape or
   contiguity is not knowable from the code, do not guess and do not time code to
   discover it: ask for typical shapes, read the config, or instrument the call
   site. Failing that, answer conditionally on the gate — "worth patching only if
   the inner dimension is well above the transcendental threshold; at or below it
   the vector kernel never runs, so expect no speedup."

5. **Collect the applicable sites**: location, the call, the minimal change, and
   the one-line reason it helps *this* code. Do not add a redundant patch to a
   surface that is already wired.

6. **Confirm the whole input was scanned** before presenting. Do not stop at the
   first hit.

7. **Present the result.** With suggestions: grouped by location, each with a diff
   or code block and a short reason, plus the install commands if packages are
   missing, the proof step per surface, and the `mkl_random` confirmation gate. If
   the code has exact-output tests or golden files, flag the numerical effect —
   none of the three is a bitwise drop-in. With no suggestions: say so plainly and
   why (only uncovered ufuncs, only the `Generator` API, only sub-threshold arrays,
   only strided views, only FFT helpers, or nothing MKL-relevant). Do not invent a
   benefit.

8. **Apply only on confirmation.** Show suggestions first and edit after the user
   agrees. `mkl_random` needs an explicit yes every time, because it changes
   results.

9. **Repair the install when a package is missing.** conda, Intel channel first:

   ```bash
   conda install -c https://software.repos.intel.com/python/conda \
     -c conda-forge --override-channels \
     "blas=*=*_intelmkl" numpy scipy mkl_fft mkl_random mkl_umath mkl-service
   ```

   pip, with Intel's index as the *primary* index:

   ```bash
   pip install --index-url https://software.repos.intel.com/python/pypi \
     numpy scipy mkl_fft mkl_random mkl_umath mkl-service
   pip install threadpoolctl   # not mirrored on Intel's index
   ```

   Use `--index-url`, not `--extra-index-url`: Intel's index is a partial mirror
   and pip takes the highest version across indexes, so an extra index lets the
   stock PyPI numpy win — and which numpy wins decides what is build-wired.

## Performance

**Static analysis decides eligibility; measurement decides benefit.** Whether a
call site dispatches at all is answerable from the code plus the gates. How much
faster it runs is not — that depends on both builds, the thread count, and the CPU.
So this skill states no speedup figures of its own, and neither should an agent
using it.

Measurement is not illegitimate, it is the next step. Once eligibility is
established:

- Point the user at measurement on their own hardware; upstream ships benchmark
  suites precisely because the answer varies per machine.
- Compare a warmed-up run of the same code with the surface patched and unpatched,
  at the shapes the code actually uses.
- Never use a stopwatch to discover a fact that can be read — shape, dtype,
  contiguity. Instrumenting a call site to log the inner-loop length is a good
  answer, and so is asking.
- Do **not** claim a sub-threshold regression. It is unsupported and
  build-dependent. If the user supplies measured numbers, reason about those.

## Gotchas & Limitations

- **Proof of "FFT active" is `np.fft.fft.__module__`, not `is_patched()`.** On a
  build-wired numpy, FFT is live while the patch counter still reads zero. Read
  together, the pair tells you *how* it was wired.
- **An Intel-built numpy arrives build-wired**, dispatching FFT and umath to MKL
  before any activation call. That wiring lives in the numpy recipe, which rebinds
  the `numpy/fft` globals directly and never touches the patch counter. A
  conda-forge or stock-PyPI numpy leaves the extensions dormant.
- **The numpy build is not the only thing that wires a process.** `mkl_fft`'s CLI
  can install a persistent `.pth` patch that applies at every interpreter start,
  invisible to `pip list`; a `sitecustomize.py` variant does the same; and an
  earlier import in the same process may already have patched. Read runtime state
  — never infer it from the installer, the channel, or a version string.
- **`mkl-service` is not optional for the SciPy FFT backend.** It provides the
  `mkl` module that the backend imports at module level, and the guard means the
  backend disappears *silently* without it rather than raising. `scipy` is required
  for that same backend even if the code only touches `np.fft` today. Outside that,
  `mkl-service` is a thread-control API with no patch surface.
- **A BLAS selector constrains BLAS only**, never FFT wiring.
- **Call activation functions on the top-level package** (`import mkl_fft;
  mkl_fft.patch_numpy_fft()`). No private submodules, no invented names.
- **Reason only from the code the user gave you.** Do not claim to have found
  something in a document.
- **Dispatch thresholds may be cited, but they are private `#define`s, not API.**
  Withholding them produces wrong advice; quoting them from memory produces stale
  advice. Read them from the installed build and say which build, or express the
  answer as a gate the user can check.
- **Every fact stated should be one the environment can confirm.** These packages
  move: APIs get added, thresholds change, coverage grows, docs lag source,
  channels shift. Prefer "check `u.sin.types`" over a dtype list. A fact with a
  probe attached stays true; a fact with a version attached expires.
- If Intel's servers are unreachable, the extensions are on public PyPI and most
  are on conda-forge. Coverage is not uniform across the family or stable over
  time, so check per package rather than assuming parity — and any fallback install
  pairs with a stock numpy, so nothing is build-wired.

## References

| File | Load it when |
|---|---|
| [`scripts/probe_env.py`](scripts/probe_env.py) | you need the install and runtime axes for the current environment in one read-only call |
| [`references/mkl_fft.md`](references/mkl_fft.md) | the code touches `np.fft`, `scipy.fft`, or `scipy.fftpack` |
| [`references/mkl_random.md`](references/mkl_random.md) | the code touches `np.random`, `RandomState`, or `default_rng` |
| [`references/mkl_umath.md`](references/mkl_umath.md) | the code applies element-wise math ufuncs to arrays |
| [`references/official-sources.md`](references/official-sources.md) | you need the current install channels, the documented activation API, or the coverage of an installed release |

Two things here must never be answered from memory: **the coverage of the
installed build** (read it from `types`, `__module__`, and `is_patched()`) and
**the current install channels and package names**, which have already changed.

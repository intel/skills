# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the question is which
distributions the installed release implements, which generator produced a sequence,
or why the same seed gives different numbers than NumPy.

## dpnp — the implemented distributions

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point; the release this skill's guidance is written against |
| [Random sampling reference](https://intelpython.github.io/dpnp/reference/random.html) | **the authoritative list of implemented distributions and their signatures.** Check here before telling a user a distribution is missing |
| [NumPy comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) | per-keyword coverage, for the case a distribution exists but rejects an argument |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | source and release notes; where a newly added distribution appears first |

## The generators

| Source | Use it for |
|---|---|
| [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | the random number generator families oneMKL provides, and their seeding and stream semantics — the backend side of the reproducibility answer |
| [NumPy random sampling](https://numpy.org/doc/stable/reference/random/index.html) | NumPy's own generator, its default bit generator, and the legacy `RandomState` behaviour — the other half of "same seed, different numbers" |
| [NumPy legacy RandomState](https://numpy.org/doc/stable/reference/random/legacy.html) | the exact guarantee NumPy makes about sequence stability, which is what a user comparing the two libraries usually assumes applies to both |

## How to use these

- **Never explain a cross-library difference as a defect.** Two different generators
  produce two different streams from one seed. The oneMKL and NumPy pages together
  are the explanation, and they are worth citing so the user stops treating it as a
  bug to be fixed.
- **Coverage claims expire.** A list of supported or missing distributions is true of
  one release. Read it from the reference page or `dir(dpnp.random)`.
- **Do not import a generation throughput figure from these pages.** Random
  generation is bandwidth-bound and machine-specific; `SKILL.md` says how to measure
  it, and only the user's own number is citable.

# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when a statement in the skill
needs to be checked against upstream documentation, or when the answer depends on
API coverage that changes between releases.

## dpnp

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point; version this skill's guidance is written against |
| [Quick start guide](https://intelpython.github.io/dpnp/quick_start_guide.html) | install paths, supported platforms, driver prerequisites |
| [API reference](https://intelpython.github.io/dpnp/reference/index.html) | whether a function exists, and which keyword arguments it accepts |
| [NumPy comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) | **the authoritative coverage answer.** Check here before telling a user an API is missing — coverage is per parameter, not per function |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | source, release notes, open issues behind a `NotImplementedError` |
| [dpnp on PyPI](https://pypi.org/project/dpnp/) | released versions, wheel availability |

## dpctl — device and queue control

| Source | Use it for |
|---|---|
| [dpctl documentation](https://intelpython.github.io/dpctl/latest/index.html) | SYCL device model, queues, USM allocation |
| [dpctl API reference](https://intelpython.github.io/dpctl/latest/api_reference/dpctl/index.html) | exact signatures for device selection and context managers |
| [IntelPython/dpctl](https://github.com/IntelPython/dpctl) | source and issues |

## Distribution

| Source | Use it for |
|---|---|
| `https://software.repos.intel.com/python/conda` | the Intel conda channel used in Quick Start; browse it to confirm a package and version exist for the user's platform |

## How to use these

- **Coverage questions are answered by the comparison table, not from memory.**
  dpnp's NumPy coverage moves every release, so an answer that was right for one
  version is a guess for another.
- **Do not paraphrase a performance number out of these pages into an answer.**
  Numbers here describe someone else's hardware. `SKILL.md` says how to measure;
  the user's own measurement is the only citable number.
- **Prefer the version-matched page.** Ask the user for `dpnp.__version__` when a
  question turns on whether a specific API is implemented.

# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the answer turns on an
install channel, a package name, an environment variable, or whether an API is
covered in the user's release — the four things in this skill that have already
changed at least once and must not be answered from memory.

## Install and runtime packages

| Source | Use it for |
|---|---|
| [dpnp quick start guide](https://intelpython.github.io/dpnp/quick_start_guide.html) | **the current install paths, channel names, and driver prerequisites.** This page, not a remembered `conda install` line, is the source for step 2 |
| [dpnp on PyPI](https://pypi.org/project/dpnp/) | released versions and wheel availability per platform |
| [dpctl on PyPI](https://pypi.org/project/dpctl/) | the same for `dpctl`, which is the package a device query actually needs |
| [intel-cmplr-lib-rt on PyPI](https://pypi.org/project/intel-cmplr-lib-rt/) | the SYCL runtime alone, for the `OSError` on `libsycl` case |

## API coverage

| Source | Use it for |
|---|---|
| [NumPy comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) | **the authoritative coverage answer.** Coverage is per keyword argument, which is why a `TypeError` on an argument is not a contradiction of "the function exists" |
| [dpnp API reference](https://intelpython.github.io/dpnp/reference/index.html) | the signature actually implemented for the call that raised |
| [IntelPython/dpnp issues](https://github.com/IntelPython/dpnp/issues) | whether a `NotImplementedError` is known, and whether a fix has landed |

## Devices and device selection

| Source | Use it for |
|---|---|
| [dpctl documentation](https://intelpython.github.io/dpctl/latest/index.html) | the device model, filter selector syntax, and what an empty device list means |
| [dpctl API reference](https://intelpython.github.io/dpctl/latest/api_reference/dpctl/index.html) | exact signatures for `get_devices`, `SyclDevice`, and explicit queue construction |
| [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) | **the current device selection environment variables.** `ONEAPI_DEVICE_SELECTOR` replaced an older variable, and a page is the only way to know which one the installed runtime honours |

## How to use these

- **Report what a probe printed, not what it should print.** Each command in Quick
  Start produces a fact; these pages only explain it.
- **Install advice must come from the quick start page.** Channel URLs and package
  names in this repository's text are a snapshot, and a stale channel produces a
  confusing resolver error rather than a clear failure.
- **Nothing here is a source of speedup figures.** When the symptom is "slower than
  NumPy", the answer is a measurement on the user's machine, not a number from a page.

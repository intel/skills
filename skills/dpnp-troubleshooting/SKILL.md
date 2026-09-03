---
name: dpnp-troubleshooting
description: >-
  Diagnosing dpnp failures on Intel CPUs and GPUs. Use when dpnp raises
  NotImplementedError or an unexpected TypeError, when the import fails or a SYCL
  runtime library is missing, when no SYCL device is visible, when dpctl reports a
  device the user did not expect, or when dpnp code runs slower than the NumPy it
  replaced. Covers the fallback pattern for unimplemented APIs, install repair,
  forcing CPU execution, and the handoff to libraries that only accept NumPy
  arrays.
license: Apache-2.0
compatibility: "Requires dpnp and dpctl. Install commands assume conda or pip with the Intel channel or index."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp troubleshooting

## Purpose

Turns a `dpnp` failure into the next command to run. Covers the five things that
actually go wrong: an API that is not implemented, an import that cannot find the
SYCL runtime, no visible device, a device that is not the one expected, and code
that got slower instead of faster.

Prefer this skill over reading the traceback and guessing. Each symptom below has
a check that produces an answer, and most of them are one line.

## When to Use This Skill

Use this skill when:

- `dpnp` raises `NotImplementedError`, `AttributeError`, or a `TypeError` about a
  keyword argument.
- `import dpnp` fails, or fails on a missing `libsycl` shared object.
- No SYCL device is found, or `dpctl` lists a different device than expected.
- `dpnp` code is slower than the NumPy it replaced.
- Another library rejects a `dpnp` array.

Do **not** use this skill to plan a migration (`dpnp-quickstart`), to size a
workload against device memory (`dpnp-memory`), or as a source of speedup
figures.

## Quick Start

Three commands answer most questions before any code changes:

```bash
python -c "import dpnp; print(dpnp.__version__)"
python -c "import dpctl; print([d.filter_string for d in dpctl.get_devices()])"
python -c "import dpnp; print(dpnp.arange(4).sycl_device)"
```

Version, what is visible, and where an array actually lands. Report what they
print rather than what they were expected to print.

## Implementation Guide

1. **`NotImplementedError` or a rejected keyword.** `dpnp` implements a subset of
   NumPy, and coverage is per keyword argument as well as per function — a
   function that exists can still reject a signature. Do not gate on
   `hasattr(dpnp, "name")`; the attribute can be there and the call still fail.
   Guard the call instead:

   ```python
   import dpnp
   import numpy

   def safe_call(device_func, host_func, x):
       try:
           return device_func(x)
       except (NotImplementedError, AttributeError, TypeError):
           host = dpnp.asnumpy(x) if isinstance(x, dpnp.ndarray) else x
           return dpnp.array(host_func(host))

   unique = safe_call(dpnp.unique, numpy.unique, dpnp.array([1, 2, 2, 3]))
   ```

2. **Import failures.** `ImportError` on the module name means it is not
   installed; `OSError` on `libsycl.so` means the package is there and the SYCL
   runtime is not. Install from the Intel conda channel, or the runtime alone
   from pip:

   ```bash
   conda install -c https://software.repos.intel.com/python/conda \
     -c conda-forge --override-channels dpnp dpctl

   pip install intel-cmplr-lib-rt      # SYCL runtime only
   ```

3. **No device, or the wrong device.** `dpctl.get_devices()` returning an empty
   list means the driver stack is not visible to SYCL; `dpnp` then has only the
   host to fall back to. To pin execution while debugging, select the device
   explicitly at allocation, which is clearer than relying on process-wide state:

   ```python
   import dpctl
   import dpnp

   cpu = dpctl.SyclDevice("opencl:cpu:0")
   arr = dpnp.arange(1000, device=cpu)
   ```

   The same restriction from outside the process is
   `ONEAPI_DEVICE_SELECTOR=opencl:cpu` (it replaced the older
   `SYCL_DEVICE_FILTER`, which no longer has an effect on current runtimes).

4. **Slower than NumPy.** Three causes, in the order they occur:

   - The array is too small, and dispatch dominates. Below roughly a thousand
     elements NumPy is the right answer; `dpnp` earns its keep on large arrays.
   - The first call was timed. It includes compilation, so time the second:

     ```python
     import time
     import dpnp

     x = dpnp.random.randn(100000)
     dpnp.sin(x)                                  # warm up, discard
     start = time.perf_counter()
     dpnp.sin(x)
     print(f"{time.perf_counter() - start:.4f}s")
     ```

   - A conversion sits inside the loop. `dpnp.asnumpy()` copies device to host
     every call; hoist it above the loop, or keep the whole loop on the device.

5. **Another library rejects the array.** pandas, scikit-learn, PyTorch, and
   TensorFlow check for a NumPy array and refuse anything else. Convert once at
   the boundary with `dpnp.asnumpy()` — `dpnp-interop` has the per-library
   patterns.

## Performance

No measured numbers ship with this skill, and a fix here is not evidence of a
speedup. When a change is meant to make something faster, measure it:

- Warm up first, then time the steady state.
- Compare against the NumPy original on the same inputs and dtype.
- Time the whole pipeline, including conversions — a loop body that got faster
  while the surrounding transfers got more frequent is a net loss.

## Gotchas & Limitations

- **`hasattr` is not a coverage check.** The attribute can exist and the call
  still raise. `try`/`except` is the only reliable gate.
- **A fallback that converts inside a loop is its own bug.** Correct, and slower
  than never having moved to `dpnp`.
- **The default device is whatever is visible.** Code that runs on a GPU
  workstation lands on a CPU in CI without raising, so "it worked locally" says
  nothing about where it ran.
- **`intel-cmplr-lib-rt` fixes the runtime, not the driver.** A GPU that the
  kernel driver does not expose stays invisible whatever is installed in the
  environment.
- Not covered: driver installation, container device passthrough, and multi-GPU
  scheduling.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the current install channels, the API coverage of the installed release, or the device selection environment variables — all three change between releases and must not be answered from memory |

Two things here should never be answered from memory: **which install channel and
package names are current**, and **whether a given NumPy API is covered in the
user's release**. Both are documented upstream and both have already changed.

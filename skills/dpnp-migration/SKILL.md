---
name: dpnp-migration
description: >-
  Porting an existing NumPy or CuPy program to dpnp on Intel CPUs and GPUs. Use
  when deciding whether a codebase can run on dpnp at all, when a call raises
  NotImplementedError or AttributeError after the import was swapped, when the
  user asks whether dpnp supports a specific NumPy function or family, or when
  CuPy code has to move to Intel hardware. Covers probing the installed release
  for what it actually implements, the families that have no device counterpart,
  the fallback wrapper for the ones that do not, and where CuPy's device model
  differs from dpnp's.
license: Apache-2.0
compatibility: "Requires dpnp. The fallback path needs numpy. Device examples need a SYCL device; the probe itself runs anywhere dpnp imports."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp migration from NumPy and CuPy

## Purpose

Answers one question about an existing program: which of its array calls `dpnp`
implements, and what to do with the rest. Swapping `import numpy as np` for
`import dpnp as np` moves the calls it covers and raises on the calls it does
not, so a port is an inventory problem before it is a performance one.

The method here is to **probe the installed release rather than consult a
coverage list**. A list of supported functions is the single most perishable
claim about `dpnp`: it is accurate for the release someone wrote it against and
silently wrong afterwards, in both directions. `hasattr` is not.

## When to Use This Skill

Use this skill when:

- A NumPy or CuPy codebase has to run on Intel hardware and the question is
  whether it can.
- A call raises `NotImplementedError` or `AttributeError` after the import swap.
- The user asks whether `dpnp` supports a specific function or family.
- CuPy code needs the `dpnp` spelling of device selection or of a host copy.

Do **not** use this skill to diagnose a broken install or a missing SYCL runtime
(`dpnp-troubleshooting`), to hand arrays to pandas, scikit-learn, PyTorch, or
TensorFlow (`dpnp-interop`), to choose a device or manage buffers
(`dpnp-memory`), to decide whether the workload belongs on a device at all
(`dpnp-quickstart`), or to migrate a CUDA-based AI repository — model code,
kernels, and framework calls are `cuda-to-xpu-migration`, not this skill.

## Quick Start

```python
import dpnp

hasattr(dpnp, "linspace")            # constructor present in this release?
hasattr(dpnp.linalg, "eigh")         # submodule member present?
hasattr(dpnp.fft, "fftn")            # same question, FFT surface
```

Three lines against the release the user has installed settle more than any
table can. Everything below is how to act on the answers.

## Implementation Guide

1. **Inventory the surface the program actually uses.** Grep for the `np.`
   call sites and reduce them to a set of names; that set, not the whole NumPy
   API, is the scope of the port.

2. **Probe each name in the installed release.** Presence is one question and
   signature is another:

   ```python
   import dpnp

   wanted = ["sort", "argsort", "einsum", "interp", "unique"]
   missing = [name for name in wanted if not hasattr(dpnp, name)]

   import inspect
   inspect.signature(dpnp.sort)      # a present name can still lack a parameter
   ```

   A name that exists but rejects a keyword the program passes fails at runtime
   just as hard as an absent one, so read the signature for anything called with
   optional arguments.

3. **Expect three outcomes, and verify each against the installed release
   rather than this list.** The families are stable enough to plan with; the
   membership is not:

   | Outcome | Families that usually land here |
   |---|---|
   | Present | array construction, element-wise arithmetic and ufuncs, reductions, `linalg`, `fft`, basic and boolean indexing |
   | Present with a narrower signature | sorting, some random distributions, anything with a `kind=` or `method=` parameter |
   | Absent by design | string arrays, `datetime64`/`timedelta64`, structured and record arrays, polynomials, masked arrays |

   The last row is not a gap waiting to be filled. Those families are host data
   structures rather than numeric kernels, so a device implementation is not
   pending — plan to keep that code on NumPy.

4. **Wrap what is missing, once, at the call site.** The fallback converts to
   the host, runs NumPy there, and comes back:

   ```python
   import dpnp
   import numpy

   def unique_counts(array):
       """dpnp where it implements this, NumPy where it does not."""
       try:
           return dpnp.unique(array, return_counts=True)
       except (NotImplementedError, AttributeError, TypeError):
           values, counts = numpy.unique(dpnp.asnumpy(array), return_counts=True)
           return dpnp.array(values), dpnp.array(counts)
   ```

   `TypeError` belongs in that tuple: a narrower signature is how a partially
   implemented function refuses, and it is the outcome step 2 warns about.

5. **Know what the conversions cost.** `dpnp.array(host_array)` copies host to
   device and `dpnp.asnumpy(device_array)` copies device to host. `dpnp.asarray`
   avoids a copy only when its input already lives in USM memory reachable by
   the target queue — a NumPy array never does, so treat both directions as
   copies unless you have measured otherwise.

6. **From CuPy, expect the device model to differ more than the array API.**
   CuPy's `cupy.cuda.Device(0).use()` has no `dpnp` counterpart: there is no
   ambient current device to set. Placement is an argument at construction time:

   ```python
   import dpnp

   x = dpnp.zeros(1024, device="gpu")             # explicit at creation
   y = dpnp.zeros(1024, sycl_queue=x.sycl_queue)  # or inherit the queue
   ```

   `cupy.asnumpy` maps onto `dpnp.asnumpy`. For anything else CuPy-specific —
   `.get()`, memory pools, `RawKernel`, `cupyx.scipy` — probe before promising an
   equivalent; the pool and kernel APIs in particular have no `dpnp` analogue to
   translate into.

7. **Record what fell back.** A port that ends with four wrapped functions and a
   note saying which they are is finished. One that ends with a wrapper around
   every call has hidden its own status, and nothing will tell you later which
   calls were ever on the device.

## Performance

No measured numbers ship with this skill. What to measure once the port runs:

- The fallback rate on the hot path. Each fallback is two transfers plus a host
  computation, so a wrapped function called per iteration can cost more than the
  whole device stage saves.
- The end-to-end time against the unported original. A program that runs on the
  device but falls back inside its inner loop is the failure this skill exists to
  prevent, and only the whole-program timing shows it.
- Warm-up separately from steady state; first-call compilation is part of a
  port's measurements too (`dpnp-quickstart` covers the timing method).

## Gotchas & Limitations

- **A coverage list is a claim with a shelf life; `hasattr` is not.** Probe the
  installed release, and say which release an answer was checked against.
- **Presence does not imply the same signature.** The parameter the program
  passes is the thing to check, not the name.
- **`NotImplementedError` and `AttributeError` are different symptoms.** The
  first is a function that exists and declines; the second is a name that is not
  there at all. A fallback that catches only one of them leaves the other
  crashing.
- **A fallback inside a loop is correct code that loses the port.** Wrap the
  function, not the iteration.
- **CuPy's current-device idiom has no translation.** Do not offer a
  context-manager equivalent; pass `device=` or `sycl_queue=` instead.
- **The host-data families will not arrive.** Strings, datetimes, structured
  arrays, polynomials, and masked arrays are not scheduled work, and telling a
  user to wait for them is wrong advice.
- Not covered: CUDA kernel sources and `RawKernel`, `cupyx.scipy`, framework and
  model migration, and any judgement about whether the ported workload is large
  enough to belong on a device.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the API surface a specific dpnp release implements, the CuPy call whose equivalent is in question, or the array API standard the two are converging on |

One question here must never be answered from memory: **what the installed
release implements**. The probe in step 2 is cheap, and a remembered coverage
table is how this skill would tell a user that a function they need is missing
when it is present, or present when it is missing.

---
name: dpnp-memory
description: >-
  Device memory management for dpnp arrays on Intel CPUs and GPUs. Use when a dpnp
  script grows in memory until it fails, when a dataset does not fit in device
  memory, when an array turns out to be on a different device than expected, or
  when a loop allocates a new array on every iteration. Covers USM allocation,
  inspecting placement and queues with dpctl, reusing an output buffer, chunking a
  workload larger than the device, and the tools that report device memory use.
license: Apache-2.0
compatibility: "Requires dpnp and dpctl. Device memory reporting needs xpu-smi (data center GPUs) or intel_gpu_top (client GPUs)."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp memory and device placement

## Purpose

Answers where a `dpnp` array lives, how much room the device has, and how to keep
a long-running script from filling it. `dpnp` arrays are allocated in SYCL unified
shared memory on a device, not on the CPU heap the way NumPy arrays are, so the
questions that matter are different: which device, whose queue, and when the
allocation is released.

Prefer this skill over guessing from symptoms — a script that slows down over
hours, an out-of-memory error, or a GPU that turns out to have been a CPU all
along are all answered by reading state the runtime already exposes.

## When to Use This Skill

Use this skill when:

- Memory use climbs over the life of a script or a notebook session.
- A dataset is larger than the device and has to be processed in pieces.
- The user needs to confirm which device or queue an array is on.
- A tight loop allocates a new array per iteration.
- The user asks how to see device memory use from outside Python.

Do **not** use this skill for host-side NumPy memory questions, for file I/O
(that is `dpnp-io`), or to decide whether `dpnp` is worth using at all.

## Quick Start

```python
import dpnp

arr = dpnp.arange(1000)
print(arr.sycl_device)                                  # e.g. level_zero:gpu:0
print(arr.sycl_device.name)                             # human-readable name
print(arr.sycl_device.global_mem_size / 1e9, "GB")      # capacity, not free space
```

`global_mem_size` is the total the device reports. There is no `dpnp` API for
*free* memory — that comes from the tools in Gotchas.

## Implementation Guide

1. **Read the placement before changing anything.** Every array carries
   `sycl_device` and `sycl_queue`; `dpctl.get_devices()` lists what is visible.
   Filter strings are `backend:device_type:index`, so `level_zero:gpu:0` and
   `opencl:cpu:0` name specific devices. Level Zero is the lower-overhead backend
   for Intel GPUs.

   ```python
   import dpctl

   for device in dpctl.get_devices():
       print(device.filter_string, device.name)
   ```

2. **Target a device explicitly when the default is wrong.** `dpnp` picks a
   default device at import time using the SYCL default selector, which scores
   the visible devices — it is not "the first GPU". Pass `device=` or
   `sycl_queue=` rather than relying on it:

   ```python
   gpu = dpctl.SyclDevice("level_zero:gpu:0")
   arr = dpnp.arange(1000, device=gpu)

   queue = dpctl.SyclQueue(gpu)
   shared = dpnp.arange(1000, sycl_queue=queue)
   ```

3. **Reuse the output buffer in loops.** Universal functions take `out=`, which
   writes into an existing allocation instead of making one:

   ```python
   a = dpnp.arange(10000, dtype=dpnp.float64)
   b = dpnp.arange(10000, dtype=dpnp.float64)
   result = dpnp.empty(10000, dtype=dpnp.float64)   # allocate once

   for _ in range(1000):
       dpnp.add(a, b, out=result)                   # no new allocation
   ```

   Use `dpnp.empty()` rather than `dpnp.zeros()` when the initial values are
   overwritten anyway, and pre-allocate the output of `dpnp.matmul(A, B, out=C)`
   the same way.

4. **Keep conversions out of the loop body.** `dpnp.asnumpy()` copies device to
   host and `dpnp.array()` copies host to device. Calling a NumPy function on a
   `dpnp` array, or mixing the two in one expression, does the same thing
   implicitly. Hoist the conversion above the loop.

5. **Chunk a workload that does not fit.** Size each chunk so the input and the
   intermediates together stay under the device capacity — roughly half to two
   thirds of it is a workable starting point — then release the arrays before the
   next iteration:

   ```python
   import gc
   import numpy
   import dpnp

   chunk = 10_000_000
   for start in range(0, 100_000_000, chunk):
       host = numpy.load(f"data_chunk_{start}.npy")
       device_array = dpnp.array(host)
       total = dpnp.sum(device_array ** 2)
       numpy.save(f"result_{start}.npy", dpnp.asnumpy(total))
       del device_array, total, host
       gc.collect()
   ```

6. **Watch the device while it runs** rather than reasoning about it afterwards:
   `xpu-smi dump -m 1` on data center GPUs, `intel_gpu_top` on client GPUs,
   `clinfo` for OpenCL limits, `ze_info` for Level Zero. Steadily climbing memory
   is the signature of a leak.

## Performance

No measured numbers ship with this skill. Whether pre-allocation or chunking is
worth it depends on array size, device, and driver, so measure the specific case:

- Pre-allocation matters most for small arrays in loops with many iterations,
  where allocation is a large share of the work. For large arrays the allocation
  cost is amortized over the compute.
- Chunking trades memory for repeated allocation and transfer. If disk I/O
  dominates, that trade is invisible; if compute dominates, it is not.
- Warm up before timing anything: the first call on a new shape includes
  compilation.

## Gotchas & Limitations

- **`del` does not free device memory immediately.** It drops a reference. The
  allocation goes back when the object is collected, and in a notebook an output
  cell can hold the last reference. `gc.collect()` encourages collection; it does
  not guarantee the allocator returns the memory at that instant.
- **There is no `memory_summary()`.** No device memory accounting API is exposed at
  the time of writing — `global_mem_size` is capacity, and free memory comes from
  `xpu-smi` or `intel_gpu_top`. `SYCL_UR_TRACE=1` traces allocations (verbose; it
  replaced `SYCL_PI_TRACE`).
- **Integrated and discrete devices are not comparable.** An integrated GPU
  shares host RAM; a discrete one has its own. The same chunk size can fit on one
  and not the other.
- **A leak looks like a slowdown first.** Device memory fills, then the run
  either falls back or fails. If a script degrades over hours, check memory
  before profiling compute.
- Not covered: multi-process or multi-device sharing of one allocation, and USM
  allocation kinds (`device`, `host`, `shared`) beyond the default.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the current dpctl device or queue API, the USM allocation kinds, or which release added a property — memory APIs move between releases and must not be answered from memory |

Two things here should never be answered from memory: **which dpctl properties
exist in the installed version**, and **how much memory the device actually has
free**. The first is in the documentation, the second only in the running system.

---
name: dpnp-io
description: >-
  Reading and writing files from dpnp code on Intel CPUs and GPUs. Use when the
  user needs to load an array into dpnp or save a dpnp result — .npy, .npz, HDF5
  via h5py, Zarr, CSV or plain text — when a file is larger than device memory and
  has to be read in chunks, or when they ask why dpnp has no save function of its
  own. Covers the NumPy conversion round trip, chunked and incremental patterns,
  and choosing a format by dataset size.
license: Apache-2.0
compatibility: "Requires dpnp and NumPy. HDF5 needs h5py, Zarr needs zarr, CSV parsing examples use pandas."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp file I/O

## Purpose

Gets data in and out of `dpnp` arrays. `dpnp` has no native binary file I/O:
every format goes through NumPy, with `dpnp.array()` on the way in and
`dpnp.asnumpy()` on the way out. This skill is that round trip, plus the chunked
variants for data larger than memory and the format choice by size.

Prefer it over reaching for a `dpnp.save()` that does not exist, and over loading
a file whole when the device cannot hold it.

## When to Use This Skill

Use this skill when:

- An array has to be loaded into `dpnp` from a file, or a result written out.
- A file is larger than host or device memory and must be streamed in pieces.
- The user is choosing between `.npy`, `.npz`, HDF5, Zarr, and CSV.
- The user asks why `dpnp` will not write their format.

Do **not** use this skill to decide device placement or chunk sizing against
device capacity — that is `dpnp-memory` — and do not expect it to make an
I/O-bound job faster: if reading dominates, moving the compute to a device
changes nothing.

## Quick Start

```python
import numpy
import dpnp

arr = dpnp.array(numpy.load("data.npy"))          # host file -> device array
result = dpnp.fft.fft2(arr) + dpnp.mean(arr)      # compute on the device
numpy.save("output.npy", dpnp.asnumpy(result))    # device array -> host file
```

The whole skill is that shape: NumPy load → `dpnp.array()` → compute →
`dpnp.asnumpy()` → NumPy save.

## Implementation Guide

1. **`.npy` and `.npz`.** One array or several, with the archive closed after
   reading:

   ```python
   with numpy.load("data.npz") as npz:
       x = dpnp.array(npz["x"])
       y = dpnp.array(npz["y"])

   numpy.savez("output.npz", x=dpnp.asnumpy(x), y=dpnp.asnumpy(y))
   ```

   Each conversion needs a full host copy of the array as well as the device
   copy, so a 4 GB array wants 4 GB of free RAM during the call.

2. **Chunked reads for a file larger than RAM.** Memory-map the source, write
   each processed chunk straight into a pre-allocated output slice rather than
   appending to a list:

   ```python
   data = numpy.load("large.npy", mmap_mode="r")
   final = numpy.empty(len(data), dtype=numpy.float64)
   chunk = 25_000_000

   for start in range(0, len(data), chunk):
       host = data[start:start + chunk]
       processed = dpnp.sqrt(dpnp.array(host)) * 2.0
       final[start:start + len(host)] = dpnp.asnumpy(processed)

   numpy.save("output.npy", final)
   ```

   A chunk of roughly a tenth to a fifth of free RAM is a workable start.

3. **HDF5 through h5py.** h5py only speaks NumPy, so the same conversion applies,
   and datasets can be written incrementally when the result is too large to
   hold:

   ```python
   import h5py

   with h5py.File("output.h5", "w") as handle:
       dset = handle.create_dataset("result", shape=(50_000_000,), dtype="float64")
       for start in range(0, 50_000_000, 5_000_000):
           dset[start:start + 5_000_000] = dpnp.asnumpy(compute_chunk(start))
   ```

4. **Zarr for very large or remote arrays.** Chunked, compressed, and reachable
   on object storage through fsspec; read and write slice by slice:

   ```python
   import zarr

   store = zarr.open("output.zarr", mode="w", shape=(10_000_000,),
                     chunks=(500_000,), dtype="float32")
   for start in range(0, 10_000_000, 500_000):
       store[start:start + 500_000] = dpnp.asnumpy(compute_chunk(start))
   ```

5. **Text and CSV.** `dpnp.loadtxt()` returns a `dpnp` array directly (it
   delegates to `numpy.loadtxt` internally, and does not support structured
   dtypes). Anything with headers, strings, or missing values goes through
   `numpy.loadtxt`/`numpy.genfromtxt` or pandas first:

   ```python
   import pandas

   frame = pandas.read_csv("data.csv")
   arr = dpnp.array(frame.values)
   numpy.savetxt("output.csv", dpnp.asnumpy(arr), delimiter=",")
   ```

6. **Pick the format by size.** `.npy`/`.npz` below about a gigabyte, HDF5 for
   multi-dataset files in the gigabyte range, Zarr above that or when the data
   lives in cloud storage, CSV only for small human-readable exports.

## Performance

No measured numbers ship with this skill. What to measure, and in which order:

- Time the I/O and the compute separately first. If reading dominates, no device
  will help and the conversion cost is irrelevant either way.
- Count conversions, not bytes. One conversion at each end of a batch of work is
  the pattern; one per iteration of a loop is the anti-pattern, and it is the
  usual reason a rewritten pipeline is no faster.
- Chunking trades peak memory against more conversions. Compare the two on the
  real file rather than assuming a ratio.
- CSV parsing is CPU-bound and dominates everything around it. Convert once to
  `.npy` or HDF5 if the same file is read repeatedly.

## Gotchas & Limitations

- **There is no `dpnp.save()` for binary formats.** `dpnp.loadtxt()` exists;
  `.npy`, HDF5, and Zarr all go through NumPy. Code that calls a `dpnp` save
  function fails at the call, not at review.
- **Conversion doubles peak memory.** Host copy plus device copy, briefly, for
  every `dpnp.array()` and `dpnp.asnumpy()`.
- **Accumulating chunks in a list defeats chunking.** The whole point is that the
  full array never exists in memory; a pre-allocated output or an incremental
  dataset write is what preserves that.
- **`mmap_mode="r"` is a NumPy facility, not a device one.** The mapped pages are
  host memory; each chunk still gets copied to the device.
- Not covered: parallel or multi-process writes, Arrow and Parquet, and anything
  about which device the array lands on — see `dpnp-memory` for that.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need to confirm what dpnp implements for a given release — whether a `loadtxt`-style entry point exists, or which NumPy I/O helpers have a dpnp counterpart — or the current h5py or Zarr chunking API |

Two questions here should not be answered from memory: **which I/O entry points
the installed `dpnp` actually has** (the list has grown between releases) and
**the current chunking API of h5py and Zarr**, both of which are documented
upstream and change on their own schedule.

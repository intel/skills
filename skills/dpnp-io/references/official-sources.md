# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the skill says a format is
not supported natively, or when the read path has to be checked against the library
that actually owns it — `dpnp` has no file format of its own, so most of the
authority here is NumPy's, h5py's, Zarr's, or pandas'.

## dpnp — what exists on the device side

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point; the release this skill's guidance is written against |
| [API reference](https://intelpython.github.io/dpnp/reference/index.html) | **whether an I/O-shaped function exists at all**, and whether it delegates to NumPy. Check here before telling a user `dpnp` can read a format |
| [NumPy comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) | per-parameter coverage of the text and array-creation functions used on the way in |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | source and issues; the place a `NotImplementedError` on a loader is explained |

## The formats themselves

| Source | Use it for |
|---|---|
| [NumPy input and output](https://numpy.org/doc/stable/reference/routines.io.html) | `save`, `savez`, `savez_compressed`, `load`, `loadtxt`, `savetxt` semantics and their keyword arguments |
| [NumPy memory-mapped files](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) | what `mmap_mode="r"` gives and what it does not: the chunked read in `SKILL.md` depends on the file staying on disk |
| [h5py documentation](https://docs.h5py.org/en/stable/) | dataset creation, `maxshape` and resizing, chunking and compression for the incremental write |
| [Zarr documentation](https://zarr.readthedocs.io/en/stable/) | chunked array storage and the store backends, for the case a dataset does not fit one file |
| [pandas IO tools](https://pandas.pydata.org/docs/user_guide/io.html) | CSV and Parquet reading before the values reach a device array |

## How to use these

- **The library that owns the format owns the answer.** A question about chunk
  shapes is h5py's or Zarr's; `dpnp` only receives the resulting host array.
- **Check delegation before promising it.** Where `dpnp` exposes a NumPy I/O name,
  it may forward to NumPy and inherit its restrictions rather than implement its
  own. That is a per-release fact, readable from the API reference.
- **Do not turn a documented format limit into a performance claim.** File size
  thresholds in `SKILL.md` are guidance about where formats become awkward, not
  measurements; the user's own timing is the only citable number.

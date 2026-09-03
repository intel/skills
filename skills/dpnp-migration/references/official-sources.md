# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the question is what a
particular `dpnp` release implements, or what a CuPy call does before you look for
its `dpnp` equivalent. Both are questions where a remembered answer is the failure
mode this skill is built to avoid.

## dpnp — what the installed release implements

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point, and the release the guidance is written against |
| [dpnp API reference](https://intelpython.github.io/dpnp/reference/index.html) | the authoritative per-function list for that release — the thing a coverage table in a skill can only approximate |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | release notes and issues; where a newly implemented function appears first, and where an absent one is either tracked or explicitly out of scope |
| [dpctl documentation](https://intelpython.github.io/dpctl/latest/index.html) | `device=` and `sycl_queue=`, and the device objects a placement argument accepts |

## The API on the other side of the port

| Source | Use it for |
|---|---|
| [NumPy API reference](https://numpy.org/doc/stable/reference/index.html) | the signature the program is currently calling, including the optional parameters a narrower `dpnp` signature may not take |
| [CuPy documentation](https://docs.cupy.dev/en/stable/) | what a CuPy call guarantees before you look for an equivalent — device contexts, memory pools, and `.get()` in particular |
| [CuPy comparison table](https://docs.cupy.dev/en/stable/reference/comparison.html) | CuPy's own NumPy coverage, which is the right reference for what the source program relied on |
| [Python array API standard](https://data-apis.org/array-api/latest/) | the common surface both libraries are converging on, which is the portable subset a port should aim at |

## How to use these

- **The API reference outranks any list in a skill, including the one in
  `SKILL.md`.** That table names families so a port can be planned; membership is
  read from the reference for the installed release.
- **Check the release, not the library.** "dpnp does not implement this" is only
  ever true of a version. Say which one was checked.
- **Read CuPy's page before claiming an equivalent exists.** The device model and
  the memory pool are where the two libraries genuinely differ, and an invented
  equivalent is worse than saying there is none.
- **Nothing here supplies performance numbers**, and none should be imported from
  these pages into `perf/`. A number that did not come from a run on the hardware
  in question does not belong in this repository.

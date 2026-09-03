# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when a statement in the skill
needs checking against upstream documentation, or when the answer depends on device
capacity, allocation semantics, or tooling that differs between platforms and
releases.

## dpnp — array allocation and device attributes

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point; the release this skill's guidance is written against |
| [API reference](https://intelpython.github.io/dpnp/reference/index.html) | whether an array constructor accepts `device=` or `sycl_queue=`, and which array attributes exist |
| [NumPy comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) | whether an `out=` variant exists for the function being reused as a buffer sink |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | source and issues behind an allocation failure |

## dpctl — the USM and device model

| Source | Use it for |
|---|---|
| [dpctl documentation](https://intelpython.github.io/dpctl/latest/index.html) | the USM allocation kinds, queues, and what a SYCL device object reports |
| [dpctl API reference](https://intelpython.github.io/dpctl/latest/api_reference/dpctl/index.html) | **the authoritative property names.** `global_mem_size`, `max_mem_alloc_size`, and the rest are read from here, not from memory — a wrong attribute name raises rather than reporting the wrong number, but a *missing* one leads to guessing |
| [IntelPython/dpctl](https://github.com/IntelPython/dpctl) | source, and the release notes when a property changes type or units |

## Measuring what the device is actually holding

| Source | Use it for |
|---|---|
| [Intel XPU Manager (`xpu-smi`)](https://github.com/intel/xpumanager) | installing and reading `xpu-smi` on data center GPUs; the metric ids behind `dump -m` |
| [Level Zero](https://github.com/oneapi-src/level-zero) | what the driver exposes about memory, when `xpu-smi` reports nothing |

`intel_gpu_top` ships with the `igt-gpu-tools` package on client systems and is the
counterpart to `xpu-smi` there. It is packaged by the distribution rather than by
Intel, so install it from the distribution's repository.

## How to use these

- **Capacity is a property of the installed device, never a documented constant.**
  Any number in an answer must come from `global_mem_size` on the user's machine.
  The documentation says which attribute to read; it does not say what it returns.
- **Allocation semantics are dpctl's, not dpnp's.** When the question is where
  memory came from or who owns it, the dpctl page is the authority and the dpnp page
  is a consumer of it.
- **Do not import a number from these pages into an answer.** Memory headroom,
  transfer cost, and allocation limits are all machine-specific; `SKILL.md` says how
  to observe them.

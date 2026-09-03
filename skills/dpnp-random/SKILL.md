---
name: dpnp-random
description: >-
  Random number generation with dpnp on Intel CPUs and GPUs, backed by oneMKL. Use
  when NumPy random calls move to dpnp, when a seeded dpnp run does not reproduce
  a NumPy sequence, when a distribution turns out not to be implemented, when
  results have to be reproducible across machines, or when random data feeds a
  training or augmentation loop. Covers the supported distributions, what seeding
  does and does not guarantee, the host fallback, and where to generate data so it
  does not bounce between host and device.
license: Apache-2.0
compatibility: "Requires dpnp with its oneMKL backend. Fallback examples use NumPy."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp random number generation

## Purpose

Generates random data with `dpnp.random`, which is backed by oneMKL on Intel CPUs
and GPUs and mirrors the NumPy API for the distributions it implements. Covers
what is implemented, what seeding actually guarantees, how to fall back to NumPy
for a missing distribution, and how to keep generation from becoming a stream of
host-device copies.

The reproducibility part is the reason this skill exists: the API looks like
NumPy's and the numbers are different, which is correct behaviour and reliably
surprises people.

## When to Use This Skill

Use this skill when:

- NumPy random calls are being moved to `dpnp`.
- A seeded `dpnp` run does not reproduce a seeded NumPy run.
- A distribution raises `NotImplementedError` or is missing.
- Results must be reproducible across machines or devices.
- Random data feeds a training loop, dropout, augmentation, or an initializer.

Do **not** use this skill when the arrays are small — NumPy is the better answer
there — and do not use it to claim a generation speedup: there are no measured
numbers here.

## Quick Start

```python
import dpnp

dpnp.random.seed(42)
x = dpnp.random.randn(1000, 1000)             # standard normal
y = dpnp.random.uniform(0, 1, size=10000)     # uniform [0, 1)
z = dpnp.random.randint(0, 100, size=500)     # integers [0, 100)
```

Seeding twice with the same value on the same device reproduces the same
sequence. It does **not** reproduce NumPy's sequence — see the Guide.

## Implementation Guide

1. **Use the NumPy spelling for what is implemented.** `rand`, `randn`,
   `random`, `uniform`, `normal`, `randint`, `choice`, `shuffle`, and the common
   univariate distributions — exponential, poisson, binomial, geometric, gamma,
   beta — keep their NumPy signatures. Confirm the specific one against the
   installed release rather than a remembered list.

2. **Set the seed once, at the top.** Reseeding inside a loop resets generator
   state on every iteration and produces neither speed nor determinism:

   ```python
   dpnp.random.seed(42)
   noise = dpnp.random.randn(1000, 256, 256)      # one call, all iterations
   for index in range(1000):
       image = clean + noise[index]
   ```

3. **Do not expect NumPy's numbers.** `dpnp.random` and `numpy.random` use
   different generators — oneMKL's on one side, NumPy's PCG64 on the other — so
   the same seed gives different sequences. This is expected, not a bug, and it
   means a reproducibility chain must not mix the two:

   ```python
   import numpy

   numpy.random.seed(42)
   dpnp.random.seed(42)
   # numpy.random.randn(5) and dpnp.random.randn(5) do not match, by design
   ```

4. **Fall back on the host for a missing distribution.** Generate with NumPy,
   then move the batch across once — the cost is the transfer, so make the batch
   large:

   ```python
   host = numpy.random.beta(a=2.0, b=5.0, size=100_000)
   device_array = dpnp.array(host)
   result = dpnp.mean(device_array ** 2)
   ```

   Distributions commonly missing include the multivariate ones — `dirichlet`,
   `multivariate_normal`, `multinomial` — and several of the long tail such as
   `chisquare`, `triangular`, `vonmises`, `wald`, `zipf`. Check before promising
   one.

5. **Persist the draw when a result has to be reproducible elsewhere.** Seeding
   guarantees a sequence on the same device and release; a different device type,
   or a different version, may produce a different one. When an experiment has to
   be replayed exactly, save the numbers rather than the seed — through NumPy,
   because `dpnp` has no binary writer of its own:

   ```python
   dpnp.random.seed(42)
   numpy.save("random_state.npy", dpnp.asnumpy(dpnp.random.randn(1_000_000)))
   later = dpnp.array(numpy.load("random_state.npy"))
   ```

6. **Generate where the data is consumed.** Random values that feed device
   compute should be drawn on the device; values that immediately go back to the
   host should be drawn with NumPy. Offsets for a crop, a dropout mask, or an
   initializer belong on the device:

   ```python
   def dropout(x, p=0.5, training=True):
       if not training:
           return x
       mask = dpnp.random.rand(*x.shape) > p
       return x * mask / (1 - p)
   ```

## Performance

No measured numbers ship with this skill. Generation is memory-bandwidth bound
rather than compute bound, which shapes what is worth measuring:

- There is a size below which kernel launch overhead dominates and NumPy wins.
  It is in the thousands of elements, not the millions; measure the real shapes.
- The first call pays for plan creation, and later calls of the same size reuse
  it. Warm up before timing, and never time a reseeded loop.
- One large draw beats many small ones, because the fixed cost is paid once.
- A host fallback costs a transfer per batch. Batch size, not distribution, is
  what makes that acceptable.

## Gotchas & Limitations

- **Same seed, different numbers from NumPy.** Different generator, by design. A
  test that compares sequences across the two libraries is testing the wrong
  thing; compare distributions or statistics instead.
- **Reseeding in a loop is the classic mistake.** It is slower and it does not
  make anything more deterministic.
- **Cross-device reproducibility is not guaranteed.** Same seed on the same device
  reproduces; CPU versus GPU may not, and neither may two `dpnp` releases.
- **`dpnp` has no binary save.** Persisting a draw goes through
  `numpy.save(dpnp.asnumpy(...))`; see `dpnp-io`.
- **Coverage claims expire.** Any list of supported or unsupported distributions
  describes one release. Check `dir(dpnp.random)` and the documentation.
- Not covered: parallel independent streams, counter-based generator state, and
  the `Generator`/`default_rng` object API.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the distributions a specific dpnp release implements, the oneMKL generator behind them, or NumPy's own generator semantics to explain a difference |

Two questions here should not be answered from memory: **which distributions the
installed release implements** and **which generator oneMKL uses for a given
call**, which is what makes a cross-library difference explainable instead of
suspicious.

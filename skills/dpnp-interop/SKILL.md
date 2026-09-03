---
name: dpnp-interop
description: >-
  Passing dpnp arrays to and from other Python libraries on Intel CPUs and GPUs.
  Use when dpnp numeric work has to feed pandas, scikit-learn, PyTorch, or
  TensorFlow, when one of those libraries raises a type error on a dpnp array, when
  a pipeline mixes device math with host-only libraries, or when the user asks
  where in a pipeline the conversion belongs. Covers the boundary conversion
  pattern per library, the Intel extensions that accelerate the host side, and why
  a conversion inside a loop erases the benefit.
license: Apache-2.0
compatibility: "Requires dpnp. Library examples need pandas, scikit-learn, PyTorch, or TensorFlow as applicable."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# dpnp interoperability

## Purpose

Connects `dpnp` to the libraries around it. None of pandas, scikit-learn,
PyTorch, or TensorFlow accepts a `dpnp` array: they check for a NumPy array, so
every handoff is a `dpnp.asnumpy()` on the way out and a `dpnp.array()` on the way
back. This skill is where that conversion goes, per library, and what it costs.

Prefer it when `dpnp` is one stage of a longer pipeline. The failure it prevents
is not a crash — it is a pipeline that converts on every iteration and ends up
slower than the NumPy version it replaced.

## When to Use This Skill

Use this skill when:

- A `dpnp` result has to reach pandas, scikit-learn, PyTorch, or TensorFlow.
- One of those libraries raises a type error on a `dpnp` array.
- A pipeline alternates between device math and host-only libraries.
- The user asks where the conversion belongs.

Do **not** use this skill for file formats (`dpnp-io`), for device placement
(`dpnp-memory`), or to decide whether the numeric stage belongs on a device at
all (`dpnp-quickstart`).

## Quick Start

```python
import dpnp

x = dpnp.random.randn(10000, 100)      # device
gram = dpnp.dot(x, x.T)                # device
host_gram = dpnp.asnumpy(gram)         # one conversion, at the boundary
```

Do the arithmetic first, convert once, then call the host library. The rule is
the whole skill; the sections below are the per-library spelling of it.

## Implementation Guide

1. **pandas.** Frames hold NumPy arrays, so convert both ways explicitly:

   ```python
   import pandas

   frame = pandas.DataFrame(dpnp.asnumpy(x), columns=list("abcde"))
   values = dpnp.array(frame.values)
   column = dpnp.array(frame["a"].values)
   ```

2. **scikit-learn.** `fit` and `predict` take host arrays; convert the features
   and the target once before training:

   ```python
   from sklearn.linear_model import LinearRegression

   features = dpnp.asnumpy(x)
   target = dpnp.asnumpy(y)
   model = LinearRegression().fit(features, target)
   predictions = dpnp.array(model.predict(features))
   ```

   The host side of this has its own Intel acceleration — the scikit-learn
   extension patches estimators in place:

   ```python
   from sklearnex import patch_sklearn

   patch_sklearn()
   ```

3. **PyTorch.** Go through NumPy in both directions, and bring a device tensor to
   the host first:

   ```python
   import torch

   tensor = torch.from_numpy(dpnp.asnumpy(x))
   back = dpnp.array(tensor.cpu().numpy())
   ```

   PyTorch has its own Intel GPU path: with a recent build, or with Intel
   Extension for PyTorch on older ones, tensors move with `.to("xpu")` and stay
   in the framework rather than passing through `dpnp` at all. When the whole
   pipeline is a model, that is the better route — this skill is for the case
   where array math and a model each own part of it.

4. **TensorFlow.** Same shape, through `tf.constant` and `.numpy()`:

   ```python
   import tensorflow as tf

   x_tf = tf.constant(dpnp.asnumpy(x))
   back = dpnp.array(x_tf.numpy())
   ```

5. **Put the conversions at the ends of a mixed pipeline**, not between stages:

   ```python
   frame = pandas.read_csv("data.csv")                     # host
   features = dpnp.array(frame[["f1", "f2", "f3"]].values) # -> device
   normalized = (features - dpnp.mean(features, axis=0)) / dpnp.std(features, axis=0)
   inputs = torch.from_numpy(dpnp.asnumpy(normalized))     # -> host, once
   ```

6. **Check the boundary when a library refuses the array.** The symptom is a type
   error naming `ndarray`, and it means the library ran an `isinstance` check.
   `dpnp.asnumpy()` at that call site is the fix; a wrapper that converts on every
   call is not.

## Performance

No measured numbers ship with this skill. What to measure when a handoff is on
the hot path:

- Count conversions per unit of work. One at each boundary is the target; one per
  loop iteration is the anti-pattern, and it is usually the reason a converted
  pipeline is no faster.
- Time the whole pipeline, not the numeric stage. A faster `dpnp` stage
  surrounded by more transfers can be a net loss.
- Compare against the all-NumPy original. If the host library dominates the
  runtime, the numeric stage is not where the time is.
- The Intel extensions for scikit-learn and PyTorch accelerate the host and
  framework side respectively; they do not remove the conversion.

## Gotchas & Limitations

- **No library here takes a `dpnp` array directly.** Treat the compatibility
  question as settled: convert, do not probe for support.
- **A conversion in a loop is the common failure.** It is correct code, and it
  can be slower than never having used a device.
- **A CUDA tensor needs `.cpu()` first.** `torch.Tensor.numpy()` on a device
  tensor raises; the host copy is not optional.
- **`asnumpy` copies.** It is not a view, and peak memory holds both copies
  during the call.
- **The Intel extensions are separate packages** with their own release cadence;
  whether `patch_sklearn` or an explicit extension import is needed depends on
  the installed versions, so check rather than assume.
- Not covered: zero-copy exchange protocols such as DLPack or the array API
  interchange, and any library not named above.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the current interoperability surface of dpnp, whether a library has gained direct support, or the install and activation steps for the Intel extensions for scikit-learn and PyTorch |

Two questions here must not be answered from memory: **whether a library has
gained direct support for device arrays** (the interchange protocols are moving,
and a claim that it has not can go stale) and **how the Intel extensions are
activated in the installed version**, which has changed more than once.

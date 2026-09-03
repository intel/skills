# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the question is whether a
library has gained direct support for device arrays, or how an Intel extension is
activated in the version the user has installed. Both have moved, and both are the
kind of claim that goes stale silently.

## dpnp — the conversion surface

| Source | Use it for |
|---|---|
| [dpnp documentation](https://intelpython.github.io/dpnp/) | entry point; the release this skill's guidance is written against |
| [dpnp API reference](https://intelpython.github.io/dpnp/reference/index.html) | `asnumpy` and the array constructors, and whether an interchange protocol has appeared |
| [IntelPython/dpnp](https://github.com/IntelPython/dpnp) | source and release notes; where zero-copy interchange would land first |
| [Python array API standard](https://data-apis.org/array-api/latest/) | the interchange the ecosystem is converging on, which is what would eventually make a conversion unnecessary |

## The libraries on the other side of the boundary

| Source | Use it for |
|---|---|
| [pandas IO tools](https://pandas.pydata.org/docs/user_guide/io.html) | how values enter and leave a frame, and what `.values` returns |
| [scikit-learn API](https://scikit-learn.org/stable/api/index.html) | the input types estimators accept, which is what the type error is about |
| [PyTorch documentation](https://pytorch.org/docs/stable/index.html) | `from_numpy`, `.numpy()`, and why a device tensor needs `.cpu()` first |
| [TensorFlow API](https://www.tensorflow.org/api_docs/python/tf) | `tf.constant` and `.numpy()` at the boundary |

## The Intel extensions for the host side

| Source | Use it for |
|---|---|
| [Intel Extension for Scikit-learn](https://github.com/uxlfoundation/scikit-learn-intelex) | **how patching is activated in the installed version.** The spelling has changed more than once, so read it here rather than trusting the `patch_sklearn()` line in `SKILL.md` |
| [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) | whether the installed PyTorch needs the extension at all for Intel GPU support, and which release moved that support upstream |
| [Intel Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) | the same question for TensorFlow |

## How to use these

- **"No library accepts a dpnp array" is a claim with a shelf life.** It is correct
  for the libraries listed at the time of writing; the array API and interchange work
  is exactly what would change it. Check before repeating it as permanent.
- **Extension activation is read from the extension's own page.** A remembered import
  line is the most common way this skill would give wrong advice.
- **The extensions accelerate the host side; they do not remove the conversion.** Do
  not present one as an alternative to the boundary pattern.

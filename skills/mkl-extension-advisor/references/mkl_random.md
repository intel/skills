# mkl_random reference

Load when the code uses `np.random.*`, `np.random.RandomState`, or `np.random.default_rng`/`Generator`.

A Python interface over oneMKL's Vector Statistics Library for bulk sampling. Covers the legacy `numpy.random` distribution API (`standard_normal`, `normal`, `rand`, `uniform`, `poisson`, `gamma`, `choice`, ... plus `seed`, `shuffle`, `permutation`).

`brng` families: MT19937 (default), SFMT19937, MT2203 (`('MT2203', id)`), R250, WH, MCG31, MCG59, MRG32K3A, PHILOX4X32X10, NONDETERM, ARS5. `method` for normal: ICDF (default), BoxMuller, BoxMuller2.

## Reproducibility hazard (always state this)

Patching changes the sampling sequence for the same seed. Upstream says so outright: "not fixed-seed backward compatible drop-in replacement for `numpy.random`". This holds even with `brng='MT19937'` - the patched interface hardcodes MT19937, so that *is* the patched path and it is still declared incompatible.

Never apply it to code that depends on reproducible values without an explicit OK from the user, every time.

`method` is a second, independent source of stream change, but it is **not reachable through the patch**: the patched `np.random.standard_normal` hardcodes `method="ICDF"` and raises `TypeError` if you pass the kwarg. Mention `method` only when recommending an explicit `MKLRandomState`.

## What it does NOT apply to

- **The modern `default_rng()` / `Generator` API.** The patch is a `setattr` loop over `mkl_random.interfaces.numpy_random.__all__`, which is legacy-only - `Generator`, `default_rng`, `BitGenerator` and `SeedSequence` appear nowhere in the package. Generator code gets nothing from patching, and since it is now the NumPy default this is the most frequent real NO.
  Accelerating it means changing the call site: `mkl_random.interfaces.numpy_random.RandomState` (the documented legacy drop-in), `mkl_random.MKLRandomState` when you need `brng`/`method` control, or the top-level `mkl_random.*` functions. All change the sampled values.
- Tiny or scalar draws, where vectorization has nothing to work on.
- Code needing bit-for-bit agreement with numpy's stream.

## Independent parallel streams

Match the mechanism to the generator - the support matrix is not uniform, and this is the most common source of wrong parallel-RNG advice:

- **Parametrized families.** MT2203 and WH expose many members: `MKLRandomState(seed, brng=('MT2203', i))`. Distinct ids are independent by construction. This is the mechanism to reach for.
- **`skipahead`.** Works on most generators, MT2203 included. Budget the jump: a worker must not consume more than `nskips` states or the streams overlap. `copy.copy(rs)` per jump, or all workers share one object.
- **`leapfrog`.** Raises for MT19937, SFMT19937 and MT2203. Rarely the answer.

Probe rather than assume:
```python
try:
    mkl_random.MKLRandomState(1, brng=('MT2203', 0)).skipahead(1000)
except ValueError:
    ...   # "... method of stream initialization is not supported for <brng>"
```

Different **seeds on one generator** is not a substitute: it gives different draws with no guarantee the streams do not overlap.

Across processes, build the states in the parent and ship one per worker; they pickle via `get_state`/`set_state`. Note `get_state` reports only the family name, but the real member is inside the saved stream bytes.

For an auditable run, mkl_random is reproducible given (seed, brng, member id). `brng='NONDETERM'` is hardware-based and is not.

Trap: `mkl_random.interfaces.numpy_random.RandomState` - the advertised drop-in - hardcodes `brng="MT19937"` and exposes no `brng`, so it cannot do multi-stream work at all.

## Decide fit

YES on bulk legacy-API sampling with large `size`, Monte Carlo loops, parallel multi-stream Monte Carlo. NO on reproducibility-dependent code unless the user accepts changed values.

## Which RandomState

Prefer `MKLRandomState(seed=None, brng='MT19937')`.

`mkl_random.RandomState` is deprecated and warns on construction - that warning is the reliable signal. Do not repeat the version it names as release history; that string has cited a release that was never published, and the same warning misdescribes what the class delegates to. The rendered docs are also stale here, autoclassing the deprecated `RandomState` and never mentioning `MKLRandomState`, so anything doc-grounded recommends the wrong one.

## Activation

```python
import mkl_random
mkl_random.patch_numpy_random()
# ... np.random.* now routes to VSL ...
mkl_random.restore_numpy_random()
```
Scoped: `with mkl_random.mkl_random():`

Explicit object, which is what multi-stream work needs:
```python
rng = mkl_random.MKLRandomState(seed=0, brng='MT19937')
x = rng.standard_normal(100_000_000, method='BoxMuller')
```
Patch/restore are reference-counted: N patches need N restores.

## Proof it is active

Both signals are reliable here and flip together (unlike FFT):

- `mkl_random.is_patched()` returns True.
- `np.random.rand.__module__ == "mkl_random.interfaces._numpy_random"`.

Critically, `np.random.RandomState.__module__` flips too - **the class itself is replaced.** So code that constructs its own `np.random.RandomState(seed)` is affected even though it never calls a `np.random.*` module function. Scan for that pattern before recommending a patch; it is the easiest way to under-report the blast radius.

# Independent parallel random streams with mkl_random

`/app/reference.py` estimates, by Monte Carlo, the probability that a unit stick
broken at two uniform points yields three pieces that form a triangle. The exact
answer is `1/4`. The reference draws every batch from a single
`numpy.random.RandomState`, one after another.

Write `/app/solution.py` that produces the same estimate, but with each worker
drawing from its **own independent `mkl_random` stream**.

Requirements:

1. Give each worker a separate generator whose stream is independent of every
   other worker's, using the **MT2203 parametrized family**: worker `i` uses
   `mkl_random.MKLRandomState(77777, brng=("MT2203", i))` for `i` in
   `range(n_workers)`. Independence comes from the family parametrization, not
   from passing different seeds to one generator — distinct seeds give no
   guarantee that the streams do not overlap.
2. Construct the generators explicitly. Do not patch `numpy.random`.
3. Keep the estimator unchanged: import `triangle_inequality` and `mc_prob` from
   the reference rather than reimplementing them.
4. For each worker, in order: draw the first 8 doubles with `rs.rand(8)`, then
   draw the batch with `rs.rand(2, batch)` and pass it to `mc_prob`. Batch size
   is `250_000`. The draw order matters — the verifier reproduces the same
   sequence and compares.
5. Print one line containing `VALID`, plus:
   - `workers=<n>` — how many independent streams were used
   - `estimate=<value>` — the mean of the per-worker estimates, 4 decimal places
   - `brng=<name>` — the generator family used
6. The estimate must be within `0.01` of the analytic `0.25`. It will **not**
   match the reference's estimate digit for digit: `mkl_random` is not
   seed-compatible with `numpy.random`, so the sampled values differ. That is
   expected; the analytic target is the contract.
7. Write the per-worker first draws to `/app/streams.json` as
   `{"brng": "MT2203", "workers": [[...], [...], ...]}` where each inner list is
   the 8 doubles that worker drew first.
8. Accept an optional CLI argument `<n_workers>` (between 2 and 64). Reject
   anything outside that range with a non-zero exit code.

Notes:

- The verifier reconstructs the expected MT2203 draws in its own process and
  compares them bitwise. `numpy.random` cannot produce them, so sampling from
  numpy will fail even if the estimate happens to land near `0.25`.
- Not every stream-splitting mechanism works with every generator family. This
  task pins MT2203 family members specifically; `leapfrog`, for instance, raises
  `ValueError` for MT2203.
- The workers may run in-process; actual multiprocessing is not required.

```bash
python3 /app/reference.py
python3 /app/solution.py
python3 /app/solution.py 8
```

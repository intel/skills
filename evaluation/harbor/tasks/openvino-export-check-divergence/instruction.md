# The exported OpenVINO model does not agree with the source model

`/app/src/reference.py` is a small NumPy model that scores the cells of a grid, with its
weights in `/app/src/weights.npz`. It is the source of truth. `/app/export/scorer.xml` is an
OpenVINO IR that someone exported from it and shipped. The exporter is not available.

The application feeds the IR the tensor that the `preprocess` function of the reference
returns, and its results differ from the reference's.

Find out why, and write:

- `/app/out/fixed.xml` with its `.bin`: an IR with the same input and output shapes that,
  for any pre-processed tensor, returns the scores the `forward` function of the reference
  returns, within 1e-4 absolute.
- `/app/out/report.json` with these keys:
  - `max_abs_diff_shipped` and `max_abs_diff_fixed`: the largest absolute difference in
    scores between each IR and the reference over the images in `/app/samples.npz`, each
    passed through `preprocess` first.
  - `causes`: a list with one entry per separate defect in the shipped IR. A defect that
    touches several nodes is one entry. Each entry is an object with `node` (the name of
    one node of the shipped IR that carries the defect), `what` (one sentence) and
    `max_abs_diff_alone` (the same largest difference, for an IR that has this defect and
    no other).

The verifier rebuilds the reference, the shipped IR and the sample images itself. It runs
the fixed IR on tensors you do not have, recomputes every number in the report and checks
each cause against the shipped IR. It uses the CPU with f32 inference precision.

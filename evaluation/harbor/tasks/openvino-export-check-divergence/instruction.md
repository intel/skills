# The exported OpenVINO model does not agree with the source model

`/app/src/reference.py` is a small NumPy model that scores the cells of a grid. It is the
source of truth. `/app/src/export.py` converts it to OpenVINO IR. Someone ran the command in
`/app/export/export_command.txt` and shipped the result, `/app/export/scorer.xml`.

The application feeds the IR the tensor that the `preprocess` function of the reference
returns, and its results differ from the reference's.

Find out why, and write:

- `/app/out/fixed.xml` with its `.bin`: an IR with the same input and output shapes that,
  for any pre-processed tensor, returns the scores the `forward` function of the reference
  returns, within 1e-4 absolute.
- `/app/out/report.json` with these keys: `max_abs_diff_shipped` and `max_abs_diff_fixed`
  (the largest absolute difference in scores between each IR and the reference over the
  images in `/app/samples.npz`, each passed through `preprocess` first), and `causes`
  (a list of short sentences, one per thing that was wrong).

The verifier runs the fixed IR on tensors you do not have and recomputes both numbers. It
uses the CPU with f32 inference precision. Do not change anything under `/app/export`.

# Quantize a classifier to INT8 and show it still holds

`/app/model/fp32.xml` is an FP32 OpenVINO IR of an image classifier with ten classes.
`/app/preprocess.py` is the pre-processing the application runs before every inference call.
`/app/data/calib_pool.npz` holds unlabelled images (`images`), and `/app/data/val.npz` holds
labelled ones (`images`, `labels`). All images are uint8, height by width by channel.

The team wants an INT8 model for an Intel CPU, made with NNCF. They accept it only if top-1
accuracy falls by at most 2.0 points against the FP32 IR.

Write:

- `/app/out/int8.xml` with its `.bin`: the same input and output as the FP32 IR, with both
  weights and activations quantized to 8 bits.
- `/app/out/report.json` with these keys: `max_drop_points` (the bar above), `fp32_accuracy`
  and `int8_accuracy` (top-1 in percent on `/app/data/val.npz`), and `passed` (true or false).

The verifier measures both IRs on a labelled split you do not have, drawn the same way as
the data in `/app/data`, and applies the same bar. It also recomputes the numbers in your
report. Both run on the CPU with f32 inference precision. Do not change `/app/model`.

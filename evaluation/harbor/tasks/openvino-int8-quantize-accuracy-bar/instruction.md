# Make the classifier faster on CPU with INT8, without losing accuracy

`/app` holds an image classifier with ten classes as it runs today: the application
`classify.py`, its FP32 OpenVINO IR `model/fp32.xml`, unlabelled images in
`data/calib_pool.npz` (`images`) and labelled ones in `data/val.npz` (`images`, `labels`).
All images are uint8, height by width by channel.

The team wants an INT8 model, made with NNCF, so that inference is faster on an Intel CPU.
They accept it only if top-1 accuracy falls by at most 2.0 points against the FP32 model.

Write:

- `/app/out/int8.xml` with its `.bin`: a drop-in replacement for the FP32 IR in the
  application, with the same input and output.
- `/app/out/report.json` with these keys: `max_drop_points` (the bar above), `fp32_accuracy`
  and `int8_accuracy` (top-1 in percent on the labelled images in `/app/data/val.npz`), and
  `passed` (true or false).

The verifier rebuilds the FP32 model and the data itself, so it does not read them from
`/app`. It measures both models on a labelled split you do not have, drawn the same way as
the data in `/app/data`, and applies the same bar. It also recomputes the numbers in your
report. Both run on the CPU with f32 inference precision.

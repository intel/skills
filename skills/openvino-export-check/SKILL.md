---
name: openvino-export-check
description: >-
  Check that a model converted to OpenVINO IR gives the same results as the original
  PyTorch or ONNX model, and find where the two diverge. Use for "OpenVINO
  gives different outputs than PyTorch", "accuracy or mAP dropped after export", "fewer
  detections after converting", "is my ov.convert_model, optimum-cli export openvino or
  Ultralytics export correct". Covers export arguments, input and output shapes,
  pre-processing done twice, NMS and score filters baked into the graph, FP16 and precision
  hints, static versus dynamic shapes. Run it before tuning or quantizing. Not for an
  export that fails with an error, or accuracy lost to INT8 quantization
  (openvino-int8-quantize).
license: Apache-2.0
---

# openvino-export-check

Prove that an OpenVINO IR reproduces its source model, or find the first place where it
stops. Do this before anyone tunes thresholds, benchmarks or quantizes. A difference that
enters at export looks like a weaker model later.

Scope: an export that succeeds and gives different results, checked on the CPU plugin.
Third-party behaviour below is stated for OpenVINO 2026.4.0, Ultralytics 8.4.215 and
optimum-intel 2.2.0. Check it again on other versions.

## 1. Inspect

- The exact export command or call, every argument, and the exporter version. Defaults
  count as arguments. If the exporter is gone, the IR is the only record of what it did.
- The IR's inputs and outputs, and the operation types at both ends. `Subtract`,
  `Multiply`, `Divide`, `Gather`, `Transpose`, `Interpolate` or `Convert` before the first
  layer mean the IR does pre-processing itself. `NonMaxSuppression`, `TopK`, `Select`,
  `Greater` or `GreaterEqual` near the outputs mean it does post-processing, with
  thresholds fixed at export time.
- The pre-processing and post-processing code of the source pipeline. Read the code, not a
  description of it.

## 2. Compare in this order and stop at the first stage that differs

Write the tolerance first. An FP32 IR on CPU agrees with the source to float rounding on
raw outputs. Take precision out of the question: save with `compress_to_fp16=False`
(`ov.save_model` compresses to FP16 by default) and compile with
`INFERENCE_PRECISION_HINT` set to `f32`. Use real samples. Random input rarely reaches a
score filter or a branch.

1. **Input tensor.** Run one real sample through both pre-processing paths and compare the
   two tensors (the token ids for a text model). If they differ, fix that first. The model
   is not the question yet.
2. **Raw outputs for one identical tensor.** Feed the same array to both, with the source
   in inference mode. A different output count or shape is another head, not rounding. For
   values, note where they differ: everywhere, only low values, only some rows.
3. **Post-processed results.** Same thresholds on both sides. Count items that match, items
   only the source has, items only the IR has.
4. **Task metric.** Only now. If stages 1 to 3 are clean and the metric differs, the two
   evaluation runs differ (threshold, resize, subset), not the models.

Do not start from the metric. It says that something differs, not where. After every fix,
start again from stage 1: fixing one cause often uncovers a second.

## 3. Causes

| Signal | Cause | Check | Fix |
|---|---|---|---|
| Stage 1 differs | The two pipelines differ in resize, letterbox, colour order, mean and scale, or layout | Diff the two functions | One function for both |
| Stage 1 equal, stage 2 differs everywhere, arithmetic before the first layer | Normalisation or channel reorder happens outside the IR and again inside it | Operations before the first layer | Remove one of the two |
| Stage 1 equal, stage 2 differs everywhere, graph looks clean | Normalisation was folded into the first layer's weight and bias, so no operation shows it and it is applied twice | Compare the first layer's constants with the source weights. A ratio that is constant per input channel and equals 1/std, with a shifted bias, is this | Put the source constants back, or stop normalising outside |
| Stage 2 differs everywhere, constants equal the source | Wrong checkpoint, or the source ran in training mode | Reload, `model.eval()`, compare again | Export again |
| Outputs differ in count or shape | The exporter picked another head: decoded or raw, end-to-end or NMS-based | Export arguments, output shapes | Export the head the source pipeline uses |
| Low-score items are missing or zero | NMS or a score filter sits in the graph with its own threshold | `NonMaxSuppression`, `Select`, `Greater` near the outputs | Set that threshold to the lowest value any consumer needs, or export without it and filter outside |
| Much smaller `.bin` than expected, small differences | The exporter compressed weights. `optimum-cli export openvino` writes 8-bit weights by default for models over one billion parameters | Size of the `.bin`, constant types | `--weight-format fp32` for the check |
| Results match, memory is higher than the source | Dynamic dimensions when the shape is in fact fixed | `?` in the input shape | `model.reshape(...)` to a static shape, then compare again |

Detector heads in Ultralytics. A model with an end-to-end (NMS-free) head keeps that branch
in the export only with `nms=False`. With `nms` unset or `nms=True` the export uses the
other branch, which is not the one the framework's own predict call runs. `nms=True` also
puts a score filter in the graph, at `conf=0.25` unless `conf` is passed. Boxes under it
never leave the model, so a tracker that needs low-score boxes loses them and no threshold
outside can bring them back. Compare each export variant at stages 2 and 3 before choosing.

Export arguments in Ultralytics 8.4.215. Write down the call and check each argument
against the source pipeline:

```python
from ultralytics import YOLO
YOLO("model.pt").export(format="openvino", imgsz=640, half=False, dynamic=False, nms=True, conf=0.01)
```

The values are examples. Take sizes and thresholds from the user's pipeline.

- `imgsz` and letterbox. A static export has one input size: `imgsz`, square unless
  `[h, w]` is passed. `predict` on the `.pt` pads each image only up to a multiple of the
  stride (a 1280x720 frame becomes a 384x640 tensor). `predict` on a static IR letterboxes
  to the full `imgsz` (640x640). The two models then see different tensors for one image,
  which is a stage 1 difference and not an export bug. Compare with one identical
  letterboxed tensor, or pass `rect=False` on the `.pt` side. Own pre-processing that
  stretches the image instead of letterboxing fails stage 1 the same way.
- `half`. `half=True` writes FP16 weights. 8.4.215 still accepts `half` and `int8` but
  forwards them to `quantize` (16 and 8). Keep FP32 for the check, then add FP16 as a
  measured variant.
- `dynamic`. `dynamic=False` fixes the input shape. `dynamic=True` leaves `?` dimensions,
  and the IR is then fed the same minimally padded tensor as the `.pt`. Use static when
  production sends one size. With dynamic, compare at two real sizes, not only at `imgsz`.
- `nms` and `conf`. See above. `nms=True` without `conf` fixes the score filter at 0.25.
- `val` mAP. Stage 4 only. Run `val` on both with the same `data`, `imgsz`, `conf` and
  `iou`. On a static IR `val` uses batch 1 and the full-size letterbox. On the `.pt` it
  uses rectangular batches, so pass `rect=False` there. Without it a small mAP gap remains
  with identical outputs, and it comes from the evaluation, not from the export.

If no row explains a stage 2 difference, bisect: add intermediate outputs with
`model.add_outputs(...)`, compare them with the same activations in the source, and report
the first layer that differs.

When you cannot export again, lower the baked-in threshold or repair the constants in the
IR, and say that this is a workaround.

## 4. Report

The export command and versions. The first stage that differed, with the largest absolute
difference on the input tensor and on each raw output. Each cause in one sentence, with the
difference that remains when only that cause is present. The fix and the same numbers after
it. Never report "matches" from a metric alone or from one random input.

## When not to use this

- The export stops with an error or an unsupported operation.
- The FP32 IR matches and the loss appears only after INT8: **openvino-int8-quantize**.
  Hand off to it once the FP32 IR is confirmed here.
- Speed. A correct IR that is slow is a tuning problem, not an export problem.

Sources: [converting a model](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/docs/articles_en/openvino-workflow/model-preparation/convert-model-to-ir.rst),
[setting input shapes](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/docs/articles_en/openvino-workflow/model-preparation/setting-input-shapes.rst),
[precision control](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/docs/articles_en/openvino-workflow/running-inference/optimize-inference/precision-control.rst),
[pre-processing in the graph](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/docs/articles_en/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.rst),
[Ultralytics exporter 8.4.215](https://github.com/ultralytics/ultralytics/blob/v8.4.215/ultralytics/engine/exporter.py),
[optimum-intel 2.2.0 export](https://github.com/huggingface/optimum-intel/blob/v2.2.0/docs/source/openvino/export.mdx).

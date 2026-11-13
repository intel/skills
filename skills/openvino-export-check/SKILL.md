---
name: openvino-export-check
description: >-
  Check that a model converted to OpenVINO IR gives the same results as the original
  PyTorch or ONNX model, and find where the two diverge. Use for "OpenVINO
  gives different outputs than PyTorch", "accuracy or mAP dropped after export", "fewer
  detections after converting", "is my ov.convert_model, optimum-cli export openvino or
  Ultralytics export correct". Covers export arguments, input and output shapes,
  pre-processing, NMS and score filters baked into the graph, FP16 and precision hints,
  static versus dynamic shapes. Run it before tuning or quantizing. Not for an export that
  fails with an error, accuracy lost to INT8 quantization (openvino-int8-quantize), or
  serving (openvino-gpu-run).
license: Apache-2.0
---

# openvino-export-check

Prove that an OpenVINO IR reproduces its source model, or find the first place where it
stops doing so. Do this before anyone tunes thresholds, benchmarks or quantizes. A
difference that enters at export is invisible later: it looks like a weaker model.

The method is to compare stage by stage, in order, and stop at the first stage that
differs. Do not start from the final metric. A metric says that something differs, not where.

## 1. Inspect

Collect three things before running any comparison.

**How the IR was made.** The exact command or call and every argument: `ov.convert_model`,
`optimum-cli export openvino`, an exporter such as Ultralytics, or ONNX first and then a
conversion. Ask for it if it is not in the repository. Defaults count as arguments. Record
the versions of OpenVINO and of the exporter.

**What the IR expects and returns.**

```python
from collections import Counter
import openvino as ov

core = ov.Core()
m = core.read_model("model_ir.xml")
for port in list(m.inputs) + list(m.outputs):
    print(port.get_names(), port.get_partial_shape(), port.get_element_type())
print(Counter(op.get_type_name() for op in m.get_ops()).most_common())
```

Read the output against the source model:

- Number, names, order, shapes and types of inputs and outputs. A different output count or
  shape means the exporter changed the head. It is not rounding.
- Static or dynamic dimensions (`?` or a range in the shape).
- Layout: NCHW or NHWC, and a `u8` input where the source takes floats.
- Work baked into the graph. At the input: `Convert`, `Transpose`, `Interpolate`, `Gather`,
  `Subtract`, `Multiply` or `Divide` before the first layer mean the IR does part of the
  pre-processing itself. At the output: `NonMaxSuppression`, `TopK`, `Select`, `Greater` or
  `GreaterEqual` mean the IR does part of the post-processing, with thresholds fixed at
  export time.
- The `rt_info` block at the end of the `.xml` file. It names the converter version, and
  some exporters store their arguments there.

**What the source pipeline does around the model.** Pre-processing (resize method,
letterbox or stretch, colour order, mean and scale, layout) and post-processing (decoding,
NMS, score threshold). Find the code. Do not rely on a description of it.

## 2. Decide what "the same" means

Write the tolerance before measuring. For an FP32 IR on CPU, raw outputs should agree with
the source to within float rounding at the scale of the output. Agreement that is only
"close" on a metric is not agreement. After post-processing, the same boxes, classes or
tokens should come out. Use a handful of real samples, not random noise alone: random input
rarely exercises a score filter or a branch.

## 3. Compare, in this order

**Stage 1. The input tensor.** Take one real sample. Run the source pipeline's
pre-processing and the OpenVINO pipeline's pre-processing. Compare the two tensors: shape,
type, and largest absolute difference. They must be identical or differ only by rounding.
If they differ, the model is not the problem yet. Fix the pre-processing and repeat.
For a text model the "tensor" is the token ids. Compare those first.

**Stage 2. Raw outputs for an identical tensor.** Feed the same array to both. Pin float
precision so the device does not pick a smaller type on its own.

```python
import numpy as np

compiled = core.compile_model(m, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
x = preprocess(sample)                  # the exact tensor the source model receives
reference = source_forward(x)           # list of numpy arrays from the source framework
result = compiled(x)
for port, expected in zip(compiled.outputs, reference):
    got = result[port]
    print(port.get_names(), got.shape, expected.shape,
          float(np.max(np.abs(got - expected))) if got.shape == expected.shape else "shape differs")
```

Put the source model in inference mode first (`model.eval()` in PyTorch). If shapes differ,
go back to the export arguments. If values differ, look at where: everywhere, only small
values, only some rows.

**Stage 3. Post-processed results.** Run each side's post-processing at the same
thresholds and compare the final items one by one. Count items that match, items only the
source has, and items only the IR has.

**Stage 4. The task metric.** Only now run the full evaluation. With stages 1 to 3 clean,
the metric should match too. If it does not, the evaluation harness differs between the two
runs (another threshold, another resize, another subset).

## 4. Usual causes

| Signal | Likely cause | Check | Fix |
|---|---|---|---|
| Stage 1 differs | Resize method, letterbox, colour order, mean and scale, or layout differ between the two pipelines | Diff the two pre-processing functions | Use one function for both |
| Stage 1 is equal, stage 2 differs everywhere | The IR also normalizes or reorders channels inside the graph, so it happens twice or in the wrong order | Ops before the first layer in step 1 | Export without embedded pre-processing, or stop doing it outside |
| Stage 2 differs everywhere, nothing baked in | Wrong checkpoint, or the source was in training mode | Reload weights, call `model.eval()`, export again | Export again |
| Stage 2 differs slightly | FP16 weights in the file, or the device runs in f16 or bf16 | `compiled.get_property("INFERENCE_PRECISION_HINT")`. Save again with `compress_to_fp16=False` | Keep FP32 for the check. Decide about FP16 later, as a measured variant |
| Outputs differ in count or shape | The exporter picked another head: decoded or raw, end-to-end or NMS-based | Export arguments, output shapes | Export with the head the source pipeline uses |
| Low-score items are missing or zero | NMS or a score filter is inside the graph with its own threshold | `NonMaxSuppression`, `Select`, `Greater` near the outputs | Set the threshold at export to the lowest value any consumer needs, or export without it and filter outside |
| Equal at the export size, wrong at other sizes | Tracing fixed a shape or a branch | Run two sizes through both | Export with dynamic shapes, or export at the size that will be used |
| Much smaller file than expected, small differences | The exporter compressed weights by default. `optimum-cli` does this for large models | Size of the `.bin`, constant types | `--weight-format fp32` for the check |
| Results match, memory is higher than the source | Dynamic shapes when the shape is in fact fixed | Input shapes in step 1 | `m.reshape(...)` to a static shape |

One case worth knowing. Detectors with an end-to-end (NMS-free) head can export two ways.
The default export and the framework's own prediction path may not go through the same
branch of the head. Export both ways and compare each at stage 2 and stage 3. If the
variant with NMS in the graph is the one that matches, remember that it also carries a
score filter. A tracker that needs low-score boxes will lose them unless the export sets
that filter low enough.

## 5. Recover

Change one thing, export again, and repeat from the first stage that failed.

```bash
optimum-cli export openvino --model <id> --weight-format fp32 <output_dir>
```

```python
from ultralytics import YOLO
YOLO("model.pt").export(format="openvino", half=False, dynamic=False, nms=True, conf=0.01)
```

```python
ov_model = ov.convert_model(source_model, example_input=x)
ov.save_model(ov_model, "model_fp32.xml", compress_to_fp16=False)

m.reshape([1, 3, 640, 640])            # fixed shape when the input size never changes
ov.save_model(m, "model_static.xml", compress_to_fp16=False)
```

The values above are examples. Take the sizes and thresholds from the user's pipeline.

When you cannot export again, you can still move a filter out of the way: set its threshold
low and apply the real threshold outside the graph. Say that this is a workaround.

If nothing explains a stage 2 difference, bisect. Add intermediate outputs to the IR with
`m.add_outputs(...)`, compare them with the same activations in the source, and report the
first layer that differs.

## 6. Report

- The export command, with versions.
- The first stage that differed, and the numbers: largest absolute difference on the input
  tensor and on each raw output, matched and unmatched items after post-processing, the
  task metric on both sides.
- The cause, in one sentence each if there are several. Fixing the first cause often
  uncovers a second one, so repeat all stages after each fix.
- The fix, and the same numbers after it.
- Any difference that remains and why it is acceptable, against the tolerance from step 2.

Never report "matches" from a metric alone, and never from a single random input.

## When not to use this

- The export stops with an error or an unsupported operation. This skill is for an export
  that succeeds and gives different results.
- The FP32 IR matches and the loss appears only after INT8: **openvino-int8-quantize**.
  Once the FP32 IR is confirmed here, hand off to it for quantization.
- Serving the model on an Intel GPU: **openvino-gpu-run**. Unsure which runtime:
  **ai-inference-deploy**. Memory fit: **model-can-it-fit**.
- Speed questions. A correct IR that is slow is a tuning problem, not an export problem.

Sources: [converting a model](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/model-preparation/convert-model-to-ir.rst),
[setting input shapes](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/model-preparation/setting-input-shapes.rst),
[precision control](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/optimize-inference/precision-control.rst),
[pre-processing in the graph](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.rst).

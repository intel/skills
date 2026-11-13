---
name: openvino-int8-quantize
description: >-
  Quantize a model to INT8 for OpenVINO on an Intel CPU with NNCF, and prove the accuracy
  still holds. Use for "make my model smaller or faster on CPU", "convert to int8",
  "post-training quantization", "nncf.quantize dropped my accuracy or mAP", "4-bit weight
  compression for an LLM", "which layers should stay in float". Covers weight-only
  compression or full quantization, calibration data, an accuracy bar set before the run,
  the task metric on held-out data, and recovering lost accuracy. Not for
  quantization-aware training, GGUF or bitsandbytes, serving (openvino-gpu-run), memory
  fit (model-can-it-fit), or an FP32 export that already differs from the source
  framework (openvino-export-check).
license: Apache-2.0
---

# openvino-int8-quantize

Quantize a model for the OpenVINO CPU plugin with NNCF, then show with numbers that the
result is still good enough for the task. A quantized file that loads is not the result.
The result is a variant that passed a bar you wrote down before you ran anything.

## 1. Inspect

Find these out before choosing a method. Ask the user for what you cannot read.

```bash
python3 -c "import openvino, nncf; print(openvino.__version__, nncf.__version__)"
lscpu | grep -iE "model name|avx512_vnni|avx_vnni|amx"
```

- The pair of versions. Each NNCF release names the OpenVINO release it was tested with, in
  the table in its
  [installation guide](https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md).
  Do not mix a new one with an old one.
- The CPU. INT8 pays off most on CPUs with VNNI or AMX. On an older CPU, measure before
  promising a gain.
- The model type: convolutional network, transformer encoder, LLM or other generative
  model, detector with a post-processing head.
- The metric the user is judged on. For a detector inside a tracker this is the tracking
  metric, not the detector metric. For a generative model it may be a similarity score
  against the FP32 output.
- The data: how many unlabelled samples exist for calibration, how many labelled samples
  exist for scoring, and whether they come from the same distribution as production.
- Why they want INT8: latency, throughput, file size or memory. The answer picks the method.

## 2. Decide

Write the acceptance bar first. Example form: "top-1 drops by at most N points against the
FP32 IR on the held-out split". Get the number from the user. If they have none, propose
one and get a yes. A bar chosen after seeing the results is not a bar.

Then pick one path:

| Situation | Path | Call |
|---|---|---|
| Convolutional network or small encoder, goal is latency or throughput | Full post-training quantization: weights and activations | `nncf.quantize` |
| LLM, diffusion or other large generative model, goal is size or memory | Weight-only compression. Activations stay in float | `nncf.compress_weights` |
| Full quantization misses the bar and a labelled validation set exists | Accuracy-aware quantization. It reverts the layers that hurt most | `nncf.quantize_with_accuracy_control` |
| One part is known to be sensitive (detection head, final layers, embeddings) | Full quantization with that part excluded | `ignored_scope=` |
| Nothing passes the bar | FP16 weights. Report that INT8 was rejected and why | `compress_to_fp16=True` |

Full quantization changes speed and accuracy. Weight-only compression always cuts file size
and load memory. Its effect on speed depends on what limits the model: token generation in
an LLM is limited by memory bandwidth and usually gets faster, a compute-bound network such
as a CNN or a VAE decoder usually does not. Measure before promising a speed-up.

## 3. Act

**Step 1. Get an FP32 IR.** Save it without FP16 compression so the baseline is exact.

```python
import openvino as ov

ov_model = ov.convert_model(source_model, example_input=example)   # or a path to an ONNX file
ov.save_model(ov_model, "model_fp32.xml", compress_to_fp16=False)
```

**Step 2. Check the FP32 IR against the source framework before quantizing.** Run the task
metric with both. They should agree closely. If they do not, stop and use
**openvino-export-check**. An export problem looks exactly like a quantization loss later,
and no calibration setting repairs it.

**Step 3. Split the data three ways.** Calibration samples, samples for tuning thresholds,
and held-out samples for the final score. Calibration needs no labels. Never score on data
that was used to calibrate or to tune. If the data is video, split by sequence, not by frame.

**Step 4. Quantize.** The calibration items must go through the same pre-processing as real
inference. Raw images, another resize, or another channel order give wrong ranges and a
large accuracy loss.

```python
import nncf
import openvino as ov

model = ov.Core().read_model("model_fp32.xml")

def transform_fn(item):
    return preprocess(item)          # exactly what inference feeds the model; drop the label

calibration = nncf.Dataset(calibration_items, transform_fn)
quantized = nncf.quantize(model, calibration, subset_size=300)
ov.save_model(quantized, "model_int8.xml")
```

Options that matter, all on `nncf.quantize`:

- `model_type=nncf.ModelType.TRANSFORMER` for attention models. Without it they lose accuracy.
- `preset=nncf.QuantizationPreset.MIXED` when activations are not symmetric (GELU, SiLU,
  models without ReLU).
- `ignored_scope=nncf.IgnoredScope(names=[...], types=[...], patterns=[...])` to keep nodes
  in float. Names are the node names in the IR.
- `fast_bias_correction=False` for a slower and more exact bias correction.

Weight-only compression:

```python
compressed = nncf.compress_weights(model)                     # 8-bit weights, no data needed
compressed = nncf.compress_weights(                           # 4-bit, data-aware
    model, mode=nncf.CompressWeightsMode.INT4_SYM, ratio=0.8, group_size=64, dataset=calibration)
```

For a Hugging Face model the same thing is one command:
`optimum-cli export openvino --model <id> --weight-format int8 <output_dir>`.

Some exporters quantize for you and make choices on the way. The Ultralytics INT8 export
keeps the detection head in float. Read what the exporter did before adding your own scope,
and treat its output as one more variant to verify.

## 4. Verify

Do all four. Each one catches a different failure.

**The IR is really quantized.** Full quantization leaves `FakeQuantize` nodes in the graph.
Weight-only compression leaves 8-bit or 4-bit constants and no `FakeQuantize`.

```python
from collections import Counter
import openvino as ov

m = ov.Core().read_model("model_int8.xml")
print(Counter(op.get_type_name() for op in m.get_ops())["FakeQuantize"])
print(Counter(op.get_output_element_type(0).get_type_name()
              for op in m.get_ops() if op.get_type_name() == "Constant"))
```

**The task metric on the held-out split.** Use the metric from step 1 and the full pipeline,
post-processing included. Score FP32 and each quantized variant with the same code.

**The operating threshold.** Quantization shifts scores. A confidence or score threshold
tuned for FP32 can be wrong for INT8. Sweep it again on the tuning split for every variant.
Report each variant at its own best threshold and at the old FP32 threshold. If the user
cannot change the threshold in production, only the second number counts.

**Speed, on the target CPU, with the same settings for every variant.**

```bash
benchmark_app -m model_fp32.xml -d CPU -hint latency -t 20
benchmark_app -m model_int8.xml -d CPU -hint latency -t 20
benchmark_app -m model_int8.xml -d CPU -hint latency -t 20 -pc | grep -iE "I8|U8" | head
```

The last command prints per-layer counters. Layers that run in 8 bits show `I8` or `U8` in
their execution type. If none do, the CPU plugin fell back to float and there will be no
gain. For a pipeline with heavy pre- or post-processing, also time the whole loop. The
model may not be the slow part.

Compare every variant to the bar from step 2. Pass or fail, nothing in between.

## 5. Recover

| Signal | Likely cause | What to do |
|---|---|---|
| Accuracy collapses to near chance | Calibration items did not go through the inference pre-processing | Fix `transform_fn`, quantize again |
| FP32 IR is already worse than the source framework | Export problem, not quantization | **openvino-export-check** |
| Moderate drop, small or narrow calibration set | Ranges do not cover production data | More samples, drawn from all conditions. Raise `subset_size` to match |
| Drop on a transformer | Default settings assume a convolutional network | `model_type=nncf.ModelType.TRANSFORMER` |
| Detector loses low-score boxes, tracker loses tracks | Scores shifted, or the head was quantized | Sweep the threshold again. Put the head in `ignored_scope` |
| Drop stays after the above | A few sensitive layers | `nncf.quantize_with_accuracy_control` with the bar as `max_drop`, or exclude layers by hand |
| Still under the bar | The model does not tolerate 8-bit activations | Weight-only compression, or FP16. Say that full INT8 was rejected |
| No speed-up | Float fallback, no VNNI, or the model is not the slow part | Read the `-pc` output, check `lscpu`, time the full pipeline |

Accuracy-aware quantization needs a validation function that returns one number where
higher is better:

```python
def validate(compiled_model, validation_items):
    return task_metric(compiled_model, validation_items)

quantized = nncf.quantize_with_accuracy_control(
    model, calibration_dataset=calibration,
    validation_dataset=nncf.Dataset(validation_items, transform_fn),
    validation_fn=validate, max_drop=0.01, drop_type=nncf.DropType.ABSOLUTE)
```

Use the tuning split here, not the held-out split. Rejecting INT8 is a valid outcome. A
variant that misses the bar is reported as failed, not shipped with a note.

## 6. Report

Return this, filled in from your own runs:

- Versions of OpenVINO and NNCF, the CPU model, the benchmark settings.
- The bar, as written before the run.
- A table with one row per variant (FP32, FP16, weight-only, full INT8, any scoped or
  accuracy-aware variant): file size, task metric at the old threshold, task metric at its
  own best threshold, latency or throughput, pass or fail.
- What was used for calibration, tuning and scoring, with counts, and that they do not overlap.
- The exact commands or script that produced each variant.
- The recommendation, and what was rejected.

## When not to use this

- The FP32 IR does not match the source framework: **openvino-export-check** first.
- "Will this model fit in memory": **model-can-it-fit**.
- Serving the result on an Intel GPU: **openvino-gpu-run**. Unsure which runtime:
  **ai-inference-deploy**.
- Quantization-aware training, or formats for other runtimes (GGUF, bitsandbytes). This
  skill covers post-training methods for OpenVINO only.
- GPU and NPU accept a different set of 8-bit operations. The steps are the same, but verify
  on the device that will run the model.

Sources: [NNCF post-training quantization](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md),
[NNCF weight compression](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md),
[benchmark_app](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst),
[precision control](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/optimize-inference/precision-control.rst).

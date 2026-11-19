---
name: openvino-int8-quantize
description: >-
  Quantize a model to INT8 for OpenVINO on an Intel CPU with NNCF, and prove the accuracy
  still holds. Use for "make my model smaller or faster on CPU", "convert to int8",
  "post-training quantization", "nncf.quantize dropped my accuracy or mAP", "4-bit weight
  compression for an LLM", "which layers should stay in float". Covers weight-only
  compression or full quantization, calibration data, an accuracy bar set before the run,
  the task metric on held-out data, and recovering lost accuracy. CPU plugin only. Not for
  GPU or NPU targets, quantization-aware training, GGUF or bitsandbytes, or an FP32 export
  that already differs from the source framework (openvino-export-check).
license: Apache-2.0
---

# openvino-int8-quantize

Quantize a model for the OpenVINO CPU plugin with NNCF, then show with numbers that it is
still good enough for the task. A quantized file that loads is not the result. The result
is a variant that passed a bar written down before anything ran.

Scope: post-training methods on the CPU plugin. GPU and NPU are not covered: they accept a
different set of 8-bit operations, and nothing here was checked on them. Third-party
behaviour below is stated for OpenVINO 2026.4.0, NNCF 3.4.0, Ultralytics 8.4.215 and
optimum-intel 2.2.0. Check it again on other versions.

## 1. Inspect

- The metric the user is judged on. For a detector inside a tracker it is the tracking
  metric, not mAP. For a generative model it may be similarity to the FP32 output.
- The data: unlabelled samples for calibration, labelled samples for scoring, and whether
  both look like production.
- Why INT8: latency, throughput, file size or memory. The answer picks the method.
- The code that feeds the model in production. Calibration has to go through it.
- The CPU (`lscpu | grep -iE "vnni|amx"`). Without VNNI or AMX, measure before promising a gain.

## 2. Decide

Write the acceptance bar first, for example "top-1 drops by at most N points against the
FP32 IR on the held-out split". Get the number from the user, or propose one and get a yes.
A bar chosen after seeing results is not a bar.

| Situation | Path |
|---|---|
| Convolutional network or small encoder, goal is latency or throughput | Full quantization of weights and activations: `nncf.quantize` |
| LLM, diffusion or other large generative model, goal is size or memory | Weight-only: `nncf.compress_weights`. Activations stay in float |
| Full quantization misses the bar and labelled tuning data exists | `nncf.quantize_with_accuracy_control`, with the bar as `max_drop` |
| One part is known to be sensitive (detection head, last layers, embeddings) | `nncf.quantize` with that part in `ignored_scope` |
| Nothing passes the bar | Keep float. Report that INT8 was rejected and why |

Weight-only compression cuts file size and load memory. Whether it is faster depends on
what limits the model: an LLM is limited by memory bandwidth and usually gains, a
compute-bound network such as a CNN usually does not.

## 3. Act

1. **Check the FP32 IR against the source framework first.** If the task metric differs,
   stop and use **openvino-export-check**. An export problem looks exactly like a
   quantization loss later, and no calibration setting repairs it.
2. **Split the data three ways:** calibration, threshold tuning, held-out scoring. Never
   score on data used to calibrate or tune. Split video by sequence, not by frame.
3. **Quantize through the production pre-processing.** The transform given to
   `nncf.Dataset` must return exactly the tensor inference feeds the model. Raw images,
   another resize or another channel order give wrong ranges.

```python
calibration = nncf.Dataset(calibration_items, lambda item: preprocess(item))
quantized = nncf.quantize(model, calibration, subset_size=len(calibration_items))
ov.save_model(quantized, "model_int8.xml", compress_to_fp16=False)
```

Save the FP32 baseline and the INT8 model with `compress_to_fp16=False`. The default is
`True`, which turns the remaining float constants into FP16 and mixes a second change
into the FP32 against INT8 comparison.

Options on `nncf.quantize` that change accuracy: `model_type=nncf.ModelType.TRANSFORMER`
for attention models, `preset=nncf.QuantizationPreset.MIXED` for activations that are not
symmetric (GELU, SiLU), `ignored_scope=nncf.IgnoredScope(names=..., types=..., patterns=...)`
to keep nodes in float, `fast_bias_correction=False` for a slower and more exact correction.

Exporters that quantize for you make choices. Ultralytics `export(format="openvino",
quantize=8, data=...)` keeps the head of detection and RT-DETR models in float through
`ignored_scope` and uses the mixed preset on CPU. `optimum-cli export openvino` writes
8-bit weights by default for models over one billion parameters. Read what the exporter
did and verify its output as one more variant.

## 4. Verify

- **Really quantized.** Full quantization leaves `FakeQuantize` nodes in the IR.
  Weight-only compression leaves 8-bit or 4-bit constants and none of those nodes.
- **Task metric on the held-out split,** with the full pipeline, same code for every variant.
- **Operating threshold.** Quantization shifts scores. Sweep the confidence threshold again
  on the tuning split for each variant. Report each variant at its own best threshold and
  at the old FP32 one. If production cannot change the threshold, only the second counts.
- **No float fallback,** on the target CPU:

```bash
benchmark_app -m model_int8.xml -d CPU -hint latency -t 20 -ip f32 -pc \
  | grep -E "layerType: +(Convolution|FullyConnected|MatMul)"
```

`-ip f32` keeps the input in float, so an 8-bit input does not hide the answer. Read the
`execType` of these compute layers. `I8` or `U8` there means they run in 8 bits. If none
of them says `I8` or `U8`, the layer runs in float (FP32, or bf16 on CPUs with AMX, shown
as `brgconv_avx512_amx_bf16`) and there is no gain. Run the same command on the FP32 IR to
see the difference.

Compare every variant to the bar. Pass or fail, nothing in between.

## 5. Recover

| Signal | Likely cause | What to do |
|---|---|---|
| Accuracy collapses to near chance | Calibration items skipped the inference pre-processing | Fix the transform of `nncf.Dataset`, quantize again |
| FP32 IR is already worse than the source | Export problem, not quantization | **openvino-export-check** |
| Moderate drop, small or narrow calibration set | Ranges do not cover production data | More samples from all conditions, `subset_size` to match |
| Drop on a transformer | Defaults assume a convolutional network | `model_type=nncf.ModelType.TRANSFORMER` |
| Detector keeps mAP, tracker loses tracks | Scores shifted, or the head was quantized | Sweep the threshold again, then put the head in `ignored_scope` |
| Drop stays | A few sensitive layers | `nncf.quantize_with_accuracy_control` on the tuning split, or exclude layers by hand |
| Still under the bar | The model does not tolerate 8-bit activations | Weight-only or float. Report full INT8 as rejected |
| No speed-up | Float fallback, no VNNI, or the model is not the slow part | The fallback check above, `lscpu`, time the whole pipeline |

Rejecting INT8 is a valid outcome. A variant that misses the bar is reported as failed, not
shipped with a note, and the bar does not move after the results are in.

## 6. Report

The bar as written before the run. One row per variant: file size, task metric at the old
threshold and at its own best, latency, pass or fail. What was used for calibration, tuning
and scoring, with counts, and that they do not overlap. Versions, CPU and commands.

## When not to use this

- The FP32 IR does not match the source framework: **openvino-export-check** first.
- GPU or NPU targets, quantization-aware training, GGUF, bitsandbytes.

Sources: [NNCF 3.4.0 post-training quantization](https://github.com/openvinotoolkit/nncf/blob/v3.4.0/docs/usage/post_training_compression/post_training_quantization/Usage.md),
[NNCF 3.4.0 weight compression](https://github.com/openvinotoolkit/nncf/blob/v3.4.0/docs/usage/post_training_compression/weights_compression/Usage.md),
[benchmark_app](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/docs/articles_en/get-started/learn-openvino/openvino-samples/benchmark-tool.rst),
[Ultralytics 8.4.215 OpenVINO export](https://github.com/ultralytics/ultralytics/blob/v8.4.215/ultralytics/utils/export/openvino.py),
[optimum-intel 2.2.0 export](https://github.com/huggingface/optimum-intel/blob/v2.2.0/docs/source/openvino/export.mdx).

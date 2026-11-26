#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/out
cat > /app/out/solve.py <<'PY'
import json
import sys

import numpy as np
import openvino as ov
from openvino import opset13 as ops

sys.path.insert(0, "/app/src")
import reference

core = ov.Core()
weights = {key: value.astype(np.float32) for key, value in np.load("/app/src/weights.npz").items()}
tensors = [reference.preprocess(image) for image in np.load("/app/samples.npz")["images"]]
shipped = core.read_model("/app/export/scorer.xml")

constants = {}
for op in shipped.get_ordered_ops():
    print(op.get_type_name(), op.get_friendly_name())
    if op.get_type_name() == "Constant":
        constants[op.get_friendly_name()] = np.array(op.get_data())

in_ir = {}
changed = []
for key, value in weights.items():
    name = next(n for n, c in constants.items() if c.shape == value.shape)
    in_ir[key] = constants[name].astype(np.float32)
    gap = float(np.max(np.abs(in_ir[key] - value)))
    print(key, name, "max abs difference from weights.npz", gap)
    if gap > 1e-6:
        changed.append(name)
ratio = in_ir["w1"] / weights["w1"]
print("w1 ratio per colour plane", [float(np.median(ratio[i * 64:(i + 1) * 64])) for i in range(3)], "1/STD", (1.0 / reference.STD).tolist())

filters = [op for op in shipped.get_ordered_ops() if op.get_type_name() == "Select"]
threshold = max(float(c.reshape(-1)[0]) for c in constants.values() if c.size == 1 and c.dtype == np.float32)


def build(w1, b1, score_threshold):
    image = ops.parameter([1, 3, 8, 8], np.float32, name="image")
    flat = ops.reshape(image, ops.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)
    hidden = ops.matmul(flat, ops.constant(w1), transpose_a=False, transpose_b=False)
    hidden = ops.relu(ops.add(hidden, ops.constant(b1)))
    logits = ops.matmul(hidden, ops.constant(weights["w2"]), transpose_a=False, transpose_b=False)
    scores = ops.sigmoid(ops.add(logits, ops.constant(weights["b2"])))
    if score_threshold is not None:
        keep = ops.greater_equal(scores, ops.constant(np.array(score_threshold, dtype=np.float32)))
        scores = ops.select(keep, scores, ops.constant(np.array(0.0, dtype=np.float32)))
    image.output(0).get_tensor().set_names({"image"})
    scores.output(0).get_tensor().set_names({"scores"})
    return ov.Model([ops.result(scores)], [image], "grid_scorer")


def max_abs_diff(model):
    compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    return max(float(np.max(np.abs(compiled(x)[out] - reference.forward(x)))) for x in tensors)


ov.save_model(build(weights["w1"], weights["b1"], None), "/app/out/fixed.xml", compress_to_fp16=False)
report = {
    "max_abs_diff_shipped": max_abs_diff(shipped),
    "max_abs_diff_fixed": max_abs_diff("/app/out/fixed.xml"),
    "causes": [
        {
            "node": changed[0],
            "what": "The first layer's weight and bias already contain the mean and scale normalisation that preprocess applies, so the input is normalised twice.",
            "max_abs_diff_alone": max_abs_diff(build(in_ir["w1"], in_ir["b1"], None)),
        },
        {
            "node": filters[0].get_friendly_name(),
            "what": f"A score filter is baked into the graph after the sigmoid: scores under {threshold:g} are replaced with 0, while the reference returns raw scores.",
            "max_abs_diff_alone": max_abs_diff(build(weights["w1"], weights["b1"], threshold)),
        },
    ],
}
with open("/app/out/report.json", "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
PY
python3 /app/out/solve.py

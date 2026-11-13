#!/usr/bin/env bash
set -euo pipefail
# The oracle: calibrate on the unlabelled pool through the application's own
# pre-processing, quantize weights and activations, then score both IRs on the labelled
# split and write the verdict against the bar.
mkdir -p /app/out
cat > /app/out/quantize.py <<'PY'
import json
import sys

import nncf
import numpy as np
import openvino as ov

sys.path.insert(0, "/app")
from preprocess import preprocess

BAR = 2.0
core = ov.Core()

pool = np.load("/app/data/calib_pool.npz")["images"]
calibration = nncf.Dataset(list(pool[:300]), preprocess)
quantized = nncf.quantize(core.read_model("/app/model/fp32.xml"), calibration, subset_size=300)
ov.save_model(quantized, "/app/out/int8.xml", compress_to_fp16=False)

fake_quantize = sum(op.get_type_name() == "FakeQuantize" for op in quantized.get_ops())
assert fake_quantize > 0, "nothing was quantized"


def accuracy(path, images, labels):
    compiled = core.compile_model(path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    hits = sum(int(np.argmax(compiled(preprocess(i))[out]) == int(y)) for i, y in zip(images, labels))
    return 100.0 * hits / len(labels)


val = np.load("/app/data/val.npz")
fp32 = accuracy("/app/model/fp32.xml", val["images"], val["labels"])
int8 = accuracy("/app/out/int8.xml", val["images"], val["labels"])
report = {
    "max_drop_points": BAR,
    "fp32_accuracy": fp32,
    "int8_accuracy": int8,
    "passed": bool(fp32 - int8 <= BAR),
}
with open("/app/out/report.json", "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report), "FakeQuantize ops:", fake_quantize)
PY
python3 /app/out/quantize.py

#!/usr/bin/env bash
set -euo pipefail
# The oracle: the shipped IR reorders the colour planes at its input and zeroes low scores
# at its output. Export again with both switched off, then measure both IRs against the
# reference on the sample images and write the evidence.
mkdir -p /app/out
python3 /app/src/export.py --input-order rgb --no-postprocess --out /app/out/fixed.xml

cat > /app/out/compare.py <<'PY'
import json
import sys

import numpy as np
import openvino as ov

sys.path.insert(0, "/app/src")
import reference

core = ov.Core()
tensors = [reference.preprocess(image) for image in np.load("/app/samples.npz")["images"]]


def max_abs_diff(path):
    compiled = core.compile_model(path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    return max(float(np.max(np.abs(compiled(x)[out] - reference.forward(x)))) for x in tensors)


report = {
    "max_abs_diff_shipped": max_abs_diff("/app/export/scorer.xml"),
    "max_abs_diff_fixed": max_abs_diff("/app/out/fixed.xml"),
    "causes": [
        "The export put a channel reorder (Gather on axis 1) at the input, so the IR expected BGR planes while the reference takes RGB.",
        "The export baked a score filter into the graph: scores under 0.25 were replaced with 0, while the reference returns raw scores.",
    ],
}
with open("/app/out/report.json", "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
PY
python3 /app/out/compare.py

#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

APP = Path("/app")
WEIGHT_SEED = 11
SAMPLE_SEED = 5
THRESHOLD = 0.25
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_weights():
    rng = np.random.default_rng(WEIGHT_SEED)
    return {
        "w1": (rng.standard_normal((192, 32)) / np.sqrt(192.0)).astype(np.float32),
        "b1": (0.1 * rng.standard_normal((1, 32))).astype(np.float32),
        "w2": (1.5 * rng.standard_normal((32, 16)) / 4.0).astype(np.float32),
        "b2": (0.5 * rng.standard_normal((1, 16))).astype(np.float32),
    }


def fold_normalisation(weights):
    mean = np.repeat(MEAN, 64)
    std = np.repeat(STD, 64)
    folded = dict(weights)
    folded["w1"] = (weights["w1"] / std[:, None]).astype(np.float32)
    folded["b1"] = (weights["b1"] - (mean / std)[None, :] @ weights["w1"]).astype(np.float32)
    return folded


def build(weights, threshold=None):
    image = ops.parameter([1, 3, 8, 8], np.float32, name="image")
    flat = ops.reshape(image, ops.constant(np.array([1, -1], dtype=np.int64), name="flat_shape"), special_zero=False, name="flat")
    hidden = ops.matmul(flat, ops.constant(weights["w1"], name="fc1_weight"), transpose_a=False, transpose_b=False, name="fc1")
    hidden = ops.relu(ops.add(hidden, ops.constant(weights["b1"], name="fc1_bias"), name="fc1_add"), name="fc1_relu")
    logits = ops.matmul(hidden, ops.constant(weights["w2"], name="fc2_weight"), transpose_a=False, transpose_b=False, name="fc2")
    scores = ops.sigmoid(ops.add(logits, ops.constant(weights["b2"], name="fc2_bias"), name="fc2_add"), name="fc2_sigmoid")
    if threshold is not None:
        keep = ops.greater_equal(scores, ops.constant(np.array(threshold, dtype=np.float32), name="head_a"), name="head_b")
        scores = ops.select(keep, scores, ops.constant(np.array(0.0, dtype=np.float32), name="head_c"), name="head_d")
    image.output(0).get_tensor().set_names({"image"})
    scores.output(0).get_tensor().set_names({"scores"})
    return ov.Model([ops.result(scores)], [image], "grid_scorer")


def preprocess(image):
    x = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def forward(weights, x):
    flat = np.asarray(x, dtype=np.float32).reshape(1, -1)
    hidden = np.maximum(flat @ weights["w1"] + weights["b1"], 0.0)
    logits = hidden @ weights["w2"] + weights["b2"]
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def max_abs_diff(model, weights, tensors):
    compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    return max(float(np.max(np.abs(np.asarray(compiled(x)[out]) - forward(weights, x)))) for x in tensors)


def main():
    weights = make_weights()
    (APP / "src").mkdir(parents=True, exist_ok=True)
    (APP / "export").mkdir(parents=True, exist_ok=True)
    np.savez(APP / "src" / "weights.npz", **weights)
    images = np.random.default_rng(SAMPLE_SEED).integers(0, 256, size=(32, 8, 8, 3), dtype=np.uint8)
    np.savez(APP / "samples.npz", images=images)
    folded = fold_normalisation(weights)
    ov.save_model(build(folded, THRESHOLD), str(APP / "export" / "scorer.xml"), compress_to_fp16=False)

    tensors = [preprocess(image) for image in images]
    both = max_abs_diff(build(folded, THRESHOLD), weights, tensors)
    constants_only = max_abs_diff(build(folded), weights, tensors)
    filter_only = max_abs_diff(build(weights, THRESHOLD), weights, tensors)
    clean = max_abs_diff(build(weights), weights, tensors)
    print(f"max abs diff: both {both:.4f}, constants {constants_only:.4f}, filter {filter_only:.4f}, clean {clean:.7f}")
    assert clean <= 1e-4
    assert filter_only > 0.05 and constants_only > 0.05
    assert abs(constants_only - filter_only) > 0.05


if __name__ == "__main__":
    main()

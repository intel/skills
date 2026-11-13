#!/usr/bin/env python3
"""Build-time generator: a small FP32 classifier IR and its data splits.

Run once by the Dockerfile and then removed. Everything is seeded, so the image is the
same on every build. tests/test_quantize.py repeats make_split() to draw its own split.
"""
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from preprocess import preprocess

APP = Path("/app")
CLASSES = 10
TEMPLATE_SEED = 20260
WEIGHT_SEED = 7
NOISE = 0.15


def make_split(seed, count):
    """uint8 HWC images and labels. Each sample has its own signal strength."""
    templates = np.sign(np.random.default_rng(TEMPLATE_SEED).standard_normal((CLASSES, 16, 16, 3)))
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, size=count)
    strength = np.exp(rng.uniform(np.log(0.02), np.log(0.3), size=count))[:, None, None, None]
    images = 0.5 + strength * templates[labels] + rng.normal(0.0, NOISE, size=(count, 16, 16, 3))
    return np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8), labels.astype(np.int64)


def backbone(x):
    rng = np.random.default_rng(WEIGHT_SEED)
    w1 = (rng.standard_normal((8, 3, 3, 3)) / np.sqrt(27.0)).astype(np.float32)
    b1 = (0.1 * rng.standard_normal((1, 8, 1, 1))).astype(np.float32)
    w2 = (rng.standard_normal((16, 8, 3, 3)) / np.sqrt(72.0)).astype(np.float32)
    b2 = (0.1 * rng.standard_normal((1, 16, 1, 1))).astype(np.float32)
    c1 = ops.convolution(x, ops.constant(w1), [2, 2], [1, 1], [1, 1], [1, 1])
    c1 = ops.relu(ops.add(c1, ops.constant(b1)))
    c2 = ops.convolution(c1, ops.constant(w2), [2, 2], [1, 1], [1, 1], [1, 1])
    c2 = ops.relu(ops.add(c2, ops.constant(b2)))
    return ops.reshape(c2, ops.constant(np.array([0, -1], dtype=np.int64)), special_zero=True)


def batch(images):
    return np.concatenate([preprocess(image) for image in images], axis=0)


def main():
    core = ov.Core()
    f32 = {"INFERENCE_PRECISION_HINT": "f32"}

    # Fit the last layer by ridge regression on the fixed backbone's features.
    train_images, train_labels = make_split(1, 4000)
    x = ops.parameter([-1, 3, 16, 16], np.float32, name="image")
    extractor = core.compile_model(ov.Model([ops.result(backbone(x))], [x], "backbone"), "CPU", f32)
    features = np.asarray(extractor(batch(train_images))[extractor.output(0)], dtype=np.float64)
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    targets = np.eye(CLASSES)[train_labels]
    gram = design.T @ design + 1e-3 * len(design) * np.eye(design.shape[1])
    solution = np.linalg.solve(gram, design.T @ targets)
    w3 = solution[:-1].astype(np.float32)
    b3 = solution[-1:].astype(np.float32)

    x = ops.parameter([1, 3, 16, 16], np.float32, name="image")
    logits = ops.add(ops.matmul(backbone(x), ops.constant(w3), transpose_a=False, transpose_b=False), ops.constant(b3))
    x.output(0).get_tensor().set_names({"image"})
    logits.output(0).get_tensor().set_names({"logits"})
    model = ov.Model([ops.result(logits)], [x], "classifier")

    (APP / "model").mkdir(parents=True, exist_ok=True)
    (APP / "data").mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(APP / "model" / "fp32.xml"), compress_to_fp16=False)

    pool_images, _ = make_split(2, 600)
    np.savez(APP / "data" / "calib_pool.npz", images=pool_images)
    val_images, val_labels = make_split(3, 1000)
    np.savez(APP / "data" / "val.npz", images=val_images, labels=val_labels)

    compiled = core.compile_model(str(APP / "model" / "fp32.xml"), "CPU", f32)
    out = compiled.output(0)
    hits = sum(int(np.argmax(compiled(preprocess(i))[out]) == int(y)) for i, y in zip(val_images, val_labels))
    accuracy = 100.0 * hits / len(val_labels)
    print(f"fp32 accuracy on val: {accuracy:.2f}")
    assert accuracy > 50.0, "the generated classifier is too weak to quantize meaningfully"


if __name__ == "__main__":
    main()

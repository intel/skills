import json
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

INT8 = Path("/app/out/int8.xml")
REPORT = Path("/app/out/report.json")
BAR = 2.0
REPORT_TOLERANCE = 0.5

CLASSES = 10
TEMPLATE_SEED = 20260
WEIGHT_SEED = 7
NOISE = 0.15
TRAIN_SEED = 1
TRAIN_COUNT = 4000
VAL_SEED = 3
VAL_COUNT = 1000
HELD_OUT_SEED = 4
HELD_OUT_COUNT = 2000
F32 = {"INFERENCE_PRECISION_HINT": "f32"}


def make_split(seed, count):
    templates = np.sign(np.random.default_rng(TEMPLATE_SEED).standard_normal((CLASSES, 16, 16, 3)))
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, size=count)
    strength = np.exp(rng.uniform(np.log(0.02), np.log(0.3), size=count))[:, None, None, None]
    images = 0.5 + strength * templates[labels] + rng.normal(0.0, NOISE, size=(count, 16, 16, 3))
    return np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8), labels.astype(np.int64)


def preprocess(image):
    x = (np.asarray(image, dtype=np.float32) / 255.0 - 0.5) / 0.25
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


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


def build_fp32():
    core = ov.Core()
    train_images, train_labels = make_split(TRAIN_SEED, TRAIN_COUNT)
    x = ops.parameter([-1, 3, 16, 16], np.float32, name="image")
    extractor = core.compile_model(ov.Model([ops.result(backbone(x))], [x], "backbone"), "CPU", F32)
    batch = np.concatenate([preprocess(image) for image in train_images], axis=0)
    features = np.asarray(extractor(batch)[extractor.output(0)], dtype=np.float64)
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    targets = np.eye(CLASSES)[train_labels]
    gram = design.T @ design + 1e-3 * len(design) * np.eye(design.shape[1])
    solution = np.linalg.solve(gram, design.T @ targets)
    w3 = solution[:-1].astype(np.float32)
    b3 = solution[-1:].astype(np.float32)
    x = ops.parameter([1, 3, 16, 16], np.float32, name="image")
    logits = ops.add(ops.matmul(backbone(x), ops.constant(w3), transpose_a=False, transpose_b=False), ops.constant(b3))
    return ov.Model([ops.result(logits)], [x], "classifier")


def accuracy(model, images, labels):
    compiled = ov.Core().compile_model(model, "CPU", F32)
    out = compiled.output(0)
    hits = sum(int(np.argmax(compiled(preprocess(i))[out]) == int(y)) for i, y in zip(images, labels))
    return 100.0 * hits / len(labels)


@pytest.fixture(scope="module")
def fp32_model():
    return build_fp32()


@pytest.fixture(scope="module")
def int8_model():
    assert INT8.is_file(), f"{INT8} was not written"
    assert INT8.with_suffix(".bin").is_file(), "the IR needs its .bin next to the .xml"
    return ov.Core().read_model(str(INT8))


@pytest.fixture(scope="module")
def report():
    assert REPORT.is_file(), f"{REPORT} was not written"
    return json.loads(REPORT.read_text())


def test_same_interface_as_fp32(int8_model, fp32_model):
    assert len(int8_model.inputs) == 1 and len(int8_model.outputs) == 1
    assert int8_model.input(0).get_partial_shape() == fp32_model.input(0).get_partial_shape()
    assert int8_model.output(0).get_partial_shape() == fp32_model.output(0).get_partial_shape()


def test_activations_are_quantized(int8_model):
    count = sum(op.get_type_name() == "FakeQuantize" for op in int8_model.get_ops())
    assert count >= 2, f"found {count} FakeQuantize ops; activations are not quantized"


def test_held_out_accuracy_within_bar(int8_model, fp32_model):
    images, labels = make_split(HELD_OUT_SEED, HELD_OUT_COUNT)
    fp32 = accuracy(fp32_model, images, labels)
    int8 = accuracy(int8_model, images, labels)
    assert fp32 > 50.0, f"the rebuilt FP32 baseline scores {fp32:.2f}"
    assert fp32 - int8 <= BAR, f"held-out top-1 fell from {fp32:.2f} to {int8:.2f}, more than {BAR} points"


def test_report_states_the_bar(report):
    assert float(report["max_drop_points"]) == BAR


def test_report_numbers_are_reproducible(report, int8_model, fp32_model):
    images, labels = make_split(VAL_SEED, VAL_COUNT)
    fp32 = accuracy(fp32_model, images, labels)
    int8 = accuracy(int8_model, images, labels)
    assert abs(float(report["fp32_accuracy"]) - fp32) <= REPORT_TOLERANCE, f"reported FP32 accuracy is not {fp32:.2f}"
    assert abs(float(report["int8_accuracy"]) - int8) <= REPORT_TOLERANCE, f"reported INT8 accuracy is not {int8:.2f}"
    assert isinstance(report["passed"], bool)
    assert report["passed"] == (fp32 - int8 <= BAR), "the pass or fail verdict does not follow from the numbers"

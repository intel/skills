"""Verifier for openvino-int8-quantize-accuracy-bar.

Checks outcomes only: the IR is really quantized, it holds accuracy on a split the agent
never had, and the report states numbers this file can reproduce.
"""
import json
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

FP32 = Path("/app/model/fp32.xml")
INT8 = Path("/app/out/int8.xml")
REPORT = Path("/app/out/report.json")
VAL = Path("/app/data/val.npz")
BAR = 2.0  # largest allowed top-1 drop, in accuracy points
REPORT_TOLERANCE = 0.5  # how far a reported accuracy may sit from the recomputed one

# Must match environment/make_task.py. The held-out split uses a seed of its own.
CLASSES = 10
TEMPLATE_SEED = 20260
NOISE = 0.15
HELD_OUT_SEED = 4
HELD_OUT_COUNT = 2000


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


def accuracy(path, images, labels):
    compiled = ov.Core().compile_model(str(path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    hits = sum(int(np.argmax(compiled(preprocess(i))[out]) == int(y)) for i, y in zip(images, labels))
    return 100.0 * hits / len(labels)


@pytest.fixture(scope="module")
def int8_model():
    assert INT8.is_file(), f"{INT8} was not written"
    assert INT8.with_suffix(".bin").is_file(), "the IR needs its .bin next to the .xml"
    return ov.Core().read_model(str(INT8))


@pytest.fixture(scope="module")
def report():
    assert REPORT.is_file(), f"{REPORT} was not written"
    return json.loads(REPORT.read_text())


def test_same_interface_as_fp32(int8_model):
    fp32 = ov.Core().read_model(str(FP32))
    assert len(int8_model.inputs) == 1 and len(int8_model.outputs) == 1
    assert int8_model.input(0).get_partial_shape() == fp32.input(0).get_partial_shape()
    assert int8_model.output(0).get_partial_shape() == fp32.output(0).get_partial_shape()


def test_activations_are_quantized(int8_model):
    # Weight-only compression leaves no FakeQuantize in the graph, and neither does a copy
    # of the FP32 file. Each of the two convolutions needs a quantized input at least.
    count = sum(op.get_type_name() == "FakeQuantize" for op in int8_model.get_ops())
    assert count >= 2, f"found {count} FakeQuantize ops; activations are not quantized"


def test_held_out_accuracy_within_bar(int8_model):
    images, labels = make_split(HELD_OUT_SEED, HELD_OUT_COUNT)
    fp32 = accuracy(FP32, images, labels)
    int8 = accuracy(INT8, images, labels)
    assert fp32 > 50.0, f"the FP32 baseline scores {fp32:.2f}; /app/model was changed"
    assert fp32 - int8 <= BAR, f"held-out top-1 fell from {fp32:.2f} to {int8:.2f}, more than {BAR} points"


def test_report_states_the_bar(report):
    assert float(report["max_drop_points"]) == BAR


def test_report_numbers_are_reproducible(report, int8_model):
    data = np.load(VAL)
    fp32 = accuracy(FP32, data["images"], data["labels"])
    int8 = accuracy(INT8, data["images"], data["labels"])
    assert abs(float(report["fp32_accuracy"]) - fp32) <= REPORT_TOLERANCE, f"reported FP32 accuracy is not {fp32:.2f}"
    assert abs(float(report["int8_accuracy"]) - int8) <= REPORT_TOLERANCE, f"reported INT8 accuracy is not {int8:.2f}"
    assert isinstance(report["passed"], bool)
    assert report["passed"] == (fp32 - int8 <= BAR), "the pass or fail verdict does not follow from the numbers"

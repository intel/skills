import json
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

FIXED = Path("/app/out/fixed.xml")
REPORT = Path("/app/out/report.json")
TOLERANCE = 1e-4
REPORT_TOLERANCE = 5e-3

WEIGHT_SEED = 11
SAMPLE_SEED = 5
THRESHOLD = 0.25
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONSTANT_NODES = {"fc1_weight", "fc1_bias", "fc1", "fc1_add"}
FILTER_NODES = {"head_a", "head_b", "head_c", "head_d"}


def make_weights():
    rng = np.random.default_rng(WEIGHT_SEED)
    return {
        "w1": (rng.standard_normal((192, 32)) / np.sqrt(192.0)).astype(np.float32),
        "b1": (0.1 * rng.standard_normal((1, 32))).astype(np.float32),
        "w2": (1.5 * rng.standard_normal((32, 16)) / 4.0).astype(np.float32),
        "b2": (0.5 * rng.standard_normal((1, 16))).astype(np.float32),
    }


W = make_weights()


def fold_normalisation(weights):
    mean = np.repeat(MEAN, 64)
    std = np.repeat(STD, 64)
    folded = dict(weights)
    folded["w1"] = (weights["w1"] / std[:, None]).astype(np.float32)
    folded["b1"] = (weights["b1"] - (mean / std)[None, :] @ weights["w1"]).astype(np.float32)
    return folded


def build(weights, threshold=None):
    image = ops.parameter([1, 3, 8, 8], np.float32, name="image")
    flat = ops.reshape(image, ops.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)
    hidden = ops.matmul(flat, ops.constant(weights["w1"]), transpose_a=False, transpose_b=False)
    hidden = ops.relu(ops.add(hidden, ops.constant(weights["b1"])))
    logits = ops.matmul(hidden, ops.constant(weights["w2"]), transpose_a=False, transpose_b=False)
    scores = ops.sigmoid(ops.add(logits, ops.constant(weights["b2"])))
    if threshold is not None:
        keep = ops.greater_equal(scores, ops.constant(np.array(threshold, dtype=np.float32)))
        scores = ops.select(keep, scores, ops.constant(np.array(0.0, dtype=np.float32)))
    return ov.Model([ops.result(scores)], [image], "grid_scorer")


def preprocess(image):
    x = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def forward(x):
    flat = np.asarray(x, dtype=np.float32).reshape(1, -1)
    hidden = np.maximum(flat @ W["w1"] + W["b1"], 0.0)
    logits = hidden @ W["w2"] + W["b2"]
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def max_abs_diff(model, tensors):
    compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    return max(float(np.max(np.abs(np.asarray(compiled(x)[out]) - forward(x)))) for x in tensors)


@pytest.fixture(scope="module")
def fixed():
    assert FIXED.is_file(), f"{FIXED} was not written"
    assert FIXED.with_suffix(".bin").is_file(), "the IR needs its .bin next to the .xml"
    return ov.Core().read_model(str(FIXED))


@pytest.fixture(scope="module")
def report():
    assert REPORT.is_file(), f"{REPORT} was not written"
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def sample_tensors():
    images = np.random.default_rng(SAMPLE_SEED).integers(0, 256, size=(32, 8, 8, 3), dtype=np.uint8)
    return [preprocess(image) for image in images]


@pytest.fixture(scope="module")
def expected(sample_tensors):
    folded = fold_normalisation(W)
    return {
        "shipped": max_abs_diff(build(folded, THRESHOLD), sample_tensors),
        "constants": max_abs_diff(build(folded), sample_tensors),
        "filter": max_abs_diff(build(W, THRESHOLD), sample_tensors),
    }


def test_interface(fixed):
    assert len(fixed.inputs) == 1 and len(fixed.outputs) == 1
    assert list(fixed.input(0).get_shape()) == [1, 3, 8, 8]
    assert list(fixed.output(0).get_shape()) == [1, 16]


def test_fixed_ir_matches_reference_on_unseen_inputs(fixed):
    rng = np.random.default_rng(99)
    tensors = [rng.uniform(-2.0, 2.0, size=(1, 3, 8, 8)).astype(np.float32) for _ in range(64)]
    low = sum(int(np.sum(forward(x) < THRESHOLD)) for x in tensors)
    assert low > 0, "the unseen inputs must include low scores"
    diff = max_abs_diff(fixed, tensors)
    assert diff <= TOLERANCE, f"scores differ from the reference by up to {diff:.6f}"


def test_report_evidence_is_reproducible(report, fixed, sample_tensors, expected):
    after = max_abs_diff(fixed, sample_tensors)
    assert expected["shipped"] > 100 * TOLERANCE
    assert abs(float(report["max_abs_diff_shipped"]) - expected["shipped"]) <= REPORT_TOLERANCE, "reported shipped difference is not reproducible"
    assert after <= TOLERANCE
    assert 0.0 <= float(report["max_abs_diff_fixed"]) <= TOLERANCE


def test_report_names_both_causes(report, expected):
    causes = report["causes"]
    assert isinstance(causes, list) and len(causes) == 2, "the shipped IR has two separate defects"
    for cause in causes:
        assert isinstance(cause, dict), "each cause is an object with node, what and max_abs_diff_alone"
        assert isinstance(cause.get("what"), str) and cause["what"].strip()
    for nodes, key in ((CONSTANT_NODES, "constants"), (FILTER_NODES, "filter")):
        hits = [cause for cause in causes if str(cause.get("node", "")).strip().lower() in nodes]
        assert len(hits) == 1, f"exactly one cause must name a node of the {key} defect"
        alone = float(hits[0]["max_abs_diff_alone"])
        assert abs(alone - expected[key]) <= REPORT_TOLERANCE, f"max_abs_diff_alone of the {key} defect is not reproducible"

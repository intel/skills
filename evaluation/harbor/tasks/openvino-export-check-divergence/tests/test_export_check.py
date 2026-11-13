"""Verifier for openvino-export-check-divergence.

The reference is rebuilt here from its seed, so nothing under /app/src is trusted. The
fixed IR is run on inputs the agent never saw, and the report's numbers are recomputed.
"""
import json
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

SHIPPED = Path("/app/export/scorer.xml")
FIXED = Path("/app/out/fixed.xml")
REPORT = Path("/app/out/report.json")
SAMPLES = Path("/app/samples.npz")
TOLERANCE = 1e-4

# Must match environment/make_task.py and environment/reference.py.
WEIGHT_SEED = 11
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


W = make_weights()


def preprocess(image):
    x = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def forward(x):
    flat = np.asarray(x, dtype=np.float32).reshape(1, -1)
    hidden = np.maximum(flat @ W["w1"] + W["b1"], 0.0)
    logits = hidden @ W["w2"] + W["b2"]
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def max_abs_diff(path, tensors):
    compiled = ov.Core().compile_model(str(path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
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
    return [preprocess(image) for image in np.load(SAMPLES)["images"]]


def test_interface(fixed):
    assert len(fixed.inputs) == 1 and len(fixed.outputs) == 1
    assert list(fixed.input(0).get_shape()) == [1, 3, 8, 8]
    assert list(fixed.output(0).get_shape()) == [1, 16]


def test_fixed_ir_matches_reference_on_unseen_inputs(fixed):
    rng = np.random.default_rng(99)
    tensors = [rng.uniform(-2.0, 2.0, size=(1, 3, 8, 8)).astype(np.float32) for _ in range(64)]
    low = sum(int(np.sum(forward(x) < 0.25)) for x in tensors)
    assert low > 0, "the unseen inputs must include low scores"
    diff = max_abs_diff(FIXED, tensors)
    assert diff <= TOLERANCE, f"scores differ from the reference by up to {diff:.6f}"


def test_report_evidence_is_reproducible(report, fixed, sample_tensors):
    shipped = max_abs_diff(SHIPPED, sample_tensors)
    after = max_abs_diff(FIXED, sample_tensors)
    assert shipped > 100 * TOLERANCE, "the IR under /app/export was changed"
    assert abs(float(report["max_abs_diff_shipped"]) - shipped) <= 0.02, f"reported shipped difference is not {shipped:.4f}"
    assert after <= TOLERANCE
    assert 0.0 <= float(report["max_abs_diff_fixed"]) <= TOLERANCE


def test_report_names_causes(report):
    causes = report["causes"]
    assert isinstance(causes, list) and causes, "the report must say what was wrong"
    assert all(isinstance(cause, str) and cause.strip() for cause in causes)

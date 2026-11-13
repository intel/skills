#!/usr/bin/env python3
"""Build-time generator: the weights, the sample images and the IR the task starts from.

Run once by the Dockerfile and then removed. tests/test_export_check.py repeats
make_weights() so the verifier does not depend on files the agent can edit.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

APP = Path("/app")
WEIGHT_SEED = 11
SAMPLE_SEED = 5
EXPORT_COMMAND = ["python3", "/app/src/export.py", "--out", "/app/export/scorer.xml"]


def make_weights():
    rng = np.random.default_rng(WEIGHT_SEED)
    return {
        "w1": (rng.standard_normal((192, 32)) / np.sqrt(192.0)).astype(np.float32),
        "b1": (0.1 * rng.standard_normal((1, 32))).astype(np.float32),
        "w2": (1.5 * rng.standard_normal((32, 16)) / 4.0).astype(np.float32),
        "b2": (0.5 * rng.standard_normal((1, 16))).astype(np.float32),
    }


def main():
    np.savez(APP / "src" / "weights.npz", **make_weights())
    images = np.random.default_rng(SAMPLE_SEED).integers(0, 256, size=(32, 8, 8, 3), dtype=np.uint8)
    np.savez(APP / "samples.npz", images=images)

    (APP / "export").mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, *EXPORT_COMMAND[1:]], check=True)
    (APP / "export" / "export_command.txt").write_text(" ".join(EXPORT_COMMAND) + "\n")


if __name__ == "__main__":
    main()

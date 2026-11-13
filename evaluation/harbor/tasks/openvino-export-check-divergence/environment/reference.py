"""NumPy reference for the grid scorer. This is the model the IR has to reproduce."""
from pathlib import Path

import numpy as np

_W = np.load(Path(__file__).with_name("weights.npz"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image):
    """One uint8 RGB image, HWC [8, 8, 3] -> float32 NCHW tensor [1, 3, 8, 8]."""
    x = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def forward(x):
    """Pre-processed tensor [1, 3, 8, 8] -> raw scores [1, 16], one per grid cell."""
    flat = np.asarray(x, dtype=np.float32).reshape(1, -1)
    hidden = np.maximum(flat @ _W["w1"] + _W["b1"], 0.0)
    logits = hidden @ _W["w2"] + _W["b2"]
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def detect(image, threshold=0.05):
    """What the application does: indices of the grid cells that score at least threshold."""
    return np.flatnonzero(forward(preprocess(image))[0] >= threshold)

"""The pre-processing the application runs before every inference call."""
import numpy as np

MEAN = 0.5
STD = 0.25


def preprocess(image):
    """One uint8 HWC image [16, 16, 3] -> float32 NCHW tensor [1, 3, 16, 16]."""
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

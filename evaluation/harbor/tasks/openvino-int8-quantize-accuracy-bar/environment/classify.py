import sys

import numpy as np
import openvino as ov

MODEL = "/app/model/fp32.xml"
MEAN = 0.5
STD = 0.25


def preprocess(image):
    """One uint8 HWC image [16, 16, 3] -> float32 NCHW tensor [1, 3, 16, 16]."""
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def classify(images, model=MODEL):
    compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    out = compiled.output(0)
    return [int(np.argmax(compiled(preprocess(image))[out])) for image in images]


if __name__ == "__main__":
    print(classify(np.load(sys.argv[1])["images"]))

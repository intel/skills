#!/usr/bin/env python3
"""Export the grid scorer in reference.py to OpenVINO IR."""
import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops


def build(weights, input_order, postprocess, score_threshold):
    image = ops.parameter([1, 3, 8, 8], np.float32, name="image")
    node = image
    if input_order == "bgr":
        # The caller passes camera frames as they come; put the planes back in RGB order.
        node = ops.gather(node, ops.constant(np.array([2, 1, 0], dtype=np.int64)), ops.constant(np.array(1, dtype=np.int64)))
    flat = ops.reshape(node, ops.constant(np.array([1, -1], dtype=np.int64)), special_zero=False)
    hidden = ops.matmul(flat, ops.constant(weights["w1"]), transpose_a=False, transpose_b=False)
    hidden = ops.relu(ops.add(hidden, ops.constant(weights["b1"])))
    logits = ops.matmul(hidden, ops.constant(weights["w2"]), transpose_a=False, transpose_b=False)
    scores = ops.sigmoid(ops.add(logits, ops.constant(weights["b2"])))
    if postprocess:
        keep = ops.greater_equal(scores, ops.constant(np.array(score_threshold, dtype=np.float32)))
        scores = ops.select(keep, scores, ops.constant(np.array(0.0, dtype=np.float32)))
    image.output(0).get_tensor().set_names({"image"})
    scores.output(0).get_tensor().set_names({"scores"})
    return ov.Model([ops.result(scores)], [image], "grid_scorer")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="path of the .xml to write")
    parser.add_argument("--weights", default=str(here / "weights.npz"))
    parser.add_argument("--input-order", choices=["bgr", "rgb"], default="bgr",
                        help="order of the colour planes the caller will feed")
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True,
                        help="keep only the scores that reach --score-threshold; the rest become 0")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    args = parser.parse_args()

    weights = {key: value.astype(np.float32) for key, value in np.load(args.weights).items()}
    model = build(weights, args.input_order, args.postprocess, args.score_threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, args.out, compress_to_fp16=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

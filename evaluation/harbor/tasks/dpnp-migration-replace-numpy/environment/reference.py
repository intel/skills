"""NumPy reference pipeline for dpnp migration task.

Generates deterministic synthetic data, applies element-wise math, computes
reduction statistics and a histogram signature, and prints a validation line.
"""
import numpy as np


def run():
    x = np.arange(1_000_000, dtype=np.float64)
    y = np.sin(x) + np.cos(x)
    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    min_val = float(np.min(y))
    max_val = float(np.max(y))

    # Histogram signature: weighted sum of bin midpoints
    counts, edges = np.histogram(y, bins='auto')
    mids = 0.5 * (edges[:-1] + edges[1:])
    hist_sig = float(np.dot(counts.astype(np.float64), mids))

    print(
        f"VALID mean={mean_val:.9e} std={std_val:.9e} "
        f"min={min_val:.9e} max={max_val:.9e} hist_sig={hist_sig:.9e}"
    )


if __name__ == "__main__":
    run()

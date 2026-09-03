"""Serial NumPy reference: stick-breaking triangle probability by Monte Carlo.

Break a unit stick at two independent uniform points. The three pieces form a
triangle only when each piece is shorter than the sum of the other two. The
exact probability is 1/4, so the Monte Carlo estimate has an analytic target
that does not depend on the generator used.

Adapted from IntelPython/mkl_random examples/stick_triangle.py.
"""
import sys

import numpy as np

ANALYTIC = 0.25


def triangle_inequality(x1, x2, x3):
    """True where the three pieces satisfy all three triangle inequalities."""
    ok = x1 < x2 + x3
    np.logical_and(ok, x2 < x1 + x3, out=ok)
    np.logical_and(ok, x3 < x1 + x2, out=ok)
    return ok


def mc_prob(draws):
    """Estimate the probability from an (2, n) array of uniform draws."""
    ws = np.sort(draws, axis=0)
    x1 = ws[0]
    x2 = ws[1] - ws[0]
    x3 = 1.0 - ws[1]
    return float(triangle_inequality(x1, x2, x3).sum()) / draws.shape[1]


def run(n_workers: int = 4, batch: int = 250_000) -> float:
    # Serial baseline: one stream, all batches drawn from it in sequence.
    rs = np.random.RandomState(77777)
    total = 0.0
    for _ in range(n_workers):
        total += mc_prob(rs.rand(2, batch))
    estimate = total / n_workers
    print(
        f"VALID reference workers={n_workers} batch={batch} "
        f"estimate={estimate:.4f} analytic={ANALYTIC}"
    )
    return estimate


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if n_workers < 2 or n_workers > 64:
        print("INVALID_ARGUMENTS: n_workers must be between 2 and 64", file=sys.stderr)
        sys.exit(2)
    run(n_workers)

"""Stock-NumPy reference: the npbench arc_distance kernel.

Great-circle distance between two sets of (theta, phi) coordinate pairs. The
kernel is a deliberate mix: sin, cos and sqrt are element-wise ufuncs mkl_umath
may accelerate, while arctan2 is not in its registry at all.

The signature is a bit-pattern sum, not a float sum. A float sum is useless as a
fingerprint here: patching changes tens of thousands of individual elements while
the sum stays bitwise identical, so a tolerance check on the sum cannot tell the
two apart. Reinterpreting the float64 bits as int64 keeps every mantissa bit.

Kernel adapted from IntelPython/mkl_umath benchmarks (npbench arc_distance).
"""
import sys

import numpy as np

UFUNCS_USED = ("sin", "cos", "sqrt", "arctan2")


def initialize(n, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random(n), rng.random(n), rng.random(n), rng.random(n))


def arc_distance(theta_1, phi_1, theta_2, phi_2):
    temp = (
        np.sin((theta_2 - theta_1) / 2) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
    )
    return 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))


def bit_signature(out: np.ndarray) -> int:
    """Bit-exact fingerprint. Distinct per ufunc implementation."""
    return int(np.ascontiguousarray(out, dtype=np.float64).view(np.int64).sum())


def run(n: int = 1_000_000) -> int:
    theta_1, phi_1, theta_2, phi_2 = initialize(n)
    sig = bit_signature(arc_distance(theta_1, phi_1, theta_2, phi_2))
    print(f"VALID reference n={n} patched=False sig={sig}")
    return sig


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    if n < 1024:
        print("INVALID_ARGUMENTS: n must be at least 1024", file=sys.stderr)
        sys.exit(2)
    run(n)

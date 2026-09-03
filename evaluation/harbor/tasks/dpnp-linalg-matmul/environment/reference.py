"""NumPy reference: deterministic matrix multiply with validation signature."""
import sys
import numpy as np


def run(n: int = 512):
    A = np.array(
        [[(i * 7 + j * 3) % 13 - 6 for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )
    B = np.array(
        [[(i * 5 + j * 11) % 17 - 8 for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )
    C = A @ B
    sig = float(np.sum(np.abs(C)))
    print(f"VALID reference sig={sig:.6f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    if n < 1:
        print("INVALID_ARGUMENTS", file=sys.stderr)
        sys.exit(2)
    run(n)

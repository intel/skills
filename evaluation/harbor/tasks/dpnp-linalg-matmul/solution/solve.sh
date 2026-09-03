#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
import sys
import dpnp as np


def run(n: int = 512):
    i_idx = np.arange(n, dtype=np.float64).reshape(n, 1)
    j_idx = np.arange(n, dtype=np.float64).reshape(1, n)
    A = (i_idx * 7 + j_idx * 3) % 13 - 6
    B = (i_idx * 5 + j_idx * 11) % 17 - 8
    C = np.matmul(A, B)
    sig = float(np.sum(np.abs(C)))
    print(f"VALID dpnp sig={sig:.6f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    if n < 1:
        import sys as _sys
        print("INVALID_ARGUMENTS", file=_sys.stderr)
        _sys.exit(2)
    run(n)
PY

echo "solution written to /app/solution.py"

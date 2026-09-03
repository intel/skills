#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
"""Oracle: activate mkl_fft so the existing np.fft call site dispatches to oneMKL."""
import sys

import numpy as np

import mkl_fft

from reference import make_signal, spectrum_signature

# Process-wide activation. The transform call site below is unchanged; the patch
# redirects np.fft.* to mkl_fft. Proof of activation is the bound function's
# __module__, not mkl_fft.is_patched().
mkl_fft.patch_numpy_fft()


def run(n: int = 65536) -> int:
    signal = make_signal(n)
    sig = spectrum_signature(np.fft.fft(signal))
    print(f"VALID mkl_fft n={n} fft_module={np.fft.fft.__module__} sig={sig}")
    return sig


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    if n < 8 or (n & (n - 1)) != 0:
        print("INVALID_ARGUMENTS: N must be a power of 2 >= 8", file=sys.stderr)
        sys.exit(2)
    run(n)
PY

echo "solution written to /app/solution.py"

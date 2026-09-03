#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
import math
import sys
import dpnp


def run(n: int = 65536):
    t = dpnp.arange(n, dtype=dpnp.float64) / n
    signal = dpnp.sin(2 * math.pi * 50 * t) + 0.5 * dpnp.sin(2 * math.pi * 120 * t)
    spectrum = dpnp.fft.fft(signal)
    power = dpnp.abs(spectrum) ** 2
    sig = float(dpnp.sum(power[: n // 2]))
    print(f"VALID dpnp sig={sig:.4f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    if n < 8 or (n & (n - 1)) != 0:
        print("INVALID_ARGUMENTS: N must be a power of 2 >= 8", file=sys.stderr)
        sys.exit(2)
    run(n)
PY

echo "solution written to /app/solution.py"

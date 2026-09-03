"""NumPy reference: FFT power spectrum of a deterministic two-tone signal."""
import math
import sys
import numpy as np


def run(n: int = 65536):
    t = np.arange(n, dtype=np.float64) / n
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    spectrum = np.fft.fft(signal)
    power = np.abs(spectrum) ** 2
    sig = float(np.sum(power[: n // 2]))
    print(f"VALID reference sig={sig:.4f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    if n < 8 or (n & (n - 1)) != 0:
        print("INVALID_ARGUMENTS: N must be a power of 2 >= 8", file=sys.stderr)
        sys.exit(2)
    run(n)

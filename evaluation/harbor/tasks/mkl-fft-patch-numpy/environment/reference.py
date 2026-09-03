"""Stock-NumPy reference: FFT of a deterministic two-tone signal.

The signature is a bit-pattern sum over the raw complex spectrum, not an energy
sum. An energy sum would be useless here: by Parseval it collapses to the closed
form 0.3125 * N**2, which a solution could print without computing anything.
Reinterpreting the float64 bits as int64 and summing keeps every mantissa bit,
so pocketfft and MKL DFTI give different signatures for the same input.
"""
import sys

import numpy as np


def make_signal(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / n
    return np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)


def spectrum_signature(spectrum: np.ndarray) -> int:
    """Bit-exact fingerprint of a complex spectrum.

    Stable across thread counts and repeat runs, but distinct per FFT
    implementation, so it identifies *which* backend produced the transform.
    """
    z = np.ascontiguousarray(spectrum, dtype=np.complex128)
    return int(z.real.view(np.int64).sum()) ^ int(z.imag.view(np.int64).sum())


def run(n: int = 65536) -> int:
    signal = make_signal(n)
    sig = spectrum_signature(np.fft.fft(signal))
    print(f"VALID reference n={n} fft_module={np.fft.fft.__module__} sig={sig}")
    return sig


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    if n < 8 or (n & (n - 1)) != 0:
        print("INVALID_ARGUMENTS: N must be a power of 2 >= 8", file=sys.stderr)
        sys.exit(2)
    run(n)

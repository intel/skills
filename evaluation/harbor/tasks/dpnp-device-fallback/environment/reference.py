"""NumPy reference: deterministic element-wise computation."""
import numpy as np


def run():
    x = np.arange(500_000, dtype=np.float64)
    y = np.sqrt(np.abs(np.sin(x) * np.cos(x) + 1.0))
    sig = np.sum(y)
    print(f"VALID reference sig={sig:.9e}")


if __name__ == "__main__":
    run()

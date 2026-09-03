"""NumPy reference: per-column statistical reductions on a deterministic batch."""
import numpy as np

ROWS = 10_000
COLS = 256


def run():
    i = np.arange(ROWS, dtype=np.float64).reshape(ROWS, 1)
    j = np.arange(COLS, dtype=np.float64).reshape(1, COLS)
    M = ((i * 31 + j * 17) % 997) - 498

    col_sum = np.sum(M, axis=0)
    col_mean = np.mean(M, axis=0)
    col_std = np.std(M, axis=0)

    # Balanced formula: all components contribute measurably
    sig = (
        float(np.sum(col_sum)) * 1e6
        + float(np.sum(col_mean)) * 1e9
        + float(np.sum(col_std)) * 1e11
    )
    print(f"VALID reference sig={sig:.6f}")


if __name__ == "__main__":
    run()

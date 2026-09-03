#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
import dpnp as np
import numpy


def run():
    x = np.arange(1_000_000, dtype=np.float64)
    y = np.sin(x) + np.cos(x)
    mean_val = float(np.mean(y))
    std_val = float(np.std(y))
    min_val = float(np.min(y))
    max_val = float(np.max(y))

    # dpnp.histogram does not support bins='auto' (string bin selectors are
    # not yet implemented); fall back to NumPy via asnumpy() for this step.
    try:
        counts, edges = np.histogram(y, bins='auto')
        counts_host = numpy.asarray(counts)
        edges_host = numpy.asarray(edges)
    except NotImplementedError:
        host = numpy.asarray(np.asnumpy(y))
        counts_host, edges_host = numpy.histogram(host, bins='auto')

    mids = 0.5 * (edges_host[:-1] + edges_host[1:])
    hist_sig = float(numpy.dot(counts_host.astype(numpy.float64), mids))

    print(
        f"VALID mean={mean_val:.9e} std={std_val:.9e} "
        f"min={min_val:.9e} max={max_val:.9e} hist_sig={hist_sig:.9e}"
    )


if __name__ == "__main__":
    run()
PY

echo "solution written to /app/solution.py"

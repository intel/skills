#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
"""Oracle: per-worker independent streams from the MT2203 generator family.

MT2203 is a parametrized family: each member id is a different generator, so
distinct ids give independent streams by construction. That is the property the
task asks for, and it is why a family beats handing different seeds to one
generator.

The draw order matters and is part of the contract: per worker, rand(8) first,
then the batch. The verifier reconstructs the same sequence from the same
(seed, member id) and compares bitwise.
"""
import json
import sys

import mkl_random as rnd

from reference import ANALYTIC, mc_prob

BRNG = "MT2203"
SEED = 77777


def worker_streams(n_workers):
    """One generator per worker, each a distinct member of the MT2203 family."""
    return [rnd.MKLRandomState(SEED, brng=(BRNG, i)) for i in range(n_workers)]


def run(n_workers: int = 4, batch: int = 250_000) -> float:
    streams = worker_streams(n_workers)

    firsts = []
    total = 0.0
    for rs in streams:
        firsts.append(list(rs.rand(8)))
        total += mc_prob(rs.rand(2, batch))
    estimate = total / n_workers

    with open("/app/streams.json", "w", encoding="utf-8") as fh:
        json.dump({"brng": BRNG, "workers": firsts}, fh)

    print(
        f"VALID mkl_random workers={n_workers} batch={batch} "
        f"estimate={estimate:.4f} analytic={ANALYTIC} brng={BRNG}"
    )
    return estimate


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if n_workers < 2 or n_workers > 64:
        print("INVALID_ARGUMENTS: n_workers must be between 2 and 64", file=sys.stderr)
        sys.exit(2)
    run(n_workers)
PY

echo "solution written to /app/solution.py"

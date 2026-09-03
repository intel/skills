#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
"""Oracle: patch mkl_umath, then read the covered set from the installed build.

The registry is the authority. `mkl_umath._ufuncs` exposes exactly the ufuncs the
build registers, and each one's `.types` lists its dtype loops. Since patching is
signature-matched, those two facts decide what can possibly be accelerated -- no
hardcoded list required, and none would stay correct across releases.
"""
import json
import sys

import mkl_umath
import mkl_umath._ufuncs as mkl_ufuncs

from reference import UFUNCS_USED, arc_distance, bit_signature, initialize

# Complex dtype characters in a ufunc type signature, e.g. "F->F", "D->D".
COMPLEX_CHARS = ("F", "D", "G")


def coverage():
    registered = set(dir(mkl_ufuncs))
    covered, not_covered, complex_capable = [], [], {}
    for name in UFUNCS_USED:
        if name in registered:
            covered.append(name)
            types = getattr(mkl_ufuncs, name).types
            complex_capable[name] = any(
                any(c in sig for c in COMPLEX_CHARS) for sig in types
            )
        else:
            not_covered.append(name)
            complex_capable[name] = False
    return {
        "patched": bool(mkl_umath.is_patched()),
        "covered": sorted(covered),
        "not_covered": sorted(not_covered),
        "complex_capable": complex_capable,
    }


def run(n: int = 1_000_000) -> int:
    mkl_umath.patch_numpy_umath()

    theta_1, phi_1, theta_2, phi_2 = initialize(n)
    sig = bit_signature(arc_distance(theta_1, phi_1, theta_2, phi_2))

    with open("/app/coverage.json", "w", encoding="utf-8") as fh:
        json.dump(coverage(), fh, indent=2, sort_keys=True)

    print(f"VALID mkl_umath n={n} patched={mkl_umath.is_patched()} sig={sig}")
    return sig


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    if n < 1024:
        print("INVALID_ARGUMENTS: n must be at least 1024", file=sys.stderr)
        sys.exit(2)
    run(n)
PY

echo "solution written to /app/solution.py"

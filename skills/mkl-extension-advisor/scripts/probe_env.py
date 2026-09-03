#!/usr/bin/env python3
"""One-shot environment probe for the mkl-extension-advisor skill.

Answers step 4 ("establish the install axis") and the per-package capability
probes in one call, so every invocation does not re-derive them from prose.

Notes on what the readings do and do not prove:
  1. threadpool_info() enumerates shared libraries ALREADY LOADED in the
     process. numpy's BLAS is a link-time dependency of its C extension, so it
     is mapped at import -- no BLAS call is needed to make it visible.
     An earlier version of this script forced a matmul first, on the belief that
     a call was required to load the pool. That was wrong: the reported pool is
     identical before and after. The matmul is kept only to exercise the path.
     Consequence worth stating plainly: on a normally-built MKL numpy, the
     ABSENCE of an mkl entry right after import IS meaningful evidence. Do not
     explain it away by re-probing after a warm-up.
  2. FFT wiring is read from np.fft.fft.__module__, never from is_patched():
     on build-wired numpy, FFT is live while is_patched() still returns False.
     Read together, the pair tells you HOW it was wired -- see fft_wiring.

Read-only: it never calls patch_*, never times anything, never writes files.
Whether is_patched() reads True after a bare import has differed across
generations of these packages, so the probe reports what it finds rather than
asserting what it should be.

Usage:  python scripts/probe_env.py
Output: a single JSON object on stdout.
"""

import json


def probe():
    out = {}

    try:
        import numpy as np
    except ImportError as exc:
        return {"error": "numpy not importable: {}".format(exc)}

    out["numpy_version"] = np.__version__

    # Read the BLAS axis BEFORE importing any mkl_* extension. Importing one
    # loads oneMKL into the process, after which the pool reports "mkl" even on
    # a stock OpenBLAS numpy -- an easy way to conclude the wrong thing.
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
        out["mkl_blas"] = any(d.get("internal_api") == "mkl" for d in pools)
        out["blas_apis"] = sorted({d.get("internal_api") for d in pools if d.get("internal_api")})
    except ImportError:
        out["mkl_blas"] = None
        out["blas_apis"] = "threadpoolctl not installed"

    # Authoritative for the BLAS question; the pool can be contaminated.
    try:
        cfg = np.__config__.show(mode="dicts")
        out["numpy_blas_name"] = cfg["Build Dependencies"]["blas"]["name"]
    except Exception:
        out["numpy_blas_name"] = None

    out["fft_module"] = getattr(np.fft.fft, "__module__", None)
    out["fft_active"] = out["fft_module"] == "mkl_fft.interfaces._numpy_fft"

    have = {}
    for module, label in (
        ("mkl_fft", "mkl_fft"),
        ("mkl_random", "mkl_random"),
        ("mkl_umath", "mkl_umath"),
        ("mkl", "mkl_service"),
        ("scipy", "scipy"),
    ):
        try:
            __import__(module)
            have[label] = True
        except ImportError:
            have[label] = False
    out["have"] = have

    # Current patch state per surface, plus the one genuine environment gate
    # (the scipy backend needs scipy and mkl-service installed).
    can = {}
    if have["mkl_fft"]:
        import mkl_fft

        can["scipy_fft_backend"] = hasattr(mkl_fft.interfaces, "scipy_fft")
        # Counter only; False alongside an active fft_module means build-wired.
        can["fft_is_patched"] = bool(mkl_fft.is_patched())
    if have["mkl_random"]:
        import mkl_random

        can["random_is_patched"] = bool(mkl_random.is_patched())
    if have["mkl_umath"]:
        import mkl_umath

        can["umath_active_now"] = bool(mkl_umath.is_patched())
    out["can"] = can

    # How FFT became active, which is more useful than whether it is.
    if not out["fft_active"]:
        out["fft_wiring"] = "dormant"
    elif can.get("fft_is_patched"):
        out["fft_wiring"] = "in-process patch (explicit call, .pth, or sitecustomize)"
    else:
        out["fft_wiring"] = "build-wired numpy (recipe patch; patch counter is 0)"

    notes = []
    if out["mkl_blas"] is None:
        notes.append("threadpoolctl missing: BLAS backend unconfirmed. pip install threadpoolctl")
    if out["mkl_blas"] is False:
        notes.append(
            "No mkl entry in the threadpool. numpy's BLAS loads at import, so this is real "
            "evidence that numpy is not MKL-backed -- not a probe-ordering artifact. A warm-up "
            "BLAS call will not change it; numpy_blas_name above is authoritative."
        )
    if have["mkl_fft"] and not have["mkl_service"]:
        notes.append(
            "mkl-service missing: mkl_fft.interfaces.scipy_fft will not exist "
            "(_scipy_fft.py does a module-level `import mkl`)."
        )
    if have["mkl_fft"] and not have["scipy"]:
        notes.append("scipy missing: SciPy FFT backend unavailable regardless of mkl-service.")
    if have["mkl_fft"] and not out["fft_active"]:
        notes.append(
            "FFT dormant: np.fft.* needs mkl_fft.patch_numpy_fft(), the mkl_fft context "
            "manager, or call mkl_fft.interfaces.numpy_fft directly."
        )
    if have["mkl_umath"] and can.get("umath_active_now") is False:
        notes.append(
            "umath dormant: needs an explicit mkl_umath.patch_numpy_umath(). Expected in "
            "current builds, where importing the package does not patch."
        )
    notes.append(
        "BLAS/LAPACK and FFT are independent axes: numpy's FFT is bundled pocketfft with no "
        "BLAS linkage, so an MKL BLAS says nothing about FFT wiring."
    )
    out["notes"] = notes

    return out


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, default=str))

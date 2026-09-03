#!/usr/bin/env python3
"""Verifier for mkl-fft-patch-numpy.

The signature the solution prints is checked against a oneMKL spectrum this
verifier computes in its own process. Nothing here trusts a string the solution
emitted, and the expected value is never hardcoded.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

APP = Path("/app")
SOLUTION = APP / "solution.py"
REFERENCE = APP / "reference.py"
TIMEOUT_SEC = 120.0
N_TEST = "8192"
PATCHED_MODULE = "mkl_fft.interfaces._numpy_fft"


def _run(script, *args):
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=TIMEOUT_SEC, cwd=str(APP),
    )
    assert result.returncode == 0, (
        f"{script} exit={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "VALID" in result.stdout, f"expected VALID in stdout, got {result.stdout!r}"
    return result.stdout


def _field(text, key, cast=str):
    m = re.search(rf"{key}=(\S+)", text)
    assert m, f"no {key}=<value> found in {text!r}"
    return cast(m.group(1))


def _expected_signatures(n):
    """Compute the stock and oneMKL signatures here, in this process."""
    import numpy as np

    sys.path.insert(0, str(APP))
    from reference import make_signal, spectrum_signature

    import mkl_fft

    signal = make_signal(n)
    stock = spectrum_signature(np.fft.fft(signal))
    mkl_fft.patch_numpy_fft()
    try:
        mkl = spectrum_signature(np.fft.fft(signal))
    finally:
        mkl_fft.restore_numpy_fft()
    return stock, mkl


def _tree():
    return ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))


def _reachable_calls(tree):
    """Call names reachable from module scope or from run().

    A call parked inside an unreferenced helper does not count: the point is that
    activation actually executes, not that the token appears in the file.
    """
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    entry_bodies = [
        n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
    ]
    if "run" in funcs:
        entry_bodies.append(funcs["run"])

    seen_funcs, names = set(), set()
    stack = list(entry_bodies)
    while stack:
        node = stack.pop()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                names.add(name)
                # Follow calls into locally defined helpers.
                if name in funcs and name not in seen_funcs:
                    seen_funcs.add(name)
                    stack.append(funcs[name])
            elif isinstance(sub, ast.With):
                for item in sub.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Call):
                        fn = ctx.func
                        names.add(
                            fn.attr if isinstance(fn, ast.Attribute)
                            else getattr(fn, "id", "")
                        )
    return names


def test_files_exist():
    assert SOLUTION.exists(), f"{SOLUTION} not found"
    assert REFERENCE.exists(), f"{REFERENCE} not found"


def test_activation_is_reachable():
    """mkl_fft must be activated on a path that actually runs."""
    names = _reachable_calls(_tree())
    assert {"patch_numpy_fft", "mkl_fft"} & names, (
        "solution.py must activate mkl_fft on a reachable path, via "
        "mkl_fft.patch_numpy_fft() or the mkl_fft context manager. A call parked "
        "in an unreferenced helper does not count."
    )


def test_solution_keeps_numpy_call_site():
    """The transform must still be written as np.fft.*, not swapped out."""
    tree = _tree()
    np_names = {
        a.asname or a.name
        for n in ast.walk(tree) if isinstance(n, ast.Import)
        for a in n.names if a.name == "numpy"
    }
    kept = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"fft", "rfft"}
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "fft"
        and isinstance(n.func.value.value, ast.Name)
        and n.func.value.value.id in np_names
        for n in ast.walk(tree)
    )
    assert kept, (
        "the transform must still be written as np.fft.fft(...); replacing the "
        "call site with a direct mkl_fft.fft(...) call does not satisfy the task"
    )


def test_solution_avoids_private_submodules():
    for node in ast.walk(_tree()):
        mod = ""
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
        elif isinstance(node, ast.Import):
            mod = ";".join(a.name for a in node.names)
        for part in mod.split(";"):
            if part.startswith("mkl_fft.") and part.split("mkl_fft.", 1)[1].startswith("_"):
                raise AssertionError(
                    f"solution.py must use the top-level mkl_fft API, not {part}"
                )


def test_reference_is_unpatched_numpy():
    assert _field(_run(REFERENCE, N_TEST), "fft_module") == "numpy.fft", (
        "reference.py must report the stock numpy.fft module; if it does not, "
        "numpy in this image is pre-wired to MKL and the task premise is broken"
    )


def test_solution_reports_patched_fft_module():
    module = _field(_run(SOLUTION, N_TEST), "fft_module")
    assert module == PATCHED_MODULE, (
        f"expected np.fft.fft.__module__ == {PATCHED_MODULE!r} after activation, "
        f"got {module!r}; FFT is still dormant"
    )


def test_signature_is_the_onemkl_spectrum():
    """The load-bearing check: the signature must be oneMKL's, computed here.

    Checked at more than one size. A solution that prints a constant harvested
    from some earlier run can match one size but not several, since each N has a
    different oneMKL signature.
    """
    for n in (int(N_TEST), 4096, 16384):
        stock, mkl = _expected_signatures(n)
        assert stock != mkl, (
            f"premise broken at N={n}: pocketfft and MKL produce the same "
            f"signature, so the task cannot distinguish them"
        )
        got = _field(_run(SOLUTION, str(n)), "sig", int)
        assert got != stock, (
            f"N={n}: signature {got} equals the stock-NumPy spectrum, so the "
            f"transform did not dispatch to oneMKL"
        )
        assert got == mkl, (
            f"N={n}: signature {got} does not match the oneMKL spectrum {mkl} "
            f"computed by the verifier"
        )


def test_reference_signature_is_the_stock_spectrum():
    """Guards the premise from the other side."""
    stock, _ = _expected_signatures(int(N_TEST))
    assert _field(_run(REFERENCE, N_TEST), "sig", int) == stock


def test_rejects_invalid_size():
    result = subprocess.run(
        [sys.executable, str(SOLUTION), "100"],
        capture_output=True, text=True, timeout=TIMEOUT_SEC, cwd=str(APP),
    )
    assert result.returncode != 0, "solution must exit non-zero for non-power-of-2 N"

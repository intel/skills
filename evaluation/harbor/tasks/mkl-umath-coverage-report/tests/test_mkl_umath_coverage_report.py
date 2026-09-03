#!/usr/bin/env python3
"""Verifier for mkl-umath-coverage-report.

Both the expected bit signature and the expected coverage report are recomputed
here from the installed package. Nothing is hardcoded and nothing the solution
prints is taken on trust.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

APP = Path("/app")
SOLUTION = APP / "solution.py"
REFERENCE = APP / "reference.py"
COVERAGE = APP / "coverage.json"
TIMEOUT_SEC = 180.0
N_TEST = 1_000_000
COMPLEX_CHARS = ("F", "D", "G")


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
    """Compute the unpatched and patched signatures here, in this process."""
    import mkl_umath

    sys.path.insert(0, str(APP))
    from reference import arc_distance, bit_signature, initialize

    arrays = initialize(n)
    if mkl_umath.is_patched():
        mkl_umath.restore_numpy_umath()
    stock = bit_signature(arc_distance(*arrays))
    mkl_umath.patch_numpy_umath()
    try:
        patched = bit_signature(arc_distance(*arrays))
    finally:
        mkl_umath.restore_numpy_umath()
    return stock, patched


def _expected_coverage():
    """Recompute the coverage truth from the installed registry."""
    import mkl_umath._ufuncs as u

    sys.path.insert(0, str(APP))
    from reference import UFUNCS_USED

    registered = set(dir(u))
    covered = sorted(n for n in UFUNCS_USED if n in registered)
    not_covered = sorted(n for n in UFUNCS_USED if n not in registered)
    complex_capable = {
        n: (
            any(any(c in sig for c in COMPLEX_CHARS) for sig in getattr(u, n).types)
            if n in registered
            else False
        )
        for n in UFUNCS_USED
    }
    return covered, not_covered, complex_capable


def _tree():
    return ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))


def _reachable_calls(tree):
    """Call names reachable from module scope or run(), following local helpers."""
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    entry = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if "run" in funcs:
        entry.append(funcs["run"])

    seen, names = set(), set()
    stack = list(entry)
    while stack:
        for sub in ast.walk(stack.pop()):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                names.add(name)
                if name in funcs and name not in seen:
                    seen.add(name)
                    stack.append(funcs[name])
            elif isinstance(sub, ast.With):
                for item in sub.items:
                    if isinstance(item.context_expr, ast.Call):
                        fn = item.context_expr.func
                        names.add(
                            fn.attr if isinstance(fn, ast.Attribute)
                            else getattr(fn, "id", "")
                        )
    return names


def test_files_exist():
    assert SOLUTION.exists(), f"{SOLUTION} not found"
    assert REFERENCE.exists(), f"{REFERENCE} not found"


def test_activation_is_reachable():
    names = _reachable_calls(_tree())
    assert {"patch_numpy_umath", "mkl_umath"} & names, (
        "solution.py must activate mkl_umath on a reachable path, via "
        "mkl_umath.patch_numpy_umath() or the mkl_umath context manager. A call "
        "parked in an unreferenced helper does not count."
    )


def test_solution_reuses_the_reference_kernel():
    reused = any(
        isinstance(n, ast.ImportFrom) and (n.module or "") == "reference"
        for n in ast.walk(_tree())
    )
    assert reused, "solution.py must import the kernel from reference.py, not rewrite it"


def test_reference_runs_unpatched():
    assert _field(_run(REFERENCE, str(N_TEST)), "patched") == "False"


def test_signature_is_the_patched_kernel():
    """Load-bearing: the signature must be mkl_umath's, computed here.

    Checked at two sizes so a constant harvested from one run cannot pass.
    """
    for n in (N_TEST, 262_144):
        stock, patched = _expected_signatures(n)
        assert stock != patched, (
            f"premise broken at n={n}: patching does not change the kernel's bit "
            f"signature, so the task cannot distinguish patched from unpatched"
        )
        got = _field(_run(SOLUTION, str(n)), "sig", int)
        assert got != stock, (
            f"n={n}: signature {got} equals the unpatched kernel, so mkl_umath "
            f"was not active for the computation"
        )
        assert got == patched, (
            f"n={n}: signature {got} does not match the patched kernel signature "
            f"{patched} computed by the verifier"
        )


def test_reference_signature_is_the_unpatched_kernel():
    stock, _ = _expected_signatures(N_TEST)
    assert _field(_run(REFERENCE, str(N_TEST)), "sig", int) == stock


def test_coverage_report_matches_the_installed_registry():
    _run(SOLUTION, str(N_TEST))
    assert COVERAGE.exists(), f"{COVERAGE} not written"
    report = json.loads(COVERAGE.read_text())
    covered, not_covered, complex_capable = _expected_coverage()

    assert sorted(report.get("covered", [])) == covered, (
        f"covered list wrong: reported {sorted(report.get('covered', []))}, "
        f"installed build registers {covered}"
    )
    assert sorted(report.get("not_covered", [])) == not_covered, (
        f"not_covered list wrong: reported "
        f"{sorted(report.get('not_covered', []))}, expected {not_covered}"
    )
    assert report.get("complex_capable") == complex_capable, (
        f"complex_capable wrong: reported {report.get('complex_capable')}, "
        f"expected {complex_capable}"
    )
    assert report.get("patched") is True, "coverage.json must record patched: true"


def test_report_identifies_at_least_one_uncovered_ufunc():
    """Guards the premise: the kernel really does mix covered and uncovered."""
    covered, not_covered, _ = _expected_coverage()
    assert covered, "premise broken: no kernel ufunc is registered"
    assert not_covered, (
        "premise broken: every kernel ufunc is registered, so the task no longer "
        "tests a coverage boundary"
    )


def test_rejects_invalid_size():
    result = subprocess.run(
        [sys.executable, str(SOLUTION), "512"],
        capture_output=True, text=True, timeout=TIMEOUT_SEC, cwd=str(APP),
    )
    assert result.returncode != 0, "solution must exit non-zero for n < 1024"

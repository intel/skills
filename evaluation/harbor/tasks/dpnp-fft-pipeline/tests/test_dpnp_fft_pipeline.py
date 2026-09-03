#!/usr/bin/env python3
"""Verifier for dpnp-fft-pipeline."""
import ast
import re
import subprocess
import sys
from pathlib import Path

SOLUTION = Path("/app/solution.py")
REFERENCE = Path("/app/reference.py")
TIMEOUT_SEC = 60.0
REL_TOL = 1e-6
N_TEST = "8192"


def _run(script, *args):
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
    assert result.returncode == 0, (
        f"{script} exit={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "VALID" in result.stdout, f"expected VALID in stdout, got {result.stdout!r}"
    return result.stdout


def _sig(text):
    m = re.search(r"sig=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    assert m, f"no sig=<value> found in {text!r}"
    return float(m.group(1))


def _uses_dpnp_fft(tree):
    """True if the solution imports dpnp and calls dpnp.fft.fft (not numpy/scipy)."""
    # Collect all dpnp import aliases (e.g. "import dpnp as xp" -> "xp")
    dpnp_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("dpnp"):
                    dpnp_names.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("dpnp"):
            for alias in node.names:
                dpnp_names.add(alias.asname or alias.name)

    if not dpnp_names:
        return False

    # Walk for a .fft call whose receiver chain starts with a dpnp-bound name
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # dpnp.fft.fft(...) -> Attribute(Attribute(Name('dpnp'), 'fft'), 'fft')
            if isinstance(func, ast.Attribute) and func.attr == "fft":
                inner = func.value
                # two-level: dpnp.fft
                if isinstance(inner, ast.Attribute) and inner.attr == "fft":
                    root = inner.value
                    if isinstance(root, ast.Name) and root.id in dpnp_names:
                        return True
                # one-level alias: fft_mod.fft where fft_mod came from dpnp
                if isinstance(inner, ast.Name) and inner.id in dpnp_names:
                    return True
    return False


def test_files_exist():
    assert SOLUTION.exists(), f"{SOLUTION} not found"
    assert REFERENCE.exists(), f"{REFERENCE} not found"


def test_solution_uses_dpnp_fft():
    tree = ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))
    assert _uses_dpnp_fft(tree), (
        "solution.py must import dpnp and call dpnp.fft.fft(); "
        "using numpy.fft does not satisfy the task"
    )


def test_signature_matches_reference():
    ref_sig = _sig(_run(REFERENCE, N_TEST))
    sol_sig = _sig(_run(SOLUTION, N_TEST))
    denom = max(abs(ref_sig), 1e-15)
    rel_err = abs(sol_sig - ref_sig) / denom
    assert rel_err <= REL_TOL, (
        f"dpnp sig {sol_sig} differs from reference {ref_sig} "
        f"(relative error {rel_err:.2e} > {REL_TOL})"
    )


def test_rejects_invalid_size():
    # N=100 is not a power of 2
    result = subprocess.run(
        [sys.executable, str(SOLUTION), "100"],
        capture_output=True, text=True, timeout=TIMEOUT_SEC
    )
    assert result.returncode != 0, "solution must exit non-zero for non-power-of-2 N"

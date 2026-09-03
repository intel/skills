#!/usr/bin/env python3
"""Verifier for dpnp-reduction-stats."""
import ast
import re
import subprocess
import sys
from pathlib import Path

SOLUTION = Path("/app/solution.py")
REFERENCE = Path("/app/reference.py")
TIMEOUT_SEC = 60.0
REL_TOL = 1e-7


def _run(script):
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=TIMEOUT_SEC
    )
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


def _uses_dpnp_reductions(tree):
    """True if solution imports dpnp and uses sum/mean/std rooted on a dpnp-bound name."""
    # Collect dpnp-bound names
    dpnp_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dpnp":
                    dpnp_names.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "dpnp":
            for alias in node.names:
                dpnp_names.add(alias.asname or alias.name)

    if not dpnp_names:
        return False

    # Find reduction calls rooted on a dpnp-bound name
    REDUCTIONS = {"sum", "mean", "std"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # dpnp.sum(...) / alias.sum(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr in REDUCTIONS
                and isinstance(func.value, ast.Name)
                and func.value.id in dpnp_names
            ):
                found.add(func.attr)
    return len(found) >= 2


def test_files_exist():
    assert SOLUTION.exists(), f"{SOLUTION} not found"
    assert REFERENCE.exists(), f"{REFERENCE} not found"


def test_solution_uses_dpnp_reductions():
    tree = ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))
    assert _uses_dpnp_reductions(tree), (
        "solution.py must import dpnp and use at least sum/mean/std reduction functions"
    )


def test_solution_uses_axis_parameter():
    tree = ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))
    # Check that at least one reduction call has axis=0 (keyword or positional)
    dpnp_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dpnp":
                    dpnp_names.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "dpnp":
            for alias in node.names:
                dpnp_names.add(alias.asname or alias.name)
    found_axis = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("sum", "mean", "std")
                and isinstance(func.value, ast.Name)
                and func.value.id in dpnp_names
            ):
                # Check keyword argument axis=0
                for kw in node.keywords:
                    if kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value == 0:
                        found_axis = True
                # Check positional argument (2nd position for sum/mean/std is axis)
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == 0:
                    found_axis = True
    assert found_axis, (
        "solution.py must pass axis=0 to a dpnp reduction (sum/mean/std) for per-column computation"
    )


def test_signature_matches_reference():
    ref_sig = _sig(_run(REFERENCE))
    sol_sig = _sig(_run(SOLUTION))
    denom = max(abs(ref_sig), 1e-15)
    rel_err = abs(sol_sig - ref_sig) / denom
    assert rel_err <= REL_TOL, (
        f"dpnp sig {sol_sig} differs from reference {ref_sig} "
        f"(relative error {rel_err:.2e} > {REL_TOL})"
    )

#!/usr/bin/env python3
"""Verifier for dpnp-migration-replace-numpy."""
import ast
import re
import subprocess
from pathlib import Path

SOLUTION = Path("/app/solution.py")
REFERENCE = Path("/app/reference.py")
TIMEOUT_SEC = 60.0
# dpnp on CPU uses float64 via oneMKL; rel error vs NumPy is at ULP level (~1e-15).
# 1e-9 is intentionally generous to tolerate any future backend differences.
REL_TOL = 1e-9
# hist_sig uses bins='auto' (intentionally — it exercises the dpnp->NumPy
# fallback for an unsupported string bin selector). Freedman-Diaconis derives the
# bin *count* from the data range/IQR, so a ULP-level difference between the
# NumPy reference and the dpnp solution's y can tip the count by +/-1 and shift
# hist_sig discontinuously (measured ~0.44 for a 71->72 flip, up to ~0.59 for
# +/-2, against a baseline magnitude ~0.28). hist_sig is also a near-symmetric
# sum that can sit close to zero, so a pure relative gate is fragile. Gate it on
# abs OR rel tolerance: HIST_ABS_TOL absorbs a discrete bin-count flip with
# margin while still rejecting a genuinely wrong pipeline (off by far more).
HIST_ABS_TOL = 1.0
HIST_REL_TOL = 1e-6


def _run(script):
    result = subprocess.run(
        ["python3", str(script)], capture_output=True, text=True, timeout=TIMEOUT_SEC
    )
    assert result.returncode == 0, (
        f"{script} exit={result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "VALID" in result.stdout, f"expected VALID in stdout, got {result.stdout!r}"
    return result.stdout


def _parse_stats(text):
    """Parse mean, std, min, max, hist_sig from output."""
    pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    mean_m = re.search(rf"mean={pattern}", text)
    std_m = re.search(rf"std={pattern}", text)
    min_m = re.search(rf"min={pattern}", text)
    max_m = re.search(rf"max={pattern}", text)
    hist_m = re.search(rf"hist_sig={pattern}", text)

    assert mean_m, f"no mean=<value> found in {text!r}"
    assert std_m, f"no std=<value> found in {text!r}"
    assert min_m, f"no min=<value> found in {text!r}"
    assert max_m, f"no max=<value> found in {text!r}"
    assert hist_m, f"no hist_sig=<value> found in {text!r}"

    return {
        "mean": float(mean_m.group(1)),
        "std": float(std_m.group(1)),
        "min": float(min_m.group(1)),
        "max": float(max_m.group(1)),
        "hist_sig": float(hist_m.group(1)),
    }


def _uses_dpnp_calls(tree):
    """True if solution uses dpnp API calls (not just imports it)."""
    module_aliases: set[str] = set()
    direct_funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dpnp":
                    module_aliases.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "dpnp":
            for alias in node.names:
                direct_funcs.add(alias.asname or alias.name)

    if not module_aliases and not direct_funcs:
        return False

    DPNP_APIS = {"arange", "sin", "cos", "mean", "std", "min", "max", "asnumpy"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in DPNP_APIS
                and isinstance(func.value, ast.Name)
                and func.value.id in module_aliases
            ):
                found.add(func.attr)
            elif isinstance(func, ast.Name) and func.id in direct_funcs & DPNP_APIS:
                found.add(func.id)
    return len(found) >= 2


def test_files_exist():
    assert SOLUTION.exists(), f"{SOLUTION} not found"
    assert REFERENCE.exists(), f"{REFERENCE} not found"


def test_solution_imports_dpnp():
    tree = ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))
    assert _uses_dpnp_calls(tree), (
        "solution.py must import dpnp and use it for real computations (arange/sin/cos/mean/std); "
        "a plain NumPy solution does not satisfy the task"
    )


def test_signature_matches_reference():
    ref_stats = _parse_stats(_run(REFERENCE))
    sol_stats = _parse_stats(_run(SOLUTION))

    # Smooth reductions must match tightly; hist_sig is binned and gets a
    # dedicated abs-OR-rel gate (see HIST_* tolerances above).
    for metric in ["mean", "std", "min", "max"]:
        ref_val = ref_stats[metric]
        sol_val = sol_stats[metric]
        denom = max(abs(ref_val), 1e-15)
        rel_err = abs(sol_val - ref_val) / denom
        assert rel_err <= REL_TOL, (
            f"{metric}: solution {sol_val:.9e} differs from reference {ref_val:.9e} "
            f"(relative error {rel_err:.2e} > {REL_TOL})"
        )

    ref_hist = ref_stats["hist_sig"]
    sol_hist = sol_stats["hist_sig"]
    abs_err = abs(sol_hist - ref_hist)
    rel_err = abs_err / max(abs(ref_hist), 1e-15)
    assert abs_err <= HIST_ABS_TOL or rel_err <= HIST_REL_TOL, (
        f"hist_sig: solution {sol_hist:.9e} differs from reference {ref_hist:.9e} "
        f"(abs error {abs_err:.2e} > {HIST_ABS_TOL} and rel error {rel_err:.2e} > {HIST_REL_TOL})"
    )

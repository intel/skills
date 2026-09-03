#!/usr/bin/env python3
"""Verifier for mkl-random-parallel-streams.

The expected MT2203 draws are reconstructed here, in this process, and compared
bitwise against what the solution reports. numpy.random cannot produce them, so
sampling from the wrong generator fails even if the Monte Carlo estimate lands
near the analytic target.
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
STREAMS = APP / "streams.json"
TIMEOUT_SEC = 300.0
ANALYTIC = 0.25
TOL = 0.01
N_WORKERS = 4
SEED = 77777
BRNG = "MT2203"
BATCH = 250_000


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


def _expected_first_draws(n_workers):
    """Reconstruct the first 8 doubles each MT2203 member yields."""
    import mkl_random as rnd

    return [
        list(rnd.MKLRandomState(SEED, brng=(BRNG, i)).rand(8))
        for i in range(n_workers)
    ]


def _expected_estimate(n_workers, batch=BATCH):
    """Reproduce the whole computation: first8 then batch, per worker, in order."""
    import mkl_random as rnd

    sys.path.insert(0, str(APP))
    from reference import mc_prob

    total = 0.0
    for i in range(n_workers):
        rs = rnd.MKLRandomState(SEED, brng=(BRNG, i))
        rs.rand(8)                      # consumed first, as the task specifies
        total += mc_prob(rs.rand(2, batch))
    return total / n_workers


def _tree():
    return ast.parse(SOLUTION.read_text(errors="replace"), filename=str(SOLUTION))


def _reachable_calls(tree):
    """Call names reachable from module scope, run(), or worker_streams()."""
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    entry = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for name in ("run", "worker_streams"):
        if name in funcs:
            entry.append(funcs[name])

    seen, names = set(), set()
    stack = list(entry)
    while stack:
        for sub in ast.walk(stack.pop()):
            if isinstance(sub, ast.Call):
                fn = sub.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                names.add(nm)
                if nm in funcs and nm not in seen:
                    seen.add(nm)
                    stack.append(funcs[nm])
    return names


def test_files_exist():
    assert SOLUTION.exists(), f"{SOLUTION} not found"
    assert REFERENCE.exists(), f"{REFERENCE} not found"


def test_solution_uses_mkl_random_explicitly():
    tree = _tree()
    imported = any(
        (isinstance(n, ast.Import)
         and any(a.name.split(".")[0] == "mkl_random" for a in n.names))
        or (isinstance(n, ast.ImportFrom)
            and (n.module or "").startswith("mkl_random"))
        for n in ast.walk(tree)
    )
    assert imported, "solution.py must import mkl_random"

    patched = {"patch_numpy_random", "mkl_random"} & _reachable_calls(tree)
    assert not patched, (
        "solution.py must construct generators explicitly, not patch numpy.random"
    )


def test_solution_reuses_the_reference_estimator():
    reused = any(
        isinstance(n, ast.ImportFrom) and (n.module or "") == "reference"
        for n in ast.walk(_tree())
    )
    assert reused, (
        "solution.py must import the estimator from reference.py so that only "
        "the random source differs"
    )


def test_streams_are_the_expected_mt2203_members():
    """Load-bearing: the reported draws must be MT2203's, reconstructed here.

    This is what makes the task unspoofable. numpy.random cannot reproduce these
    values, and neither can a different brng or seed.
    """
    _run(SOLUTION, str(N_WORKERS))
    assert STREAMS.exists(), f"{STREAMS} not written"
    data = json.loads(STREAMS.read_text())
    reported = data.get("workers")
    assert isinstance(reported, list) and len(reported) == N_WORKERS, (
        f"expected {N_WORKERS} per-worker draw lists, got "
        f"{len(reported) if isinstance(reported, list) else type(reported)}"
    )

    expected = _expected_first_draws(N_WORKERS)
    for i, (got, want) in enumerate(zip(reported, expected)):
        assert len(got) >= 8, f"worker {i} reported fewer than 8 draws"
        assert [float(x) for x in got[:8]] == want, (
            f"worker {i} draws do not match MT2203 member {i} seeded with {SEED}. "
            f"got {got[:3]}..., expected {want[:3]}... -- the draws did not come "
            f"from the required mkl_random stream"
        )


def test_streams_are_pairwise_distinct():
    _run(SOLUTION, str(N_WORKERS))
    workers = json.loads(STREAMS.read_text())["workers"]
    seen = set()
    for i, draws in enumerate(workers):
        key = tuple(round(v, 12) for v in draws[:8])
        assert key not in seen, (
            f"worker {i} produced draws identical to an earlier worker: the "
            f"streams are not independent"
        )
        seen.add(key)


def test_estimate_reproduces_the_expected_computation():
    """The estimate must be the one this exact stream sequence produces."""
    out = _run(SOLUTION, str(N_WORKERS))
    got = _field(out, "estimate", float)
    expected = _expected_estimate(N_WORKERS)
    assert abs(got - expected) < 5e-5, (
        f"estimate {got} does not reproduce the expected MT2203 computation "
        f"{expected:.6f}; the draw sequence differs from the one the task specifies"
    )
    assert abs(got - ANALYTIC) <= TOL, (
        f"estimate {got} is not within {TOL} of the analytic {ANALYTIC}"
    )


def test_reference_also_matches_analytic():
    """Guards the premise: the workload itself converges."""
    estimate = _field(_run(REFERENCE, str(N_WORKERS)), "estimate", float)
    assert abs(estimate - ANALYTIC) <= TOL


def test_reported_worker_count_and_brng():
    out = _run(SOLUTION, "6")
    assert _field(out, "workers", int) == 6
    brng = _field(out, "brng").upper().strip("()',\"")
    assert BRNG in brng, f"expected the {BRNG} family, got {brng!r}"
    data = json.loads(STREAMS.read_text())
    assert len(data["workers"]) == 6, "streams.json not rewritten for 6 workers"
    expected = _expected_first_draws(6)
    assert [float(x) for x in data["workers"][5][:8]] == expected[5], (
        "worker 5 draws do not match MT2203 member 5: the member id must track "
        "the worker index"
    )


def test_rejects_invalid_worker_count():
    for bad in ("1", "0", "65"):
        result = subprocess.run(
            [sys.executable, str(SOLUTION), bad],
            capture_output=True, text=True, timeout=TIMEOUT_SEC, cwd=str(APP),
        )
        assert result.returncode != 0, (
            f"solution must exit non-zero for n_workers={bad}"
        )

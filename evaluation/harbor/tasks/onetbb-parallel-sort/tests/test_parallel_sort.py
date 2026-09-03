#!/usr/bin/env python3
"""
Tests for the oneTBB parallel_sort task.
Verifies:
  1. Binary /app/sort_parallel exists and is executable.
  2. It prints "SORTED" and exits 0.
  3. It completes within 5 seconds (wall clock).
  4. The source code references tbb (i.e. agent actually used oneTBB).
"""
import os
import subprocess
import time
from pathlib import Path

import pytest

BINARY = Path("/app/sort_parallel")
SOURCE_CANDIDATES = [
    Path("/app/sort_parallel.cpp"),
    Path("/app/sort_benchmark.cpp"),  # agent may have modified the original
]
TIMEOUT_SEC = 5.0


def test_binary_exists():
    """The compiled binary must exist at /app/sort_parallel."""
    assert BINARY.exists(), f"Binary {BINARY} not found — did the agent compile it?"
    assert os.access(BINARY, os.X_OK), f"{BINARY} is not executable"


@pytest.fixture(scope="module")
def binary_result():
    """Run the binary once and cache stdout, returncode, and elapsed time."""
    start = time.perf_counter()
    result = subprocess.run(
        [str(BINARY)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SEC + 10,  # generous timeout for slow machines
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def test_binary_runs_and_prints_sorted(binary_result):
    """Running the binary must print SORTED and exit 0."""
    result, _ = binary_result
    assert result.returncode == 0, (
        f"Binary exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SORTED" in result.stdout, (
        f"Expected 'SORTED' in stdout, got: {result.stdout!r}"
    )


def test_binary_completes_within_timeout(binary_result):
    """The parallel sort must finish within {TIMEOUT_SEC}s wall-clock time."""
    _, elapsed = binary_result
    assert elapsed < TIMEOUT_SEC, (
        f"Sort took {elapsed:.2f}s — exceeds {TIMEOUT_SEC}s limit. "
        "Make sure you are using tbb::parallel_sort, not std::sort."
    )


def test_source_uses_tbb():
    """At least one source file must reference oneTBB headers/API."""
    tbb_keywords = [
        "tbb::parallel_sort",
        "tbb/parallel_sort",
        "oneapi/tbb",
    ]
    found = False
    checked = []
    for src in SOURCE_CANDIDATES:
        if src.exists():
            text = src.read_text(errors="replace")
            checked.append(str(src))
            if any(kw in text for kw in tbb_keywords):
                found = True
                break

    # Also scan /app for any .cpp that mentions tbb
    if not found:
        for cpp in Path("/app").glob("*.cpp"):
            text = cpp.read_text(errors="replace")
            checked.append(str(cpp))
            if any(kw in text for kw in tbb_keywords):
                found = True
                break

    assert found, (
        f"No TBB usage found in sources: {checked}. "
        "The solution must use tbb::parallel_sort (include oneapi/tbb or tbb/parallel_sort.h)."
    )

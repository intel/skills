#!/usr/bin/env python3
"""Verifier for onetbb-transpose."""
import os
import re
import subprocess
import time
from pathlib import Path

BINARY = Path("/app/transpose_tbb")
SERIAL = Path("/app/transpose_serial")
TARGET_SOURCE = Path("/app/transpose_tbb.cpp")
ARGS = ['1024', '10', '32']
KEYWORDS = ['parallel_for', 'blocked_range2d', 'oneapi/tbb']
TIMEOUT_SEC = 8.0


def _run(cmd):
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC + 5)
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, f"{cmd} exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "VALID" in result.stdout, f"expected VALID in stdout, got {result.stdout!r}"
    return result.stdout, elapsed


def _last_number(text):
    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    assert nums, f"no numeric validation value found in {text!r}"
    return float(nums[-1])


def test_binary_exists():
    assert BINARY.exists(), f"{BINARY} not found; compile the required oneTBB binary"
    assert os.access(BINARY, os.X_OK), f"{BINARY} is not executable"
    assert SERIAL.exists(), f"{SERIAL} not found; serial reference binary is required"
    assert os.access(SERIAL, os.X_OK), f"{SERIAL} is not executable"


def test_binary_matches_serial_reference():
    serial_args = ARGS[:2]
    serial_out, _ = _run([str(SERIAL), *serial_args])
    parallel_out, elapsed = _run([str(BINARY), *ARGS])
    serial_value = _last_number(serial_out)
    parallel_value = _last_number(parallel_out)
    tolerance = max(1e-6, abs(serial_value) * 1e-8)
    assert abs(serial_value - parallel_value) <= tolerance, (
        f"parallel result {parallel_value} differs from serial reference {serial_value}"
    )
    assert elapsed < TIMEOUT_SEC, f"execution took {elapsed:.2f}s, expected under {TIMEOUT_SEC}s"


def test_source_uses_onetbb():
    assert TARGET_SOURCE.exists(), f"{TARGET_SOURCE} not found"
    text = TARGET_SOURCE.read_text(errors="replace")
    missing = [kw for kw in KEYWORDS if kw not in text]
    assert not missing, f"missing required oneTBB/source markers {missing} in {TARGET_SOURCE}"

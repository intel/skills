#!/usr/bin/env python3
"""Verifier for onetbb-parallel-scan: parallel_scan prefix sum."""
import os
import re
import subprocess
import time
from pathlib import Path

BINARY = Path("/app/scan_tbb")
SERIAL = Path("/app/scan_serial")
TARGET_SOURCE = Path("/app/scan_tbb.cpp")
ARGS = ['3000000']
KEYWORDS = ['parallel_scan', 'is_final_scan', 'oneapi/tbb']
TIMEOUT_SEC = 8.0


def _run(cmd):
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC + 5)
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, f"{cmd} exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "VALID" in result.stdout, f"expected VALID in stdout, got {result.stdout!r}"
    return result.stdout, elapsed


def _last_integer(text):
    nums = re.findall(r"\d+", text)
    assert nums, f"no integer validation value found in {text!r}"
    return int(nums[-1])


def test_binary_exists():
    assert BINARY.exists(), f"{BINARY} not found; compile the required oneTBB binary"
    assert os.access(BINARY, os.X_OK), f"{BINARY} is not executable"
    assert SERIAL.exists(), f"{SERIAL} not found; serial reference binary is required"
    assert os.access(SERIAL, os.X_OK), f"{SERIAL} is not executable"


def test_binary_matches_serial_reference_exactly():
    serial_out, _ = _run([str(SERIAL), *ARGS])
    parallel_out, elapsed = _run([str(BINARY), *ARGS])
    serial_value = _last_integer(serial_out)
    parallel_value = _last_integer(parallel_out)
    assert parallel_value == serial_value, (
        f"parallel signature {parallel_value} differs from serial reference {serial_value}"
    )
    assert elapsed < TIMEOUT_SEC, f"execution took {elapsed:.2f}s, expected under {TIMEOUT_SEC}s"


def test_source_uses_required_onetbb_api():
    assert TARGET_SOURCE.exists(), f"{TARGET_SOURCE} not found"
    text = TARGET_SOURCE.read_text(errors="replace")
    missing = [kw for kw in KEYWORDS if kw not in text]
    assert not missing, f"missing required oneTBB/source markers {missing} in {TARGET_SOURCE}"

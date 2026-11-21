"""Verifier for xe2-attention-tile-build.

Builds the agent's /app/attention.hpp into a harness ahead of time for Battlemage and
checks the generated GPU code. When an Intel GPU is visible in the container it also
runs the kernel and compares the result with a CPU reference.
"""
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HEADER = Path("/app/attention.hpp")
WORK = Path("/tmp/verify")
ISA_DIR = WORK / "isa"
BINARY = WORK / "harness"


@pytest.fixture(scope="session")
def build():
    assert HEADER.is_file(), "/app/attention.hpp does not exist"
    shutil.rmtree(WORK, ignore_errors=True)
    ISA_DIR.mkdir(parents=True)
    shutil.copy("/tests/harness.cpp", WORK / "harness.cpp")
    env = dict(os.environ, IGC_ShaderDumpEnable="1", IGC_DumpToCustomDir=str(ISA_DIR))
    cmd = ["icpx", "-fsycl", "-fsycl-targets=spir64_gen", "-Xs", "-device bmg", "-O2",
           "-I/app", str(WORK / "harness.cpp"), "-o", str(BINARY)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    log = proc.stdout + proc.stderr
    asm = {}
    for f in glob.glob(str(ISA_DIR / "*.asm")):
        name = os.path.basename(f)
        if name.startswith(("VC_asm", "OCL_asm")):
            # instruction lines only: drop comments
            asm[name] = "\n".join(l for l in Path(f).read_text(errors="replace").splitlines()
                                  if l.strip() and not l.lstrip().startswith("//"))
    return proc.returncode, log, asm


def count(asm, pattern):
    return sum(len(re.findall(pattern, text)) for text in asm.values())


def test_builds_for_battlemage(build):
    rc, log, asm = build
    assert rc == 0, f"build failed:\n{log[-3000:]}"
    assert asm, "build produced no GPU kernel assembly"


def test_uses_xmx_matrix_engine(build):
    rc, _, asm = build
    assert rc == 0
    assert count(asm, r"\bdpas\.") > 0, "no DPAS (XMX) instruction in the generated code"


def test_vnni_operand_from_memory(build):
    """The matrix-engine operand that needs VNNI order is produced by the memory
    subsystem (a transforming 2D block load), not rearranged in registers."""
    rc, _, asm = build
    assert rc == 0
    assert count(asm, r"\bload_block2d\.ugm\.d(8|16)v\b") > 0, \
        "no VNNI-transforming 2D block load in the generated code"


def test_combines_through_work_group_local_memory(build):
    rc, _, asm = build
    assert rc == 0
    assert count(asm, r"\.slm\.") > 0, "no shared local memory access in the generated code"
    assert count(asm, r"\bsync\.bar\b") > 0, "no work-group barrier in the generated code"


def test_no_register_spills(build):
    rc, log, asm = build
    assert rc == 0
    assert not re.search(r"spill", log, re.IGNORECASE), f"compiler reports spills:\n{log[-2000:]}"
    assert count(asm, r"\b(load|store)\.scratch\b") == 0, "scratch (spill) memory traffic in the generated code"


def gpu_visible():
    if not glob.glob("/dev/dri/renderD*"):
        return False
    proc = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=120)
    return "level_zero:gpu" in proc.stdout


@pytest.mark.skipif(not gpu_visible(), reason="no Intel GPU visible in the container")
def test_numerics_on_gpu(build):
    rc, _, _ = build
    assert rc == 0
    proc = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert "RESULT PASS" in out, f"numerical check failed:\n{out[-2000:]}"

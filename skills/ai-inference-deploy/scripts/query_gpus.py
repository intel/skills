#!/usr/bin/env python3
"""Query the Intel GPUs on this host. Read-only; prints JSON, one entry per device.

Nothing here is a table of known parts: every value comes from a tool that answered.
  torch.xpu   name, total_memory          (any Intel GPU PyTorch sees)
  xpu-smi     name, memory, PCI BDF and state (cards XPU Manager supports)
  OpenCL      CL_DEVICE_HOST_UNIFIED_MEMORY -> integrated if true;
              CL_DEVICE_GLOBAL_MEM_SIZE as the last memory fallback (Windows, or no torch/xpu-smi)
A unified-memory device shares system RAM, so its budget is capped by available RAM
(GlobalMemoryStatusEx on Windows, MemAvailable from /proc/meminfo elsewhere).
"""
import ctypes
import json
import re
import shutil
import sys
import subprocess


def run(cmd):
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def from_torch():
    try:
        import torch
        if not torch.xpu.is_available():
            return []
        props = [torch.xpu.get_device_properties(i) for i in range(torch.xpu.device_count())]
        return [{"name": p.name, "memory_bytes": p.total_memory} for p in props]
    except Exception:
        return []


def json_or_empty(text):
    try:
        return json.loads(text or "{}")
    except ValueError:  # an error message or a warning instead of JSON
        return {}


def from_xpu_smi():
    out = []
    for dev in json_or_empty(run(["xpu-smi", "discovery", "-j"])).get("device_list", []):
        if "device_id" not in dev:
            continue
        detail = json_or_empty(run(["xpu-smi", "discovery", "-d", str(dev["device_id"]), "-j"]))
        mem = detail.get("memory_physical_size_byte")
        out.append({"name": dev.get("device_name"), "bdf": dev.get("pci_bdf_address"),
                    "memory_bytes": int(mem) if mem else None, "state": dev.get("device_state")})
    return out


def from_opencl():
    """[{name, unified, memory_bytes}] for Intel GPUs, read through the OpenCL loader library
    itself, so no clinfo binary is needed."""
    try:
        cl = ctypes.WinDLL("OpenCL.dll") if sys.platform == "win32" else ctypes.CDLL("libOpenCL.so.1")
    except (OSError, AttributeError):
        return []
    cl.clGetPlatformIDs.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    cl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    cl.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t,
                                   ctypes.c_void_p, ctypes.c_void_p]
    for f in (cl.clGetPlatformIDs, cl.clGetDeviceIDs, cl.clGetDeviceInfo):
        f.restype = ctypes.c_int32
    def ids(get, *args):
        n = ctypes.c_uint32()
        if get(*args, 0, None, ctypes.byref(n)) or not n.value:
            return []
        arr = (ctypes.c_void_p * n.value)()
        return list(arr) if get(*args, n.value, arr, None) == 0 else []
    def info(dev, param, ctype):
        val = ctype()
        return None if cl.clGetDeviceInfo(dev, param, ctypes.sizeof(val), ctypes.byref(val), None) else val
    out = []
    for plat in ids(cl.clGetPlatformIDs):
        for dev in ids(cl.clGetDeviceIDs, plat, 4):  # CL_DEVICE_TYPE_GPU
            vendor = info(dev, 0x1001, ctypes.c_uint32)  # CL_DEVICE_VENDOR_ID
            if vendor is None or vendor.value != 0x8086:  # Intel's PCI vendor ID
                continue
            name = info(dev, 0x102B, ctypes.c_char * 256)  # CL_DEVICE_NAME
            unified = info(dev, 0x1035, ctypes.c_uint32)  # CL_DEVICE_HOST_UNIFIED_MEMORY
            mem = info(dev, 0x101F, ctypes.c_uint64)  # CL_DEVICE_GLOBAL_MEM_SIZE
            out.append({"name": name.value.decode(errors="replace").strip() if name else None,
                        "unified": bool(unified.value) if unified else None,
                        "memory_bytes": mem.value or None if mem else None})
    return out


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def mem_available():
    if sys.platform == "win32":
        try:
            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))  # the call fails without it
            return stat.ullAvailPhys if ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(stat)) else None
        except (OSError, AttributeError):
            return None
    try:
        text = open("/proc/meminfo").read()
        return int(re.search(r"MemAvailable:\s+(\d+) kB", text).group(1)) * 1024
    except (OSError, AttributeError):
        return None


def pick(entries, i, name, used):
    """The entry for device i, never one already handed to another device. Prefer the same
    index with the same name (identical cards), then any entry with that name, then the same
    index (the tools can spell one GPU's name differently)."""
    same = [i] if i < len(entries) else []
    named = [j for j, e in enumerate(entries) if e["name"] == name]
    for j in [j for j in same if j in named] + named + same:
        if j not in used:
            used.add(j)
            return entries[j]
    return {}


def main():
    torch_devs, smi_devs, cl_devs = from_torch(), from_xpu_smi(), from_opencl()
    avail = mem_available()
    devices, smi_used, cl_used = [], set(), set()
    for i, dev in enumerate(torch_devs or smi_devs or cl_devs):
        smi = pick(smi_devs, i, dev["name"], smi_used)
        cl = pick(cl_devs, i, dev["name"], cl_used)
        entry = {"index": i, "name": dev["name"], "bdf": smi.get("bdf"),
                 "memory_bytes": smi.get("memory_bytes") or dev.get("memory_bytes") or cl.get("memory_bytes"),
                 "unified_memory": cl.get("unified"),
                 "state": smi.get("state"),
                 "source": [s for s, hit in (("torch", torch_devs), ("xpu-smi", smi), ("opencl", cl)) if hit]}
        if entry["unified_memory"] and entry["memory_bytes"] and avail:
            entry["memory_bytes"] = min(entry["memory_bytes"], avail)
        entry["class"] = {True: "igpu", False: "dgpu"}.get(entry["unified_memory"], "unknown")
        devices.append(entry)
    print(json.dumps({"devices": devices, "mem_available_bytes": avail}, indent=2))


if __name__ == "__main__":
    main()

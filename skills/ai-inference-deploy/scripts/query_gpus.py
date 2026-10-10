#!/usr/bin/env python3
"""Query the Intel GPUs on this host. Read-only; prints JSON, one entry per device.

Nothing here is a table of known parts: every value comes from a tool that answered.
  torch.xpu   name, total_memory          (any Intel GPU PyTorch sees)
  xpu-smi     name, memory, PCI BDF, device type and state (cards XPU Manager supports)
  OpenCL      CL_DEVICE_HOST_UNIFIED_MEMORY -> integrated if true
A unified-memory device shares system RAM, so its budget is capped by MemAvailable.
"""
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


def from_xpu_smi():
    out = []
    for dev in json.loads(run(["xpu-smi", "discovery", "-j"]) or "{}").get("device_list", []):
        detail = json.loads(run(["xpu-smi", "discovery", "-d", str(dev["device_id"]), "-j"]) or "{}")
        mem = detail.get("memory_physical_size_byte")
        kind = (dev.get("device_type") or "").lower()
        out.append({"name": dev.get("device_name"), "bdf": dev.get("pci_bdf_address"),
                    "memory_bytes": int(mem) if mem else None, "state": dev.get("device_state"),
                    "unified": True if "integrated" in kind else False if "discrete" in kind else None})
    return out


def from_opencl():
    """{device name: unified?} from OpenCL's CL_DEVICE_HOST_UNIFIED_MEMORY, via the loader
    library itself, so no clinfo binary is needed."""
    import ctypes
    try:
        cl = ctypes.CDLL("OpenCL.dll" if sys.platform == "win32" else "libOpenCL.so.1")
    except OSError:
        return {}
    def ids(get, *args):
        n = ctypes.c_uint32()
        if get(*args, 0, None, ctypes.byref(n)) or not n.value:
            return []
        arr = (ctypes.c_void_p * n.value)()
        get(*args, n.value, arr, None)
        return list(arr)
    def info(dev, param, ctype):
        val = ctype()
        cl.clGetDeviceInfo(ctypes.c_void_p(dev), param, ctypes.sizeof(val), ctypes.byref(val), None)
        return val
    unified = {}
    for plat in ids(cl.clGetPlatformIDs):
        for dev in ids(cl.clGetDeviceIDs, ctypes.c_void_p(plat), ctypes.c_uint64(4)):  # GPU
            if info(dev, 0x1001, ctypes.c_uint32).value != 0x8086:  # vendor: Intel
                continue
            name = info(dev, 0x102B, ctypes.c_char * 256).value.decode().strip()
            unified[name] = bool(info(dev, 0x1035, ctypes.c_uint32).value)  # HOST_UNIFIED_MEMORY
    return unified


def mem_available():
    try:
        text = open("/proc/meminfo").read()
        return int(re.search(r"MemAvailable:\s+(\d+) kB", text).group(1)) * 1024
    except (OSError, AttributeError):
        return None


def main():
    torch_devs, smi_devs, unified = from_torch(), from_xpu_smi(), from_opencl()
    devices = []
    base = torch_devs or smi_devs or [{"name": n} for n in unified]
    for i, dev in enumerate(base):
        smi = next((s for s in smi_devs if s["name"] == dev["name"]), smi_devs[i] if i < len(smi_devs) else {})
        entry = {"index": i, "name": dev["name"], "bdf": smi.get("bdf"),
                 "memory_bytes": smi.get("memory_bytes") or dev.get("memory_bytes"),
                 "unified_memory": unified.get(dev["name"], smi.get("unified")),
                 "state": smi.get("state"),
                 "source": [s for s, hit in (("torch", torch_devs), ("xpu-smi", smi), ("opencl", dev["name"] in unified)) if hit]}
        if entry["unified_memory"] and entry["memory_bytes"] and (avail := mem_available()):
            entry["memory_bytes"] = min(entry["memory_bytes"], avail)
        entry["class"] = {True: "igpu", False: "dgpu"}.get(entry["unified_memory"], "unknown")
        devices.append(entry)
    print(json.dumps({"devices": devices, "mem_available_bytes": mem_available()}, indent=2))


if __name__ == "__main__":
    main()

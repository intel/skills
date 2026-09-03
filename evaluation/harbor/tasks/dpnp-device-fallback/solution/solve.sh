#!/usr/bin/env bash
set -euo pipefail

cat > /app/solution.py <<'PY'
import dpctl
import dpnp


def get_device():
    try:
        # Use generic "gpu" for portability across backends (Level Zero, OpenCL).
        # If targeting Level Zero only, use "level_zero:gpu" instead.
        # Note: iGPU may not support float64 — fall back to cpu for fp64 precision.
        device = dpctl.SyclDevice("gpu")
        return device
    except Exception:
        return dpctl.SyclDevice("cpu")


def run():
    sycl_device = get_device()
    print(f"INFO device={sycl_device.device_type.name.lower()}")
    x = dpnp.arange(500_000, dtype=dpnp.float64, device=sycl_device)
    y = dpnp.sqrt(dpnp.abs(dpnp.sin(x) * dpnp.cos(x) + 1.0))
    sig = dpnp.sum(y)
    print(f"VALID dpnp sig={sig:.9e}")


if __name__ == "__main__":
    run()
PY

echo "solution written to /app/solution.py"

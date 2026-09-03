# dpnp portable computation with explicit device selection and CPU fallback

You are given `/app/reference.py`, a NumPy program that generates a deterministic
dataset, applies element-wise operations, and prints a validation signature.
The environment intentionally has **no GPU** (simulating a CI/cloud host) —
only a CPU SYCL device is available. The task tests that your code handles
this gracefully: try GPU first, catch the error, fall back to CPU.

Create `/app/solution.py` that runs the same computation portably with **dpnp**,
using proper device management:

1. Use `dpctl` to detect the available SYCL device:
   - Try `dpctl.SyclDevice("gpu")` first.
   - If a GPU is unavailable (raises `dpctl.SyclDeviceCreationError` or any
     exception), fall back to `dpctl.SyclDevice("cpu")`.
   - Print `INFO device=<device_name>` to stdout (e.g. `INFO device=cpu`).
2. Create dpnp arrays on the selected device using `device=` parameter:
   - `x = dpnp.arange(500_000, dtype=dpnp.float64, device=sycl_device)`
3. Compute: `y = dpnp.sqrt(dpnp.abs(dpnp.sin(x) * dpnp.cos(x) + 1.0))`
4. Print a line containing `VALID` and `sig=<value>` where the signature is
   `float(dpnp.sum(y))` formatted to 9 decimal places in scientific notation.

Requirements:

- The solution **must** use `dpctl` to query the device before creating arrays —
  a hardcoded `"cpu"` string without a try/gpu-first attempt does not satisfy
  the task.
- Results must match the NumPy reference within `1e-9` relative tolerance
  (assumes float64 execution on CPU; iGPU devices may not support float64
  and would require a relaxed tolerance of ~1e-6).
- The solution must not crash when no GPU is present.

```bash
python3 /app/reference.py
python3 /app/solution.py
```

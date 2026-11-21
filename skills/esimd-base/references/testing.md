# Testing New ESIMD Kernels

Part of the `esimd-base` skill.

## Testing New Kernels

Every new kernel must have both an **accuracy test** and a **performance test**. Follow these rules:

### Rules

| Rule | Detail |
|------|--------|
| **Always add both tests** | Accuracy test + performance test for every new kernel |
| **Random input** | Use `rand()` / `srand(42)` for both tests — never all-zeros or trivially structured data |
| **Accuracy: small input** | Use a small input size to keep the CPU reference fast |
| **Accuracy: run once** | Run kernel once, run reference once, compare all outputs |
| **Accuracy: absolute + relative error** | Compute both `abs_err = |out - ref|` and `rel_err = abs_err / (|ref| + eps)`. Report both. PASS if either is below its threshold (e.g. abs < 1e-2 **or** rel < 1e-3). Pure absolute error gives false FAILs for large-magnitude outputs and false PASSes for small-magnitude ones. |
| **Accuracy: print worst elements** | For the element with `max_abs_err` and the element with `max_rel_err`, print the index, `out` value, and `ref` value. This immediately shows whether errors are isolated outliers or systematic, and reveals near-zero reference values that inflate relative error. |
| **Perf: large input** | Use a larger input size, with its own buffers, so the whole GPU is busy |
| **Perf: 5 runs, best result** | Run kernel 5 times, report the **minimum** (best) time — no warmup needed |
| **Perf: report FLOPS + BW** | Compute and print both TFLOPS and GB/s where applicable |

### Accuracy Test Pattern

```cpp
// Small input — fast CPU reference
constexpr int ACC_LEN = 1024;  // smaller than perf test

half* in_host  = sycl::malloc_host<half>(ACC_LEN, q);
half* out_host = sycl::malloc_host<half>(ACC_LEN, q);
half* in_dev   = sycl::malloc_device<half>(ACC_LEN, q);
half* out_dev  = sycl::malloc_device<half>(ACC_LEN, q);

// Random input
srand(42);
for (int i = 0; i < ACC_LEN; i++)
    in_host[i] = half(((float)rand() / RAND_MAX) * 2.f - 1.f);

q.memcpy(in_dev, in_host, ACC_LEN * sizeof(half)).wait();

// Run kernel once
my_kernel(q, in_dev, out_dev, ACC_LEN).wait();
q.memcpy(out_host, out_dev, ACC_LEN * sizeof(half)).wait();

// CPU reference once — check both absolute and relative error, record worst elements
double max_abs_err = 0.0, max_rel_err = 0.0;
int    max_abs_idx = 0,   max_rel_idx = 0;
double max_abs_ref = 0.0, max_rel_ref = 0.0;
double max_abs_out = 0.0, max_rel_out = 0.0;
for (int i = 0; i < ACC_LEN; i++) {
    float ref = cpu_reference(in_host, i);
    double abs_err = std::abs((double)out_host[i] - (double)ref);
    double rel_err = abs_err / (std::abs((double)ref) + 1e-6);
    if (abs_err > max_abs_err) {
        max_abs_err = abs_err; max_abs_idx = i;
        max_abs_ref = ref;     max_abs_out = (double)out_host[i];
    }
    if (rel_err > max_rel_err) {
        max_rel_err = rel_err; max_rel_idx = i;
        max_rel_ref = ref;     max_rel_out = (double)out_host[i];
    }
}
// PASS if either abs OR rel is within threshold (handles large/small magnitude outputs)
bool pass = max_abs_err < 1e-2 || max_rel_err < 1e-3;
printf("  accuracy: max_abs_err=%.6f (idx=%d out=%.6f ref=%.6f)"
       "  max_rel_err=%.6f (idx=%d out=%.6f ref=%.6f)  %s\n",
       max_abs_err, max_abs_idx, max_abs_out, max_abs_ref,
       max_rel_err, max_rel_idx, max_rel_out, max_rel_ref,
       pass ? "PASS" : "FAIL");
```

### Performance Test Pattern

```cpp
// Large input with its own buffers (the accuracy buffers hold only ACC_LEN elements)
constexpr int PERF_LEN = 1 << 20;  // larger than accuracy test
constexpr int RUNS = 5;

half* perf_in_host = sycl::malloc_host<half>(PERF_LEN, q);
half* perf_in_dev  = sycl::malloc_device<half>(PERF_LEN, q);
half* perf_out_dev = sycl::malloc_device<half>(PERF_LEN, q);

// Random input (same srand(42) for reproducibility)
srand(42);
for (int i = 0; i < PERF_LEN; i++)
    perf_in_host[i] = half(((float)rand() / RAND_MAX) * 2.f - 1.f);
q.memcpy(perf_in_dev, perf_in_host, PERF_LEN * sizeof(half)).wait();

// Run 5 times, keep best (use profiling events for kernel-only time)
double best_ns = std::numeric_limits<double>::max();
for (int r = 0; r < RUNS; r++) {
    sycl::event ev = my_kernel(q, perf_in_dev, perf_out_dev, PERF_LEN);
    ev.wait();
    double ns = (double)(
        ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
        ev.get_profiling_info<sycl::info::event_profiling::command_start>());
    if (ns < best_ns) best_ns = ns;
}

double ms    = best_ns * 1e-6;
double bytes = (double)(PERF_LEN) * sizeof(half) * 2;  // read + write
double bw_gb = (bytes / (best_ns * 1e-9)) / 1e9;
double flops = 2.0 * PERF_LEN;  // adjust per kernel
double tflops= (flops / (best_ns * 1e-9)) / 1e12;

printf("  perf: %.3f ms   %.2f TFLOPS   %.1f GB/s\n", ms, tflops, bw_gb);
```

### Notes

- Use `sycl::property::queue::enable_profiling{}` on the queue for event-based timing.
- Event profiling measures **kernel-only** time, excluding host↔device memcpy; use it rather than wall-clock timing to measure a kernel.
- For kernels with no FLOP count (e.g., pure memory ops like topk), report only GB/s.
- The "best of 5" policy avoids GPU scheduler noise without requiring a separate warmup pass.
- Always check `max_err < threshold` appropriate to FP16 precision (typically `< 1e-2` for accumulating kernels).

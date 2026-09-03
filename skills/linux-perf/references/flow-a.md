<!-- (C) 2026 Intel Corporation, MIT license -->
# Flow A: execution steps

### Interpret the output

| Metric | Signal |
|--------|--------|
| Instructions per cycle (IPC) | < 1.0 → stalled; > 3.0 → efficient |
| Cache-miss rate (misses/refs) | > 3–5% → memory-bound |
| Branch-miss rate (misses/branches) | > 3–5% → branch-heavy |
| % of time in kernel (task-clock) | > 30% → syscall/I/O bound |

After presenting the output, always add an interpretation section. Apply whichever patterns match:

- **IPC < 1.0 + cache-miss rate > 3%** → **memory-bound**: data layout, cache blocking, prefetching
- **IPC < 1.0 + branch-miss rate > 3%** → **branch-heavy**: branch elimination, `cmov`, prediction hints
- **High kernel time (> 30% of task-clock)** → **syscall/I/O bound**: reduce syscall frequency, batch I/O, use async
- **IPC ≥ 3.0 + low miss rates** → **CPU-efficient**: the bottleneck is likely elsewhere
- **No single dominant pattern** → report the raw numbers and suggest Flow B

---

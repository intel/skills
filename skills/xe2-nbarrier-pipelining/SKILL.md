---
name: xe2-nbarrier-pipelining
description: >
  Named (split) barriers in Intel ESIMD (SYCL explicit SIMD) kernels for Intel
  GPUs such as Arc (Xe2, Battlemage) and Lunar Lake: named_barrier_init,
  named_barrier_signal and named_barrier_wait, the raw __esimd_nbarrier*
  intrinsics found in existing kernels, and the pattern of putting independent
  work (global loads, prefetches, register-only math) between signal and wait
  to hide latency. Use it when splitting a work-group barrier, pipelining work
  across a sync point, or reading kernels that use named barriers. Not for basic
  shared local memory use or a plain barrier(): see esimd-lsc-slm.
compatibility: "Requires the oneAPI DPC++ compiler (icpx -fsycl). Code checked with DPC++ 2025.3.3 and 2026.1.1 on an Intel Arc B580 (Xe2)."
---

# Named Barrier Pipelining in ESIMD (Intel Xe2)

A `barrier()` makes every thread of the work-group stall until all have arrived. A named barrier splits that into two calls:

1. **signal** (non-blocking): "my writes before this point are done"; the thread keeps running.
2. **wait** (blocking): stalls until every thread has signalled.

Work placed between signal and wait runs while slower threads are still finishing, instead of being stall time.

## API

Functions in `sycl::ext::intel::esimd` (there is no `nbarrier` class):

```cpp
named_barrier_init<NUM_BARRIERS + 1>();                // once, at kernel start (id 0 is barrier()'s)
named_barrier_signal(id, 0, NUM_THREADS, NUM_THREADS); // id, mode, producers, consumers
named_barrier_wait(id);                                // blocks until all have signalled
```

- With mode 0 and producers = consumers = the work-group's thread count, every thread signals and waits: a split version of `barrier()`. This is the form this skill covers. The header documents only 0x1 (producer only) and 0x2 (consumer only); intel/llvm's ESIMD named-barrier tests use 0 for threads that are both.
- `named_barrier_signal` issues a memory fence before signalling by default, so the thread's prior SLM and global writes are visible after the wait.
- Barrier ids go from 1 to `NUM_BARRIERS`: id 0 is reserved for the work-group `barrier()` (as in intel/llvm's ESIMD tests), and `named_barrier_init` counts it. Use a separate id for each independent sync point.
- The oneAPI headers document these functions as available only on PVC, but they work on Xe2 (verified on an Arc B580 with oneAPI DPC++ 2026.1).

Existing kernels often call the intrinsics behind these functions directly:

| Raw intrinsic | Public function |
|---|---|
| `__esimd_nbarrier_init(N)` | `named_barrier_init<N>()` |
| `__esimd_nbarrier_arrive(id, mode, producers, consumers)` | `named_barrier_signal(id, mode, producers, consumers)` (adds the fence) |
| `__esimd_nbarrier(0, id, count)` | `named_barrier_wait(id)` (the first argument 0 means wait) |

## Pattern: work between signal and wait

```cpp
named_barrier_init<2>();                                 // one named barrier, id 1
// ... compute this thread's part of a tile ...
slm_block_store<half, 128>(my_offset, s_tile);          // others read this after the wait
named_barrier_signal(1, 0, NUM_THREADS, NUM_THREADS);

// Independent work: reads no SLM that others are still writing,
// writes no SLM that others read after the wait.
simd<half, 256> v_tile = load_2d<half, 16, 16, 1, false, true>(V, w, h, p, x, y, hints);
prefetch_2d<half, 16, 16, 1>(K, w, h, p, x, y + 16, hints);   // next iteration
acc *= scale;                                            // registers only

named_barrier_wait(1);
simd<half, 128> s_other = slm_block_load<half, 128>(other_offset);
```

Good candidates for the window are global loads and prefetches needed soon after the wait, and math on the thread's own registers.

## Rules

- **Every thread signals and waits**, the same number of times. A thread that skips either call leaves the others waiting forever: the kernel hangs.
- **One wait per signal.** Do not signal the same barrier twice before waiting on it.
- **The window must not touch shared data:** no reading SLM another thread is still writing, no writing SLM another thread reads after the wait.
- **A nearby `barrier()` may already be enough.** If a full barrier elsewhere already makes the same writes visible, a named barrier for them adds nothing.
- **It only pays off with enough independent work** between signal and wait; with nothing to put there, a plain `barrier()` is simpler.

## Checklist

1. Find a `barrier()` that separates SLM writes from SLM reads by other threads.
2. Check that no other barrier nearby already provides the same guarantee.
3. Collect independent work for the window: global loads, prefetches, register-only math.
4. Replace the barrier with `named_barrier_signal`, the work, then `named_barrier_wait`; add `named_barrier_init<N + 1>()` for N named barriers at kernel start.
5. Check the window against the shared-data rule above, and test on the GPU under a timeout so a hang fails fast.

## Related Skills

| Skill | Use for |
|-------|---------|
| `esimd-lsc-slm` | Shared local memory, `slm_init`, plain `barrier()` |
| `esimd-base` | ESIMD kernel structure, vector operations, correctness traps |

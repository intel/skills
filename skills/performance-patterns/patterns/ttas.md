<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: test-and-set spinlock → test-and-test-and-set (TTAS)

## When to apply

Apply when a spinlock is implemented as a bare Test-and-Set loop — a `cmpxchg`
(or equivalent atomic exchange) with no preceding ordinary read. Every iteration
of the spin loop acquires the cache line **exclusively**, even on failure, causing
high cache-coherence traffic under contention.

### Source code signals

- `while (!cmpxchg(&lock, UNLOCKED, LOCKED)) _mm_pause();` — no prior `read`
- `while (__sync_lock_test_and_set(&lock, 1)) { /* spin */ }` — no pre-read
- Any spinlock body that performs only an atomic RMW without a preceding shared
  read of the lock variable
- `pthread_spin_lock` wrappers that are suspected to be TAS internally

### Profiling signals

- `lock cmpxchg` (or `lock xchg`, `lock cmpxchg8b`) clusters in `perf annotate`
  with high sample density inside a tight spin loop
- `spin_lock`, `cmpxchg`, `try_lock`, or `cas` wrappers prominent in `perf report`
- Throughput **drops** (not just plateaus) as thread count increases — a strong
  indicator of cache-line thrashing
- `perf c2c` shows true sharing on a lock variable where accessor functions
  contain `cmpxchg` in a loop

---

## Why this is slow

A simple Test-and-Set loop:

```c
/* Test-and-Set — DO NOT USE under contention */
while (!cmpxchg(&lock, UNLOCKED, LOCKED))
    _mm_pause();
```

`cmpxchg` always acquires the cache line **exclusively**, even on failure. When
thread A holds the lock and threads B and C are spinning:

- B and C repeatedly race each other for exclusive ownership of the lock's cache line
- The cache line bounces between B and C at high frequency
- This *also* steals the line away from thread A — even when A is trying to
  release the lock
- If protected data shares the same cache line as the lock, that data is caught
  in the same bounce (shows as false sharing in `perf c2c`)

The more waiters, the worse this scales — bus traffic grows as O(N²) under
contention.

---

## The fix: Test-and-Test-and-Set

Read the lock non-atomically first (shared cache line, no bus traffic) and only
attempt the atomic operation when there is a realistic chance of success:

```c
/* Test-and-Test-and-Set */
while (1) {
    /* Phase 1: spin cheaply on a shared read until lock looks free */
    while (atomic_read(&lock) == LOCKED)
        _mm_pause();
    /* Phase 2: attempt the atomic acquire */
    if (cmpxchg(&lock, UNLOCKED, LOCKED) == UNLOCKED)
        break;
    /* cmpxchg failed — another thread won the race; back to phase 1 */
}
```

Waiters now hold the cache line in **shared** mode during the spin. The exclusive
acquisition only happens when the lock genuinely transitions to unlocked, so bus
traffic drops from O(N²) to O(N) under contention.

### Optimized variant (try-first)

For the common uncontended case, try the atomic operation first to avoid the extra
read latency on an uncontended lock, then fall back to the spin-read loop:

```c
/* TTAS with try-first — lower latency when lock is uncontended */
if (cmpxchg(&lock, UNLOCKED, LOCKED) != UNLOCKED) {
    do {
        while (atomic_read(&lock) == LOCKED)
            _mm_pause();
    } while (cmpxchg(&lock, UNLOCKED, LOCKED) != UNLOCKED);
}
```

The first `cmpxchg` succeeds immediately when the lock is free (the common case),
incurring no extra overhead. Only under contention does it fall through to the
spin-read loop.

### Lock and data on the same cache line

If the `perf c2c` offset analysis shows that protected data shares a cache line
with the lock variable, separating them onto different cache lines compounds the
benefit significantly:

```c
struct my_lock {
    atomic_int lock;
    char _pad[60];      /* push protected data to the next cache line */
} __attribute__((aligned(64)));
```

---

## Verification

After applying the fix:

1. **Correctness** — run the existing test suite; lock semantics are unchanged.
2. **Profiling** — rerun `perf annotate` and confirm the `lock cmpxchg` density
   drops (the shared-read loop should be the dominant instructions now).
3. **Scaling** — measure throughput at 1, 2, 4, 8, … threads; the curve should
   no longer decline under contention.
4. **c2c** (if false sharing was present) — rerun `perf c2c` and confirm HITM
   drops on the lock's cache line.

---

## Presenting this to the user

1. Show the current spin loop from the source code with `/* ◄ exclusive acquire */`
   annotating the `cmpxchg` line.
2. Explain the exclusive-acquire / cache-line-bounce mechanism in one short
   paragraph.
3. Present both the basic TTAS form and the try-first variant, and recommend
   the try-first for production code.
4. Note whether the lock and protected data share a cache line (visible from
   the c2c Offset analysis) — if so, flag the padding fix as a compound benefit.

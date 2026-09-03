<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: shared statistics counter → per-CPU statistics aggregation

## When to apply

Apply when a single global counter (hits, errors, bytes, packets, total) is
incremented atomically from many threads in a hot path. This is **true sharing**
(all threads write the *same* field, not different fields) — distinguished from
false sharing, where threads write different fields on the same line.

### Source code signals

- `global_counter++` or `atomic_inc(&counter)` in a hot, frequently-called path
- `atomic_fetch_add`, `__sync_fetch_and_add`, `_Atomic` increment in a tight loop
- Field name suggests accounting: `count`, `total`, `hits`, `misses`, `errors`,
  `stat`, `bytes`, `packets`, or similar
- No `cmpxchg` retry loop (if present, the TTAS pattern applies instead)
- The counter is updated from many call sites or a highly-parallelized loop

### Profiling signals

- `lock add` / `lock inc` / `lock xadd` prominent in `perf annotate` on the hot
  path
- `perf c2c` shows **true sharing** on the counter's cache line: the same byte
  offset is written by many different CPUs
- The function appears as a scaling bottleneck in the multi-core profile (Flow D)
  but is barely visible at 1 core — contention scales linearly with thread count
- No `cmpxchg` instruction in the annotate output (differentiates from TTAS)

---

## Why frequent shared stats hurt under scaling

Every atomic increment on a shared counter requires the updater to hold the cache
line **exclusively**. With N threads all updating the same counter:

- N-1 threads must wait for the exclusive transfer on every update
- The cache line bounces between LLC slices at near-memory latency (~100–300 ns)
- The more threads, the more bouncing — this is the defining signature of
  scaling-limited true sharing
- Even `memory_order_relaxed` does not help: the **hardware** still enforces
  exclusive ownership for any write, regardless of the software memory order

---

## Choosing between options

Three options are available. Apply them in order of preference:

| Signal | Preferred option |
|--------|-----------------|
| Increment is **inside a loop local to one function** AND ≤ 2 hot profile sites | **Option A: Local batching** |
| More than 2 hot profile sites, or increments scattered across call sites | **Option B: CPU-indexed bucket array** |
| Code is a shared library that cannot own a global array, or counter is isolated | **Option C: `__thread` TLS (fallback)** |

When multiple counters are updated in the same loop, Options A and B are still
the primary choices: Option A batches all of them cheaply with one flush per
counter per batch; Option B calls `sched_getcpu()` once per increment site.

---

## Option A: Local batching

Accumulate into a local (stack) variable inside the loop and flush to the global
only at the batch boundary and unconditionally at function exit:

```c
#include <stdatomic.h>

_Atomic uint64_t global_counter;

#define COUNTER_BATCH_SIZE 16    /* tune to workload */

void update_items(struct item *array, int n)
{
    uint64_t local_counter = 0;
    for (int i = 0; i < n; i++) {
        perform_update(&array[i]);
        local_counter++;
        if (local_counter > COUNTER_BATCH_SIZE) {
            atomic_fetch_add_explicit(&global_counter, local_counter,
                                      memory_order_relaxed);
            local_counter = 0;
        }
    }
    /* flush remainder */
    atomic_fetch_add_explicit(&global_counter, local_counter,
                              memory_order_relaxed);
}
```

**Removing the threshold check:** if the loop body is purely CPU-bound (no I/O,
no blocking), remove the mid-loop threshold check entirely and flush once at the
end. This eliminates all mid-loop atomics.

**Multiple counters:** give each counter its own local variable and flush all of
them in the same conditional block — no extra `sched_getcpu()` overhead.

Use `memory_order_relaxed` for the flush — no ordering dependency exists.

---

## Option B: CPU-indexed bucket array

A fixed-size array of cache-line-padded structs, indexed by the current CPU
number. Robust, zero-overhead on the read side (just sum across buckets), no
flush-interval decisions needed.

**Group all statistics fields into one struct** — this amortizes the padding
overhead and means `sched_getcpu()` is called once, not once per counter.

### Step 1 — Define the per-CPU struct (in a shared header)

```c
#include <stdint.h>
#include <assert.h>

struct statistics {
    /* All counters here. Keep total size ≤ 56 bytes for one cache line. */
    uint64_t read_counter;
    uint64_t insert_counter;
    uint64_t error_counter;
    uint64_t byte_counter;
    uint64_t _padding[4];    /* adjust so sizeof == 64 */
};

_Static_assert(sizeof(struct statistics) % 64 == 0,
               "struct statistics must be a multiple of the cache line size (64 bytes)");

/* COUNT must be a power of 2; 64 is a good default */
#define STATISTICS_COUNT  64
#define STATISTICS_MASK   (STATISTICS_COUNT - 1)

extern struct statistics statistics[STATISTICS_COUNT]
    __attribute__((aligned(64)));    /* declaration in header */
```

The `& STATISTICS_MASK` on the CPU index is a cheap modulo: if the live CPU count
exceeds `STATISTICS_COUNT`, some sharing occurs but the contention factor drops
from N threads sharing 1 line to at most N/STATISTICS_COUNT threads sharing 1 line.

**Sizing guidance:**

| Situation | Recommended STATISTICS_COUNT |
|-----------|------------------------------|
| General use / unsure | 64 |
| Low thread count or infrequent updates | 16 or 32 |
| Contention still visible after 32 | Increase to 64 |
| Read cost too high (frequent reads of many buckets) | Decrease by one tier |

### Step 2 — Define the array and increment helpers (in one `.c` file)

```c
#include <sched.h>       /* sched_getcpu() */
#include <stdatomic.h>
#include "statistics.h"

struct statistics statistics[STATISTICS_COUNT] __attribute__((aligned(64)));

static inline void inc_read_counter(void)
{
    atomic_fetch_add_explicit(
        &statistics[sched_getcpu() & STATISTICS_MASK].read_counter,
        1, memory_order_relaxed);
}
/* repeat for each counter */
```

**Atomic API priority** — adapt to the existing codebase style:

| Priority | API | When to use |
|----------|-----|-------------|
| 1st | `<stdatomic.h>` — `atomic_fetch_add_explicit(..., memory_order_relaxed)` | C11 or later |
| 2nd | `<atomic>` — `std::atomic<uint64_t>::fetch_add(1, std::memory_order_relaxed)` | C++11 or later |
| 3rd | GCC `__atomic_add_fetch(&x, 1, __ATOMIC_RELAXED)` | GCC, pre-C11 |
| 4th | GCC `__sync_fetch_and_add(&x, 1)` | Legacy GCC — implies full barrier, slower |

Use `memory_order_relaxed` for counter increments — no ordering dependency exists,
and a relaxed atomic is still atomically correct.

### Step 3 — Define the aggregation helpers

```c
uint64_t get_read_counter(void)
{
    uint64_t sum = 0;
    for (int i = 0; i < STATISTICS_COUNT; i++)
        sum += atomic_load_explicit(
            &statistics[i].read_counter, memory_order_relaxed);
    return sum;
}
/* repeat for each counter */
```

Using `atomic_load` (a plain load on x86) keeps the code consistent and avoids
torn reads on `uint64_t`. For display-only statistics the value is inherently
approximate at any point in time.

### Step 4 — If per-CPU stats already exist but contention persists

If the code already has per-CPU or per-thread statistics but contention persists,
the flush interval may be too short — flushing is happening inside the hot path:

```c
/* Over-eager flush — called on every request */
void handle_request(...) {
    do_work();
    flush_stats();   /* ◄ moves data to global too frequently */
}
```

Increase the flush interval (batch N requests, use a timer, or flush only at
thread exit) and re-verify with `perf c2c`.

---

## Option C (fallback): `__thread` TLS

If the bucket-array approach is impractical (e.g., a shared library that cannot
own a global array, or an isolated counter not worth a full struct):

```c
static __thread uint64_t local_hit_count;
static _Atomic uint64_t  global_hit_count;

static inline void inc_hit_count(void) { local_hit_count++; }

static void flush_hit_count(void) {
    atomic_fetch_add_explicit(&global_hit_count, local_hit_count,
                              memory_order_relaxed);
    local_hit_count = 0;
}
/* Call flush_hit_count() at thread exit or periodically */
```

Caveats: `__thread` / `thread_local` can have non-trivial initialization overhead
in shared libraries (`.so`), and the flush decision adds complexity. Prefer the
bucket-array approach when possible.

---

## Verification

1. **Correctness** — the total counter value is now approximate (threads may not
   have flushed yet); confirm this is acceptable for the use case (usually yes
   for statistics/monitoring).
2. **Profiling** — rerun `perf annotate` and confirm `lock add`/`lock xadd`
   disappears or drops sharply from the hot path.
3. **c2c** — rerun `perf c2c` and confirm HITM drops on the counter's cache line.
4. **Scaling** — measure throughput at N threads; the linear regression should be
   gone.

---

## Presenting this to the user

1. Show the shared counter declaration and the hot accessor with `/* ◄ contention */`.
2. Explain cache-line bouncing in plain terms — every increment forces all other
   threads to reload the line before they can increment it.
3. Apply the decision table to choose Option A, B, or C. Briefly explain the
   choice (one sentence is enough).
4. For Option A: show the local variable, the threshold flush, and the exit flush.
5. For Option B: show the struct definition with padding, the array definition,
   and one increment + one read helper. Let the agent adapt field names and API
   to the local codebase style.
6. Recommend re-running `perf c2c` after the change to confirm HITM drops.

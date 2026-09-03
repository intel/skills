<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: false sharing → structured false-sharing fix

## When to apply

Apply when a cache line is heavily written by multiple threads, but each thread
accesses **different fields** at different byte offsets within that line. The
fields are logically independent — no thread needs the data another thread writes
— but the cache-coherence protocol treats the entire line as the unit of ownership,
causing spurious invalidations.

### Source code signals

- A struct with multiple fields, where different fields are written by different
  threads (e.g., per-thread counters, state flags, or work items packed into a
  shared struct)
- No `alignas(64)` or `__attribute__((aligned(64)))` between thread-private fields
- Fields written at high frequency in a tight per-thread loop share a struct with
  other frequently-written fields

### Profiling signals

- `perf c2c` reports **HITM > 5%** on a cache line (Tot Hitm column)
- The per-line access map shows **different offsets** being written by different
  CPU/thread IDs — key distinction from true sharing
- The hotspot function appears prominently only in the multi-core profile (Flow D),
  not at 1 core — the regression scales linearly with thread count
- `perf c2c` `lcl hitm` (local HITM, same socket) is the dominant component rather
  than `rmt hitm` (cross-socket)

---

## Why false sharing hurts under scaling

A cache line is the smallest unit of coherence — typically 64 bytes. When thread A
writes field X and thread B writes field Y — even though X and Y are completely
unrelated — both writes invalidate each other's cached copy of the **entire** line.
Every write forces all other holders to reload the full 64 bytes from the L3 or
memory. This traffic grows linearly with thread count, which is why the function
only becomes prominent in a multi-core profile.

---

## The fix: structured three-step protocol

### Step 1 — Generous padding (proof of concept)

Add `alignas(64)` to the struct and insert explicit `char _pad[N]` fields between
the contested fields to push them to separate cache lines. Use *more padding than
you think you need* at this stage — proving the diagnosis is more important than
minimizing size:

```c
struct example {
    int written_by_thread_a;        /* thread A's field */
    char _pad0[60];                 /* push to next cache line */
    int written_by_thread_b;        /* thread B's field */
    char _pad1[60];
    int read_a_lot;                 /* shared read-mostly field */
} __attribute__((aligned(64)));
```

### Step 2 — Verify the fix

Rebuild and rerun the workload with the same `perf c2c` collection. Confirm that:
- The HITM percentage for this cache line drops sharply (target: below 1%)
- Overall workload throughput improves noticeably at the same thread count

If HITM does not drop, the diagnosis was wrong — the sharing may be true sharing,
or additional fields are involved. Re-examine the `perf c2c` per-offset access map
before proceeding.

### Step 3 — Optimize struct layout

Once the fix is confirmed, minimize the added size by grouping struct members by
access pattern rather than inserting padding bluntly:

| Group | What to put here |
|-------|-----------------|
| **Write-often (thread A)** | Fields written frequently from thread A only |
| **Write-often (thread B)** | Fields written frequently from thread B only |
| **Read-mostly** | Fields read by many threads but rarely written (config, thresholds, lookup tables) |
| **Rarely touched** | Initialization-time or infrequently accessed fields |

Each **write-often** group for a different thread must occupy its own cache line.
**Read-mostly** fields from all threads can safely share a cache line — shared
reads do not cause HITM. **Rarely-touched** fields can fill remaining space.

```c
/* After optimization */
struct example {
    /* Thread A's write-often fields — one cache line */
    int written_by_thread_a;
    uint64_t thread_a_counter;
    char _pad0[52];             /* fill to 64 bytes */

    /* Thread B's write-often fields — next cache line */
    int written_by_thread_b;
    uint64_t thread_b_counter;
    char _pad1[52];             /* fill to 64 bytes */

    /* Read-mostly shared fields — can share a line */
    int config_threshold;
    int config_limit;
} __attribute__((aligned(64)));
```

After each layout change, re-verify with `perf c2c` that HITM stays suppressed.
Stop when the padding is minimal and HITM is still below 1%.

### Alternative: per-thread structs

If the false sharing stems from a struct that was designed as a global-but-sharded
object, consider moving thread-private fields into a true per-thread struct
(allocated separately per thread or indexed by thread ID) and removing them from
the shared struct entirely. This eliminates the problem structurally rather than
papering over it with padding.

---

## Verification

1. **Step 2 above is the primary verification** — `perf c2c` before and after.
2. **Benchmark** — measure throughput at N threads; the regression should be gone.
3. **Size check** — `sizeof(struct example)` should be what you expect after
   the Step 3 optimization; use `_Static_assert` to guard it.

---

## Presenting this to the user

1. Show the struct definition annotated with byte offsets (`/* +0x00 */`,
   `/* +0x04 */`, etc.) so the conflict is visually obvious.
2. Show the fields involved from the `perf c2c` Offset column and explain why
   they conflict even though the data is unrelated.
3. Apply Step 1 (generous padding) and show the modified struct.
4. Ask the user to rebuild and rerun — **do not skip to Step 3 without Step 2
   confirmation**; the padding may be wrong.
5. After the user confirms HITM dropped, propose the optimized layout and show
   the new struct with grouping comments.

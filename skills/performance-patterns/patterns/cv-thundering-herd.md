<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: condition variable thundering herd

## When to apply

Apply when a condition variable (`pthread_cond_broadcast` / `notify_all`, or a
`notify_one` loop) wakes far more threads than can make progress. At high core
count (HCC) scale,
the wasted wakeups dominate runtime: woken threads race for the mutex, find no
work (or the predicate already consumed), and immediately re-block.

### Source code signals

- `pthread_cond_broadcast(&cv)` or `cv.notify_all()` used to wake a worker pool
  where only a subset of waiters have work available
- `notify_one()` called in a loop to wake N threads sequentially (O(N) futex
  syscalls)
- A dispatcher/leader that wakes all threads unconditionally, regardless of how
  many jobs are pending: `for (i = 0; i < nthreads; i++) wake(thread[i])`
- Thread pool workers that call `sched_yield()` or immediately re-block after
  wakeup — a sign they were woken unnecessarily

### Profiling signals

- `futex_wake`, `try_to_wake_up`, `wake_up_q` prominent in `perf report` —
  these are the kernel wakeup path, not lock contention functions
- `__pthread_cond_broadcast` / `__pthread_cond_signal` hot in flamegraph
- Context-switch rate anomaly: `perf stat -e context-switches` shows cs/s
  scaling with N waiters per broadcast, far exceeding the natural work rate
- IPC collapse combined with high CPU utilization — cores are busy doing
  scheduler work (wakeup dispatch, mutex re-acquisition), not useful computation
- `sched_yield` or `__futex_abstimed_wait_common` prominent when threads find
  no work after waking
- `perf trace -e futex 2>&1 | grep WAKE` shows `FUTEX_WAKE, val=INT_MAX` at
  high frequency (thundering herd) or rapid bursts of `val=1` (notify_one loop)
- `perf sched latency` shows high `max_lat` — woken threads cannot run
  immediately because all CPUs are occupied by the N-1 other woken threads

### Distinguishing from lock contention

CV scaling problems are **invisible to lock profiling tools** (`perf lock stat`
shows normal hold/wait times). The key differentiator:

| Symptom | Lock contention | CV thundering herd |
|---------|----------------|-------------------|
| `perf top` hot symbol | `native_queued_spin_lock_slowpath` | `futex_wake`, `try_to_wake_up` |
| `lock_stat` wait_total | High | Normal |
| Context-switch rate | Moderate | Very high |
| `futex` WAKE frequency | Low | High |

---

## Why this is slow

When `notify_all()` fires with N waiters:

```
Thundering Herd: notify_all with N waiters

  t0: notify_all() → N threads become runnable simultaneously
  t1: All N race for mutex re-acquisition (mandatory by CV semantics)
      → only 1 wins, N-1 immediately block on the mutex
  t2: Each of the N-1 losers pays a full futex round-trip
      (wake → schedule → attempt acquire → fail → sleep)
  t3: Threads wake one-by-one as holder releases mutex
      Most re-check predicate, find nothing, go back to sleep
```

The cost per `notify_all` call:
- O(N) context switches
- O(N) futex syscalls
- O(N) scheduler IPI dispatches (cross-core interrupts)
- Burst of RFO traffic on the mutex cache line as N cores attempt acquire

With a sequential `notify_one` loop, each call is a separate futex syscall +
IPI + context switch. At 160 threads: `T_wakeup ≈ N × 5µs ≈ 800µs` of pure
wakeup overhead per dispatch round.

**Why this worsens super-linearly with core count.** If T threads wake for J
jobs (J << T):

```
Wasted syscalls per round  = T - J   (grows with T)
Failed mutex acquisitions  ≈ T - J   (grows with T)
yield()/re-block calls     ≈ T - J   (grows with T)
```

On a 64-core system with 40 jobs, 24 threads waste — modest. On 160 cores,
120 threads waste — 75% of all wakeup effort is pure overhead.

---

## The fix

Choose the strategy that matches the wakeup topology:

### Fix 1: Precise wakeup count (dispatcher → workers)

When a dispatcher knows how many jobs are available, wake exactly that many
threads:

```cpp
// Before: wake all threads regardless of available work
for (int i = 0; i < num_threads; i++)
    cv.notify_one();

// After: wake only as many as there are jobs, with early exit
int threads_to_wake = std::min(num_threads, num_jobs);
for (int i = 0; i < threads_to_wake; i++) {
    if (all_jobs_claimed()) break;  // early termination
    cv.notify_one();
}
```

This eliminates wasted wakeups entirely. Combined with early termination
(checking whether all jobs have been claimed during the wakeup loop), it
handles the case where early-woken threads consume multiple jobs.

### Fix 2: Tree wakeup (leader → N followers)

When a leader must wake all N followers (all have work to do), replace the
O(N) sequential loop with an O(sqrt(N)) or O(log N) tree:

```
// Sequential: Leader wakes N writers one by one
Leader → W1 → W2 → W3 → ... → WN     (O(N) elapsed time)

// Tree (K=3): Leader wakes K helpers, then all wake in parallel
Step 1: Leader → W1, W2, W3            (O(K) elapsed time)
Step 2: Leader  → W4, W8, W12
        W1      → W5, W9, W13          (O(N/K) elapsed time)
        W2      → W6, W10, W14
        W3      → W7, W11, W15

Total elapsed: O(K + N/(K+1)) ≈ O(sqrt(N))  when K = sqrt(N)
```

Implementation sketch:

```cpp
void tree_wakeup(int total, int branch_factor) {
    int helpers = std::min(branch_factor, total);
    // Phase 1: wake helpers
    for (int i = 0; i < helpers; i++)
        wake(thread[i]);  // these threads know to help wake others

    // Phase 2: leader + helpers wake remaining threads in parallel
    int my_start = helpers;
    int stride = helpers + 1;
    for (int i = my_start; i < total; i += stride)
        wake(thread[i]);
}

// Each helper runs a similar loop for its assigned range
```

### Fix 3: Replace `notify_all` with `notify_one` + hand-off

When only one thread can make progress at a time (single-consumer pattern),
wake exactly one. That thread, after finishing, wakes the next:

```cpp
// Producer: wake exactly one consumer
{
    std::lock_guard lock(mtx);
    queue.push(item);
}
cv.notify_one();  // notify after releasing mutex to avoid waking into contention

// Consumer: after processing, pass the baton if work remains
{
    std::unique_lock lock(mtx);
    cv.wait(lock, [&]{ return !queue.empty(); });
    auto item = queue.pop();
    bool more = !queue.empty();
    lock.unlock();
    if (more)
        cv.notify_one();  // wake next consumer after releasing mutex
    process(item);
}
```

### Fix 4: Replace CV with semaphore (counter-based predicates)

When the condition is simply "N units are available", a semaphore avoids the
mutex re-acquisition race entirely:

```cpp
// Before: CV + counter (pthreads)
pthread_mutex_lock(&mtx);
while (count == 0)
    pthread_cond_wait(&cv, &mtx);
count--;
pthread_mutex_unlock(&mtx);

// After: semaphore (no mutex re-acquisition per wakeup)
sem_wait(&sem);  // blocks only if count == 0
```

POSIX `sem_post` / `sem_wait` avoid the mutex re-acquisition storm that makes
thundering herd expensive.

### When NOT to apply

- **All waiters genuinely need to wake** (barrier synchronization where all N
  threads must proceed) — use `std::barrier` or `std::latch` instead of manual
  CV, as they implement wakeup more efficiently
- **The predicate check is the expensive part** — if re-checking the predicate
  (not the wakeup machinery) is the bottleneck, the thundering herd pattern
  does not apply

---

## Verification

After applying the fix:

1. **Correctness** — verify no lost wakeups: every job is eventually processed;
   no thread sleeps forever. Run stress tests with varying job counts and
   thread counts.
2. **Futex trace** — `perf trace -e futex 2>&1 | grep WAKE` should show
   reduced WAKE frequency or smaller `val` arguments.
3. **Context switches** — `perf stat -e context-switches` should show a
   proportional reduction (from O(N) per dispatch to O(J) where J = job count).
4. **IPC recovery** — IPC should improve as cores spend time on useful work
   instead of scheduler thrash.
5. **Scaling test** — throughput at high core counts should improve; the symptom
   "performance gets worse past N cores" should disappear or shift to a higher
   core count.

---

## Presenting this to the user

1. Show the perf evidence: `futex_wake` / `try_to_wake_up` cycle percentage,
   context-switch rate, or `sched_yield` dominance.
2. Identify the `notify_all` or `notify_one` loop in source code and state the
   ratio: "N threads woken for J jobs means N-J wasted wakeups per round."
3. Explain in one paragraph: "Each wasted wakeup costs a futex syscall + IPI +
   context switch + mutex re-acquisition attempt. At 160 cores with 40 jobs,
   that's 120 × ~5µs = 600µs of pure overhead per dispatch — often exceeding
   the actual work time."
4. Present the appropriate fix (precise wakeup, tree wakeup, hand-off, or
   semaphore) based on the wakeup topology.
5. Note expected improvement: 49–84% for precise wakeup (OpenCV case), 3× for
   tree wakeup (RocksDB case), depending on the ratio of wasted wakeups to
   useful work.

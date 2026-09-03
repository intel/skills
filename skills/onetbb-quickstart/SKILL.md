---
name: onetbb-quickstart
description: >-
  Getting started with Intel oneTBB for C++ parallelism on Intel CPUs. Use when a
  C++ loop or reduction should run on multiple threads with oneTBB, when the user
  needs the headers, namespace, or CMake wiring for a first oneTBB program, when a
  parallel_for body has a data race, or when a reduction is accumulating into a
  shared variable. Covers parallel_for and parallel_reduce over blocked_range, the
  build setup, and the pitfalls of the task-based model.
license: Apache-2.0
compatibility: "Requires oneTBB and a C++17 compiler. CMake examples need the TBB package config that ships with oneTBB."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# oneTBB quickstart

## Purpose

Parallelizes C++ loops and reductions with Intel oneAPI Threading Building Blocks
on Intel CPUs: the umbrella header, `parallel_for` and `parallel_reduce` over a
`blocked_range`, the CMake lines that link it, and the assumptions the task-based
model breaks.

Prefer oneTBB over hand-rolled threads when the work is a bounded loop or a
reduction over a container: the partitioner decides the split, and the runtime
composes with other oneTBB-based libraries in the same process instead of
oversubscribing the machine.

## When to Use This Skill

Use this skill when:

- A C++ loop or reduction is a candidate for multithreading.
- A first oneTBB program needs headers, namespace, and CMake wiring.
- A `parallel_for` body has a race on shared state.
- A reduction accumulates into one variable from many threads.

Do **not** use this skill for GPU offload, for OpenMP or `std::thread` questions,
or for tuning an existing oneTBB program's grain size and partitioners — that is
past "getting started" and belongs with a profile in hand.

## Quick Start

```cpp
#include <oneapi/tbb.h>            // umbrella header

int main() {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, n),
        [&](const oneapi::tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i)
                out[i] = f(in[i]);
        });
}
```

`using namespace oneapi::tbb;` shortens the calls; qualifying them keeps the
origin visible in code that mixes threading libraries.

## Implementation Guide

1. **Parallelize a loop with `parallel_for` over a range.** The body receives a
   subrange, not a single index — iterate inside it, as above. The range type
   carries the index type, so `blocked_range<size_t>` and `blocked_range<int>`
   are different instantiations.

2. **Reduce with `parallel_reduce`, not a shared accumulator.** The body folds a
   subrange into a partial result and the last argument combines two partials:

   ```cpp
   double sum = oneapi::tbb::parallel_reduce(
       oneapi::tbb::blocked_range<size_t>(0, n), 0.0,
       [&](const auto& r, double acc) {
           for (size_t i = r.begin(); i != r.end(); ++i) acc += a[i];
           return acc;
       },
       std::plus<double>());
   ```

3. **Link it in CMake.** oneTBB ships a package config, so the two lines are the
   whole build change:

   ```cmake
   find_package(TBB REQUIRED)
   target_link_libraries(my_app PRIVATE TBB::tbb)
   ```

4. **Leave the grain size alone at first.** The auto-partitioner chooses the
   split; a hand-set grain size is a tuning decision that needs a measurement
   behind it, and a wrong one is worse than none.

5. **Make shared state safe or remove it.** If the body must write to a shared
   structure, use one of the `concurrent_*` containers or restructure as a
   reduction. A mutex around the body of a `parallel_for` usually gives back the
   parallelism it was added to protect.

## Performance

No measured numbers ship with this skill, and a parallel version is not
automatically a faster one. What to measure:

- Compare against the serial loop on the same input, with the same compiler flags
  and optimization level.
- Watch for a body too small to cover the task overhead: at that size the
  partitioner's fixed cost shows up as a slowdown.
- Check whether the loop is memory-bandwidth bound before adding threads — more
  threads on a saturated bus do not help.
- Count the threads in the process. Nested parallelism from another library, or
  an OpenMP region around a oneTBB call, oversubscribes the cores and the
  slowdown is not in either loop.

## Gotchas & Limitations

- **The body runs many times, concurrently.** It is not called once per loop and
  not once per thread; the range is split as the runtime sees fit. Anything
  captured by reference and written to is shared mutable state.
- **`parallel_reduce` is not deterministic in floating point.** The combination
  order varies between runs, so sums can differ in the last bits. Use
  `parallel_deterministic_reduce` when a reproducible result matters more than
  speed.
- **Exceptions propagate out of the algorithm**, not out of the body where they
  were thrown — one is rethrown on the caller's thread and the rest are lost.
- **`find_package(TBB)` needs oneTBB's own config**, which the environment script
  or the package install puts on `CMAKE_PREFIX_PATH`. A build that cannot find it
  usually has not sourced the environment.
- Not covered: flow graph, task groups, arenas and thread affinity, and the
  deprecated `tbb::` (pre-oneAPI) spellings.

## References

| File | Load it when |
|---|---|
| [`references/official-sources.md`](references/official-sources.md) | you need the current oneTBB API for an algorithm, the deprecation status of a `tbb::` name, or the supported CMake integration for the installed version |

Two things here should not be answered from memory: **the current name and
signature of an algorithm** (oneTBB renamed and dropped parts of the pre-oneAPI
API, and the old spellings still compile in some builds) and **how the package is
found by CMake in the installed layout**.

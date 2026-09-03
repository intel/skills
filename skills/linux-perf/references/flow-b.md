<!-- (C) 2026 Intel Corporation, MIT license -->
# Flow B: execution steps

### Phase 0 — Ensure debug symbols

Run **Building block: Ensure debug symbols** (Part 4) on the binary before recording. If the binary lacks DWARF and the user agrees to recompile, do so now — before Phase 1 — so that the recording and annotation use the same binary. If the user declines, note the limitation and proceed.

### Phase 1 — Record

Use **Building block: perf record** (Part 4).

**After proposing this, ask the user:** *"Would you also like call graph data? It shows why functions are hot (the callers that led to them), not just which functions are hot."* If the prompt is already clear on this, skip the question. If the user agrees, pass `--call-graph dwarf`.

### Phase 2 — Report

```bash
perf report --stdio --no-header -F+srcfile -F+srcline | head -100
```

Key columns: **Overhead**, **Source:Line**, **Command**, **Shared Object**, **Symbol** (`[k]` = kernel, `[.]` = userspace).

**If symbols show as raw addresses**: the binary is stripped — recompile with `-g` and re-record, or use **Building block: resolve address to source** (Part 4) to correlate manually.

### Phase 3 — Interpret

- **Top entry `[k]` > 40%** → syscall/I/O bound; suggest `strace -c -- <command>`
- **malloc/free/new/delete dominates** → allocation-heavy; consider object pools or arena allocators
- **No clear hotspot (< 5% per function)** → well-distributed; look for parallelism or algorithmic improvements
- **One function > 20%** → strong optimization candidate; proceed to Phase 4 (annotate)
- **Many small kernel helpers** (`copy_to_user`, `mutex_lock`) → lock contention or syscall overhead
- **Spinlock / CAS wrappers prominent** (`spin_lock`, `cmpxchg`, `try_lock`, `cas`) → possible Test-and-Set spin pattern; proceed to Phase 4 to confirm `lock cmpxchg` clusters, then apply **Resolution strategy: Test-and-Test-and-Set** (Part 5)

### Phase 4 — Drill-down with `perf annotate`

**Mandatory**: if any function accounts for **≥ 20% of samples**, use **Building block: perf annotate (assembly view)** (Part 4) — do not just offer it, actually run it. If the binary is stripped, fall back to **Building block: resolve address to source** (Part 4).

Once you have the annotate output, run **Building block: Annotate pattern scan** (Part 4). For each pattern returned in the scan output, apply the **Suggested RS** shown in the table — using the step-by-step RS instructions in Part 5.

Additionally, after a SIMD fix (upconversion or accumulator rewrite), always verify with:

```bash
perf stat -e fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double -- <workload>
```

If `256b_packed_double` or `512b_packed_double` is non-zero and `scalar_double` is negligible, SIMD is active. If scalar is still dominant, the vectorizer is still blocked — check for aliasing (`restrict` missing?), loop-carried dependencies, or non-unit stride.

**Note for Scalar FP pattern**: the most common vectorization blocker is a **strided inner loop** — pointer arithmetic of the form `base + k*stride` where stride is large (e.g., `N*8` bytes for a row-major matrix column access). The fix is a **loop reorder** to make the innermost loop stride-1 — then rebuild and re-run the annotate pattern scan to confirm SIMD is now active.

---


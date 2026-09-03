<!-- (C) 2026 Intel Corporation, MIT license -->
# Flow C: execution steps

### Phase 1 — Collect hot cache lines

Use **Building block: c2c hot cache lines** (Part 4).

This produces the recording, the `c2c_report.txt` file, and the summary table of hot cache lines. Pass the threshold and N from the user's request to the building block.

### Phase 2 — Map access locations per cache line

For each cache line above the threshold, use **Building block: c2c access map for a cache line** (Part 4).

This extracts which code locations are accessing the cache line, classifies the sharing pattern (false vs true) via the Offset column, and returns a per-line access table.

### Phase 3 — Gather source code

Use **Building block: resolve address to source** (Part 4) if source lines are not already populated from the Pareto output — treat it as a fallback.

Gather the source code for:
- the functions that access the cache line under contention
- the definition of the data structure (struct, class, etc.)

Then verify the Phase 2 preliminary conclusion of true vs false sharing by inspecting the field name in the struct:
- Same field name (or a non-struct variable) → True sharing
- Different field names → False sharing



### Phase 4 — Report findings

Produce a Markdown report. Start with a one-line summary table, then one detailed section per hot cache line.

#### Annotation conventions used throughout the report

**Struct field offsets** — annotate each field with its byte offset using `/* +0xNN */`. This lets the reader directly cross-reference fields against the `Offset` column from the perf c2c data without any arithmetic. Always show the **complete** struct definition — no truncation, even for large structs:
```c
struct foo {
    int written_a_lot;      /* +0x00 */
    int innocent_bystander; /* +0x04 */
    int read_a_lot;         /* +0x08 */
};
```

**Access markers** — mark the specific line performing the contested access with `/* ◄ write */` or `/* ◄ read */`:
```c
    s->written_a_lot++;   /* ◄ write */
    if (s->read_a_lot)    /* ◄ read  */
```

**Function display rules:**
- **≤ 20 lines**: show the entire function body, no truncation.
- **> 20 lines**: use this summarization strategy — show the function signature and opening brace with 1–2 lines of setup context, then `...`, then at least 3 lines of context around each key line (the `/* ◄ */` markers), then `...` if more code follows, and always end with the closing `}`:

```c
void long_function(struct example *s, int n)
{
    int i, result = 0;
    ...
    for (i = 0; i < n; i++) {
        s->written_a_lot++;   /* ◄ write */
        result += s->read_a_lot;
    }
    ...
}
```

The goal is that the reader sees enough to understand the calling convention and the context of the hot access without wading through unrelated code.

#### Report template

````markdown
# c2c Analysis — Cache Line Contention Report

This report covers the test run with `<command>`.

## Summary

| # | Address | Tot Hitm | Sharing type |
|---|---------|----------|--------------|
| 0 | `0xffff8ede402e0800` | 15.96% | False sharing |
| 1 | `0xffff8ede400e8800` | 6.10% | True sharing |

---

## Case N: False sharing  (<Tot Hitm>% of cycles)

### The data structure

`<path/header.h, line NN>`
```c
struct example {
    int written_a_lot;      /* +0x00 */
    int innocent_bystander; /* +0x04 */
    int read_a_lot;         /* +0x08 */
};
```

### Accessor 1: `<function_name>` at `<path/file.c:NN>`  (<LclHitm>% of HITM)

The field `written_a_lot` (+0x00) is **written** here (function is short — shown in full):
```c
void write_me(struct example *s)
{
    s->written_a_lot++;   /* ◄ write */
}
```

Called from:

| Caller | Location | % of calls |
|--------|----------|-----------|
| `try_write()` | `path/file2.c:42` | 80% |
| `do_write()` | `path/file2.c:80` | 20% |

### Accessor 2: `<function_name>` at `<path/file.c:NN>`  (<LclHitm>% of HITM)

The field `read_a_lot` (+0x08) is **read** here (function exceeds 20 lines — summarized):
```c
bool complex_reader(struct example *s, int flags)
{
    bool result = false;
    ...
    if (flags & FLAG_CHECK) {
        if (s->read_a_lot)    /* ◄ read  */
            result = true;
        else
            result = false;
    }
    ...
}
```

Called from:

| Caller | Location | % of calls |
|--------|----------|-----------|
| `try_read()` | `path/file3.c:12` | 100% |

Different fields (`written_a_lot` vs `read_a_lot`) at different offsets (0x00 vs 0x08) → **false sharing**.

Apply **Resolution strategy: Structured false-sharing fix** (Part 5).

---

## Case N: True sharing  (<Tot Hitm>% of cycles)

### The data structure

`<path/header.h, line NN>`
```c
struct example {
    atomic_t refcount;    /* +0x00 */
};
```

### Accessor 1 & 2: `<function_name>` at `<path/file.c:NN>`  (both at +0x00)

The same field `refcount` (+0x00) is accessed from multiple threads:
```c
void do_something(struct object *obj)
{
    atomic_inc(&obj->refcount);   /* ◄ write */
    if (atomic_dec_and_test(&obj->refcount))   /* ◄ write */
}
```

Same field (`refcount`) and offset (0x00) → genuine contention on the same datum → **true sharing**.

**Test-and-Set spin detection** — if the accessor functions contain `cmpxchg` / `lock cmpxchg` in a loop, this may be a Test-and-Set spin pattern rather than plain data contention. Every waiter competes for exclusive ownership of the cache line, including threads that currently have no chance of acquiring the lock — which also steals the line from the holder. Apply **Resolution strategy: Test-and-Test-and-Set** (Part 5) before considering other strategies.

Common resolution strategies for true sharing without the spin pattern:
- **Atomic / lock-free** — `std::atomic` or `_Atomic` for counters or flags (may already be in use here)
- **Per-CPU / per-thread sharding** — each thread owns its copy; merge at the end
- **RCU** — for read-heavy data rarely written
- **Finer lock granularity** — split a coarse lock so fewer threads compete
- **Convert to a R/W primitive** — if reads dominate, a reader/writer lock reduces exclusive contention

Recommend the best fit based on data type and access pattern, then stop and ask the user for further direction.
````

### Phase 5 — If no significant entries found

Report explicitly: no cache lines above the threshold were found; the workload may not be contention-bound, or the recording was too short — suggest recording for longer or under higher load.

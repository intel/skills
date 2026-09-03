<!-- (C) 2026 Intel Corporation, MIT license -->
# Pattern: Library version upgrade

## When to apply

A function from a **system library** (a `.so` not owned by the application)
appears in the hot path of a `perf report` or `perf annotate` profile, AND
that symbol appears in the version table in `references/library-versions.md`.

This is a low-effort performance win: the optimized code already exists and
is production-tested — the only question is whether the installed version is
new enough to include it.

## Why it matters

System libraries accumulate targeted optimizations over time — often ISA-specific
(AVX-512, SHA-NI, VPCLMULQDQ, etc.) — that are invisible to application code.
A version gap can mean the application is silently running an unoptimized
implementation despite the hardware supporting a much faster path.

## Detection

### Step 1 — Identify the hot library symbol

From `perf report` or `perf annotate` output, find symbols where the DSO column
shows a `.so` file rather than the application binary.  A line like:

```
  12.34%  myapp  libcrypto.so.3  [.] gcm_init_avx
```

gives you the symbol name (`gcm_init_avx`) and the `.so` file (`libcrypto.so.3`).

### Step 2 — Look up the symbol

Read `references/library-versions.md` and check whether the symbol appears in
the version table.

- **Match found** → proceed to Step 3.
- **No match** → this pattern does not apply; continue with other analysis.

### Step 3 — Detect the installed version

Use the version detection primitives in `references/library-versions.md`.
Try the **generic `readlink` method** first — most libraries encode their full
version in the real `.so` filename (e.g. `libz.so.1 → libz.so.1.2.11`).
If that does not yield a useful version string, fall back to the
library-specific primitive listed there.

### Step 4 — Compare and report

Parse the installed version string and compare it against the
**Suggested minimum version** from the table.

Report to the user:

**When installed version is below the minimum:**

```
Library version gap detected
────────────────────────────
Hot symbol : gcm_init_avx  (libcrypto.so.3)
Library    : OpenSSL
Installed  : 3.1.4
Minimum    : 3.3
Gap        : 2 minor versions behind
Reason     : Significant AVX-512 optimizations for AES-GCM in 3.3+
```

**When installed version meets or exceeds the minimum:**

```
Library version check: OK
Hot symbol : gcm_init_avx  (libcrypto.so.3)
Library    : OpenSSL  3.6.2  (minimum: 3.3)
The installed version already includes the known optimizations.
If this symbol is still hot, the bottleneck is elsewhere — consider
applying the linux-perf skill to profile further inside the function.
```

## Presenting to the user

- State the gap precisely (installed vs. minimum), not just "update."
- Include the reason so the user understands what optimization they are missing.
- Do not prescribe *how* to upgrade — that depends on the user's environment
  (package manager, containerized build, enterprise distribution, etc.).
- If the installed version already meets the minimum, say so explicitly and
  redirect attention to further profiling rather than leaving the user without
  a next step.

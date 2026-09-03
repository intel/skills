<!-- (C) 2026 Intel Corporation, MIT license -->
# Performance Patterns — Design Document

This file records the architectural decisions for the `performance-patterns` skill.
Update it whenever a structural decision is made or revised.

---

## Goals

- Provide a single skill that takes a user from symptom (profile signal or bad
  source code) all the way to a working fix — without requiring other skills to
  be installed.
- Grow gracefully: adding a new pattern should require no structural changes.
- Keep context load minimal: agents load only what is relevant to their current
  situation (profile data vs. source code; detection vs. implementation).

---

## Directory layout

```
performance-patterns/
├── SKILL.md                    ← routing logic only; no pattern content
├── design.md                   ← this file
├── triggers/
│   ├── from-profile.md         ← pattern identification from profiling output
│   └── from-source.md          ← pattern identification from source code
├── guidelines/
│   └── new-code.md             ← write-time checklist for new performance-sensitive code
├── patterns/
│   ├── <name>.md               ← detection + root cause + fix (or fix summary)
│   └── <name>-impl.md          ← full implementation guide (only when fix > ~150 lines)
└── library/
    └── <capability>.md         ← reusable implementation modules (shared across patterns)
```

---

## Three-tier loading

1. **`SKILL.md`** — always in context; contains only the routing decision.
   Three contexts: load `triggers/from-profile.md`, `triggers/from-source.md`,
   or `guidelines/new-code.md` based on whether the agent has profiling data,
   is reviewing existing code, or is writing new code.
2. **Trigger / guideline file** — loaded once to identify which pattern applies
   (trigger files) or to get the write-time checklist (guideline file). Separated
   by context so agents with profile data never load source-code triggers and
   vice versa. This matters as the catalog grows to 20+ patterns.
3. **Pattern file + optional impl file** — demand-loaded after a pattern is
   identified. The pattern file always covers detection and root cause. The
   impl file is loaded separately only when the agent is about to write the fix.

The `guidelines/new-code.md` file is a write-time checklist that cross-references
`patterns/` files for rationale. It is the inverted form of the patterns catalog:
the same knowledge reformatted as "do this from the start" rather than "here's
what's wrong and how to fix it".

---

## Pattern file anatomy

Every `patterns/<name>.md` must contain these sections, in order:

1. **When to apply** — source code signals and/or profiling signals
2. **Why this is slow** — root cause; motivates the fix without being a tutorial
3. **The fix** — step-by-step resolution with code examples
4. **Verification** — how to confirm the fix worked
5. **Presenting this to the user** — before/after structure, what to highlight

---

## Two-file split rule

Keep detection and fix in **one file** until the fix section alone would exceed
roughly **150 lines**. At that point, split:

```
patterns/<name>.md        ← detection + root cause + fix *summary* + link to impl
patterns/<name>-impl.md   ← full implementation: algorithms, code templates, edge cases
```

The impl file lives alongside the pattern file (not in a separate directory) so
the relationship is obvious. An agent doing detection reads `<name>.md` and
frequently never needs `<name>-impl.md`. An agent writing the fix loads it
on demand.

---

## Three classes of pattern

**Class 1 — Self-contained** *(TTAS, false-sharing, per-cpu-stats)*
Fix is explanation + code snippets that fit in the main file. Single-file
structure. No split unless the pattern grows substantially.

**Class 2 — Complex implementation** *(SIMD upconversion, parallel accumulator
once absorbed)*
Detection is compact; implementation is a multi-step algorithm (hundreds of
lines). Use the two-file split. The `<name>.md` carries detection and a
self-contained quick fix for simple cases; `<name>-impl.md` carries the full
algorithm, edge cases, and dispatch wiring.

**Class 3 — Delegating** *(simd-upconversion, parallel-accumulator currently)*
Detection lives here; implementation lives in another installed skill (currently
`vector-x86`). Acceptable short-term to avoid duplication and keep content
in sync. Each delegating pattern file is annotated with
`*(Class 3 — migration candidate)*` so the intent is visible.

### Tipping point for absorbing a Class 3 pattern

Absorb when **both** of these are true:
- A user relying solely on this skill would be blocked from completing the task
  (installation friction is a real cost, not a theoretical one)
- The delegated skill's content is stable enough to maintain a copy here

When the tipping point is reached: move the implementation content into
`<name>-impl.md`, remove the delegation call from `<name>.md`, and update the
delegated skill to note it has been superseded (don't delete it immediately —
other callers may depend on it).

### Current Class 3 delegation map

| Pattern file | Delegated to | Migration target |
|---|---|---|
| `patterns/simd-upconversion.md` | `vector-x86` (zipper algorithm, dispatch) | `patterns/simd-upconversion-impl.md` + `library/cpu-dispatch.md` |
| `patterns/parallel-accumulator.md` | `vector-x86` (vector-seq templates) | `patterns/parallel-accumulator-impl.md` |

---

## The `library/` directory

Reusable implementation modules that apply to **two or more patterns**. A library
file is a standalone how-to guide — not tied to any single pattern's detection
story. Pattern files (or impl files) reference library files when they need the
shared capability.

### Current library

| File | Purpose | Used by |
|------|---------|---------|
| `library/cpu-dispatch.md` | Runtime CPU dispatch: `target_clones` and `__builtin_cpu_supports` | simd-upconversion, parallel-accumulator, any future multi-variant pattern |

### Adding a library module

Create a library module when:
- The same implementation technique is needed by two or more patterns
- The technique is generic enough that it could apply to future patterns not yet written
- Writing it inline in each pattern file would create duplication that would diverge

Name the file after the technique, not the pattern: `cpu-dispatch.md` not
`simd-dispatch.md`. Add it to the table above and reference it from the relevant
pattern files.

### Current references

| File | Purpose |
|------|---------|
| `references/library-versions.md` | Known library symbols with a newer high-performance version; used by the library-version-upgrade pattern |
| `references/known-algorithms.md` | Compact index: algorithm name → common function names. Inlined into trigger files so detection costs no extra file load; add a row here when adding a new algorithm. |
| `references/known-algorithms-impl.md` | Per-algorithm implementation details: ISA levels, dispatch guards, key notes. Demand-loaded only after a function name match is confirmed — do not load speculatively. |

### Adding a reference file

Create a reference file when data needs to grow independently of pattern
logic — versioned library lists, algorithm catalogs, CPU feature tables.
Name the file after what it indexes, not the pattern that reads it.

`linux-perf` owns data collection and hotspot identification. When it identifies
a known pattern, it invokes `performance-patterns`. The `linux-perf` resolution
strategies are being migrated here over time; `linux-perf/references/resolution-strategies.md`
retains the content during the transition with a pointer to this skill.

### `vector-x86`

**ABSORBED (2026-05-18)** — `vector-x86` content has been fully absorbed into
`performance-patterns`. The `vector-x86/SKILL.md` is retained with a deprecation
notice to avoid breaking existing references, but it should not be invoked for new
work.

Absorption mapping:

| `vector-x86` capability | Destination in `performance-patterns` |
|--------------------------|--------------------------------------|
| Zipper algorithm (Capability 1) | `patterns/simd-upconversion.md` → `patterns/simd-upconversion-impl.md` |
| Vector-sequential optimization (Capability 2) | `patterns/parallel-accumulator.md` + `patterns/simd-upconversion-impl.md` (AVX-512) |
| Runtime CPU dispatch (Capability 3) | `library/cpu-dispatch.md` |

Decision to absorb: users working exclusively with `performance-patterns` were
blocked on SIMD implementation tasks because the zipper algorithm lived in a
separate skill. Single-skill completeness outweighed the modularity benefit once
the content was stable enough to maintain a copy.

The `cpu-dispatch.md` library module in this skill synthesizes the dispatch
content from `vector-x86/references/dispatch.md`. When `vector-x86` is
eventually absorbed, its zipper and vector-seq content maps to
`patterns/simd-upconversion-impl.md` and `patterns/parallel-accumulator-impl.md`.

### `cpuid-check`

Any dispatch code generated by this skill should be validated by `cpuid-check`.
This is called out in `library/cpu-dispatch.md` and in relevant pattern files.

---

## Trigger file maintenance

Each trigger file (`triggers/from-profile.md`, `triggers/from-source.md`)
contains:
1. A **quick-match table** — one row per pattern, key signal → file
2. A **description paragraph** per pattern — 4–8 sentences, enough to confirm
   a match without opening the pattern file

When adding a pattern, both trigger files must be updated. The tables are the
agent's primary lookup — keep them tight and scannable.

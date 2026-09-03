# Performance Summary — <skill-name>

<!--
Every number below is a placeholder. Fill in only real measurements from real
hardware. Never carry a number over from documentation, a blog post, or this
template — those are not data.

This file is the ONLY place in a skill where measured numbers live. SKILL.md says
how to measure and links here; a number in the body cannot be tied to a SKU, a
driver, or a batch size, so it becomes a claim nobody can check.

The arm without the skill is called the "reference config" here, never the
"baseline" — in this repository "baseline" means the Harbor no_skill arm, which is
a different measurement (an agent run, not hardware).

Publishing absolute numbers on named Intel SKUs needs Intel's approval, which a
maintainer arranges, not the contributor. For a first PR it is safer to report only
the relative gain (Nx) — that needs none.
-->

## Result

**<N>x throughput improvement** (<X> vs <Y> FPS) for <workload> <precision> on
<Intel HW SKU> versus <reference config>.

## Configuration

- Model / workload: <name and version>
- Hardware: <Intel HW SKU>, <RAM>
- OS / driver: <OS version>, <driver version if GPU or NPU>
- Reference config: <framework + version + precision>, no Intel optimization
- With skill: <Intel tool + version + optimization applied>

## Key Numbers

| Metric | Reference config | With skill | Gain |
|---|---|---|---|
| Throughput (FPS) | <X> | <Y> | **<N>x** |
| Latency P50 (ms) | <X> | <Y> | **<N>x faster** |
| Accuracy Top-1 (if applicable) | <X>% | <Y>% | <delta>% |

## Reproducibility

Run with the harness, environment and system state pinned in
`perf/benchmark_config.json`. Raw both-arm data: `perf/hw-results.json`.

run_id: <uuid — the same id as in hw-results.json>

## Caveats

- Results are specific to <HW SKU>. Other hardware will show different numbers.
- <Accuracy delta, batch size, driver version, anything that would move the number.>

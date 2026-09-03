# Templates

Copy-paste starting points. Nothing here is loaded by an agent or read by CI —
`tools/validate_skills.py` scans `skills/` only, so a placeholder in this directory
cannot fail a build, and a template cannot silently become a skill.

| Template | Copy to | Needed for |
|---|---|---|
| [`SKILL.md`](SKILL.md) | `skills/<name>/SKILL.md` | every skill |
| [`task_example.md`](task_example.md) | reference, not a file to copy | a skill written here, not imported |
| [`evals.json`](evals.json) | `skills/<name>/evals/evals.json` | optional |
| [`perf/hw-results.json`](perf/hw-results.json) | `skills/<name>/perf/hw-results.json` | `validated` |
| [`perf/summary.md`](perf/summary.md) | `skills/<name>/perf/summary.md` | `validated` |
| [`perf/benchmark_config.json`](perf/benchmark_config.json) | `skills/<name>/perf/benchmark_config.json` | `validated` |

Field reference and the rules behind each of these: [CONTRIBUTING.md](../CONTRIBUTING.md).
What `validated` needs: [MAINTAINERS.md](../MAINTAINERS.md).

## Using them

```bash
mkdir -p skills/your-skill-name
cp templates/SKILL.md skills/your-skill-name/SKILL.md
# edit it, then:
python3 tools/validate_skills.py
```

`templates/SKILL.md` ends in an HTML comment addressed to you. **Delete it before
committing** — an agent loading the skill would pay context for it.

`templates/evals.json` is schema-valid as it stands — deliberately, so that
`run_evals.py --validate` fails on *your* content being wrong rather than on the template's
shape. One thing it does not pass on a straight copy: `skill_name` is still
`your-skill-name`, and the validator requires it to equal the directory name. That is the
first error you will see, and it is the intended reminder that the file is unedited.

## Do not fill in `perf/` from a template

Every number in the `perf/` templates is a placeholder. They exist to show the *shape* —
two arms in one file, one `run_id`, the same hardware and workload on both sides, only the
software and the agent's code differing. Real numbers come from a measured run on real
Intel hardware, and no skill needs them to merge: `perf/` is for `validated`, and a
maintainer arranges the run as part of the promotion.

`perf/summary.md` is the only place in a skill where measured numbers belong.

One field in `hw-results.json` is computed rather than written: `skill_behavior_sha256`,
the digest of the files that determine what the agent does. Take it from the tree the run
measured, never by hand:

```bash
python3 tools/behavior_digest.py skills/your-skill-name
```

The validator recomputes it on every `validated` skill and fails the skill when it has
drifted — that is the check that stops numbers from outliving the text they measured.

## There is no template for an imported skill

A skill brought here from another repository keeps upstream's text as it was, and the
directory is not written by hand at all: the pin goes in `skills.yaml`, `python3
tools/sync_external.py --write` generates the files and the `.source.json` beside
`SKILL.md`, and `--check` re-fetches the pinned commit and byte-compares. Do not start from
this template and paste upstream's content into it — reorganising the text means existing
measurements of the original no longer describe the file here, and the next `--check` would
fail on the difference anyway.

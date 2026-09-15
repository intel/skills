# Contributing a skill

This file states the whole merge bar: the walkthrough, the format field by field, what you
can run locally, and what CI blocks on. [README.md](README.md) is the landing page;
[MAINTAINERS.md](MAINTAINERS.md) is what happens *after* a merge and is not asked of you.

To contribute you need a GitHub account, knowledge of the subject, and an editor. No Intel
hardware, no benchmark data, no credentials — nothing a fork cannot reach.

There are two ways in, and they ask for different things.

| | You are bringing an existing skill | You are writing a new skill |
|---|---|---|
| Where it came from | it already lives in an Intel project repository | you are writing it here, now |
| What is required | `SKILL.md` + two lines in `skills.yaml` | the same, plus one Harbor task |
| Why | it has been used and reviewed where it was written; re-proving that here buys nothing | a skill nobody has exercised needs one runnable check that it describes something real |

Everything else this repository can do — `evals/evals.json`, `perf/` measurements, a
capability suite, the three-arm differential — is available to you and required of nobody.
Add it when you want the stronger claim.

## What makes a skill worth merging

Product documentation says what an API supports. A skill encodes how someone who has done
the task before applies it, which is a different document:

| Documentation | Expert workflow |
|---|---|
| describes the available APIs and options | starts from an outcome the user asked for |
| assumes the reader knows their own environment | inspects hardware, driver, runtime, versions, limits |
| presents several valid choices | picks one supported path and says why |
| gives commands | runs an ordered procedure with safeguards |
| explains the expected behaviour | tests whether it actually happened |
| lists known issues | recognizes the failure signal and takes a recovery path |
| ends when the feature is explained | returns a result and the evidence for it |

Before writing, answer these — a reviewer will ask them, and the answers are the skill's
outline:

1. What user intent should activate this skill? (this is the `description`)
2. What must the agent inspect before it acts?
3. What decisions does it have to make, and on what evidence?
4. What ordered actions does it take?
5. How is success verified — what does the agent run to see that it worked?
6. Which failures have known recovery paths?
7. What result and evidence does it return?
8. When should it hand off to another skill instead of continuing?

A skill that answers 1, 4 and 7 only is a tutorial. The shape to aim for:

```text
Recognize → Inspect → Decide → Act → Verify → Recover → Report
```

[`skills/vllm-xpu-run`](skills/vllm-xpu-run) is the worked example. It recognizes a request
for an OpenAI-compatible endpoint on an Intel GPU; inspects image, model architecture,
`/dev/dri` access and available memory; decides dtype, attention backend, quantization and
KV-cache pairing, and whether the transformers backend fallback is needed; launches the
container; sends a real generation request and confirms the work landed on the GPU;
diagnoses an unsupported architecture, an OOM, a oneCCL initialization failure or a missing
device node; and hands off to `vllm-xpu-bench` the moment the question becomes "how fast is
it?". That is the part a Markdown copy of the vLLM documentation does not carry.
[`skills/linux-perf`](skills/linux-perf) is the same shape on a CPU scalability problem,
including two hand-offs: benchmark setup to `phoronix-test-suite`, a diagnosed pattern to
`performance-patterns`.

## 1. Fork, clone, create the directory

```bash
git clone https://github.com/<your-username>/skills && cd skills
mkdir -p skills/your-skill-name          # kebab-case, ≤ 64 characters
cp templates/SKILL.template.md skills/your-skill-name/SKILL.md
```

The directory name is the skill's identity — it must equal `name:` in the frontmatter.

## 2. Write `SKILL.md`

### Frontmatter

Two required keys — the two the [agentskills.io](https://agentskills.io) specification
requires, and no more. Every key below is one that specification defines. Anything else you
add is optional and is passed through untouched.

| Key | Required | Rules |
|---|---|---|
| `name` | yes | kebab-case, ≤64 characters, equal to the directory name |
| `description` | yes | what the skill covers and when to load it. ≤1024 characters; the ones here run 450–600 |
| `license` | no | an SPDX identifier. Leave it out and the skill is Apache-2.0, the repository's own licence; if you do write it, it has to match what `skills.yaml` says |
| `compatibility` | no | which agents or platforms the skill assumes |
| `allowed-tools` | no | the tools the skill expects to be able to use |
| `metadata` | no | free-form mapping: tags, languages, a version, an author |

Intel catalog fields do not go here. They go in `skills.yaml`, which is what keeps
`SKILL.md` portable: the file an agent loads carries nothing specific to this repository.

### `description` is the hardest field in the file

Under progressive disclosure it is the **only** text in the agent's context when it decides
whether to open your skill at all. A description missing its own domain vocabulary makes
the skill unreachable no matter how good the body is, and it fails silently: nothing
errors, the agent simply never picks it.

Write it for the words a user types, not for the canonical product name. Then test it:
write down the requests a user would really type, in their own words, covering the
different ways the subject gets asked about, and check which of them the description alone
would route to your skill. Each one it misses names a word the description lacks. There is
no target number — a handful of requests you have actually been asked is worth more than a
generated list. Saying what the skill is *not* for is worth as much as saying what it is
for: it stops the agent opening it for the wrong request.

### Body

No required headings. Write the document your reader needs. The rules below are what CI
checks; [agentskills.io/skill-creation](https://agentskills.io/skill-creation) has general
writing guidance, which is advice rather than a schema this repository enforces.

| Rule | Why |
|---|---|
| ≤500 lines (a warning at 250) | past that an agent stops reading before the end |
| every file the skill ships is mentioned by the path it lives at | an unmentioned file is an instruction nobody declared. A warning, not a failure, for an imported skill — see below |
| every path mentioned exists | a dead path makes the agent improvise |
| no measured numbers | your text has to stay true on hardware you did not test |
| nothing this repository will not publish | see below |

A measured number belongs in `perf/`, not in the body. "Faster on a GPU for large arrays"
is a claim the text can carry anywhere; "3.4× faster" is a claim about one machine on one
day.

Every file the skill ships — the body, `references/`, `scripts/`, anything else — is
scanned for content that would make an agent act against the person running it: an install
piped into a shell, a destructive delete, an instruction addressed to the agent's operator,
a route for a secret out, or a way to switch a protection off. `evals/` and `perf/` are
excluded, because an eval case has to be able to name the phrase it tests for.

Worked example: [`skills/dpnp-quickstart/SKILL.md`](skills/dpnp-quickstart/SKILL.md).

### Optional directories

| Path | What goes in it |
|---|---|
| `references/` | detail the skill loads on demand, so the body stays short |
| `scripts/` | helpers the skill runs. Keep them readable; they are instructions too |
| `evals/evals.json` | cases recording what a correct answer contains |
| `perf/` | hardware measurements. Three files together — see `templates/perf/` |

None of these is required to merge. They are how a skill earns `validated` later, which is
[MAINTAINERS.md](MAINTAINERS.md).

## 3. Add the catalog entry

Two lines in [`skills.yaml`](skills.yaml) — the skill's name, and a GitHub handle to route a
bug report to:

```yaml
- name: your-skill-name
  maintainer: "your-github-handle"
```

`maintainer` is a GitHub handle, not a corporate username — the two are rarely the same.
Nothing validates it beyond "not empty", so check that `github.com/<handle>` is you.

Everything else in the entry is optional:

| Field | What it does |
|---|---|
| `status` | defaults to `published`. A maintainer moves it, in a separate pull request |
| `license` | defaults to `Apache-2.0`. Set it when the skill arrived under other terms, and that is the value review reads |
| `intel-products` | comma-separated product names. Fill it in and the validator checks your description carries their vocabulary; leave it out and that check does not run |
| `intel-hw-class` | `cpu`, `gpu`, `npu`, or `accelerator` |
| `intel-hw-validated-on` | specific SKUs. Required only for `validated` |
| `intel-source-ledger` | path to `references/official-sources.md`. Required only for `validated` |

An empty value means "does not apply", not "to be filled in later". Leave the field out
rather than writing an empty string.

### If the skill is maintained in another repository

Do not copy it by hand. Add the pin to your `skills.yaml` entry — `external-repo`,
`external-commit` (a full 40-character SHA, because a branch or tag would let the copy
drift), `external-path`, `external-license` — and run:

```bash
python3 tools/sync_external.py --write
```

That generates the directory and the `.source.json` beside `SKILL.md`, so what review reads
is the pin and the diff it produced:

```json
{
  "repo": "https://github.com/intel/some-repo",
  "path": "skills/some-skill",
  "commit": "<full sha>",
  "license": "MIT"
}
```

That file is what tells review the skill is an import rather than a first draft, and it is
why an import does not need a Harbor task. It also relaxes one rule: the body is upstream's
text, kept as it arrived, so a file it never mentions by path warns instead of failing.
Editing their document to satisfy our validator would break the thing that makes an import
worth having — that the text has already been used and measured where it was written. The
content rules do not relax: every file an import ships is scanned like every other.

## 4. Run the local gate

Keyless, offline, no setup beyond Python 3.11 or newer — standard library only, nothing to
install:

```bash
python3 tools/validate_skills.py                 # every offline check CI blocks on
python3 tools/validate_skills.py --check-links   # also checks external URLs; needs network
python3 tools/run_evals.py --validate            # only if you wrote evals/evals.json
python3 tools/sync_external.py --check           # only if you imported a skill
```

The first and third need no network. If they pass, the blocking checks left are about the
repository rather than your text: the workflow linters and the installer round trip.

### The security scan

Every skill is also scanned by [NVIDIA SkillSpector](https://github.com/NVIDIA/skillspector)
— static patterns, an AST pass over shipped scripts, YARA signatures, and an OSV.dev lookup
for named dependencies. It needs the scanner, so it is not part of the offline gate above,
but it is one command with [`uv`](https://docs.astral.sh/uv/) installed:

```bash
commit=$(sed -n 's/.*SKILLSPECTOR_COMMIT: \([0-9a-f]\{40\}\).*/\1/p' \
  .github/workflows/skillspector.yml)
uv run .github/scripts/skillspector_baseline.py --skill your-skill-name --output base.yaml
uvx --python 3.12 --from "git+https://github.com/NVIDIA/skillspector.git@${commit}" \
  skillspector scan skills/your-skill-name --no-llm --format json \
  --baseline base.yaml --output report.json
uv run .github/scripts/skillspector_gate.py --report report.json --skill your-skill-name
```

The commit is read out of the workflow so the version you run is the version CI runs.

Two thresholds, because the two kinds of skill can act on a finding differently:

- **A skill written here fails above 20** — SkillSpector's `SAFE` band. A finding in it can
  be fixed in the pull request that reports it.
- **An imported skill fails above 50** — where the scanner itself says `DO_NOT_INSTALL`. Its
  body stays byte-for-byte the pinned upstream commit, so the repair lands upstream.

Active HIGH/CRITICAL findings below the threshold are reported, not failed.

If the finding is real, fix it. If it is a false positive or a pattern this catalog
documents on purpose, add a rule to [`.skillspector-baseline.yaml`](.skillspector-baseline.yaml)
with a `reason`, and list your skill under `skills:` — a rule without it applies to every
skill in the catalog. The file is the audit trail, so each suppression must be justified.
Rules are not pinned to a specific version of your skill: they keep applying after the
surrounding text is reworded.

## 5. If you are writing a new skill, add a Harbor task

One task under [`evaluation/harbor/tasks/`](evaluation/harbor/tasks): a `task.toml` naming
your skill, a container, an instruction, an oracle solution, and a verifier. CI runs the
oracle arm — no model and no API key, so it works on a pull request from a fork — and
requires only that the task is solvable and its verifier emits a reward.

Be clear on what that does and does not do. The oracle applies your reference solution; it
never reads `SKILL.md`. So the task does not score your skill — it makes your skill
**measurable**, which is what lets a maintainer later run the differential that does score
it. At merge time the judgement of your skill's content is a human reading it. One task is
the floor; a skill reaching `validated` needs five, two of which discriminate — the policy
is in [`evaluation/harbor/suites.json`](evaluation/harbor/suites.json).

Layout and a worked example: [`templates/task_example.md`](templates/task_example.md).

Before you push, check that the task can still tell the arms apart:

```bash
python3 tools/lint_task_leakage.py --task your-skill-first-task --show
```

It reports which of the symbols your skill teaches the instruction already hands over. A
task whose `instruction.md` contains the answer is passed with or without the skill and
measures nothing, at the same price as one that measures something. Keyless and offline,
like the rest of the local gate.

If your skill cannot be exercised without an Intel GPU, say so in the pull request and a
maintainer will decide — a task only that team's hardware can run is not a gate, it is a
favour someone does.

## 6. Open the pull request

Describe the skill, say which of the two ways in it came through, and complete the PR
checklist.

## What CI checks

Blocking, keyless, and runnable on a fork:

- `SKILL.md` parses; `name` is kebab-case, ≤64 characters, and equals the directory name
- `description` is present and ≤1024 characters
- the body is ≤500 lines (250 warns), mentions every file the skill ships — a warning
  rather than a failure for an imported skill, whose body belongs to another team — and
  every path it mentions exists
- the licence `skills.yaml` states is one this repository publishes, and a `license` in
  `SKILL.md` agrees with it
- no file the skill ships carries content this repository will not publish — a piped
  install script, a destructive delete, an instruction aimed at the agent's operator, a
  route for a secret out, or a way to switch a protection off
- `skills.yaml` has an entry with a maintainer, and the catalog and the tree agree
- the workflows themselves lint clean (`actionlint`, `zizmor`)
- SkillSpector scores the skill within the threshold its origin is held to — 20 for a skill
  written here, 50 for an imported body — with suppressions and their reasons in
  `.skillspector-baseline.yaml`
- for a new skill: its Harbor task is solvable, oracle reward 1.0
- for an imported skill: `skills.yaml`, `.source.json` and `NOTICE` agree, and the copy is
  still byte-for-byte the pinned upstream commit
- `npx … install` writes every skill in the catalog, and `verify` accepts each one and
  rejects an installed copy that was altered
- a link that answers 404 or 410 — a pointer an agent would follow into nothing; a warning
  rather than a failure in an imported body, for the same reason as the mentions check, and
  because the repair has to land upstream and arrive here through a moved pin. A timeout, a
  5xx or rate limiting only warns, so an outage elsewhere cannot hold up a pull request

Reported but not blocking: the coverage gaps between what a suite claims and what it
implements, a dead link in a body this repository copied rather than wrote, and a
SkillSpector HIGH/CRITICAL finding in a skill whose score is still within its threshold.

## Evaluation levels

Three levels, split by what each can afford to require. The split is the point: a gate that
needs a paid API key cannot block a pull request, and a gate that cannot block is not a
gate.

| | Level 1 — structure | Level 2 — differential | Level 3 — discoverability |
|---|---|---|---|
| Question | is the skill well-formed and reachable? | can an agent complete real work with it? | does an agent reach for it unprompted? |
| Needs | nothing | inference + Docker | inference |
| Runs | every PR, including forks | on promotion, by maintainers | on promotion, by maintainers |
| Blocking | **yes** | no | no |
| You run it | yes | no | no |

Level 1 is the only one you run, and the only one that can stop a merge. Level 2 is the
three-arm differential in [`evaluation/harbor/`](evaluation/harbor) — the agent attempts
real containerized tasks with no skill, with the previous version, and with the candidate,
and the gate is `candidate − no_skill`, because a skill that does not beat the no-skill arm
has not shown it does anything. Level 3 asks whether an agent opens the skill when nothing
names it. Both need an inference credential no fork can hold, so both are run by maintainers
by hand with the results attached to the pull request, and neither is asked of a
contributor.

Why these two and not a question set — and how prose deliverables are scored inside the same
differential — is in [MAINTAINERS.md](MAINTAINERS.md).

## Skill lifecycle

| Status | What it means | What it requires |
|---|---|---|
| `published` | in the catalog, and agents load it | `SKILL.md`, a catalog entry, and — for a skill written here — a Harbor task |
| `validated` | carries benchmark evidence from real Intel hardware | plus `perf/`, `references/official-sources.md`, `intel-hw-validated-on`, a task portfolio meeting the suite policy, and a differential run that cleared its gate |

Status lives in [`skills.yaml`](skills.yaml), never in `SKILL.md`. A first pull request
lands at `published`. `validated` is where a maintainer adds hardware evidence, in a
separate pull request; an external contributor can reach it but is never asked to. A skill
is also a claim — *give an agent this text and it does better work* — and this repository
exists to test that claim, which is why measurement is not a condition of merging and is a
condition of `validated`.

## Licence

The paragraph below is Intel's standard contributor text, with the project name and the
link to [LICENSE](LICENSE) filled in. It is what makes a contribution inbound-licensed, and
it is the only thing this repository asks you to agree to — there is no separate sign-off,
no per-commit trailer, and no CLA.

Intel Skills is licensed under the terms in [LICENSE](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Review

CI covers form. A reviewer supplies what no keyless check can: whether an agent that loaded
this would actually do better work, whether the description uses the words a user would
type, and whether every technical claim is true. Expect questions about the description —
it is the part of a skill most often rewritten before merge.

# Maintainers

Everything this repository can do beyond merging a skill. Nothing here is asked of a
contributor: [README.md](README.md) states the whole merge bar, and this file describes
what happens after — how a skill earns `validated`, what the measurement machinery is,
and what is deliberately not measured.

If you are contributing a skill, you do not need this file.

## Repository layout

| Path | What is in it |
|---|---|
| `skills/<name>/SKILL.md` | the skill: the only required file, the only one an agent must read |
| `skills/<name>/references/` | detail the skill loads on demand |
| `skills/<name>/scripts/` | helpers the skill ships |
| `skills/<name>/evals/evals.json` | eval cases: what a correct answer contains. Optional |
| `skills/<name>/perf/` | hardware measurements. Required for `validated`, optional before |
| `skills/<name>/.source.json` | present when the skill was imported: upstream repo, path, commit |
| `skills.yaml` | the catalog. Maintainer, status, and the optional Intel product fields |
| `evaluation/harbor/` | the task suites, the runner config, and the suite policy |
| `templates/` | starting points: `SKILL.md`, `task_example.md`, `evals.json`, `perf/` |
| `schemas/` | JSON Schema for the files that have one |
| `tools/` | every check, all stdlib-only Python, all runnable offline |
| `bin/intel-skills.mjs` | the installer `npx github:intel/skills` runs. Node 20+, no dependencies |
| `.github/workflows/` | CI. What blocks a merge is listed under [CI](#ci) below |

## Lifecycle and promotion

Two statuses, in `skills.yaml` and nowhere else:

| Status | Means | Requires |
|---|---|---|
| `published` | in the catalog, agents load it | `SKILL.md`, a catalog entry, and — for a skill written here — one Harbor task |
| `validated` | carries benchmark evidence from real Intel hardware | everything below |

A first pull request lands at `published`. Nothing a contributor does moves a skill to
`validated`; a maintainer does, in a separate pull request from the one that added the
skill, and that pull request carries the evidence.

`validated` requires all of:

- a task portfolio meeting the suite policy — five tasks, two of them discriminating,
  covering all five capability classes
- a Level 2 differential run that clears the gate, with the report attached
- `perf/` measurements from named hardware, with `intel-hw-validated-on` naming the SKUs
- `references/official-sources.md`, and `intel-source-ledger` pointing at it

An imported skill never reaches `validated` here. Its body is upstream's bytes at a pinned
commit; promoting it would be a claim about a file this repository does not own. What is
verified instead is that the copy still matches the commit it is pinned to, and that the
commit still resolves.

## Level 1 — structure

`python3 tools/validate_skills.py`. Keyless, offline, runs on every pull request
including from a fork, and is the only level that can block a merge. What it enforces is
in the README; what it reports without blocking is a dead link in an imported body — a
dead link in a skill written here fails — and the coverage gaps between what a suite
claims and what it implements.

Two things it deliberately does not do. It does not check that `SKILL.md` carries no
measured numbers — that rule is enforced by review, because a validator cannot tell a
benchmark result from an API constant. And it does not check a description's vocabulary
unless the skill's catalog entry fills in `intel-products`; without it, the check reports
that it did not run rather than passing silently.

## Level 2 — the differential

The three-arm run in `evaluation/harbor/`. An agent attempts real containerized tasks in
three conditions — with no skill, with the skill's previous version, and with the
candidate — and the arms are compared on mean reward.

```
tools/compare_harbor_skill.py --skill <name> --report-only
```

Policy lives in `evaluation/harbor/suites.json`:

| Setting | Value |
|---|---|
| Gate | `candidate − no_skill ≥ 0.10` on mean reward |
| Arms | `no_skill` (never optional), `previous` (skipped on a first evaluation), `candidate` |
| Attempts | 1 for a development probe, 3 for calibration, 5 for promotion |
| Maximum regression | 0.10 per task, 0.03 across the suite |
| Held constant across arms | task revision, agent, model, attempt count, timeouts |

Three properties of this design are load-bearing:

**The baseline is the no-skill arm, not a target score.** A skill that cannot beat an
agent working without it has not shown that it does anything, however high its absolute
reward.

**Tasks are in this repository, not pinned by reference.** A number nobody outside Intel
can reproduce is not evidence.

**A tie is a verdict on the suite, not on the skill.** If every arm scores 1.0, the tasks
are at the ceiling and cannot show a delta in either direction. Such tasks are marked
`calibration: "ceiling"` in `suites.json` and no skill may cite an improvement from that
run. `tools/lint_task_leakage.py` ranks how much of a task's answer its own instruction
leaks, which is the usual cause.

Before trusting a suite, run it against a deliberately falsified copy of the skill as one
arm. A suite that scores a lying skill as highly as the real one is inert. Two cautions
learned from doing it: keep the falsification blind, because anything the skill says
about itself — a version string, a warning comment — is part of the treatment and an
agent will read it and discard the skill; and treat a three-attempt result as
probabilistic, since the same false claim is followed on one attempt and ignored on the
next.

## Level 3 — discoverability

The skill sits in the agent's skills directory, requests a user would really type arrive
with nothing naming the skill, and the report says how often it was opened and what was
opened instead. Reported rather than blocking: what it finds is a description, and a
description is cheap to rewrite. The authoring-time version of the same check is the
description test in CONTRIBUTING.md.

## Why no level scores answers against a question set

A question set is the right instrument for comparing models, harnesses, or documentation
sources: it holds the task fixed and varies the thing under test. It is the wrong
instrument for promoting a skill.

A judge scoring an answer against expected topics rewards *mentioning* the right things,
and a skill can raise topic coverage without changing what the agent does. The change in
behaviour is the claim a skill makes, so the evidence has to be behaviour.

Where prose genuinely is the deliverable — an honest no, a refusal to promise a speedup —
it goes through an answer-track task with a rubric inside the same three-arm run, so the
reward stays comparable across arms and reproducible by anyone outside Intel. Criteria in
a task's `tests/rubric.json` are evaluated as regular expressions; check terms in
`evals.json` are not (see below).

## Task portfolio

Per skill, for `validated`:

| Requirement | Value |
|---|---|
| Tasks | 5 |
| Discriminating tasks | 2 |
| Capability classes covered | `correctness`, `selection`, `integration`, `debugging`, `performance` |

A shortfall blocks promotion. It does not block a pull request, and it does not block a
skill from sitting at `published` with a partial portfolio — the validator reports the
gap on every run so it stays visible rather than forgotten. Current gaps are whatever
`tools/validate_skills.py` prints as `WARN suites.json: …`.

A task is *discriminating* when an agent without the skill plausibly fails it. A task
that any competent agent solves from general knowledge measures the model, not the skill.

Adding a task: `templates/task_example.md`, then `evaluation/harbor/README.md` for the
runner's own options.

A contributor's task arrives claimed by no suite, and the validator warns about it on
every run until one exists. Claiming it is this side of the line: the capability it
covers, its class, whether it discriminates, and — the first time a skill gets an
implemented task — `evaluation/harbor/instructions/use-<skill>.md`, the neutral
instruction the skill arms receive. None of that is a judgement a first-time author has
the portfolio in view to make, and the task is run by the oracle job either way.

## Eval files

`skills/<name>/evals/evals.json` records what a correct answer contains: a real user
request, prose describing a correct response, and literal check terms.

```
tools/run_evals.py --validate                    # schema gate, runs on every PR
tools/run_evals.py --answers recorded.json        # scoring, needs answers from elsewhere
```

Matching is literal substring containment after lowercasing and whitespace collapse.
There is no regex: a pattern written into `must_include` can never match, and one written
into `must_not_include` silently matches nothing, which is worse. Anything needing a
pattern belongs in a Harbor task rubric.

An eval case may contain a phrase a skill is forbidden to contain — that is the point of
`must_not_include` — which is why the content-safety scan excludes `evals/` and `perf/`
and scans everything else the skill ships.

## Performance evidence

`skills/<name>/perf/` needs three files together: `hw-results.json`,
`benchmark_config.json`, and `summary.md`. Templates for all three are in
`templates/perf/`. The measured numbers live here and are summarised in `perf/summary.md`
— never in `SKILL.md`, whose text has to stay true across hardware an agent may be
running on.

Every claim in `perf/` needs the hardware named in `intel-hw-validated-on` and a source
in `references/official-sources.md`.

## Importing a skill from another repository

What is written by hand is the pin: `external-repo`, `external-commit`, `external-path`
and `external-license` in `skills.yaml`. The directory under `skills/` is then generated
from it — `python3 tools/sync_external.py --write` — so review is review of the pin and of
the diff it produces, never of hand-copied bytes. The generated copy ships a
`.source.json` repeating those four, so an installed skill carries its own provenance away
from this catalog, and `NOTICE` names each upstream this repository republishes.

An import does not need a Harbor task: the skill has been used and reviewed where it was
written, and that is what the pin records. Keep the imported text as it was — reorganising
it means existing measurements of the original no longer describe the file here. The one
edit the generator does make is to repository-relative command paths, which would
otherwise point at upstream's layout and resolve to nothing once the skill is installed.
Every file it touches is listed in `.source.json` under `modified-files` and carries a
one-line notice saying so, and the validator fails if either is missing.

That is also why two checks report an imported body as a warning where they would fail a
skill written here: an unmentioned file, and a link that has gone 404. Both name a real
defect and neither can be fixed in this repository — editing the body breaks the
byte-compare in `sync_external.py --check`, which is the thing that proves the copy is
still what was reviewed. The route is a pull request upstream, then move
`external-commit` and re-run `--write`. A warning that outlives a release is worth
raising with the upstream maintainer rather than living with; if upstream will not take
the fix, the pin is the wrong pin.

## CI

| Workflow | Job | Runs on | Blocks? |
|---|---|---|---|
| `validate.yml` | `validate` — `validate_skills.py`, `run_evals.py --validate`, link check | every PR | yes |
| `validate.yml` | `install` — the installer resolves, lists, and installs from the catalog | every PR | yes |
| `dco.yml` | sign-off on every non-merge commit, failing by commit SHA | every PR | yes |
| `harbor-smoke.yml` | the oracle arm over every task in `tasks/` | PRs touching tasks or skills | yes |
| `security.yml` | `actionlint`, `zizmor` | every PR | yes |
| `codeql.yml` | code scanning, Python | PRs, push, weekly | reports |

`codeql.yml` runs its job only where the repository is public, which it reads from the event
rather than being told. Uploading results needs GitHub Advanced Security, which a public
repository has and a private one does not, so in a private copy of this tree the analysis
would succeed and only the upload would fail — a check that is always red, which teaches
people to ignore red checks. Anyone holding this tree privately should not count it as
coverage.

There is no secrets-scanning job either: the action for it needs an organisation licence key
this repository has no secret for, so the job could only ever fail. GitHub's own secret
scanning covers it instead.

The oracle arm applies each task's `solution/solve.sh` and never reads `SKILL.md`. It
proves a task is solvable and its verifier emits a reward — nothing about the skill. It
runs on a pull request because it needs no model and no API key.

Levels 2 and 3 do need inference credentials, so they are run by hand and their reports
attached to the pull request. Wiring an arm that needs a credential into a workflow that
must also run on forks is unsolved; a gate nobody can run on a fork is not one this
repository will pretend to have.

## Tools

All stdlib-only Python 3.11 or newer — 3.11 for `tomllib`, and the two tools that need it
say so rather than failing as a missing module. None of them ships in the installable
package. Two reach the
network and both say so when they cannot: `validate_skills.py --check-links` and
`sync_external.py`. Everything else runs offline.

| Script | What it does |
|---|---|
| `validate_skills.py` | Level 1: every structural check, and the catalog ↔ tree bijection |
| `run_evals.py` | validates eval files against their schema; scores recorded answers |
| `compare_harbor_skill.py` | runs and reports the three-arm differential, with cost and time |
| `check_harbor_job.py` | asserts a harbor run's trial count and reward floor |
| `lint_task_leakage.py` | ranks how much of its own answer each task's instruction leaks |
| `behavior_digest.py` | digests the skill bytes a measurement was taken against, so a later edit to `SKILL.md` cannot leave `perf/` describing text that no longer exists |

Two more exist for the imported skills: `sync_external.py` regenerates a copy from its pin
with `--write`, or with `--check` re-fetches the pinned commit and byte-compares what is
here against it, and `upstream_git.py` fetches just the pinned subtree instead of the
repository around it.

## Current state

Recorded here rather than implied, because a repository about evidence should not be
vague about its own:

- **No skill carries measured numbers yet.** The maintained skills were adapted from Intel
  material that was measured elsewhere, and rewritten on the way in, so every existing run
  measured different bytes. `perf/` stays empty and no skill claims `validated` until it is
  re-measured against the text that is actually here.
- **Most skills have no runnable task.** A runnable task is not a measured skill, and a
  declared task is not a runnable one; `suites.json` distinguishes `implemented` from
  `planned` and the validator reports the difference.
- **Level 2 has been run, and returned a tie.** Every arm scored 1.0 on the tasks
  available, so the suite was at the ceiling. The tasks that could produce a nonzero
  delta are the discriminating ones still marked `planned`.

## Reviewing a pull request

CI covers form. What a reviewer has to supply is the judgement no keyless check can:

1. Would an agent that loaded this actually do better work, or does it read as
   documentation? A skill is a claim about behaviour.
2. Does the description contain the words a user would type — not the canonical product
   names? Test it against requests you invent yourself.
3. Is every technical claim true and free of a number that will age?
4. If the skill was imported, does `.source.json` point at a commit that exists, and is
   the licence recorded correctly?
5. Is `maintainer` a GitHub handle belonging to the person you think it does? Nothing
   validates it beyond "not empty", so open `github.com/<handle>` once. A plausible
   handle can quietly credit a stranger.

# Contributing a skill

The field-by-field reference. [README.md](README.md) has the walkthrough and states the
whole merge bar; this file is what to look up while you write.

To contribute you need a GitHub account, knowledge of the subject, and an editor.
No Intel hardware, no benchmark data, no credentials. Start from
[`templates/SKILL.md`](templates/SKILL.md).

```bash
git clone https://github.com/<your-username>/skills && cd skills
mkdir -p skills/your-skill-name
cp templates/SKILL.md skills/your-skill-name/SKILL.md
# write it, then:
python3 tools/validate_skills.py
git commit -s -m "Add your-skill-name skill"      # -s is the DCO sign-off
```

## `SKILL.md` frontmatter

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

Under progressive disclosure it is the **only** text in the agent's context when it
decides whether to open your skill at all. A description missing its own domain
vocabulary makes the skill unreachable no matter how good the body is, and it fails
silently: nothing errors, the agent simply never picks it.

Write it for the words a user types, not for the canonical product name. Then test it:
write down the requests a user would really type, in their own words, covering the
different ways the subject gets asked about, and check which of them the description alone
would route to your skill. Each one it misses names a word the description lacks. There is
no target number — a handful of requests you have actually been asked is worth more than a
generated list. Saying what the skill is *not* for is worth as much as saying what it is
for — it stops the agent opening it for the wrong request.

## Body

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

A measured number belongs in `perf/`, not in the body. "Faster on a GPU for large
arrays" is a claim the text can carry anywhere; "3.4× faster" is a claim about one
machine on one day.

Every file the skill ships — the body, `references/`, `scripts/`, anything else — is
scanned for content that would make an agent act against the person running it: an
install piped into a shell, a destructive delete, an instruction addressed to the agent's
operator, a route for a secret out, or a way to switch a protection off. `evals/` and
`perf/` are excluded, because an eval case has to be able to name the phrase it tests for.

### Optional directories

| Path | What goes in it |
|---|---|
| `references/` | detail the skill loads on demand, so the body stays short |
| `scripts/` | helpers the skill runs. Keep them readable; they are instructions too |
| `evals/evals.json` | cases recording what a correct answer contains |
| `perf/` | hardware measurements. Three files together — see `templates/perf/` |

None of these is required to merge. They are how a skill earns `validated` later, which
is [MAINTAINERS.md](MAINTAINERS.md).

## `skills.yaml`

Two lines:

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

If the skill came from another repository, ship a `.source.json` beside `SKILL.md`:

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

## A Harbor task — new skills only

Required only if you are writing the skill here rather than bringing one that already
lives in an Intel project repository. One task under `evaluation/harbor/tasks/`: a
container, an instruction, an oracle solution, and a verifier. CI runs the oracle arm —
no model, no API key, so it works from a fork — and requires only that the task is
solvable and its verifier emits a reward.

The oracle applies your reference solution and never reads `SKILL.md`, so the task does
not score your skill; it makes the skill measurable, which is what lets a maintainer run
the differential later. Layout and a worked example:
[`templates/task_example.md`](templates/task_example.md).

Before you push, check that the task can still tell the arms apart:

```bash
python3 tools/lint_task_leakage.py --task your-skill-first-task --show
```

It reports which of the symbols your skill teaches the instruction already hands over. A
task whose `instruction.md` contains the answer is passed with or without the skill and
measures nothing, at the same price as one that measures something. Keyless and offline,
like the rest of the local gate.

If your skill cannot be exercised without an Intel GPU, say so in the pull request and a
maintainer will decide. A task only that team's hardware can run is not a gate.

## The local gate

```bash
python3 tools/validate_skills.py                 # every offline check CI blocks on
python3 tools/validate_skills.py --check-links   # also checks external URLs; needs network
python3 tools/run_evals.py --validate            # only if you wrote evals/evals.json
```

Keyless, stdlib-only, and Python 3.11 or newer — the first and third need no network. If
all three pass, the blocking checks left are about the repository rather than your text: the
DCO sign-off on every commit, the workflow linters, the installer round trip, and — if you
imported a skill from another repository — `python3 tools/sync_external.py --check`.

## Licence and sign-off

The two sections below are Intel's standard contributor text. The project name and the link
to [LICENSE](LICENSE) are filled in; the wording and the Developer Certificate of Origin it
quotes are unchanged, and the DCO itself may not be changed by anyone.

### License

Intel Skills is licensed under the terms in [LICENSE](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

Two shortcuts for when you forget:

```bash
git commit --amend -s                # the last commit
git rebase HEAD~N --signoff          # several
```

CI checks every non-merge commit and fails by commit SHA.

## Review

CI covers form. A reviewer supplies what no keyless check can: whether an agent that
loaded this would actually do better work, whether the description uses the words a user
would type, and whether every technical claim is true. Expect questions about the
description — it is the part of a skill most often rewritten before merge.

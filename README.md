# Intel Skills

Agent skills for Intel software. A skill is a small Markdown directory an agent reads
to understand how to help with a specific task on Intel hardware. The agent loads it
when your request matches, and then gives Intel-specific guidance instead of a
generic answer.

> You say: *"This NumPy code takes minutes — can I run it on my GPU?"*
>
> Without a skill: the agent reaches for CuPy or `torch.cuda`, neither of which runs on
> an Intel GPU.
>
> With an Intel skill: the agent recognizes the Intel GPU, moves the array code to dpnp —
> NumPy's API on Intel devices — shows how to confirm the work actually landed on the GPU,
> and says where the switch is *not* worth making.

## What is a skill?

A directory under [`skills/`](skills) with one required file, `SKILL.md`: Markdown the
agent reads, headed by a short frontmatter block that says what the skill covers and
when to load it. Optional beside it — `references/` for detail loaded on demand,
`scripts/` for helpers the skill ships, `evals/` for the cases that prove it works,
`perf/` for hardware measurements. Nothing is compiled and nothing is installed; an
agent reads text.

Frontmatter has two required keys — `name` and `description` — and the body
has **no required headings**: write the document your reader needs, keep it under 500 lines,
and keep measured numbers out of it. Field by field: [CONTRIBUTING.md](CONTRIBUTING.md).

Intel catalog fields — maintainer, status, products, hardware, licence — live in
[`skills.yaml`](skills.yaml), never in `SKILL.md`. That is what keeps a skill portable:
the file an agent loads carries nothing specific to this repository. Two of them are
required, `name` and `maintainer`; the rest are optional and describe a skill more
precisely once someone wants to.

A skill is also a claim: *give an agent this text and it does better work.* This
repository exists to test that claim. Measurement is not a condition of merging — but it
is required for skills to be tagged as `validated`, and nothing here states a number it has not
measured.

## Using skills

To install and manage skills, use the skills CLI (requires Node 20 or newer, no dependencies, and nothing from npm):

```bash
npx github:intel/skills list                                     # the catalog, and which kind
npx github:intel/skills install linux-perf --target claude-code  # -> ~/.claude/skills
npx github:intel/skills install --all --target agents            # -> ~/.agents/skills
npx github:intel/skills install xpu-port --dir .agents/skills    # into a project, to commit
```

`list` is the catalog: every skill in [`skills/`](skills), marked `own` where it is
maintained here and `imported` where it is a copy of another Intel repository's, with the
commit that copy was taken from. There is no npm package: `npx github:intel/skills` runs
this repository.

**Where to install it.** `--target claude-code` is `~/.claude/skills`, read by Claude Code
and by Copilot in VS Code. `--target agents` is `~/.agents/skills`, the convention Codex,
Copilot, Cursor and other harnesses read — `--target codex` is an alias for it, because
Codex documents that path and not `~/.codex/skills`. Both are per-user. An agent working
in a repository also reads `.claude/skills` and `.agents/skills` *inside* it, so use
`--dir` for a skill your team should get from a `git clone`. Check your agent's own
documentation for the directory it reads.

**Install writes what is in this repository, and nothing else.** Every skill in the catalog
is here in full, including the ones another Intel repository maintains: those directories
are copies of a pinned upstream commit, each carrying a `.source.json` that names the
repository, path, commit and licence it came from. So `install` is a copy of a directory —
no fetch to fail, be intercepted, or disagree with the bytes a reviewer read. `show
<skill>` prints that provenance; `verify` re-checks an installed skill against the catalog,
byte for byte:

```bash
npx github:intel/skills verify linux-perf --path ~/.claude/skills/linux-perf
```

No command needs the network, and none needs anything but Node.

No installer, or an agent with no skills directory: `git clone https://github.com/intel/skills`
and point it at `skills/<name>/SKILL.md` — `#file:` in a Copilot chat, a path in an agent
config, or pasted into a system prompt or a Project knowledge base. A skill directory is
self-contained: `SKILL.md` plus the files it names.

## Contributing a skill

You need a GitHub account, working knowledge of the thing your skill covers, and an editor.
No Intel hardware, no benchmark data and no credentials — nothing that a fork cannot get to.
Field-by-field reference: [CONTRIBUTING.md](CONTRIBUTING.md).

There are two ways in, and they ask for different things.

| | You are bringing an existing skill | You are writing a new skill |
|---|---|---|
| Where it came from | it already lives in an Intel project repository | you are writing it here, now |
| What is required | `SKILL.md` + two lines in `skills.yaml` | the same, plus one Harbor task |
| Why | it has been used and reviewed where it was written; re-proving that here buys nothing | a skill nobody has exercised needs one runnable check that it describes something real |

Everything else this repository can do — `evals/evals.json`, `perf/` measurements, a
capability suite, the three-arm differential — is available to you and required of
nobody. Add it when you want the stronger claim.

### 1. Fork, clone, create the directory

```bash
git clone https://github.com/<your-username>/skills
cd skills
mkdir -p skills/your-skill-name          # kebab-case, ≤ 64 characters
cp templates/SKILL.md skills/your-skill-name/SKILL.md
```

The directory name is the skill's identity — it must equal `name:` in the frontmatter.

### 2. Write `SKILL.md`

Treat `description` as the hardest field in the file. Under progressive disclosure it is
the **only** text in the agent's context when it decides whether to open your skill at
all. A description missing its own domain vocabulary makes the skill unreachable no
matter how good the body is — and it fails silently: nothing errors, the agent simply
never picks it.

Test it before you push: write down the requests a user would really type, in their own
words, and check which of them the description alone would route to your skill. Each one it
misses names a word the description is missing.

The body is yours. Under 500 lines, no measured numbers, and every file you ship
mentioned by the path it lives at. Worked example:
[`skills/dpnp-quickstart/SKILL.md`](skills/dpnp-quickstart/SKILL.md).

### 3. Add the catalog entry

Two lines in [`skills.yaml`](skills.yaml) — the skill's name, and a GitHub handle to
route a bug report to:

```yaml
- name: your-skill-name
  maintainer: "your-github-handle"
```

Everything else in that file is optional. `intel-products` is worth filling in if you
want the validator to check your description against the vocabulary a user would type;
leave it out and that check simply does not run.

If the skill is maintained in another repository, do not copy it by hand. Add the pin to
your `skills.yaml` entry — `external-repo`, `external-commit` (a full 40-character SHA),
`external-path`, `external-license` — and run `python3 tools/sync_external.py --write`.
That writes the directory and the `.source.json` beside `SKILL.md`, so what review reads
is the pin and the diff it produced.

### 4. Run the gate locally

Keyless, offline, no setup beyond Python 3.11 or newer — standard library only, nothing to
install:

```bash
python3 tools/validate_skills.py
```

Two more checks need the network, and CI runs both. Neither is required of you, but both
are cheaper to see here than in a pull request:

```bash
python3 tools/validate_skills.py --check-links   # every link in a skill still resolves
python3 tools/sync_external.py --check           # an imported copy still matches its pin
```

### 5. If you are writing a new skill, add a Harbor task

One task under [`evaluation/harbor/tasks/`](evaluation/harbor/tasks): a `task.toml` naming
your skill, a container, an instruction, an oracle solution, and a verifier. CI runs the
oracle arm on it — no model and no API key, so it works on a pull request from a fork — and
requires only that the task is solvable and its verifier emits a reward.

Be clear on what that does and does not do. The oracle applies your reference solution;
it never reads `SKILL.md`. So the task does not score your skill — it makes your skill
**measurable**, which is what lets a maintainer later run the differential that does
score it. At merge time the judgement of your skill's content is a human reading it.
One task is the floor; a skill reaching `validated` needs five, two of which
discriminate — the policy is in
[`evaluation/harbor/suites.json`](evaluation/harbor/suites.json).

Layout and a worked example: [`templates/task_example.md`](templates/task_example.md).

If your skill cannot be exercised without an Intel GPU, say so in the pull request and a
maintainer will decide — a task only that team's hardware can run is not a gate, it is a
favour someone does.

### 6. Open the pull request

Open a pull request that clearly describes the skill, completes the PR checklist, and includes the required DCO-signed commit.

```bash
git commit -s -m "Add your-skill-name skill"      # -s is the DCO sign-off
```

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
- every commit carries a DCO sign-off
- the workflows themselves lint clean (`actionlint`, `zizmor`)
- for a new skill: its Harbor task is solvable, oracle reward 1.0
- for an imported skill: `skills.yaml`, `.source.json` and `NOTICE` agree, and the copy is
  still byte-for-byte the pinned upstream commit
- `npx … install` writes every skill in the catalog, and `verify` accepts each one and
  rejects an installed copy that was altered
- a link that answers 404 or 410 — a pointer an agent would follow into nothing. A timeout,
  a 5xx or rate limiting only warns, so an outage elsewhere cannot hold up a pull request

Reported but not blocking: the coverage gaps between what a suite claims and what it
implements.

Everything else — eval cases, hardware measurements, the differential, discoverability —
is described in [MAINTAINERS.md](MAINTAINERS.md) and gates promotion, not merging.

## Evaluation levels

Three levels, split by what each can afford to require. The split is the point: a gate
that needs a paid API key cannot block a pull request, and a gate that cannot block is
not a gate.

| | Level 1 — structure | Level 2 — differential | Level 3 — discoverability |
|---|---|---|---|
| Question | is the skill well-formed and reachable? | can an agent complete real work with it? | does an agent reach for it unprompted? |
| Needs | nothing | inference + Docker | inference |
| Runs | every PR, including forks | on promotion, by maintainers | on promotion, by maintainers |
| Blocking | **yes** | no | no |
| You run it | yes | no | no |

Level 1 is the only one you run, and the only one that can stop a merge. Level 2 is the
three-arm differential in [`evaluation/harbor/`](evaluation/harbor) — the agent attempts
real containerized tasks with no skill, with the previous version, and with the
candidate, and the gate is `candidate − no_skill`, because a skill that does not beat the
no-skill arm has not shown it does anything. Level 3 asks whether an agent opens the
skill when nothing names it. Both need an inference credential no fork can hold, so both
are run by maintainers by hand with the results attached to the pull request, and neither
is asked of a contributor.

Why these two and not a question set — and how prose deliverables are scored inside the
same differential — is in [MAINTAINERS.md](MAINTAINERS.md).

## Skill lifecycle

| Status | What it means | What it requires |
|---|---|---|
| `published` | In the catalog, and agents load it | `SKILL.md`, a catalog entry, and — for a new skill — a Harbor task |
| `validated` | Carries benchmark evidence from real Intel hardware | plus `perf/`, `references/official-sources.md`, and the hardware it was validated on |

Status lives in [`skills.yaml`](skills.yaml), never in `SKILL.md`. A first pull request
lands at `published`. `validated` is where Intel's team adds hardware evidence; an
external contributor can reach it but is never asked to.

## Focused repositories

This repository is a collection hub. Intel product teams also maintain focused skill
collections of their own — [intel/gpu-ai-skills](https://github.com/intel/gpu-ai-skills)
and [intel/intel-performance-skills](https://github.com/intel/intel-performance-skills)
are both here in the catalog. Skill names are unique across the hub. Contributing a broad
platform skill, or unsure where yours belongs?
[Open an issue](https://github.com/intel/skills/issues/new) before writing anything.

## Licence and governance

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). A skill brought here from
another repository keeps its own licence, recorded in `skills.yaml` and in its
`.source.json`.

**DCO.** Every commit needs a `Signed-off-by` line, certifying you have the right to
submit the contribution under Apache-2.0 — the same pattern as the Linux kernel.
`.github/workflows/dco.yml` checks every non-merge commit on the pull request and fails
it by commit SHA, so this is a gate rather than a request.

```bash
git commit -s -m "your message"      # sign off
git commit --amend -s                # forgot on the last commit
git rebase HEAD~N --signoff          # forgot on several
```

**Security.** Report vulnerabilities via [SECURITY.md](SECURITY.md) or Intel PSIRT.

## Notices and disclaimers

A skill is instructions, not software Intel runs. `SKILL.md` is Markdown an agent reads;
some skills also ship reference documents and scripts. Intel executes none of it — your
agent does: the harness, the model and the version you chose, on the machine and account
you chose, against your data. What a skill *causes* is the product of that combination,
not of the file.

So the same skill gives different results across harnesses, models, model versions,
hardware and driver stacks. `validated` in [`skills.yaml`](skills.yaml) means someone
measured a skill under a stated configuration — not that its output is warranted. Nothing
here is validated for safety-critical or regulated use, or as a control on a production
system. Measurements under `perf/` are point observations on the configuration recorded
beside them; performance varies by use, configuration and other factors — see
[www.intel.com/PerformanceIndex](https://www.intel.com/PerformanceIndex).

Installing a skill, letting an agent load it, and acting on what the agent then does are
your decisions, and the consequences are yours. Read a skill and the scripts it ships
before you use it: they run with your privileges. The skills here are not an Intel product
and carry no support commitment. This section explains what these files are; it does not
add to or narrow the terms you received them under — warranty and liability are disclaimed
by [LICENSE](LICENSE) itself.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel
Corporation or its subsidiaries. Other names and brands may be claimed as the property of
others.

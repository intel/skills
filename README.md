# Intel Agent Skills

**Official Intel-maintained expert workflows for AI coding agents.**

Build, port, run, profile and optimize software on Intel platforms. Each skill captures a
practical workflow rather than product documentation: it inspects the environment, chooses
a supported path, does the work, verifies the outcome, and handles the failures that
actually happen.

Current coverage: Intel GPU setup and diagnosis, AI runtimes and model serving, deployment
planning and benchmarking, CUDA-to-XPU migration, CPU performance analysis, Python
acceleration on Intel devices, and oneTBB development.

[Browse the catalog →](https://intel.github.io/skills/) ·
[Contribute](CONTRIBUTING.md) ·
[How validation works](MAINTAINERS.md)

## Quickstart

Install with the standard skills CLI, which reads this repository directly:

```bash
npx skills add intel/skills
```

That opens the catalog and lets you pick. Three variants worth knowing:

```bash
# browse without installing anything
npx skills add intel/skills --list

# one skill, one agent
npx skills add intel/skills --skill vllm-xpu-run --agent claude-code

# for every project on the machine rather than this one
npx skills add intel/skills --skill linux-perf --agent claude-code --global
```

Installs are project-scoped by default — `./.claude/skills/<name>` for Claude Code, and the
equivalent directory for each other agent you name. `--global` writes the per-user location
instead. Check your agent's own documentation for the directory it reads.

A pinned, offline, byte-verifiable install path also exists, for controlled environments:
[Reproducible and offline installation](#reproducible-and-offline-installation).

## Using what you installed

Describe the task in your own words and start a new session. The agent reads each installed
skill's `description` and opens the one that matches, so naming it is optional:

> *"Serve `Qwen/Qwen3-8B` on my Intel GPU with an OpenAI-compatible endpoint."*

Naming it works too, and is the fastest way to check an install took:

> *"Use the vllm-xpu-run skill to serve `Qwen/Qwen3-8B`."*

```bash
npx skills list              # what is installed, where, and for which agent
npx skills update            # pull the current version of each installed skill
npx skills remove            # take one back out
```

If the agent does something generic and never opens a skill you know covers the task, name
the skill explicitly to confirm it is installed and being read — then
[open an issue](https://github.com/intel/skills/issues/new) with the wording you used. A
skill an agent does not reach for is a defect in its `description`, it fails silently, and
the request you actually typed is the only thing that finds it.

Which skill covers what: **[intel.github.io/skills](https://intel.github.io/skills/)**,
searchable and filterable by product and hardware, generated from this repository on every
push. Several skills chain — a deployment plan calls preflight, sizing and configuration
before a runtime skill; a CPU profile hands off to benchmarking and to optimization
patterns — and each skill's own text says when it hands off and to what.

## Trust and validation

**Official** describes this catalog: Intel-hosted, reviewed here, with every entry's
maintainer named in [`skills.yaml`](skills.yaml). What that buys you is the same for every
entry in it. Each one parses against the [agentskills.io](https://agentskills.io) format,
is scanned for content that would make an agent act against the person running it, is read
by a human who knows the subject, and — where it was written here — ships one runnable task
proving it describes something real.

It does not mean every entry carries the same evidence, and this repository does not pretend
otherwise. A skill is a claim — *give an agent this text and it does better work* — and no
keyless check tests that claim. Where a measurement exists it is in the repository next to
the skill and says which machine produced it; nothing here states a number it has not
measured, and no entry is labelled as more proven than the evidence beside it.
[MAINTAINERS.md](MAINTAINERS.md) has what review covers and how a skill is measured after
it lands.

Provenance is checkable rather than asserted. For a skill maintained in another Intel
repository, `python3 tools/sync_external.py --check` re-fetches the exact upstream commit
this repository pinned and byte-compares it against the copy here, so "unmodified" is a
thing CI proves rather than a thing the README says —
[Catalog federation](#catalog-federation) has how the pin works. The licence a skill
arrived under is preserved and recorded, and [NOTICE](NOTICE) names what this repository
republishes.

## Reproducible and offline installation

The standard CLI above is the right default. This repository also ships its own installer,
for the case where what matters is that the bytes are exactly the ones a reviewer read:

```bash
npx github:intel/skills list                                     # the catalog, and which kind
npx github:intel/skills install linux-perf --target claude-code  # -> ~/.claude/skills
npx github:intel/skills install --all --target agents            # -> ~/.agents/skills
npx github:intel/skills install xpu-port --dir .agents/skills    # into a project, to commit
npx github:intel/skills verify linux-perf --path ~/.claude/skills/linux-perf
```

Node 20 or newer, no dependencies, nothing from npm, and no network call after the
repository itself has been obtained — including for an imported skill, because its
directory is already here in full at its pinned commit. `verify` re-checks an installed
skill against the catalog byte for byte, and `show <skill>` prints its provenance.

| Path | Best for |
|---|---|
| `npx skills add intel/skills` | normal use: broad agent compatibility, update and removal, ecosystem discovery |
| `npx github:intel/skills …` | pinned, offline, byte-verifiable installs in controlled environments |

One difference to know about: the two paths install slightly different payloads. The
standard CLI copies the public skill directory as it stands, `evals/` included; this
repository's installer ships the runtime payload only, excluding `evals/` and `perf/`,
which exist to test and measure a skill rather than to be read by an agent.

Neither is required. With no installer at all, `git clone https://github.com/intel/skills`
and point your agent at `skills/<name>/SKILL.md` — `#file:` in a Copilot chat, a path in an
agent config, or pasted into a system prompt. A skill directory is self-contained:
`SKILL.md` plus the files it names.

## Catalog federation

This repository is a hub. A skill is either written and maintained here, or an exact copy of
one maintained by an Intel product team in its own repository — the Intel stack is large
enough that the team shipping a runtime is the team that should own the workflow for it.
Which of the two any skill is, is recorded rather than inferred. An imported entry in
[`skills.yaml`](skills.yaml) carries its origin:

```yaml
- name: some-skill
  maintainer: "some-github-handle"
  external-repo: https://github.com/intel/some-product-repo
  external-commit: <full 40-character sha>
  external-path: skills/some-skill
  external-license: MIT
```

The directory here is generated from that pin, never hand-copied, so it is the upstream
directory at that exact commit, byte for byte — and the same pin is repeated in a
`.source.json` beside `SKILL.md`, so it travels with an install. A full SHA rather than a
branch is what makes that checkable: a moving ref would let the copy drift silently. An
upstream change reaches users only by moving the pin, which is a reviewed diff like any
other. An import keeps the name upstream gives it, so an agent routing by name finds the
same skill in either place, and names are unique across the hub.

Where to report a problem follows from that. A defect in an imported body has to be fixed
upstream and arrive here through a moved pin — editing the copy here would only break the
byte-compare against its own commit. `.source.json` in the skill directory names the
repository to open the issue against. For anything about the catalog itself — a skill that
is not being reached for, an install problem, a missing workflow —
[open an issue here](https://github.com/intel/skills/issues/new).

## Contributing

Contributions are welcome from Intel product teams, partners and the community. You need a
GitHub account, knowledge of the subject, and an editor — no Intel hardware, no benchmark
data, no credentials, nothing a fork cannot reach.

A skill can be written here or imported from another Intel repository with exact source
provenance; the two ask for different things, and neither asks for a measurement.

[CONTRIBUTING.md](CONTRIBUTING.md) has the walkthrough, the skill format field by field,
the local gate, what CI blocks on, and the evaluation levels.
[MAINTAINERS.md](MAINTAINERS.md) has what happens after a merge: the differential,
discoverability, and hardware evidence. Contributing a broad platform skill,
or unsure where yours belongs? [Open an
issue](https://github.com/intel/skills/issues/new) before writing anything.

## Help, licence and security

Questions and bug reports: [GitHub issues](https://github.com/intel/skills/issues).
Vulnerabilities: [SECURITY.md](SECURITY.md) or Intel PSIRT — not a public issue.

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). A skill brought here from another
repository keeps its own licence, recorded in `skills.yaml` and in its `.source.json`.
Opening a pull request here licenses what is in it under Apache-2.0, which
[CONTRIBUTING.md](CONTRIBUTING.md) states and section 5 of the licence says for any
contribution intentionally submitted for inclusion. Nothing else is asked: no sign-off
trailer, no CLA, no separate agreement.

## Notices and disclaimers

A skill is instructions, not software Intel runs. `SKILL.md` is Markdown an agent reads;
some skills also ship reference documents and scripts. Intel executes none of it — your
agent does: the harness, the model and the version you chose, on the machine and account
you chose, against your data. What a skill *causes* is the product of that combination,
not of the file.

So the same skill gives different results across harnesses, models, model versions,
hardware and driver stacks. Where this repository records a measurement, it means someone
observed that result under the configuration stated beside it — not that a skill's output is
warranted. Nothing here is validated for safety-critical or regulated use, or as a control
on a production system. Measurements under `perf/` are point observations on the configuration recorded
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

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

That opens the catalog and lets you pick. The other three commands worth knowing:

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
instead. `npx skills update` and `npx skills remove` manage what you installed. Check your
agent's own documentation for the directory it reads.

Then describe the task in your own words. An agent selects a skill from its `description`,
so naming the skill is optional — and a skill that is not reached for is a skill whose
description is wrong, which is why that field gets the most review here.

A pinned, offline, byte-verifiable install path also exists, for controlled environments:
[Reproducible and offline installation](#reproducible-and-offline-installation).

## What can Intel Skills help with?

Things worth asking a coding agent once these are installed:

- *"Prepare this Ubuntu host for Intel GPU inference and tell me what is missing."*
- *"Will this model fit on my Intel GPU, and which serving configuration should I use?"*
- *"Serve this Hugging Face model with vLLM on Intel GPU and verify a real response."*
- *"Port this CUDA PyTorch workload to XPU without silently changing what it computes."*
- *"This C++ service stops scaling past eight cores. Profile it and name the bottleneck."*
- *"Parallelize this C++ loop with oneTBB and check the result is still correct."*
- *"Move this NumPy workload to an Intel GPU, and tell me where it should stay on the CPU."*

## Browse by workflow

The catalog is searchable and filterable by product, hardware and source:
**[intel.github.io/skills](https://intel.github.io/skills/)**. It is generated from
[`skills.yaml`](skills.yaml) and the skills themselves, so it is current and this table is
deliberately not a full list.

| What you are doing | Representative skills |
|---|---|
| Set up and diagnose a host | `xpu-discover`, `xpu-system-setup`, `xpu-runtime-preflight`, `xpu-container-run` |
| Plan and configure | `model-can-it-fit`, `model-config-recommend`, `xpu-model-type-detect`, `xpu-deploy-plan` |
| Run and serve models | `torch-xpu-run`, `vllm-xpu-run`, `sglang-xpu-run`, `llamacpp-xpu-run` |
| Port and migrate | `cuda-to-xpu-migration`, `xpu-port`, `dpnp-migration` |
| Profile, benchmark, optimize | `linux-perf`, `performance-patterns`, `torch-xpu-profile`, `xpu-profile-unitrace`, `vllm-xpu-bench`, `phoronix-test-suite` |
| Develop with Intel libraries | `onetbb-quickstart`, `dpnp-quickstart`, `dpnp-linalg-fft`, `mkl-extension-advisor` |

Several of these are designed to hand work to each other: `xpu-deploy-plan` calls
preflight, sizing and configuration before a runtime skill; `linux-perf` routes a
benchmark to `phoronix-test-suite` and a diagnosed pattern to `performance-patterns`.

## Expert workflows, not copied documentation

Product documentation says what an API supports. A skill encodes how someone who has done
the task before applies it.

| Documentation | Expert workflow |
|---|---|
| describes the available APIs and options | starts from an outcome the user asked for |
| assumes the reader knows their own environment | inspects hardware, driver, runtime, versions, limits |
| presents several valid choices | picks one supported path and says why |
| gives commands | runs an ordered procedure with safeguards |
| explains the expected behaviour | tests whether it actually happened |
| lists known issues | recognizes the failure signal and takes a recovery path |
| ends when the feature is explained | returns a result and the evidence for it |

The shape a good skill has:

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

This is also the authoring standard — [CONTRIBUTING.md](CONTRIBUTING.md) asks each of those
questions of a new skill.

## Trust and validation

**Official** describes this catalog: Intel-hosted, reviewed here, with every entry's
maintainer named in [`skills.yaml`](skills.yaml). It does not mean every entry carries the
same evidence, so the status is per skill and stated in one place:

| Status in `skills.yaml` | What it claims |
|---|---|
| `published` | in the catalog and agents load it: structurally checked, content-scanned, reviewed by a human, and — for a skill written here — carrying one runnable task proving it describes something real |
| `validated` | hardware evidence stands behind it: `perf/` measurements from named Intel SKUs, a task portfolio meeting the suite policy, and a differential run that cleared its gate |

Today every entry is `published`. `validated` is a bar a maintainer raises a skill to in a
separate pull request that carries the evidence, and nothing in this repository states a
number it has not measured. What the two levels require, and why an imported skill cannot
reach `validated` here, is in [MAINTAINERS.md](MAINTAINERS.md).

Provenance is checkable rather than asserted. Every imported skill ships a `.source.json`
naming the upstream repository, path, commit and licence it came from; `skills.yaml` pins
the same full commit SHA; and `python3 tools/sync_external.py --check` re-fetches that
commit and byte-compares. The licence a skill arrived under is preserved and recorded, and
[NOTICE](NOTICE) names what this repository republishes.

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

This repository is a hub. Some skills are written and maintained here; most are exact
copies of skills maintained by an Intel product team in their own repository —
[intel/gpu-ai-skills](https://github.com/intel/gpu-ai-skills) and
[intel/intel-performance-skills](https://github.com/intel/intel-performance-skills) today.
An import keeps the name upstream gives it, so an agent routing by name finds the same
skill in either place, and skill names are unique across the hub.

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
[MAINTAINERS.md](MAINTAINERS.md) has what happens after a merge: promotion, the
differential, discoverability, and hardware evidence. Contributing a broad platform skill,
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

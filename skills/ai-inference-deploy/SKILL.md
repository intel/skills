---
name: ai-inference-deploy
description: >-
  Top-level router for an Intel GPU inference request that has not settled what it wants
  yet. Use for the bare ask — "deploy qwen", "serve a model on my Arc", "get llama running
  on my Intel graphics card", "which one do I use, vllm or sglang?", "what do I need to run
  this model on my GPU?" — where the runtime, the host OS, the install path and the
  readiness of the machine are all still open. It picks the runtime for a user who has no
  basis to pick one, translates the catalog's "xpu" skill names into the GPU the user
  actually has, names the branches this catalog does not cover instead of improvising them,
  and hands off. Not for a request already settled: a written deployment plan is
  xpu-deploy-plan, a vLLM launch is vllm-xpu-run, an SGLang launch is sglang-xpu-run,
  throughput is the bench skills, and a CUDA repo is cuda-to-xpu-migration.
---

# ai-inference-deploy

The entry point for "deploy a model on an Intel GPU" before anyone has said which runtime,
which host, or which install path. It turns that request into a resolved set of inputs and
the name of the skill that owns the next step.

This skill **routes only**. It states no image tag, no serve flag and no environment
variable, because those live in the runtime skills and change there. If you find yourself
writing a launch command while in this skill, you have gone one stage too far — stop and
hand off.

## Say "GPU", not "XPU"

Nearly every skill this router hands off to has `xpu` in its name. **XPU is Intel's name for
the GPU as a compute device** — the same Intel GPU the user is asking about, discrete card or
laptop graphics. It is not a separate product, a separate device, or something they need to go and
buy.

Users do not type "XPU". They type GPU, graphics card, Arc, or the name on the box or laptop, and a request phrased that way is this catalog's request. So:

- Recognise "XPU" when a user does say it, but **write "GPU"** back to them.
- When handing off, say what the skill does before its name — "the skill that serves models
  through vLLM, `vllm-xpu-run`" — so the name is not the first thing they have to decode.
- Do not make a user learn the term to get an answer. If they ask what XPU means, one
  sentence: it is the Intel GPU they already have.

## Stage 0 — is this catalog the right one at all

| The request is about | Do this |
|---|---|
| Running or serving a model on an Intel GPU — discrete or integrated | Continue to stage 1 |
| An NVIDIA or AMD GPU | Stop. This catalog is Intel-only; say so and name no Intel skill |
| A CPU-only workload | Stop, and point at the CPU skills — **linux-perf**, **performance-patterns**, **onetbb-quickstart**, **mkl-extension-advisor** |
| Training or fine-tuning rather than inference | Stop. This catalog covers inference; do not improvise a training recipe |

## Stage 1 — which verb

Classify before routing. The verbs are the ones the catalog is already organised by, and
each one owns a different set of skills.

| The user wants | Verb | Go to |
|---|---|---|
| A machine prepared from bare metal | setup | **xpu-system-setup**, then stage 5. Done |
| To know whether it fits, or the best config | plan | **model-can-it-fit**, **model-config-recommend**. Done |
| To actually serve or run the model | run | Stage 2 |
| A number for how fast it is | bench | **vllm-xpu-bench** / **sglang-xpu-bench** / **torch-xpu-bench**. Done |
| To know why it is slow | profile | **torch-xpu-profile** / **vllm-xpu-profile** / **xpu-profile-unitrace**. Done |
| To move a CUDA repo over | migrate | **cuda-to-xpu-migration** (assess) or **xpu-port** (execute). Done |

Only the **run** verb continues. Everything else is one hand-off and this skill is finished.

## Stage 2 — which GPU, then which runtime

Query the card; never assume one. Run `scripts/query_gpus.py` where the GPU runtime is — on
the host, or inside the runtime's container if the host has none. It prints each device's
name, memory and `class`: `dgpu`, `igpu` (shares system RAM), or `unknown`. On `unknown`, say
so and ask; do not guess. A `state` other than `normal` means the GPU itself needs attention:
report it and stop.

| Signal | dgpu | igpu |
|---|---|---|
| GGUF weights | **llamacpp-xpu-run** | **llamacpp-xpu-run** |
| Wants an endpoint, nothing else said | **vllm-xpu-run** (default) | **openvino-gpu-run** (default) |
| In-process Transformers / Diffusers | **torch-xpu-run** | **torch-xpu-run** |
| Prefix caching or grammar-constrained output | **sglang-xpu-run** | not supported; offer **openvino-gpu-run** |
| Asks for vLLM by name | **vllm-xpu-run** | **vllm-xpu-run**, warning it is unvalidated on integrated graphics |

If the choice is genuinely open, or the user asks which to pick, read
`references/runtime-choice.md` before answering. Apply the table rather than handing the
decision back: a user who asked which runtime to use is asking for a recommendation, not for
a menu. State the pick and the signal behind it, and say plainly that they can override it.

Unknown model id? Run **xpu-model-type-detect** first: a vision-language or diffusion model
routes differently from a text generator, and getting it wrong surfaces as an unexpected
keyword argument at load time rather than as a routing error.

## Stage 3 — which host OS

| Host | Route |
|---|---|
| Linux | Continue to stage 4 |
| Windows | **openvino-gpu-run** — the one runtime here with a native Windows build. Every other path is Linux-only: say so, do not adapt the Linux steps |


## Stage 4 — which install path

| Path | Route |
|---|---|
| Container (**the default, and what every run skill here documents**) | **xpu-container-run** owns the device-access flags; the runtime skill owns the rest. Continue to stage 5 |
| Build from source | Only **llamacpp-xpu-run** documents a from-source build, and it builds a container image from an upstream Dockerfile at a pinned tag. For vLLM or SGLang there is no from-source skill here: say so, and offer the container path instead of inventing a build |

Do not treat "build from source" as the way to fix a container problem. A failing container
is stage 5's question, not a reason to compile.

## Stage 5 — is the host ready

Skip only if a preflight has already run in this session.

| State | Route |
|---|---|
| Unknown hardware | **xpu-discover** — what GPUs are present, and is the driver healthy |
| Hardware known, readiness not | **xpu-runtime-preflight** — the read-only go/no-go |
| Preflight reports `FAIL` | **xpu-system-setup**. Re-run preflight; do not launch past a `FAIL` |
| Preflight fails only its `xpu-smi` rows, and the query found the GPU | Not a blocker: `xpu-smi` does not enumerate every Intel GPU. Read `references/gpu-targets.md` |

## Stage 6 — hand off

Resolve these, then call the skill stage 2 chose, or **xpu-deploy-plan** (dgpu only) if the
user wants the whole chain written down as a plan:

| Input | If not given |
|---|---|
| Model id or local path | **Ask.** This is the one thing never to guess |
| Runtime | Stage 2's pick |
| Target GPU, memory, count | From the query. Pass them explicitly to **model-can-it-fit** and **xpu-deploy-plan**; never rely on their defaults |
| Context length | The runtime skill's own default |
| Concurrency | One, for a first boot |

Ask at most three questions in total. Summarize the resolved inputs back to the user before
handing off, so a wrong branch is caught before any work starts.

## What this skill does NOT cover

Everything downstream of the hand-off. Device flags and container launch —
**xpu-container-run**. Images, serve flags, quantization pairing and out-of-memory recovery
— the runtime skill stage 2 chose. Driver, groups and `/dev/dri` — **xpu-runtime-preflight**
and **xpu-system-setup**. Memory sizing — **model-can-it-fit**. The written end-to-end plan,
with preflight and fit and rollback in one document — **xpu-deploy-plan**.

## Reporting

Return the branch, not just the destination: the verb, the runtime and why it was picked,
the host OS, the install path, the readiness verdict, and the skill being handed to. If a
stage stopped, name the stage and what is missing — an honest dead end is the correct
output for a branch this catalog does not cover.

---
name: intel-inference-microservices
description: >-
  Serve a large language model on an Intel Xeon CPU with Intel Inference Microservices.
  One Docker image per model detects the processor, applies a validated serving profile,
  and exposes an OpenAI-compatible API. Use when someone wants to serve, run, or deploy
  a model on Xeon or a CPU server with no GPU, or mentions Inference Microservices, a
  catalog model image, intel/inference-xeon, or a drop-in OpenAI endpoint on Intel CPUs.
  Not for an Intel GPU (ai-inference-deploy, vllm-xpu-run, openvino-gpu-run), a hand-built
  vLLM container, a full platform install, or training.
license: Apache-2.0
---

# intel-inference-microservices

One container serves one model on an Intel Xeon CPU and speaks the OpenAI API. At startup
it detects the processor and starts the engine with the built-in configuration for that
model. Leave `INFERENCE_ENGINE` unset unless the user asked for a specific engine. Do not
add engine flags unless they asked for a feature that configuration does not include.
The engines, the flags, and which variables to leave alone are in
[references/settings.md](references/settings.md).

Copy the image name from the model catalog. Do not substitute `:latest`, and never build
a name by lowercasing a Hugging Face id. How to read a row is in
[references/catalog.md](references/catalog.md).

https://swcatalog.intel.com/developer/docs/intel-inference-microservices/models-and-hardware/model-catalog

## When not to use this

| The request | Do this |
|---|---|
| An Intel GPU, Arc, or graphics card | **ai-inference-deploy** |
| OpenVINO Model Server on a GPU | **openvino-gpu-run** |
| vLLM or SGLang on an Intel GPU | **vllm-xpu-run** or **sglang-xpu-run** |
| Training or fine-tuning | Stop. This skill serves a model |
| A full platform (gateway, auth, RAG, cluster install) | Stop. This skill is the model container only |

## Inspect

1. Confirm the host is an Intel Xeon CPU (Xeon 4, 5, or 6: Sapphire Rapids, Emerald Rapids,
   Granite Rapids, Sierra Forest). `lscpu` is enough. A GPU request does not belong here.
2. Linux with Docker 20.10 or newer for one machine. Kubernetes 1.24 or newer only when
   they asked for replicas, secrets, or probes.
3. Open the model catalog and find the Hugging Face id. A row means that Docker Image.
   No row means the base image. The Qwen3-4B command below is only the quickstart.
4. Llama and Gemma are gated. The user must already have accepted the license. Pass
   `-e HF_TOKEN` only when that variable is already set, or read it from a file they have.
   On Kubernetes, use a secret. Do not invent a token, and do not write one into the
   command or the manifest.
5. The documented memory floor is 32 GB for every model. A larger model needs more. See
   [references/settings.md](references/settings.md).
6. Check that port 8000 is free. If it is not, pick one other port and use that same
   number for `INFERENCE_PORT`, `-p`, and the URL.

## Decide

| Situation | Run |
|---|---|
| The model is in the catalog | That image. Do not set `INFERENCE_MODEL_ID` |
| The model is not in the catalog | `intel/inference-xeon-base:0.1.0` and `-e INFERENCE_MODEL_ID=<Org/Model>` |
| Nobody named an engine | Leave `INFERENCE_ENGINE` unset. The image picks the validated engine |
| They named an engine | Set `INFERENCE_ENGINE` to `vllm`, `sglang`, or `openvino`. `dry-run` first. `profile_id` must start with that engine. If every `list-profiles` row is `engine_mismatch`, this image has no configuration for it |
| They need tools, JSON output, or a shorter context | Keep the catalog image. Pass only that flag through `INFERENCE_ENGINE_ARGS`, then `dry-run` before `serve`. The flag names belong to the engine you are running. Shapes for vLLM are in `references/settings.md` |

A model image refuses to start when nothing in the image matches the host. The base image
starts on a general profile instead, without model-specific tuning. Prefer the catalog
image whenever it lists the model.

## Launch

Catalog image. Qwen3-4B is the public model that needs no token:

```bash
docker run --rm -p 8000:8000 \
  --cap-add SYS_NICE \
  --shm-size=2g \
  intel/inference-xeon-qwen-qwen3-4b:0.1.0
```

`--cap-add SYS_NICE` lets the engine pin memory to the CPU socket. Without `--shm-size=2g`
the container exits: Docker's default shared memory is too small for the engine to start.

Model that is not in the catalog:

```bash
docker run --rm -p 8000:8000 \
  --cap-add SYS_NICE \
  --shm-size=2g \
  -e INFERENCE_MODEL_ID="Org/Model-Name" \
  intel/inference-xeon-base:0.1.0
```

The base image exits with `INFERENCE_MODEL_ID is required but not set` when that variable is missing.

Read the resolved profile before loading weights. `dry-run` does not bind a port:

```bash
docker run --rm --cap-add SYS_NICE \
  intel/inference-xeon-qwen-qwen3-4b:0.1.0 \
  dry-run --format json
```

Use `profile_id` and `command_script`. `detect-hardware` and `list-profiles` also skip the
weights. The command is one token: `list-profiles`, not `list profiles`.

Keep the weights across restarts. The container runs as uid 1001, gid 0, so the host
directory has to be writable by that group:

```bash
mkdir -p /data/cache
chmod -R g+rwX /data/cache
find /data/cache -type d -exec chmod g+s {} +
docker run --rm \
  --cap-add SYS_NICE \
  -v /data/cache:/workspace/model-cache \
  intel/inference-xeon-qwen-qwen3-4b:0.1.0 \
  download-to-cache
```

Mount the same path on the serve command. A container path other than
`/workspace/model-cache` needs `INFERENCE_CACHE_PATH` set to that same path.

Another host port means both the variable and the publish map change together:

```bash
docker run --rm -p 9000:9000 \
  --cap-add SYS_NICE \
  --shm-size=2g \
  -e INFERENCE_PORT=9000 \
  intel/inference-xeon-qwen-qwen3-4b:0.1.0
```

`-p 9000:8000` without `INFERENCE_PORT=9000` leaves the process on 8000 inside the
container. `docker run -P` always publishes 8000, because that is the image's exposed port.

On Kubernetes, use this same image. It is already the server. Map the Docker flags as in
`references/settings.md`: `SYS_NICE` on the container, a 2Gi memory volume at `/dev/shm`,
and a startup probe on `/health` before readiness and liveness. Do not put port 8000 on a
public LoadBalancer or NodePort. The port has no TLS and no authentication.

## Verify

The first start downloads weights. Wait until `/health` is 200, then chat. The `"model"`
field is the Hugging Face id, not the image name.

```bash
curl -sf http://localhost:8000/health && echo alive
curl -s http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

When the id is not one you already know, take it from `/v1/models`. A healthy server still
rejects a request whose `"model"` does not match, with `model ... does not exist`.

Clients need a non-empty `api_key` even though the server does not check it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
```

`docker ps` moves from `(health: starting)` to `(healthy)` on the same condition.

## Recover

Match the log line, then change one thing. Commands for each case are in
[references/failures.md](references/failures.md).

| What you see | What to change |
|---|---|
| `pull access denied` or `repository does not exist` | Copy the Docker Image cell from the model catalog. Check `docker images` before pulling again |
| `INFERENCE_MODEL_ID is required but not set` | You started the base image. Pass the id, or switch to the catalog image |
| `contains custom code which must be executed` | The model needs its own Python. Confirm `trust_remote_code` with `dry-run` before setting `INFERENCE_TRUST_REMOTE_CODE=true` |
| `No compatible profile found` | Run `detect-hardware`, then `list-profiles`. Do not enable the general-profile fallback unless the user accepts untuned serving |
| Exit 137 | Not enough RAM. Use a smaller model, or shorten context with `INFERENCE_ENGINE_ARGS` after `dry-run` |
| `Available memory on node 0` and `less than desired` | One NUMA node is short, not the whole machine. Lower `gpu-memory-utilization` through `INFERENCE_ENGINE_ARGS` |
| `HuggingFaceHub API error: 401` | Token missing, or the license is not accepted yet |
| `Permission denied` under `hub/.locks` | The cache mount is not writable by uid 1001, gid 0 |
| `numa_set_membind failed` or `Operation not permitted` | Add `--cap-add SYS_NICE` |
| `connection refused` on 8000 | The container exited, or `INFERENCE_PORT` and `-p` disagree |
| `/health` stays 503 | Often still downloading. If it never turns 200, rerun with `INFERENCE_LOG_LEVEL=DEBUG` |

Malformed `INFERENCE_ENGINE_ARGS` is ignored with a warning, and the built-in profile still
serves. Valid JSON that names a flag the engine rejects fails at startup. `dry-run` tells
those two apart: it prints `command_script`, or it fails with the same error `serve` would.

## Report

Return the image and tag, the Hugging Face model id, the base URL
(`http://<host>:<port>/v1`), whether a cache is mounted, and `profile_id` when you ran
`dry-run`. Say that the engine was left to the image.

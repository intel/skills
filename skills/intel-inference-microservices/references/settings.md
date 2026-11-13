# Flags and settings

Published behavior. The container starts an inference engine. Configuration is `INFERENCE_*`
environment variables, passed with `-e` before the image name, or under `env:` on
Kubernetes. The container reads them once, at startup.

Docs: https://swcatalog.intel.com/developer/docs/intel-inference-microservices/overview/meet-intel-inference-microservices

## Docker flags on every serve

| Flag | Why |
|---|---|
| `--cap-add SYS_NICE` | Lets vLLM pin memory to the CPU socket. Without it, a multi-socket host logs `numa_set_membind failed` and still serves, off-socket |
| `--shm-size=2g` | Docker's default shared memory is 64 MiB, and the engine exits without more |
| `-p <port>:<port>` | The publish map and `INFERENCE_PORT` are the same number. Default port is 8000 |

`docker run -P` publishes 8000 only. `-e` after the image name is a container command, not an environment variable.

## Engine

Leave `INFERENCE_ENGINE` unset unless the user asked for one. Unset means the image picks
the engine and the configuration validated for this model and this processor. The API,
the port, and `/health` stay the same either way.

| Engine | `INFERENCE_ENGINE` |
|---|---|
| vLLM | `vllm` |
| SGLang | `sglang` |
| OpenVINO | `openvino` |

Values are lowercase. Anything else fails with `ConfigError`. Set it before the image name, and run `dry-run` before `serve`:

```bash
docker run --rm --cap-add SYS_NICE \
  -e INFERENCE_ENGINE=sglang \
  "$IMAGE" \
  dry-run --format json
```

`profile_id` starts with the engine name, for example `sglang-xeon-bf16-tp1`. Then serve with the same `-e` and drop `dry-run`. On Kubernetes the same name goes under `env:`.

An image carries configurations only for the engines it was validated with. `list-profiles` with that same `-e` marks the others `engine_mismatch`. If every row says `engine_mismatch`, pick an engine that has a row, or another image.

`INFERENCE_ENGINE_ARGS` is passed to whichever engine is running. A flag from one engine is meaningless to another and fails startup. After a switch, metric names change too: `vllm:num_requests_waiting` exists only on vLLM.

SGLang or OpenVINO on an Intel GPU is **sglang-xpu-run** or **openvino-gpu-run**, not this container.

Also leave these unset unless the user has a reason:

| Variable | What it is today |
|---|---|
| `INFERENCE_ACCELERATOR_TYPE` | `xeon` |
| `INFERENCE_PRECISION` | `bf16`. `auto` selects that |
| `INFERENCE_ALLOW_GENERAL_PROFILE_FALLBACK` | `true` on the base image, `false` on a model image |

## Variables you do set

| Variable | When |
|---|---|
| `INFERENCE_MODEL_ID` | Base image only. A model image already has the id |
| `HF_TOKEN` | The catalog listing's `docker run` includes it. Never put the token in the command, the image, or a committed manifest |
| `INFERENCE_PORT` | Host port 8000 is taken. Must match `-p` |
| `INFERENCE_CACHE_PATH` | The mount is not `/workspace/model-cache`. The variable and the container side of `-v` are the same path |
| `INFERENCE_LOG_LEVEL` | Startup is failing. `DEBUG` shows detection and the selected profile. It does not turn vLLM up |
| `INFERENCE_LOG_LEVEL_ROOT` | You need the vLLM log. Default `WARNING` |
| `INFERENCE_PROFILE_ID` | Pin one row from `list-profiles`. Leave unset to take `[primary]` |
| `INFERENCE_ACCELERATOR_COUNT` | A cgroup hides cores. `dry-run` shows it at `detection.accelerator_count` |
| `INFERENCE_TRUST_REMOTE_CODE` | `false` forbids model-published Python and wins over the image and over `INFERENCE_ENGINE_ARGS`. `true` only after the user accepts that the model's code runs in the container |
| `INFERENCE_ENGINE_ARGS` | A JSON object of extra flags for the engine that is running, merged over the built-in configuration |

Malformed `INFERENCE_ENGINE_ARGS` is ignored with a warning and the built-in profile still
serves. Valid JSON that names a flag vLLM rejects fails at startup. `dry-run --format json`
prints `command_script`, or it fails with the same error `serve` would.

## vLLM flags, only when asked

These apply when the engine is vLLM, including the default. They are pass-throughs, not the
validated configuration. Check `command_script` before serving. On SGLang or OpenVINO, use
that engine's own flag names instead.

Shorter context, the first thing to try when memory is short:

```bash
-e INFERENCE_ENGINE_ARGS='{"max-model-len": 4096}'
```

NUMA node reservation. vLLM's CPU backend uses `gpu-memory-utilization` for the fraction of
one NUMA node's memory. The default is `0.92`. Lower it when startup exits 1 with
`Available memory on node 0 ... less than desired`:

```bash
-e INFERENCE_ENGINE_ARGS='{"gpu-memory-utilization": 0.5}'
```

Tool calling is off in the built-in configuration. The parser has to match the model's
chat template. `hermes` is an example, not the parser for every model:

```bash
-e INFERENCE_ENGINE_ARGS='{"enable-auto-tool-choice": true, "tool-call-parser": "hermes"}'
```

Structured output:

```bash
-e INFERENCE_ENGINE_ARGS='{"structured-outputs-config": {"backend": "xgrammar"}}'
```

The request then sets `response_format`. A reasoning parser, when one is required, is
`{"reasoning-parser": "<parser>"}`, and that parser is also model-specific.

## Memory

32 GB is the documented floor for every model. Weights at bf16 are about 2 GB per billion
parameters, and a mixture-of-experts model is sized by its total parameters. Starting
points from the platform page: 32 GB through 9B, 48 GB through 16B, 64 GB through 24B,
96 GB through 30B. Long context and concurrency sit on top of that.

## Kubernetes, same image

| Docker | Kubernetes |
|---|---|
| `--cap-add SYS_NICE` | `capabilities.drop: ["ALL"]` and `capabilities.add: ["SYS_NICE"]`, with `allowPrivilegeEscalation: false` |
| `--shm-size=2g` | `emptyDir` with `medium: Memory` and `sizeLimit: 2Gi`, mounted at `/dev/shm` |
| probes | `startupProbe`, then `readinessProbe` and `livenessProbe`, all `httpGet` `/health` |
| cache writable by gid 0 | pod `securityContext.fsGroup: 0` and `runAsNonRoot: true` |
| `-e HF_TOKEN` | `secretKeyRef` named `hf-credentials`, key `token` |
| `-m` / memory | request at least `32Gi`. Raise it for the model size above. Set CPU requests and no CPU limit |

`startupProbe` in the published manifest uses `periodSeconds: 15` and `failureThreshold: 80`.
Raise the threshold for a larger model. `INFERENCE_MODEL_ID` is commented out unless the
image is `intel/inference-xeon-base:0.1.0`.

Do not give the Service type `LoadBalancer` or `NodePort` on a public network. Port 8000
has no TLS and no authentication.

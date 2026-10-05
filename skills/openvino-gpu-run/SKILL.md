---
name: openvino-gpu-run
description: >-
  Serve an LLM on an Intel GPU with OpenVINO Model Server (OVMS) behind an OpenAI-compatible
  API — /v1/chat/completions and /v1/models. The default serving path for Intel integrated
  graphics (Core Ultra laptop GPUs that share system RAM), and the one runtime in this catalog
  with a native Windows build; it also runs on discrete Intel GPUs. Covers picking a model
  source (pre-converted OpenVINO models, GGUF, or a Hugging Face model that needs
  conversion), the container launch, pinning the GPU so it does not fall back to the CPU,
  and checking the output. Not for vLLM (vllm-xpu-run) or in-process PyTorch
  (torch-xpu-run).
---

# openvino-gpu-run

Serve a model with OpenVINO Model Server on an Intel GPU, then prove it ran on the GPU.

## 1. Inspect

- The GPU: run `query_gpus.py` from **ai-inference-deploy** for name, memory and class. An
  integrated GPU shares system RAM, so size against what that reports, not the RAM total.
- The model: is there a pre-converted copy in the `OpenVINO` organization on Hugging Face
  (`OpenVINO/<model>-int4-ov` and similar)? That is the simplest source — no conversion.

## 2. Decide the model source

| Source | Image | Extra arguments |
|---|---|---|
| `OpenVINO/...` model (pre-converted) | `openvino/model_server:latest-gpu` | none |
| GGUF repository | `openvino/model_server:latest-gpu` | `--gguf_filename <file>.gguf --task text_generation` |
| Plain Hugging Face model (PyTorch weights) | `openvino/model_server:latest-py` (has `optimum-cli`) | `--weight-format int4` or `int8`, `--task text_generation` |

## 3. Launch

Linux, Docker:

```bash
mkdir -p ~/models
docker run --user $(id -u):$(id -g) -d --rm -p 8000:8000 \
  --device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1) \
  -v ~/models:/models:rw \
  openvino/model_server:latest-gpu \
  --source_model <model> --model_repository_path /models \
  --target_device GPU --rest_port 8000
```

Windows: `ovms.exe` with the same arguments (`--model_repository_path c:\models`).

Always pass `--target_device`. Without it OVMS picks a device itself and falls back to the
CPU when it cannot use the GPU — the server comes up and answers, just not on the GPU. On a
host with more than one GPU use `GPU.<index>`; the startup log lists the indices.

The first run downloads the model into `~/models`; later runs reuse it.

## 4. Verify — HTTP 200 is not enough

```bash
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "<model>", "max_tokens": 30, "temperature": 0,
       "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
docker exec <container> sh -c 'grep -h "^drm-resident" /proc/1/fdinfo/*'
```

Done when: `/v1/models` lists the model, the reply is plausible text (quote it to the user),
and the server process holds GPU memory — non-zero `drm-resident-*` lines (`vram` on a discrete
GPU, `system`/`gtt` on integrated). The startup log only lists *available* devices; it does
not show where the model loaded.

## 5. Recover

| Signal | Fix |
|---|---|
| No `drm-resident-*` lines | not on the GPU: `--target_device GPU` missing, or `/dev/dri` / render group not passed |
| `Permission denied` on `/dev/dri/renderD*` | the `--group-add` value must be the render node's group id |
| Model fails to load: missing tokenizer | the IR model needs tokenizer files; re-export, or use an `OpenVINO/...` model |
| Out of memory on an integrated GPU | smaller or int4 model, or cap the KV cache with `--cache_size <GB>` |
| Podman: `short-name resolution enforced` | use the full name, `docker.io/openvino/model_server:latest-gpu` |

## Hand off

Unsure which runtime fits → **ai-inference-deploy**. A CUDA codebase → **cuda-to-xpu-migration**.
No OVMS benchmark skill exists here yet; say so rather than inventing one.

Sources: [LLM quickstart](https://github.com/openvinotoolkit/model_server/blob/main/docs/llm/quickstart.md),
[pulling models](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_hf_models.md),
[parameters](https://github.com/openvinotoolkit/model_server/blob/main/docs/parameters.md).

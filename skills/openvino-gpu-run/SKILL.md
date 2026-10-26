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

- The GPU: run `query_gpus.py` from **ai-inference-deploy** for name, memory and class. If
  only this skill is installed, use `clinfo` (Linux) or Task Manager → Performance → GPU
  (Windows), or ask the user. An integrated GPU shares system RAM, so size against the free
  RAM, not the total. If no memory figure is available, ask; do not assume one.
- The model: is there a pre-converted copy in the `OpenVINO` organization on Hugging Face
  (`OpenVINO/<model>-int4-ov` and similar)? That is the simplest source — no conversion.

## 2. Decide the model source

| Source | Image | Extra arguments |
|---|---|---|
| `OpenVINO/...` model (pre-converted) | `openvino/model_server:latest-gpu` | none |
| GGUF repository | `openvino/model_server:latest-gpu` | `--gguf_filename <file>.gguf --task text_generation` |
| Plain Hugging Face model (PyTorch weights) | Convert with `openvino/model_server:latest-py`, then serve with `latest-gpu` | see below |

`latest-py` is the image that ships `optimum-cli`; `latest-gpu` does not convert. Convert once
with `--pull`, which exports the model into `~/models` and exits (no GPU needed for this step):

```bash
docker run --user $(id -u):$(id -g) --rm -v ~/models:/models:rw \
  openvino/model_server:latest-py \
  --pull --source_model <model> --model_repository_path /models \
  --task text_generation --weight-format int4 --target_device GPU
```

Then launch from step 3 with the same `--source_model`; the server finds the exported copy.

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

Windows, native (no container):

1. Install the Microsoft Visual C++ Redistributable and the Intel graphics driver.
2. Download `ovms_windows_*_python_on.zip` from the
   [model_server releases](https://github.com/openvinotoolkit/model_server/releases) page and
   unpack it (`tar -xf <zip>`).
3. In every new shell, run `ovms\setupvars.bat` (cmd) or `ovms\setupvars.ps1` (PowerShell).
4. `ovms.exe` with the same arguments as above, and `--model_repository_path c:\models`. The
   `python_on` build can convert a plain Hugging Face model itself with `--pull`.

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

On Windows, check the GPU memory of the `ovms` process instead (PowerShell):

```powershell
$p = (Get-Process ovms).Id
Get-Counter "\GPU Process Memory(pid_${p}_*)\Dedicated Usage", "\GPU Process Memory(pid_${p}_*)\Shared Usage"
```

Task Manager → Details with the "Dedicated GPU memory" / "Shared GPU memory" columns shows the
same.

Done when: `/v1/models` lists the model, the reply is plausible text (quote it to the user),
and the server process holds GPU memory — non-zero `drm-resident-*` lines (`vram` on a discrete
GPU, `system`/`gtt` on integrated), or on Windows a non-zero dedicated or shared GPU memory
counter for `ovms`. The startup log only lists *available* devices; it does
not show where the model loaded.

## 5. Recover

| Signal | Fix |
|---|---|
| `/v1/models` returns `"data":[]`, chat says graph `not found` | the server is up but the model is still downloading or loading; poll until the model id is listed, then test |
| No `drm-resident-*` lines | not on the GPU: `--target_device GPU` missing, or `/dev/dri` / render group not passed |
| Windows: `curl error ... Error code: 60`, then `Could not infer model task` | the download failed on TLS: a fresh Windows install has not fetched the root certificate yet. Run `Invoke-WebRequest https://huggingface.co` once (or open it in a browser), then relaunch |
| Windows: `ovms` GPU memory stays at zero | not on the GPU: `--target_device GPU` missing, or the Intel graphics driver is not installed |
| `optimum-cli` not found during `--pull` | a plain Hugging Face model was given to `latest-gpu`; convert with `latest-py` first |
| `Permission denied` on `/dev/dri/renderD*` | the `--group-add` value must be the render node's group id |
| Model fails to load: missing tokenizer | the IR model needs tokenizer files; re-export, or use an `OpenVINO/...` model |
| Out of memory on an integrated GPU | smaller or int4 model, or cap the KV cache with `--cache_size <GB>` |
| Podman: `short-name resolution enforced` | use the full name, `docker.io/openvino/model_server:latest-gpu` |

## Hand off

Unsure which runtime fits → **ai-inference-deploy**. A CUDA codebase → **cuda-to-xpu-migration**.
No OVMS benchmark skill exists here yet; say so rather than inventing one.

Sources: [LLM quickstart](https://github.com/openvinotoolkit/model_server/blob/main/docs/llm/quickstart.md),
[pulling models](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_hf_models.md),
[pulling with conversion](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_optimum_cli.md),
[bare-metal install](https://github.com/openvinotoolkit/model_server/blob/main/docs/deploying_server_baremetal.md),
[parameters](https://github.com/openvinotoolkit/model_server/blob/main/docs/parameters.md).

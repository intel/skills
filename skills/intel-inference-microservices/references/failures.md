# Failure recovery

`$IMAGE` is the Docker Image cell from the model catalog
(https://swcatalog.intel.com/developer/docs/intel-inference-microservices/models-and-hardware/model-catalog),
or `intel/inference-xeon-base:0.1.0` when the model is not listed. These commands do not
load weights unless the block is a serve command.

## Image name

```bash
docker images
```

Run the name that is already on disk. A local tag and an `intel/` name are different
images. If it is not listed, copy the Docker Image cell from the model catalog and pull that.

## Base image with no model

```bash
docker run --rm -p 8000:8000 \
  --cap-add SYS_NICE \
  --shm-size=2g \
  -e INFERENCE_MODEL_ID="$MODEL_ID" \
  intel/inference-xeon-base:0.1.0
```

## Nothing matches this host

```bash
docker run --rm --cap-add SYS_NICE "$IMAGE" detect-hardware
docker run --rm --cap-add SYS_NICE "$IMAGE" list-profiles
```

A model image stops here on purpose. Serve on the base image if the user accepts a
general profile. A cgroup that hides cores is `INFERENCE_ACCELERATOR_COUNT`. Rows marked
`numa_mismatch` need more NUMA nodes than this host has.

## Out of memory, exit 137

Shorten the context, then `dry-run` and look at `command_script`:

```bash
docker run --rm --cap-add SYS_NICE \
  -e INFERENCE_ENGINE_ARGS='{"max-model-len": 4096}' \
  "$IMAGE" \
  dry-run --format json
```

## NUMA node reservation

This exits 1 during startup while the machine still has free RAM elsewhere. On CPU,
`gpu-memory-utilization` is the fraction of one NUMA node's memory:

```bash
docker run --rm --cap-add SYS_NICE \
  -e INFERENCE_ENGINE_ARGS='{"gpu-memory-utilization": 0.5}' \
  "$IMAGE" \
  dry-run --format json
```

Confirm `--gpu-memory-utilization` is in `command_script` before you serve with it.

## Token rejected

A 401 with a token that looks valid means the model license is not accepted yet. Add
`-e HF_TOKEN` to the catalog image. Do not write the token into the command.

## Cache not writable

The log names uid 1001. Fix the host directory, then start again:

```bash
chmod -R g+rwX /data/cache
find /data/cache -type d -exec chmod g+s {} +
```

`Permission denied: '/.cache'` is a different failure: an overridden `HOME` the uid
cannot write. Drop the `HOME` override.

## Startup logs without engine noise

```bash
docker run --rm -p 8000:8000 \
  --cap-add SYS_NICE \
  --shm-size=2g \
  -e INFERENCE_LOG_LEVEL=DEBUG \
  -e INFERENCE_LOG_LEVEL_ROOT=INFO \
  "$IMAGE"
```

`INFERENCE_LOG_LEVEL` is the container's own log. `INFERENCE_LOG_LEVEL_ROOT` is the
engine. A `/health` of 503 for several minutes on a first start is the download.

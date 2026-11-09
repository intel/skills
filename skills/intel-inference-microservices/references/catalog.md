# Model catalog

The catalog is this page. Open it and copy the image. Do not use a name remembered from
an older copy of this skill.

https://swcatalog.intel.com/developer/docs/intel-inference-microservices/models-and-hardware/model-catalog

| Column | Use it for |
|---|---|
| Model | The Hugging Face id, unchanged, including the slash and capitals. This is the `"model"` field in the API request |
| Docker Image | The image to run, tag included. Copy the cell. Do not derive it from the id |
| HF token required | `Yes` means pass `-e HF_TOKEN` from the environment the user already has |

A model with no row on that page is not a catalog image. Serve it with the base image
`intel/inference-xeon-base:0.1.0` and `-e INFERENCE_MODEL_ID` set to the Hugging Face id.

The page also says to copy a full `docker run` from the model's Deployment tab when the
listing has one. Prefer that command over assembling one, then still add `--cap-add
SYS_NICE` and `--shm-size=2g` if the copied command omits them.

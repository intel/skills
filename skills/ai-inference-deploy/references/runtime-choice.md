# Choosing the runtime

Stage 2 of the router, in detail. Read this only when the choice is genuinely open — the
table in `SKILL.md` settles most requests without it.

## The fork that comes first: weight format

| Format | Runtime | Why |
|---|---|---|
| GGUF | llama.cpp SYCL — **llamacpp-xpu-run** | GGUF is llama.cpp's own format. Nothing else here loads it |
| safetensors | one of the three below | The Hugging Face format the rest of the catalog assumes |

Getting this backwards is the most common mis-route, and it fails late: the user gets a
loader error from a runtime that was never going to read the file.

## Among the safetensors runtimes

All three run on the user's Intel GPU. The `-xpu` in the names below is Intel's term for
that GPU, not a fourth option — say "GPU" when explaining the choice to them.

| | vLLM | SGLang | PyTorch |
|---|---|---|---|
| Skill | **vllm-xpu-run** | **sglang-xpu-run** | **torch-xpu-run** |
| Serves HTTP | yes, OpenAI-compatible | yes, OpenAI-compatible | no — in-process only |
| Continuous batching | yes | yes | no |
| Picked for | the default: broad current model coverage on Intel, and a fallback path for architectures the engine does not implement natively | prefix-cache-heavy work (RadixAttention) and grammar-constrained output | Transformers, Accelerate or Diffusers called directly from Python |
| Rule out when | the user wants no server at all | the request says nothing about caching or constrained decoding — prefer vLLM for breadth | the user asked for an endpoint |

**vLLM is the default.** Pick it when the request says only "serve" or "deploy". Pick
SGLang when the user names a reason to; "it might be faster" is not one, and belongs to the
bench skills rather than to a routing decision.

## Workloads that are not text generation

Two questions decide these, and in this order: what the model is, then whether the user
wants an endpoint or a Python process.

| Workload | Route |
|---|---|
| Pooling, embedding or reranking, served over HTTP | **vllm-xpu-run** — it serves these model types alongside generation, and owns the dtype requirement that comes with them |
| Embeddings, encoder-only or classification, called in-process | **torch-xpu-run** |
| Diffusion / image generation | **torch-xpu-run**. The serving runtimes here are for text |
| Unknown or unfamiliar model id | **xpu-model-type-detect** first — it reports what the model actually is, which decides the rows above |

Do not assume that "not a chat model" means "not vLLM". It is the endpoint question, not the
model family, that moves a request off the serving runtimes.

## Paths that are not runtimes

Two names come up in requests and are neither a choice nor supported here:
`intel-extension-for-pytorch` (ipex) and ipex-llm. Upstream PyTorch supersedes both;
**torch-xpu-run** states that and is the skill that owns the explanation. Route there rather
than answering it in the router.

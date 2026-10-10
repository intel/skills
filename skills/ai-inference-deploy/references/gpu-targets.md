# GPU targets

**Query the GPU; do not look it up.** `scripts/query_gpus.py` reports name, memory and class
for whatever is installed, including parts no skill here was written for. Use those values.

**No tuning data for a part means none.** Recommendations that name a specific part — the
config tables in **model-config-recommend**, the setup notes in **xpu-system-setup** — apply
to that part only. For any other GPU, size from the queried memory with **model-can-it-fit**
and say that no tuned config exists. Do not borrow another part's numbers.

**`xpu-smi` does not enumerate every Intel GPU.** It is absent or empty on many integrated
GPUs. If the query found the device through another source, an `xpu-smi` failure in
**xpu-discover** or **xpu-runtime-preflight** is a gap in that tool, not a broken host.

**Integrated GPUs share system RAM.** The query caps their memory at what the OS has free.
Treat it as an upper bound; other processes compete for the same pool.

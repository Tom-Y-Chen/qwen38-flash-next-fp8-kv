# Qwen3.8-Flash-Next — fp8 KV Cache (Experimental)

> **中文版 →** [README.zh-CN.md](README.zh-CN.md)

> **⚠️ THIS IS AN EXPERIMENT / TEST ARTIFACT, NOT A RIGOROUS OR PRODUCTION BUILD.**
> It is a community patch, **not** an official vLLM or NVIDIA release. It was
> validated on **one** machine (an NVIDIA DGX Spark, GB10 / sm_121) against
> **one** pinned vLLM fork build. Different vLLM versions, attention
> backends, or GPUs may not work (or may silently regress). The fp8 scales are
> **uncalibrated (scale = 1.0)**. Use at your own risk.

This repo enables **fp8 (`fp8_e4m3`) KV cache** for the **Qwen Sparse
Attention (QSA)** Triton kernel of **Qwen3.8-Flash-Next**, so that a single
DGX Spark (128 GB unified memory) can serve **~11 concurrent requests at full
262k context** — instead of the ~4 concurrent that a bf16 KV cache allows.

The QSA sparse-attention layers run their own Triton kernel (not FlashAttention),
so vLLM's generic fp8-KV path does not cover them. These two patches add
in-kernel fp8 → bf16 dequantization and lift the dtype gates that previously
hard-rejected fp8 KV for this model.

---

## What it does

Two files are modified inside NVIDIA's vLLM fork:

| File | Change |
|---|---|
| `vllm/models/qwen3_8_flash_next/nvidia/qsa.py` | Accept `fp8*` KV cache dtypes; inject the dtype past FlashAttention's fp8 check; re-view the uint8 (raw-byte) fp8 cache as `float8`; pass the per-layer KV scales into the kernel. |
| `vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py` | Add `k_scale`/`v_scale` params to the Triton kernel; dequantize fp8 K/V to bf16 in-kernel; set `num_stages=1`. |

**Two root causes fixed:**
1. **`_k_scale` read off the wrong object** — `forward_qsa` read `float(self._k_scale)` on the impl instead of the layer. Fixed to read `layer._k_scale_float` / `_v_scale_float` (plain floats, safe under `torch.compile`).
2. **Shared-memory overflow on sm_121** — the kernel was tuned on GB300 (227 KB smem/SM). At `num_stages=2` it exceeds the 100 KB smem cap on DGX Spark's sm_121: fp8 needed 106 KB, bf16 needed 136 KB. `num_stages=1` fits both (pipeline depth only; numerics unchanged).

---

## Results (measured on one DGX Spark)

- **fp8 KV @ GPU_MEM 0.94 → `2,971,999` KV tokens → `11.34×` concurrent at 262k context.**
  That is roughly **2.8×** the bf16 ceiling (~4×), leaving ~40% headroom above the 8-concurrent target.
- **Correctness:** short prompts all correct; long sparse-attention retrieval
  (6,335 tokens / 120 facts) selected the exact fact; **8-way concurrent
  serving returned 0 errors**.
- **Drift vs bf16:** same prompts, `temperature=0` — **final answers matched
  7/7**. Only the free-form `<thinking>` wording shifted slightly (fp8
  rounding perturbs logits a little), which does **not** flip the answers.

---

## Scope — which models it applies to

**Applies to:** any **Qwen3.8-Flash-Next** checkpoint of the **same
architecture**, whatever the **weight** quantization (NVFP4, FP8 weights,
etc.). fp8 KV is chosen at runtime via `--kv-cache-dtype`, orthogonal to how
the weights are quantized. The patches are architecture-bound, not
checkpoint-bound.

**Does NOT apply to:**
- A **different vLLM build** — the exact line numbers / internal APIs
  (`current_platform`, `set_default_quant_scales`, the `torch.uint8` storage
  quirk, the FA2 dtype check) are specific to the pinned fork. Porting is
  needed.
- A **different model architecture** with its own custom attention kernel —
  it would need its own analogous fp8 support in *its* kernel.
- **nvfp4 KV**: requires datacenter Blackwell SM100 (TMEM); GB10 / sm_121
  has none.

**Hardware note:** `num_stages=1` is required to fit sm_121's 100 KB smem.
On other GPUs (e.g. GB300 / sm_100) this is unnecessary and may cost a little
performance — see `patches/qsa_ops.py.patch` and re-tune if you target those.

---

## Repository layout

```
patches/qsa.py.patch        Unified diff (patch -p1) for the attention owner
patches/qsa_ops.py.patch    Unified diff (patch -p1) for the Triton kernel
patched/qsa.py              The ready-to-copy modified file
patched/qsa_ops.py          The ready-to-copy modified file
Dockerfile.fp8              Builds qwen38-flash-dgx-fp8 FROM qwen38-flash-dgx
launch_fp8_x8.sh            Example launch (262k context, 8 concurrency)
NOTICE                      Attribution / provenance
LICENSE                     Apache License 2.0
```

---

## Build

**Prerequisite — build the base image first** (NOT in this repo; it's NVIDIA's
vLLM fork container, `qwen38-flash-dgx`):

```
# Assemble from NVIDIA's vLLM fork (vLLM 0.1.dev20073+g8e685d198, FlashAttention v2),
# the Qwen3.8-Flash-Next model code, PLE-mmap support, and FlashInfer.
# Expected container entrypoint = vllm, and the two base files present at:
#   .../vllm/models/qwen3_8_flash_next/nvidia/qsa.py
#   .../vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py
```

Then either apply the patches in-tree and rebuild, or build the provided image:

```bash
docker build -t qwen38-flash-dgx-fp8 -f Dockerfile.fp8 .
```

---

## Usage

```bash
bash launch_fp8_x8.sh          # 262k context, 8 concurrent, fp8 KV
```

Key flags (must match the architecture):
- `--kv-cache-dtype fp8_e4m3` — the fp8 KV cache (the whole point of this patch).
- `--max-model-len 262144 --max-num-seqs 8 --gpu-memory-utilization 0.94`.
- `-cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="[...]"` — required for this
  hybrid architecture.
- `VLLM_PLE_MMAP=1` — mmap the ~47 GB n-gram/PLE table to RAM.

Then query the OpenAI-compatible endpoint, e.g.:

```bash
curl http://127.0.0.1:18307/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"temperature":0,"seed":42}'
```

---

## Known limitations / caveats

- **Uncalibrated fp8 (scale = 1.0).** The KV values are simply rounded to
  fp8_e4m3; there is no per-tensor calibration. Even so, the tested answers
  were preserved vs bf16. Real deployments should calibrate per-tensor scales.
- **fp8 is lossy.** Do not claim "lossless". Under 8 bits (nvfp4 / 3.5-bit /
  similar) precision is inherently degraded.
- **The bf16 baseline must ALSO run `num_stages=1` on sm_121** — the original
  kernel spills shared memory on this hardware regardless of dtype.
- **Only DGX Spark / sm_121 was validated.** Timing, memory and exact numbers
  may differ elsewhere.
- **Model weights are NOT included.** Use a checkpoint you are licensed to
  download (the patch does not depend on a specific checkpoint).

---

## License & copyright

- This repository is a **patch** derived from **NVIDIA's vLLM fork**.
- The upstream **vLLM** project is licensed under the **Apache License 2.0**
  (Copyright the vLLM Project and its contributors).
- The **Qwen / Qwen3.8-Flash-Next** weights are by the Qwen team (Alibaba) and
  are governed by their own license. **No model weights are bundled here.**
- This repository is distributed under the **Apache License 2.0** — see
  [`LICENSE`](LICENSE). Attribution and provenance are in
  [`NOTICE`](NOTICE).

**This is a test artifact, not a rigorous build.** It is not endorsed,
supported, or maintained by vLLM or NVIDIA. No warranty. Use at your own risk.

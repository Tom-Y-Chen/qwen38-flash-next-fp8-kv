# NVFP4 KV Cache — Exploration Record (PAUSED)

> **Status: PAUSED (2026-08-28).** The QSA-kernel-side NVFP4 port is **complete and
> validated**, but the feature cannot boot because of a **vLLM-core allocator invariant**
> — not a kernel bug. Fixing it requires editing vLLM core and maintaining a divergent
> fork. **Decision: do not self-compile vLLM for this.** Wait for upstream to land
> SM120 NVFP4 KV + the Qwen3.8-Flash-Next architecture handler, then re-use the code here
> as the known-good wiring reference. The running path today is **FP8 KV + MTP num_spec=2**
> (see `launch_fp8_x8.sh`).
>
> This branch (`explore/nvfp4-kv`) is **kept un-merged** so the work is preserved. It is
> the archive of the exploration, not a deliverable to ship.

## 决策摘要 (Chinese summary)

Qwen3.8-Flash-Next（176B 混合：GDN-Mamba + QSA 稀疏注意力 + N-gram PLE）跑在单台 DGX
Spark（GB10/sm_121）上。目标：把 KV cache 从 bf16/fp8 换成 **NVFP4**（打包 fp4 + 每 16 元素
fp8 block scale，0.5625 B/元素，内存是纯收益：KV token 上限约为 bf16 的 3.56×）。

- **已完成**：QSA 内核移植完毕并经内核/配置级验证。`patched/qsa.py` 正确发出 NVFP4 规格
  （uint8、full_dim=144、num_head_slots=4、state_content_bytes=144），并接好 flashinfer
  的写路径（`reshape_and_cache_flash "nvfp4"`）与读路径（gather→dequant→bf16→原 QSA 内核）。
  `patched/qsa_ops.py` 增加 `qsa_sparse_paged_attention_nvfp4`。`--kv-cache-dtype nvfp4`
  能干净加载，日志显示 `Using nvfp4 data type to store kv cache`。
- **阻塞**：boot 死在 vLLM **core** 的 `_get_kv_cache_groups_csa_linear`。CSA+linear 让每个
  QSA main_kv 缓存与一个 mamba-GDN state 槽位**共用一块物理张量、同一个 block id**，所以
  强制 `main_kv_page >= mamba_page`。
  - `main_kv_page = 4 × block_size × 144`，block_size 由**分配器自动派生**（无视 `--block-size`，实测 3200/1648）。
  - mamba GDN 页**与 block_size 无关**：float32 = 3,248,128 B / bf16 = 1,675,264 B。
  - 要过检查需 block_size ≥ 5639（f32）/ 2908（bf16）；分配器总选更小 → 一启动就崩。
- **为什么暂停**：修复点必须在 vLLM core（重建 fork）；且**光 pad-up 不够**——用小的
  block_size（~1648）把 main_kv 页垫到 3.25MB，摊下来 ~1971 B/token ≈ fp8，「纯内存收益」
  基本被吃掉。要拿真收益还得改分配器派生的 block_size（对 core 更深开刀）。上游正活跃
  开发 NVFP4 KV，但 **SM120/DGX-Spark 目前完全没有 NVFP4 路径**（issue #49011）；且
  Qwen3.8-Flash-Next 架构 handler 本身也未合并（PR #53896）。单机维护分叉成本高、收益低、
  merge 必冲突。
- **运行路径**：现用 FP8 KV（稳定、上游推荐）。NVFP4 KV **不是速度杠杆**（QSA 稀疏、不卡
  KV 带宽；A/B fp8≈bf16 13.37 vs 13.49 tok/s），只提升长上下文 token 上限。

---

## What was done (complete & validated)

This is a **port of vLLM's existing NVFP4 KV path to this model's custom QSA owner**. The
Qwen3.8-Flash-Next QSA sparse-attention owner uses a custom Triton kernel, not vLLM's generic
flashinfer backend — so the nvfp4 path that exists in vLLM was **not wired to this hybrid's
QSA layer**. This is exactly the gap the existing FP8 patch (`076b89b`) closed for fp8; this
port adds the missing nvfp4 branches.

- `patched/qsa.py` — QSA attention owner + `FlashAttentionImpl`:
  - Config gates accept `nvfp4*`.
  - `get_kv_cache_spec`: for `kv_quant_mode.is_nvfp4`, emits a `FullAttentionSpec` mirroring
    flashinfer — `dtype=torch.uint8`, `head_size = nvfp4_kv_cache_full_dim(head_dim)`,
    `state_content_bytes = full_dim*1`, `kv_quant_mode=KVQuantMode.NVFP4`.
  - `do_kv_cache_update`: for nvfp4 branches to flashinfer
    `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` (writes packed fp4 + block scales).
  - `forward_qsa`: for nvfp4, reads the `(num_blocks, 2*num_kv_heads, block_size, full_dim)`
    uint8 layout; each side dequantized via flashinfer two-pass
    (`nvfp4_split_data_scale` → `nvfp4_kv_dequantize`) to bf16, then fed to the existing
    `qsa_sparse_paged_attention`.
  - K/V scales are on-device CUDA tensors (CUDA-graph-safe).
- `patched/qsa_ops.py` — added `qsa_sparse_paged_attention_nvfp4` (bf16 gather/compaction
  feeding the dequant; mirrors SGLang's "compact packed rows, then dequant").
- Image `qwen38-flash-dgx-nvfp4` built (see `Dockerfile.nvfp4`); runtime `--kv-cache-dtype
  nvfp4` is accepted, log shows `Using nvfp4 data type to store kv cache`.

NVFP4 cache geometry (must match flashinfer exactly — a mismatch silently corrupts):

| field | value |
|---|---|
| shape | `(num_blocks, 2*num_kv_heads, block_size, full_dim)` |
| dtype | `torch.uint8` |
| `full_dim` | `head_size//2 + head_size//16` = 144 for head_dim=256 |
| `num_head_slots` | `2*num_kv_heads` = 4 (num_kv_heads=2) |
| `state_content_bytes` | 144 |
| slot split | K = slots `[0:num_kv_heads]`, V = `[num_kv_heads:2*num_kv_heads]` |

## The blocker (vLLM-core, not the kernel)

Boot dies in `vllm/v1/core/kv_cache_utils.py` → `_get_kv_cache_groups_csa_linear` at the
check `if unpadded_page > main_kv_page: raise ValueError("CSA+linear mamba cache owner ...
needs X bytes, but a main_kv tensor page has Y bytes.")`.

**Root cause.** The CSA+linear allocation scheme *shares one physical tensor* between each
QSA `main_kv` cache and a mamba-state slot, addressed by a single block id. Therefore it
requires `main_kv_page >= mamba_unpadded_page`.

Numbers (from runtime debug dumps):
- `main_kv_page = 4 × block_size × 144` (NVFP4).
- **block_size is auto-derived by the allocator** — it ignores `--block-size`. Observed
  3200 and 1648 across two runs; because of this the page lands at ~0.5625× the mamba page
  and is *always* below it.
- mamba GDN state page is **block_size-independent** (shapes fixed by the architecture):
  `conv_state (5,10240) bf16` + `temporal_state (48,128,128) float32` = **3,248,128 B**;
  with a bf16 override = **1,675,264 B**.
- To fit: `block_size ≥ mamba_page / 576` → **5639** (float32) or **2908** (bf16). The
  allocator picks below both.

Other constraints discovered:
- `MambaDType = Literal["auto","float32","float16","bfloat16"]` — mamba state cannot go
  below bf16.
- The model registers under the Qwen3.5 config family; its HF config sets
  `mamba_ssm_dtype='float32'`. Overriding to bfloat16 logs a warning (`config.py:799`) and is
  a **real correctness deviation** for the delta-net `temporal_state`.
- Enlarging `--block-size` does **not** grow the main_kv page (the allocator ignores it /
  re-derives it).

## Fix options analyzed

| Option | Description | Result |
|---|---|---|
| **A. bf16 mamba** | `--mamba-ssm-cache-dtype bfloat16` to shrink the GDN temporal state | **Insufficient.** mamba page drops to 1,675,264 B but the allocator re-derives block_size=1648 → page 949,248, still below. Also a correctness deviation. |
| **B. Pad main_kv up (float32)** | In core allocator, pad the main_kv page up to the mamba page | **Keeps float32** (no quality deviation), but with the auto-derived small block_size (~1648) it yields ~1971 B/token ≈ fp8 — the memory win is largely eaten. A real win *also* requires raising block_size → deeper core surgery on `_unify_page_size_bytes` / `_get_kv_cache_groups_csa_linear`. |
| **C. Decouple mamba cache from main_kv group** | Separate allocations so each is sized independently | **Full win** in theory, but invasive/risky in vLLM core. |

The fix must live in vLLM core (`_get_kv_cache_groups_csa_linear` / `_unify_page_size_bytes`),
**not** the QSA kernel.

## Decision & rationale (2026-08-28)

**Pause the custom fork; wait for upstream.** Do not self-compile vLLM for this on a single box.

1. The bug is in vLLM **core** → requires a fork rebuild, and B alone doesn't deliver the
   win without also changing block_size derivation.
2. Upstream is actively developing NVFP4 KV, but **SM120 / DGX-Spark has no NVFP4 KV path
   at all** (GitHub issue #49011). Several 2026 PRs are in flight, most requiring unreleased
   FlashInfer builds or carrying open bugs:
   - [#44851 — Add SM120 NVFP4 KV cache support](https://github.com/vllm-project/vllm/pull/44851)
   - [#46963 — FlashInfer for pre-SM100 NVFP4 KV updates](https://github.com/vllm-project/vllm/pull/46963)
   - [#37332 — Add nvfp4 support to reshape_and_cache_flash](https://github.com/vllm-project/vllm/pull/37332)
   - [#42464 — Patch SlidingWindowSpec.real_page_size_bytes for nvfp4](https://github.com/vllm-project/vllm/pull/42464)
   - [#44389 — Triton software NVFP4 KV cache](https://github.com/vllm-project/vllm/pull/44389) (referenced in AEON containers)
3. The Qwen3.8-Flash-Next architecture handler itself is **unmerged** upstream
   ([PR #53896](https://github.com/vllm-project/vllm/pull/53896)) — no released build loads
   this checkpoint yet. Everything floating on a divergent core fork is high-churn, low-payoff,
   and guaranteed to conflict on merge.
4. NVFP4 KV is **not a speed lever** — QSA is sparse and not KV-bandwidth-bound (A/B fp8≈bf16:
   13.37 vs 13.49 tok/s). It only raises the long-context token ceiling.

## Resume triggers

Re-engage this branch when BOTH land upstream:
1. **SM120 NVFP4 KV** — watch vllm issue [#49011](https://github.com/vllm-project/vllm/issues/49011),
   PRs #44851 / #46963 (and #37332 if the FlashInfer kernel becomes available).
2. **Qwen3.8-Flash-Next architecture handler** — PR [#53896](https://github.com/vllm-project/vllm/pull/53896) (or a released build that loads the checkpoint).

When they land: re-use `patched/qsa.py` + `patched/qsa_ops.py` as the **known-good wiring
reference** for this model's QSA layer. The patched files are **additive** (the fp8 path is
retained), so the same image serves `bf16|fp8|nvfp4` at runtime.

## Verification checklist (once bootable)

1. **Boot** — log shows `Resolved architecture`, `nvfp4` accepted, `Application startup
   complete`, no dtype/spread errors.
2. **Memory** — read the KV-pool token ceiling; expect ~**2.85M tokens** (3.56× bf16) at
   `--gpu-memory-utilization 0.90`.
3. **Correctness** — `nvfp4_correctness.sh` (short fact, sparse long retrieval, structured
   few-shot), 8-way concurrent → 0 errors, drift-compare with `temperature=0`.
4. **Speed** — concurrency-1 benchmark with `num_spec=2`; confirm **not slower** than fp8/bf16
   (~15 tok/s).

## Artifacts on this branch

| path | description |
|---|---|
| `patched/qsa.py` | QSA owner + `FlashAttentionImpl` with nvfp4 spec + write/read paths |
| `patched/qsa_ops.py` | Triton QSA kernels + `qsa_sparse_paged_attention_nvfp4` |
| `Dockerfile.nvfp4` | builds `qwen38-flash-dgx-nvfp4` |
| `launch_nvfp4.sh` | launch for NVFP4 KV, MTP `num_spec=2` (default) |
| `nvfp4_correctness.sh` | curl smoke tests (completions on port 18307) |
| `NVFP4-KV-EXPLORATION.md` | this record |

Debug tooling (not tracked) used to diagnose the blocker: a bind-mounted copy of
`vllm/v1/core/kv_cache_utils.py` with `DEBUG_CSA` prints in `_get_kv_cache_groups_csa_linear`
(fn ~line 1786, failing check ~line 1891) and `_unify_page_size_bytes` (~1050–1110).

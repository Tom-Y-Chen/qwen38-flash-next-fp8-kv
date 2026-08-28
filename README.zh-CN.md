# Qwen3.8-Flash-Next — fp8 KV Cache（实验性）

> **English** → [README.md](README.md)

> **⚠️ 这是一个实验 / 测试产物，不是严谨或生产构建。**
> 这是一个社区补丁，**不是**官方 vLLM 或 NVIDIA 发布。它仅在**一台**机器（NVIDIA DGX Spark，GB10 / sm_121）上，针对**一个**固定版本的 vLLM fork 构建进行了验证。不同的 vLLM 版本、attention 后端或 GPU 可能无法工作（或可能静默回退）。fp8 scales 是**未校准的（scale = 1.0）**。风险自负。

本仓库为 **Qwen3.8-Flash-Next** 的 **Qwen Sparse Attention (QSA)** Triton kernel 启用了 **fp8 (`fp8_e4m3`) KV cache**，使单台 DGX Spark（128 GB 统一内存）能够在完整 262k 上下文下服务 **~11 个并发请求**，而不是 bf16 KV cache 允许的 ~4 个并发。

QSA 稀疏注意力层运行自己的 Triton kernel（不是 FlashAttention），因此 vLLM 的通用 fp8-KV 路径不覆盖它们。这两个补丁添加了 kernel 内 fp8 → bf16 反量化，并解除了此前对该模型硬性拒绝 fp8 KV 的 dtype 门控。

---

## 它做了什么

在 NVIDIA 的 vLLM fork 中修改了两个文件：

| File | Change |
|---|---|
| `vllm/models/qwen3_8_flash_next/nvidia/qsa.py` | 接受 `fp8*` KV cache dtypes；将 dtype 注入到 FlashAttention 的 fp8 check 之后；将 uint8（raw-byte）fp8 cache 重新视为 `float8`；将 per-layer KV scales 传入 kernel。 |
| `vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py` | 向 Triton kernel 添加 `k_scale`/`v_scale` 参数；在 kernel 内将 fp8 K/V 反量化为 bf16；设置 `num_stages=1`。 |

**修复了两个根本原因：**
1. **`_k_scale` 从错误对象读取** — `forward_qsa` 在 impl 上读取 `float(self._k_scale)`，而不是在 layer 上。已修复为读取 `layer._k_scale_float` / `_v_scale_float`（普通浮点数，在 `torch.compile` 下安全）。
2. **sm_121 上的共享内存溢出** — 该 kernel 在 GB300（227 KB smem/SM）上调优。在 `num_stages=2` 时，它超过了 DGX Spark 的 sm_121 上 100 KB smem 上限：fp8 需要 106 KB，bf16 需要 136 KB。`num_stages=1` 两者都适配（仅流水线深度；数值不变）。

---

## 结果（在一台 DGX Spark 上测量）

- **fp8 KV @ GPU_MEM 0.94 → `2,971,999` KV tokens → 在 262k 上下文下 `11.34×` 并发。**
  这大约是 bf16 上限（~4×）的 **2.8×**，在 8 并发目标之上留有 ~40% 余量。
- **正确性：** 短 prompt 全部正确；长稀疏注意力检索（6,335 tokens / 120 facts）选中了准确的事实；**8 路并发服务返回 0 个错误**。
- **与 bf16 的漂移：** 相同 prompts，`temperature=0` — **最终答案 7/7 匹配**。只有自由形式的 `<thinking>` 措辞略有变化（fp8 舍入会轻微扰动 logits），这**不会**翻转答案。

---

## 适用范围 — 适用于哪些模型

**适用于：** 任何**相同架构**的 **Qwen3.8-Flash-Next** checkpoint，无论**权重**量化方式如何（NVFP4、FP8 weights 等）。fp8 KV 在运行时通过 `--kv-cache-dtype` 选择，与权重如何量化正交。这些补丁绑定于架构，而不是绑定于 checkpoint。

**不适用于：**
- **不同的 vLLM 构建** — 精确的行号 / 内部 API（`current_platform`、`set_default_quant_scales`、`torch.uint8` 存储怪癖、FA2 dtype check）特定于该固定 fork。需要进行移植。
- 具有自己的自定义 attention kernel 的**不同模型架构** — 它需要在*其* kernel 中提供类似的 fp8 支持。
- **nvfp4 KV**：需要 datacenter Blackwell SM100（TMEM）；GB10 / sm_121 没有。

**硬件说明：** 需要 `num_stages=1` 才能适配 sm_121 的 100 KB smem。在其他 GPU（例如 GB300 / sm_100）上，这并非必要，并且可能会损失一点性能 — 如果你面向这些 GPU，请参阅 `patches/qsa_ops.py.patch` 并重新调优。

---

## 仓库布局

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

## 构建

**前提条件 — 先构建基础镜像**（不在本仓库中；它是 NVIDIA 的 vLLM fork 容器，`qwen38-flash-dgx`）：

```
# Assemble from NVIDIA's vLLM fork (vLLM 0.1.dev20073+g8e685d198, FlashAttention v2),
# the Qwen3.8-Flash-Next model code, PLE-mmap support, and FlashInfer.
# Expected container entrypoint = vllm, and the two base files present at:
#   .../vllm/models/qwen3_8_flash_next/nvidia/qsa.py
#   .../vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py
```

然后要么在树内应用补丁并重新构建，要么构建提供的镜像：

```bash
docker build -t qwen38-flash-dgx-fp8 -f Dockerfile.fp8 .
```

---

## 用法

```bash
bash launch_fp8_x8.sh          # 262k context, 8 concurrent, fp8 KV
```

关键 flags（必须与架构匹配）：
- `--kv-cache-dtype fp8_e4m3` — fp8 KV cache（本补丁的全部意义）。
- `--max-model-len 262144 --max-num-seqs 8 --gpu-memory-utilization 0.94`。
- `-cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="[...]"` — 此混合架构所必需。
- `VLLM_PLE_MMAP=1` — 将 ~47 GB n-gram/PLE table mmap 到 RAM。

然后查询 OpenAI 兼容 endpoint，例如：

```bash
curl http://127.0.0.1:18307/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"temperature":0,"seed":42}'
```

---

## 已知限制 / 注意事项

- **未校准的 fp8（scale = 1.0）。** KV 值只是被舍入到 fp8_e4m3；没有逐张量校准。即便如此，与 bf16 相比，测试得到的答案仍被保留。真实部署应校准逐张量 scales。
- **fp8 是有损的。** 不要声称“无损”。在 8 bits 以下（nvfp4 / 3.5-bit / 类似）精度本质上会退化。
- **bf16 基线在 sm_121 上也必须运行 `num_stages=1`** — 原始 kernel 在此硬件上无论 dtype 如何都会溢出共享内存。
- **仅验证了 DGX Spark / sm_121。** 其他地方的时间、内存和精确数字可能不同。
- **不包含模型权重。** 使用你有权下载的 checkpoint（该补丁不依赖特定 checkpoint）。

---

## 许可与版权

- 本仓库是源自 **NVIDIA 的 vLLM fork** 的**补丁**。
- 上游 **vLLM** 项目采用 **Apache License 2.0** 许可（版权归 vLLM Project 及其贡献者所有）。
- **Qwen / Qwen3.8-Flash-Next** 权重由 Qwen team（Alibaba）提供，并受其自身许可约束。**此处不包含任何模型权重。**
- 本仓库以 **Apache License 2.0** 分发 — 参见 [`LICENSE`](LICENSE)。归属与来源见 [`NOTICE`](NOTICE)。

**这是一个测试产物，不是严谨构建。** 它未获得 vLLM 或 NVIDIA 的认可、支持或维护。无担保。风险自负。

#!/usr/bin/env bash
# EXPERIMENTAL: launch Qwen3.8-Flash-Next with fp8 KV cache on a single
# DGX Spark (GB10 / sm_121), targeting ~8 concurrent requests @ 262k context.
#
# Adjust the model SNAP path to your own Qwen3.8-Flash-Next checkpoint.
# Any quantized-weight checkpoint of the SAME architecture works (fp8 KV is
# orthogonal to weight quantization). The image must be the patched one from
# Dockerfile.fp8 (qwen38-flash-dgx-fp8).
set -euo pipefail

NAME=qwen38-flash-fp8-x8
SNAP=/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
IMAGE=qwen38-flash-dgx-fp8
# fp8_e4m3 = 8-bit float E4M3 KV cache; this is what unlocks the concurrency.
KVD=fp8_e4m3

# Required for this hybrid model on this build:
SPLIT="[\"vllm::unified_attention_with_output\",\"vllm::unified_mla_attention_with_output\",\"vllm::mamba_mixer2\",\"vllm::mamba_mixer\",\"vllm::short_conv\",\"vllm::qwen3_8_flash_next_ple_short_conv\",\"vllm::qwen3_8_flash_next_qsa_with_output\",\"vllm::linear_attention\",\"vllm::qwen_gdn_attention_core\",\"vllm::qwen_gdn_attention_core_fused_norm_packed\",\"vllm::sparse_attn_indexer\",\"vllm::ple_mmap_lookup\"]"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 16g -p 18307:8000 \
  -v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  "$IMAGE" "$SNAP" --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format safetensors \
    --max-model-len 262144 --max-num-seqs 8 --gpu-memory-utilization 0.94 \
    --kv-cache-dtype "$KVD" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --no-enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="$SPLIT" \
    --no-enable-flashinfer-autotune

echo "started $NAME KVD=$KVD seq=8 ctx=262144 GMU=0.94"

#!/usr/bin/env bash
# NVFP4 correctness smoke test (once the server is ready).
set -euo pipefail
HOST=127.0.0.1:18307
q() {
  curl -s -X POST "http://$HOST/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "$1"
}

echo "=== T1 short fact prompt ==="
q '{"model":"qwen3.8-flash-next","prompt":"What is the capital of France? Answer with just the city.","max_tokens":40,"temperature":0}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d.get('choices',[{}])[0].get('text',''))[:400])"

echo "=== T2 short factual reasoning ==="
q '{"model":"qwen3.8-flash-next","prompt":"If a train travels 60 km in 40 minutes, what is its speed in km/h?","max_tokens":80,"temperature":0}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d.get('choices',[{}])[0].get('text',''))[:400])"

echo "=== T3 long-context retrieval ==="
q '{"model":"qwen3.8-flash-next","prompt":"The secret codeword for the Acme project is ZEBRA-9421. Do not forget this. What is the secret codeword for the Acme project?","max_tokens":60,"temperature":0}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d.get('choices',[{}])[0].get('text',''))[:400])"

echo "=== T4 structured few-shot ==="
q '{"model":"qwen3.8-flash-next","prompt":"Complete: 2,4,6,8,","max_tokens":20,"temperature":0}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d.get('choices',[{}])[0].get('text',''))[:200])"

echo "=== DONE ==="
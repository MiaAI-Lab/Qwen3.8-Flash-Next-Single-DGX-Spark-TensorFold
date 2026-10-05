#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
PLE_ON_SSD="${PLE_ON_SSD:-1}"
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"

export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
export TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"
if [[ "$PLE_ON_SSD" == 1 ]]; then
  export TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-2048}"
fi

if [[ "$PARALLEL" -lt 2 && "$VISION" == 1 ]]; then
  echo "Notice: CUDA Flash Next vision requires --parallel >= 2. Setting PARALLEL=2 to enable vision."
  PARALLEL=2
fi

echo "=================================================="
echo "Starting TensorFold for $MODEL_ID"
echo "  Served Name : $SERVED_NAME"
echo "  Listen      : $HOST:$PORT"
echo "  Parallel    : $PARALLEL"
echo "  Context     : $CONTEXT"
echo "  KV Dtype    : $KV_DTYPE"
echo "  PLE on SSD  : $PLE_ON_SSD"
echo "  Vision      : $VISION"
echo "=================================================="

# Check if model is cached or needs pulling
if [[ "${PULL_IF_MISSING:-1}" == 1 ]]; then
  if ! tensorfold info "$MODEL_ID" >/dev/null 2>&1; then
    echo "Model weights not fully cached. Running tensorfold pull $MODEL_ID..."
    tensorfold pull "$MODEL_ID"
  else
    echo "Model weights verified in cache."
  fi
fi

SERVE_ARGS=(
  --name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --parallel "$PARALLEL"
  --context "$CONTEXT"
  --kv-dtype "$KV_DTYPE"
  --mtp-drafts "$MTP_DRAFTS"
  --mtp-confidence "$MTP_CONFIDENCE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --max-tokens "$MAX_TOKENS"
)

[[ "$PLE_ON_SSD" == 1 ]] && SERVE_ARGS+=(--ple-on-ssd)
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
[[ "$VISION" == 1 && "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
[[ "$VISION" == 1 && -n "$VISION_MAX_IMAGES" ]] && SERVE_ARGS+=(--vision-max-images "$VISION_MAX_IMAGES")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi

if [ $# -gt 0 ]; then
  exec "$@"
else
  exec tensorfold serve "$MODEL_ID" "${SERVE_ARGS[@]}"
fi

# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. ./.env (KEY=value lines, read, never run)
# Settings marked PROVISIONAL follow TensorFold's Zig engine (tensorfold-native) and change when a gated commit lands.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# One checkpoint: azampatti's INT4-AutoRound (top-5 routing, 4.8B active, GPTQ int4 experts, FP8 n-gram table
# in ple-table/). The engine reads the format from the checkpoint. There is no second quant in this recipe.
MODEL_ID="${MODEL_ID:-azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound}"
MODEL_REVISION="${MODEL_REVISION:-1464274120d36a4d8fcaa934552334a7d83ce0fd}"
CKPT_GIB=122
QUANT_LABEL="INT4-AutoRound"

# The engine: TensorFold's Zig engine, built from TF_REPO at TF_REF (branch zig-flashnext) with patches/*.patch
# applied (git apply in the checkout's root), with Zig ZIG_VERSION. Only the Zig build goes into the image.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-db281878ddb836fd0df510d8771ecb7e0fe47d26}"   # zig-flashnext; patches bring the gated Flash Next CUDA engine
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_TAG}}"
KERNELS_PATH=/opt/tensorfold/share/tensorfold/cuda/sm121
# Triton's line info records each kernel source's mtime. Pin it so cubins match the reference kernel set.
KERNEL_SOURCE_MTIME="${KERNEL_SOURCE_MTIME:-1791318675}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE kernel-mtime-$KERNEL_SOURCE_MTIME tp-1"
  ) | sha256sum | cut -c1-12
}
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
prebuilt_image() {
  local tag="${TF_TAG}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Native window 262,144. Past it the engine applies Qwen's YaRN (factor 4) when TF_FLASHNEXT_YARN=4, up to 1,048,576.
# On one Spark a full 1M-token fp8 cache is 17.46 GiB (bf16 is 30.06 GiB) on top of ~66 GiB of weights; a full 1M
# prompt's scratch on top of that is not the default. The default stays the native window.
CONTEXT="${CONTEXT:-262144}"
if [[ "$CONTEXT" =~ ^[0-9]+$ ]] && (( CONTEXT > 262144 )); then _yarn=4; else _yarn=0; fi
export TF_FLASHNEXT_YARN="${TF_FLASHNEXT_YARN:-$_yarn}"
unset _yarn
# Requests decoded together. The engine admits each one inside its own budget (MemAvailable after load, less
# TENSORFOLD_MEMORY_RESERVE_GIB, and with VISION=1 less another 2 GiB kept for the vision helper) and refuses one
# that would not fit. The pool is shared: streams grow on demand, so PARALLEL is a cap, not a promise that every
# stream can sit at a full 262,144-token window at once. The default is filled in after the FP8+vision measurement.
PARALLEL="${PARALLEL:-8}"
# MTP: up to this many drafts a round, kept while confidence stays at least this. Drafts are checked, so replies match
# serial decoding. PROVISIONAL with the gated engine (6eb39c1).
export TF_FLASHNEXT_DEPTH="${TF_FLASHNEXT_DEPTH:-15}"
# Unset TF_FLASHNEXT_CONFIDENCE: the engine uses the running product (-0.4) while at most
# TF_FLASHNEXT_PRODUCT_STREAMS streams are live, and confidence 0.5 above that. Setting
# TF_FLASHNEXT_CONFIDENCE=0.5 forces the per-draft rule at every width.
export TF_FLASHNEXT_PRODUCT_STREAMS="${TF_FLASHNEXT_PRODUCT_STREAMS:-2}"
export TF_FLASHNEXT_PREFILL_TAIL="${TF_FLASHNEXT_PREFILL_TAIL:-512}"
# Block-FP8 split-K projections of 2048+ columns below 256 rows on the fused kernel (one block a column tile, its K
# slices added in slice order, as the cluster kernel does) and the int4 lm_head in 32-row items (patches/0018).
# 0 / 1: the cluster kernel and 16-row items. The same bits either way.
export TF_FLASHNEXT_FP8_FUSED_SPLIT="${TF_FLASHNEXT_FP8_FUSED_SPLIT:-4}"
export TF_FLASHNEXT_HEAD_ROW_TILES="${TF_FLASHNEXT_HEAD_ROW_TILES:-2}"
# KV cache (--kv-dtype). fp8 is the default: about 1.84x the pool of bf16, and lossy (~98.8% top-1 agreement with a
# bf16 cache, so a free-running reply can differ). bf16 is exact: KV_DTYPE=bf16. FP8 works together with --vision.
KV_DTYPE="${KV_DTYPE:-fp8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
DRAFTS="${DRAFTS:-1}"
# Sampling defaults (a request's own values win). The Zig server honours these flags.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
# Image and video. VISION=1 (default) passes --vision: rank 0 starts the Python helper (Pillow, PyAV, the 27-layer
# tower). VISION=0 is text only and returns the 2 GiB workspace to the KV budget. VISION_URLS=1 also accepts public
# https:// URLs; the default is data URLs only.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
MAX_VIDEOS="${MAX_VIDEOS:-4}"                                          # videos a request (--vision-max-videos)
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-2048}"
# The engine keeps this much MemAvailable free when it sizes the KV pool. prepare.sh and start.sh refuse a Spark
# that has less than this free.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-10}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

# There is no int8 cache and no SSD n-gram reader: the FP8 n-gram table is loaded with the weights
# (~65.7 GiB measured, table included).
# Memory at start. Weights ~66 GiB. One full window's cache is kv_gib below. Prefill scratch is outside the engine's
# cache budget; 8 GiB here is slack for a 2048-row chunk, not a measured 1M-prompt peak. MEM_FLOOR_GIB is the line
# start.sh stops the server at while it loads, and the line prepare.sh refuses to start below. MEM_CHECK=0 turns
# both into a warning.
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"
# kv_gib <tokens>: one sequence at TP=1, from State.cacheBytes (cuda_state.zig).
# bf16: 30,784 bytes a token. fp8: 17,888 (keys carry both scales, values are codes, indexer and pooled stay bf16).
kv_gib() {
  local ctx=${1:-$CONTEXT} bytes=30784
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  [[ "$KV_DTYPE" == fp8 ]] && bytes=17888
  echo $(( (bytes * ctx + 1073741823) / 1073741824 ))
}
# mem_need_gib <context>: MemAvailable a start needs: weights, one full window, scratch slack, the vision workspace
# when VISION=1, floor, 2 GiB.
mem_need_gib() {
  local ctx=${1:-$CONTEXT} vision=0
  [[ "$VISION" == 1 ]] && vision=2
  echo $(( 66 + $(kv_gib "$ctx") + 8 + vision + MEM_FLOOR_GIB + 2 ))
}

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-flash-next-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-qwen38/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-140}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label rev
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  rev=$(snapshot_rev)
  echo "model=$MODEL_ID@$rev image=$label patches=$hash"
}

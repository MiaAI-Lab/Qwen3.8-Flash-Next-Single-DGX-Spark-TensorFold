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

# Checkpoint. The default is NVIDIA's quantization-aware-distilled FP4 (NVFP4 routed experts, MXFP8 and bf16 elsewhere,
# FP8 n-gram table, MTP head): nvidia/Qwen3.8-Flash-Next-FP4-Local, ~96 GiB. azampatti's INT4-AutoRound (top-5 routing,
# GPTQ int4 experts, ~122 GiB) stays available: CHECKPOINT=int4 ./start.sh, or set MODEL_ID and MODEL_REVISION yourself.
# The engine reads the format from the checkpoint. WEIGHTS_GIB is what loading takes of MemAvailable (measured on the Zig engine:
# 81 GiB for the NVIDIA checkpoint with vision on).
CHECKPOINT="${CHECKPOINT:-nvfp4}"
_pin=1; [[ -z "${MODEL_ID:-}" ]] || _pin=0       # a MODEL_ID set by hand takes no preset revision pin
case "$CHECKPOINT" in
  nvfp4)
    : "${MODEL_ID:=nvidia/Qwen3.8-Flash-Next-FP4-Local}"
    (( ! _pin )) || : "${MODEL_REVISION:=b9ce18a6887e18c089fdea8509a33e63429a3300}"
    CKPT_GIB=96; WEIGHTS_GIB="${WEIGHTS_GIB:-81}"
    QUANT_LABEL="NVIDIA FP4 QAD: NVFP4 experts, MXFP8" ;;
  int4)
    : "${MODEL_ID:=azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound}"
    (( ! _pin )) || : "${MODEL_REVISION:=1464274120d36a4d8fcaa934552334a7d83ce0fd}"
    CKPT_GIB=122; WEIGHTS_GIB="${WEIGHTS_GIB:-66}"
    QUANT_LABEL="INT4-AutoRound" ;;
  *) echo "CHECKPOINT is nvfp4 or int4, not $CHECKPOINT" >&2; exit 1 ;;
esac
MODEL_REVISION="${MODEL_REVISION:-}"
unset _pin

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
# On one Spark a full 1M-token fp8 cache is 17.46 GiB (bf16 is 30.06 GiB) on top of the weights (~66 GiB INT4, ~81 NVIDIA FP4); a full 1M
# prompt's scratch on top of that is not the default. The default stays the native window.
CONTEXT="${CONTEXT:-262144}"
if [[ "$CONTEXT" =~ ^[0-9]+$ ]] && (( CONTEXT > 262144 )); then _yarn=4; else _yarn=0; fi
export TF_FLASHNEXT_YARN="${TF_FLASHNEXT_YARN:-$_yarn}"
unset _yarn
# Requests decoded together. The engine admits each one inside its own budget (MemAvailable after load, less
# TENSORFOLD_MEMORY_RESERVE_GIB, and with VISION=1 less another 2 GiB kept for the vision helper) and refuses one
# that would not fit. The pool is shared: streams grow on demand, so PARALLEL is a cap, not a promise that every
# stream can sit at a full 262,144-token window at once. Measured on a 121 GiB Spark (NVIDIA FP4, fp8 KV, vision on): the
# engine reports a 24.3 GiB sequence pool after load, one full 262,144 window is 4.4 GiB, so 5 streams x 262,144 fit
# (21.9 GiB); 8 streams at full windows do not (35 GiB), they share the pool. 5 is the default for that reason.
PARALLEL="${PARALLEL:-5}"
# MTP: up to this many drafts a round, kept while confidence stays at least this. Drafts are checked, so replies match
# serial decoding. 6 drafts: on the NVIDIA FP4 checkpoint deeper chains (the engine's 15) cost verify rows and lose 5 to 20%
# at 3 to 5 streams (measured, 1 to 5 streams, depths 4 to 15); 4 to 8 are within noise of each other.
export TF_FLASHNEXT_DEPTH="${TF_FLASHNEXT_DEPTH:-6}"
# Unset TF_FLASHNEXT_CONFIDENCE: the engine uses the running product (-0.4) while at most
# TF_FLASHNEXT_PRODUCT_STREAMS streams are live, and confidence 0.5 above that. Setting
# TF_FLASHNEXT_CONFIDENCE=0.5 forces the per-draft rule at every width. 5 (= PARALLEL's default): the product at every
# width the default serves.
export TF_FLASHNEXT_PRODUCT_STREAMS="${TF_FLASHNEXT_PRODUCT_STREAMS:-5}"
export TF_FLASHNEXT_PREFILL_TAIL="${TF_FLASHNEXT_PREFILL_TAIL:-512}"
# Prompt experts as a shared-memory tile (patch 0014): the routed experts of prompt chunks of 1024 rows or more run
# through a 64-pair tile kernel that gives bit-identical results to the older kernels and cold prefill +14% (16k prompt).
# 2 = gate/up and down, 1 = gate/up only, 0 = the older kernels. (Decode and chunks under 1024 rows are not affected.)
export TF_FLASHNEXT_PROMPT_EXPERT_TILE="${TF_FLASHNEXT_PROMPT_EXPERT_TILE:-2}"
# Fair prefill (patch 0015), OFF by default: with several long prompts queued, prompts are filled in slices, shortest first,
# so a short request waits ~2 s (not minutes) behind five 100k prompts and running decoders keep a ~0.15 s gap. Costs ~20% on
# the total time of a mixed load (five 100k prompts + shorts: 280 s -> 342 s); idle prefill is unchanged. One rank only.
# Opt in: TF_FLASHNEXT_FILL_FAIR=1 (TF_FLASHNEXT_FILL_LAYERS slice size, default 2; TF_FLASHNEXT_FILL_AGING rows, default 1024).
# Prompt cache room (patch 0029), ON by default, one rank only. Kept prompt states share the sequence pool with live requests;
# without this a prompt that does not fit is refused (NoRoom) even when idle kept states hold the room it needs.
# TF_FLASHNEXT_ADMIT_RECLAIM: admission frees kept states (superseded first, else least recently used, never the prompt's
# own resume) when, and only when, that makes the prompt fit. TF_FLASHNEXT_SUPERSEDE: a conversation's next turn frees the
# state it resumed once it keeps a longer one. 0 turns either off. Freeing a sequence always waits for the GPU's queued work.
export TF_FLASHNEXT_ADMIT_RECLAIM="${TF_FLASHNEXT_ADMIT_RECLAIM:-1}"
export TF_FLASHNEXT_SUPERSEDE="${TF_FLASHNEXT_SUPERSEDE:-1}"
# Image prompts in the prompt cache (patch 0030), ON by default, one rank only: a conversation with images keeps and
# resumes its prompt states like a text one, each image told apart by a hash of its bytes, so a repeated image turn
# prefills only its new rows. Videos are not kept. TF_FLASHNEXT_MEDIA_REUSE=0 restores the old rule (image prompts
# prefill from their start every turn).
export TF_FLASHNEXT_MEDIA_REUSE="${TF_FLASHNEXT_MEDIA_REUSE:-1}"
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

# There is no int8 cache and no SSD n-gram reader: the FP8 n-gram table is loaded with the weights (WEIGHTS_GIB,
# table included: 65.7 GiB measured for INT4-AutoRound, about 81 measured for the NVIDIA checkpoint: MemAvailable fell from 117.3 to 36.3 GiB at load with vision on and 8 slots, which includes the 0.84 GiB vision tower and kernels).
# Memory at start. Weights WEIGHTS_GIB. One full window's cache is kv_gib below. Prefill scratch is outside the engine's
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
  echo $(( WEIGHTS_GIB + $(kv_gib "$ctx") + 8 + vision + MEM_FLOOR_GIB + 2 ))
}

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-flash-next-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-qwen38/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-$((CKPT_GIB + 18))}"
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

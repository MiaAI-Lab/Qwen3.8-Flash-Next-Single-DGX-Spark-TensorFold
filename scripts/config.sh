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

# ABLIT=1 serves NVIDIA's FP4-Local with drowzeys's Keys "house-qsa" abliteration overlay grafted in: the 12 QSA layers' self_attn.o_proj
# (L3, 7, ... 47; MXFP8 g32, requantized after a rank-1 refusal-direction projection) come from the overlay's ~190 MB sidecar. GDN out_proj,
# every expert, the MTP head, the n-gram table, vision and the chat template stay NVIDIA's. The overlay is gated, free, with automatic approval after you accept
# its Responsible Use Agreement at https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47 while logged in;
# export HF_TOKEN or `hf auth login` with that account. It only fits the nvfp4 checkpoint at the revision it was made from. The model has had safety
# refusals removed. Its author's own checks (refusal32 32/32) ran on the vLLM/TensorFold recipe, not on this build: validate it here (tools/exact.py,
# tools/needle.py) before relying on it.
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" =~ ^[01]$ ]] || { echo "ABLIT must be 0 or 1, not $ABLIT" >&2; exit 1; }
ABLIT_ID=drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47
ABLIT_REVISION="${ABLIT_REVISION:-c44f5b92ee3d307b82aab1c55f4cc4ae1d440344}"
ABLIT_SIDECAR=fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors
ABLIT_STOCK_REVISION=b9ce18a6887e18c089fdea8509a33e63429a3300     # the nvidia/Qwen3.8-Flash-Next-FP4-Local snapshot the sidecar was made from
if [[ "$ABLIT" == 1 ]]; then
  if [[ "$CHECKPOINT" != nvfp4 || "$_pin" != 1 || "$MODEL_REVISION" != "$ABLIT_STOCK_REVISION" ]]; then
    echo "ABLIT=1 grafts onto nvidia/Qwen3.8-Flash-Next-FP4-Local @ ${ABLIT_STOCK_REVISION:0:8} (CHECKPOINT=nvfp4, no MODEL_ID or MODEL_REVISION of your own)" >&2
    exit 1
  fi
  QUANT_LABEL+=" + Keys house-qsa ablit"
fi
unset _pin

# The engine: TensorFold's Zig engine (tensorfold-native), built from TF_REPO at TF_REF (the v1.2.7 release) with patches/*.patch
# applied (git apply in the checkout's root), with Zig ZIG_VERSION. Only the Zig build goes into the image, plus the two
# image decoders the engine wants (libjpeg-turbo for JPEG, libwebp's decoder for WebP), built from pinned sources.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-93a2e93ecebff8aa52b88e58c5dd222516e69c5f}"   # v1.2.7
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
LIBJPEG_TURBO_VERSION=3.1.4.1
LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
LIBWEBP_VERSION=1.6.0
LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
# libtfgrammar.so (structured output, tool_choice "required" and named functions) comes prebuilt from TensorFold's own release archive, checked against its SHA-256;
# the archive ships it with xgrammar's Apache-2.0 notices. TF_RELEASE is the release that matches TF_REF.
TF_RELEASE_VERSION=1.2.7
TF_RELEASE_AARCH64_SHA256=250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b
TF_TAG="v1.2.7-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE jpeg-$LIBJPEG_TURBO_VERSION-$LIBJPEG_TURBO_SHA256 webp-$LIBWEBP_VERSION-$LIBWEBP_SHA256 grammar-$TF_RELEASE_VERSION-$TF_RELEASE_AARCH64_SHA256"
  ) | sha256sum | cut -c1-12
}
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"
# The published image of the current patch set (scripts/publish-image.sh): prepare.sh pulls it by digest while the patch hash matches IMAGE_TAG.
# None is published for the v1.2.7 patch set yet, so prepare.sh tries $GHCR_IMAGE:<tag> and otherwise builds locally.
# The v1.1.4 image stays in the registry: v1.1.4-eab8582-4baf16137ecd, sha256:5c6c42b9747318c1cd8b6cb85cb69493fff2e82cb23b61e49b9cf54cd09ac28e.
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
# Native window 262,144; the Flash Next CUDA engine of TensorFold 1.2.7 has no YaRN extension (it refuses a legacy rope_scaling), so this is also the maximum.
CONTEXT="${CONTEXT:-262144}"
# Requests decoded together. The engine admits each one inside its own budget (MemAvailable after load, less
# TENSORFOLD_MEMORY_RESERVE_GIB, and with VISION=1 less another 2 GiB kept for the vision helper) and refuses one
# that would not fit. The pool is shared: streams grow on demand, so PARALLEL is a cap, not a promise that every
# stream can sit at a full 262,144-token window at once. Measured on a 121 GiB Spark (NVIDIA FP4, fp8 KV, vision on): the
# engine reports a 24.3 GiB sequence pool after load, one full 262,144 window is 4.4 GiB, so 5 streams x 262,144 fit
# (21.9 GiB); 8 streams at full windows do not (35 GiB), they share the pool. 5 is the default for that reason.
PARALLEL="${PARALLEL:-5}"
# Prompt rows a pass (v1.2.7's default is 1,536; the v1.1.4 build's patch 0006 used 3,072). 3,072 prefills about 4% faster on a Spark
# (bench_streams.py, 4k to 126k tokens) for 0.8 GiB of the prompt cache; a stream decoding beside a prompt waits up to one pass for its token.
export TF_FLASHNEXT_CHUNK="${TF_FLASHNEXT_CHUNK:-3072}"
# MTP: up to this many drafts a round (the engine's default is 6; drafts are checked, so replies match serial decoding).
export TF_FLASHNEXT_DRAFTS="${TF_FLASHNEXT_DRAFTS:-6}"
# v1.2.7 carries most of what patches 0004 and 0006 to 0017 of the v1.1.4 series did; these are its settings for them. None of them
# changes a reply's tokens. Already the engine's default (listed so a .env can turn them off): sampled streams in the batched draft
# pass (was 0007), the GPU top-k candidates of a sampled draw (0008, TF_FLASHNEXT_GPU_TOPK), pending DeltaNet rows (0012), the shared
# expert on a side stream and the lanes' DeltaNet steps in one launch (0016). The n-gram table is read with pread on Linux by default
# (0013's reader, TF_FLASHNEXT_PLE_IO).
export TF_FLASHNEXT_SAMPLED_HEADS="${TF_FLASHNEXT_SAMPLED_HEADS:-1}"
export TF_FLASHNEXT_KEEP_LATE="${TF_FLASHNEXT_KEEP_LATE:-1}"
export TF_FLASHNEXT_SHARED_SIDE="${TF_FLASHNEXT_SHARED_SIDE:-1}"
export TF_FLASHNEXT_GDN_LANES="${TF_FLASHNEXT_GDN_LANES:-1}"
# Opt-in upstream, on here: the next prompt chunk's n-gram rows hinted to the page cache during the current one (0013), and the
# NVFP4 prompt experts on the shared-memory K32 tiles adapted from our patch 0014 (2: gate/up and down; 1 gate/up only; 0 off).
export TF_FLASHNEXT_WARM_AHEAD="${TF_FLASHNEXT_WARM_AHEAD:-1}"
export TF_FLASHNEXT_PROMPT_EXPERT_TILE="${TF_FLASHNEXT_PROMPT_EXPERT_TILE:-2}"
# Prefill (patch 0004, the v1.1.4 series' 0006 and 0015 ported): prompt passes run the MXFP8 projections 1,024 columns or wider on a
# 128 x 128 tile with the block scales folded into the weights (one fp32 sum over K; 1.4-1.65x the exact lane at a 3,072-row chunk,
# +7-9% prefill; calls of 128 rows or fewer take its 64 x 64 form, same bits, so short prompts are not slower). Unlike the switches above it changes a prompt's numbers (within the exact kernels' rounding bound; each row's bits
# do not depend on its chunk, so replies stay independent of drafting and batching); 0 keeps the exact kernels.
export TF_FLASHNEXT_PROMPT_MX128="${TF_FLASHNEXT_PROMPT_MX128:-1024}"
# The draft chain stops on the running product of its draft probabilities: keep draft j while p1*...*pj >= 0.4 (v1.2.7's default
# TF_FLASHNEXT_CONFIDENCE, the rule patch 0017 restored). Beside other open streams v1.2.7 switches greedy chains to a 0.6 floor a
# draft; TF_FLASHNEXT_SHARED_DRAFTS=0 keeps the product rule there too, as patch 0017 did up to 5 streams. Only draft rows change.
export TF_FLASHNEXT_CONFIDENCE="${TF_FLASHNEXT_CONFIDENCE:--0.4}"
export TF_FLASHNEXT_SHARED_DRAFTS="${TF_FLASHNEXT_SHARED_DRAFTS:-0}"
# The MTP draft head's rows in 4 bits (a fourth of the bytes a chained draft reads; v1.1.4's patch 0005 made it the default,
# v1.2.7 keeps bf16 by default). Drafts are only proposals, so replies are the target's either way. bf16 for the exact head.
export TF_FLASHNEXT_DRAFT_HEAD="${TF_FLASHNEXT_DRAFT_HEAD:-affine4}"
# KV cache (--kv-cache). fp8 is the default (patch 0003): about 1.84x the pool of bf16, and lossy (~98.8% top-1 agreement with a
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
# 9 GiB since v1.1.4's patch 0014 (five full 262,144-token windows need 23.9 GiB; a 10 GiB reserve with the 2 GiB vision workspace left 23.8).
# Keep nothing else big running on the Spark.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-9}"
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

# The folder under HF_CACHE that holds the checkpoint the engine reads: the snapshot, or with ABLIT=1 the view scripts/ablit_view.py builds
# (the snapshot's files linked, plus the overlay's sidecar). It stays under HF_CACHE because that is all the container mounts.
ABLIT_VIEW_SUB="tensorfold-qwen38/ablit-${MODEL_REVISION:0:8}-${ABLIT_REVISION:0:8}"
ablit_cache_dir() { echo "$HF_CACHE/hub/models--${ABLIT_ID//\//--}"; }
model_sub() {
  if [[ "$ABLIT" == 1 ]]; then echo "$ABLIT_VIEW_SUB"
  else echo "hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"; fi
}

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label rev extra=""
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  rev=$(snapshot_rev)
  [[ "$ABLIT" != 1 ]] || extra=" ablit=$ABLIT_ID@$ABLIT_REVISION"
  echo "model=$MODEL_ID@$rev image=$label patches=$hash$extra"
}

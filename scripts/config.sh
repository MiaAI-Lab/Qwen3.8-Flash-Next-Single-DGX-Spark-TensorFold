# Shared settings for start.sh, stop.sh and scripts/*.sh. Any value can be overridden from the environment,
# e.g. `PORT=9000 ./start.sh` or `PULL=0 scripts/prepare.sh`, or set in ./.env: KEY=value lines, read here (never
# run as a script); a variable already set in the environment wins over the file. .env is yours, not the repository's.
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

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"   # MLX 4-bit, group size 32, with the MTP head
# The patches and start.sh's flags are made for TensorFold v0.6.1 exactly (17c73e1). After changing
# TF_VERSION, TF_REPO or BASE_IMAGE, run `scripts/prepare.sh --rebuild`.
TF_VERSION="${TF_VERSION:-v0.6.1}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# Replies mostly in Chinese or Japanese: DRAFT_LANGUAGE=zh or ja (in .env) serves the second image, which adds that
# language's tokens to the ones MTP drafts may propose (patches/languages/): faster decoding there, the same
# output. English and code get a little slower with it, so leave it unset otherwise. Also accepted, not measured to help:
# de, fr, pt, ru, nl (experimental); several: "nl,de". See docs/nl.md for Dutch validation and benchmarks.
DRAFT_LANGUAGE="${DRAFT_LANGUAGE:-}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_VERSION}${DRAFT_LANGUAGE:+-languages}}"   # the local image prepare.sh builds or pulls
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"          # the server's container
# The prebuilt images: prepare.sh pulls $GHCR_IMAGE:<TF_VERSION>-<patches hash>; publish-image.sh pushes it (and
# :latest, or :languages for the DRAFT_LANGUAGE image).
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"   # the model id clients see in /v1/models and replies (tensorfold --name)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Serving defaults (./start.sh arguments come after them and win). All streams share one memory pool (~103-104 GiB
# budget on a 128 GB Spark, 75 GiB of it weights), so window x streams x KV bytes must fit: 4 streams x 262,144 tokens
# at int8 KV is ~97.8 GiB, 5 streams ~102.6 GiB (~4.5 GiB a stream). Other fits: 3 streams bf16 at 262k, 6 streams
# int4 at 262k, 8 streams int4 at ~250k (tight), 6 streams int8 at ~220k. int4 and bf16 KV change the output slightly.
PARALLEL="${PARALLEL:-5}"          # requests decoded together (streams)
CONTEXT="${CONTEXT:-262144}"       # prompt + reply window per stream (the model's native maximum)
KV_DTYPE="${KV_DTYPE:-int8}"       # bf16 | int8 | int4
PLE_ON_SSD="${PLE_ON_SSD:-1}"      # 1: read the 29.8 GiB n-gram tables from SSD, leaving that RAM to the KV cache
# Image input (TensorFold's Flash Next vision; video and many images from patch 0002): the model's own vision tower,
# 0.84 GiB. Its ~0.8 GiB of scratch is taken only while an image or video encodes and handed back right after, so
# startup reserves none for it. VISION=0: text only.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"    # 1: also accept public https:// image and video URLs (default: data URLs only)
# Images a request may carry (--vision-max-images; a chat's turns all count). They share TENSORFOLD_IMAGE_TOKENS.
# Empty: TensorFold's own limit (4).
VISION_MAX_IMAGES="${VISION_MAX_IMAGES-${TENSORFOLD_MAX_IMAGES:-50}}"
# MTP drafting: at most MTP_DRAFTS drafts a round, a chain stopping before a draft under MTP_CONFIDENCE.
# Swept 2026-09-29 (identical output in every arm): 6/0.60 beat the stock 6/0.30 by ~3% on
# prose and ~4% on code, the best balance of both; 4/0.50, 3/0.30 and 7/0.75 matched it on prose but not on code.
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
# Thinking mode (Qwen's recommended sampling): temperature 1.0, top_p 0.95, top_k 20. A request's own values win.
# min_p 0.0, presence_penalty 0.0 and repetition_penalty 1.0 are what TensorFold always does (it has no such
# settings: those values mean "off"). THINKING=0 serves without a think block by default; a request can still set
# "chat_template_kwargs": {"enable_thinking": true|false}.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
THINKING="${THINKING:-1}"
# Reply length for a request that sets no max_tokens (or max_completion_tokens). TensorFold's own default, 4,096,
# can end a thinking reply before it answers (finish_reason "length", no content or tool call). The value is clamped
# to the room left in the stream's window and reserves no memory; a request's own max_tokens wins.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# TensorFold switches (start.sh passes every TENSORFOLD_* variable into the container).
# Prompt piece rows. Unset, TensorFold v0.6.1 picks 2,048 with vision and 4,096 without (while nothing decodes,
# if memory allows). With the n-gram tables on SSD (PLE_ON_SSD=1) 4,096 measured 10-30% slower from 5k to 16k
# tokens and ~3% slower at 31k (2026-10-02), so this recipe keeps 2,048 there. TENSORFOLD_PREFILL_ROWS=N (256 to
# 16,384, patch 0002) forces N-row pieces, admitted with the window; empty: TensorFold's choice.
if [[ "$PLE_ON_SSD" == 1 ]]; then export TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS-2048}"; fi
# What startup reserves for the vision tower's scratch (MiB); 0: it comes from the system reserve while it encodes.
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"
# The tokens a request's images share (VISION_MAX_IMAGES above), each image at most 4,096 (one image is sized as
# before). The tower encodes them 16,384 patches at a time, the scratch one image needs.
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
# The whole video's token budget (Qwen3-VL's per-frame sizing; 2 frames a second, at most 256 frames).
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
# Startup reserve (since v0.6.0): GiB left out of MemAvailable. Unset, TensorFold takes max(4 GiB, a tenth of RAM)
# and refuses 5 x 262,144. The knob's floor is 2 GiB; that still fits the measured ~102.5 GiB admission.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
# Prompt-lookup drafts ahead of MTP (patch 0007; with PARALLEL >= 2): +6% on replies that repeat the prompt, prose and
# code unchanged. 0: off.
export TENSORFOLD_MTP_COPY="${TENSORFOLD_MTP_COPY:-1}"
# The draft list the language image serves (DRAFT_LANGUAGE above).
[[ -z "$DRAFT_LANGUAGE" ]] || export TENSORFOLD_DRAFT_VOCAB="$DRAFT_LANGUAGE"
# No "is there a newer TensorFold" call to GitHub at each start: the patches are for v0.6.1 anyway. 0: check.
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Persists compiled CUDA kernels (torch extensions + triton) so only the first start pays the compile.
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"

MIN_FREE_GB="${MIN_FREE_GB:-125}"   # free disk the checkpoint download needs (it is ~114 GB)
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"   # free disk under Docker's root that pulling or building the image needs

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }

# start.sh and scripts/*.sh (not stop.sh, which must stop the server whatever the settings) check DRAFT_LANGUAGE.
check_draft_language() {
  local one='(de|fr|ja|nl|pt|ru|zh)'
  [[ -z "$DRAFT_LANGUAGE" || "$DRAFT_LANGUAGE" =~ ^$one(,$one)*$ ]] || \
    die "DRAFT_LANGUAGE=$DRAFT_LANGUAGE: zh or ja (recommended), de, fr, nl (experimental), pt or ru, or several like nl,de"
}
# The patches baked into $IMAGE, in order: patches/*.patch, plus patches/languages/*.patch for DRAFT_LANGUAGE.
patch_files() { ls patches/*.patch; [[ -z "$DRAFT_LANGUAGE" ]] || ls patches/languages/*.patch; }
patches_hash() { patch_files 2>/dev/null | xargs -r cat | sha256sum | cut -c1-12; }

# What scripts/prepare.sh last left ready (it writes this line to PREPARED_MARKER when it succeeds); start.sh runs
# prepare.sh again whenever the current line differs: a missing or stale image, new patches, another model.
PREPARED_MARKER="$KERNEL_CACHE/.prepared"
prepared_state() {
  local hash label model=missing
  hash=$(patches_hash)
  label=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || echo missing)
  ls -d "$(model_cache_dir)"/snapshots/*/ >/dev/null 2>&1 && model=present
  echo "model=$MODEL_ID($model) image=$IMAGE($label) patches=$hash"
}

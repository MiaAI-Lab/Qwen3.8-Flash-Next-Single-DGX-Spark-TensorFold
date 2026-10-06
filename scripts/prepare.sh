#!/usr/bin/env bash
# Prepare everything needed to serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 with TensorFold on one DGX Spark:
#   1. preflight checks (docker, GPU runtime, disk space)
#   2. the image: TensorFold plus patches/*.patch (and patches/languages/*.patch with DRAFT_LANGUAGE) on NVIDIA's
#      PyTorch container, pulled prebuilt from $GHCR_IMAGE when a matching tag is reachable (PULL=0 skips that), else
#      built locally
#   3. download the checkpoint into the Hugging Face cache (~106 GB, resumable)
#   4. verify the checkpoint with `tensorfold info`
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh
check_draft_language

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$KERNEL_CACHE"
exec 9>"$KERNEL_CACHE/.prepare.lock"
flock -n 9 || die "another prepare.sh is already running (it holds the download locks); wait for it or stop it: pgrep -af prepare.sh"
log "Preflight checks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found on the host"
elif ! nvidia-smi -L; then warn "nvidia-smi failed: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime; --gpus all may fail"

mkdir -p "$HF_CACHE/hub" "$KERNEL_CACHE/torch_extensions" "$KERNEL_CACHE/triton"

mkdir -p patches
PATCHES_HASH=$(patches_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)

# Disk: the checkpoint (~106 GB) if it is not downloaded yet, and the image (~24 GB, more while it unpacks) if it is
# not built from these patches yet; both on one filesystem when Docker's root shares it with the HF cache.
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
fs_of()   { df --output=target "$1" 2>/dev/null | tail -1; }
need_model=0; need_image=0
[[ -d "$(model_cache_dir)/snapshots" ]] || need_model=$MIN_FREE_GB
[[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_image=$IMAGE_FREE_GB
docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)
if [[ -z "$docker_root" || "$(fs_of "$docker_root")" == "$(fs_of "$HF_CACHE")" ]]; then
  need=$((need_model + need_image)); have=$(free_gb "$HF_CACHE")
  (( need == 0 || have >= need )) || die "only ${have} GB free under $HF_CACHE, need ~${need} GB (checkpoint ${need_model} + image ${need_image})"
else
  have=$(free_gb "$HF_CACHE")
  (( need_model == 0 || have >= need_model )) || die "only ${have} GB free under $HF_CACHE, the checkpoint needs ~${need_model} GB"
  have_img=$(free_gb "$docker_root")
  (( need_image == 0 || ${have_img:-0} >= need_image )) || die "only ${have_img} GB free under $docker_root, the image needs ~${need_image} GB"
fi
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE"

# ---------------------------------------------------------------- 2. image
# Local fixes in ./patches (unified diffs against site-packages, applied with patch -p0) are baked into the image.
# The image is rebuilt when they change; the TensorFold install layer stays cached, so that takes seconds.
prebuilt="$GHCR_IMAGE:${TF_VERSION}-${PATCHES_HASH}"
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && "$built_hash" != "$PATCHES_HASH" ]]; then
  log "Pulling the prebuilt image $prebuilt (~11 GB; PULL=0 builds instead)"
  if docker pull "$prebuilt" && \
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$PATCHES_HASH" ]]; then
    docker tag "$prebuilt" "$IMAGE"
    built_hash=$PATCHES_HASH
    log "Using $prebuilt as $IMAGE"
  else
    warn "could not pull $prebuilt (no image for these patches, the package is not public, or no network): building it locally"
  fi
fi
if [[ $REBUILD -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1 || [[ "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }

  log "Building $IMAGE (TensorFold $TF_VERSION, patches $PATCHES_HASH: $(patch_files 2>/dev/null | xargs -rn1 basename | paste -sd' ' || true))"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  # the build context: this image's patches, side by side in the order they apply
  context=$(mktemp -d)
  trap 'rm -rf -- "$context"' EXIT
  patch_files | xargs -r cp -t "$context"
  docker build "${nocache[@]}" -t "$IMAGE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_SPEC="git+${TF_REPO}@${TF_VERSION}" \
    --build-arg PATCHES_HASH="$PATCHES_HASH" \
    -f - "$context" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG TF_SPEC
# transformers (the vision tower's modules) and PyAV (video decoding) for --vision
RUN pip install --no-cache-dir --upgrade "${TF_SPEC}" && pip install --no-cache-dir "transformers==5.17.0" av && \
    tensorfold --version
COPY . /opt/tf-patches
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.server, tensorfold.vision.qwen_cuda, tensorfold.vision.videos, av; \
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel"
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH}
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
DOCKERFILE
else
  log "Image $IMAGE already built with patches $PATCHES_HASH (use --rebuild to force)"
fi
docker run --rm --entrypoint tensorfold "$IMAGE" --version 2>/dev/null | tail -1

# Run tensorfold inside the image with the HF cache and kernel cache mounted (no GPU, and without NVIDIA's entrypoint
# banner: these steps only read files, and the GPU memory may belong to a running server).
tf_run() {
  # a token only by name, and only when set (never on the command line); else huggingface_hub finds the token file
  # in the mounted cache. The model is public, so none is needed.
  local token=(); [[ -n "${HF_TOKEN:-}" ]] && token=(-e HF_TOKEN)
  docker run --rm --ipc=host --network host --entrypoint tensorfold "${token[@]}" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$KERNEL_CACHE":/cache \
    "$IMAGE" "$@"
}

# ---------------------------------------------------------------- 3. download
log "Downloading $MODEL_ID into $HF_CACHE/hub (resumes if interrupted)"
if command -v hf >/dev/null; then
  # Host CLI: resumable, parallel, writes the standard HF cache layout.
  hf download "$MODEL_ID" --cache-dir "$HF_CACHE/hub"
else
  warn "host 'hf' CLI not found, downloading from inside the container"
fi
# `tensorfold pull` is the documented path; with the files already cached it only checks/completes them.
tf_run pull "$MODEL_ID"

snapshot=$(ls -d "$(model_cache_dir)"/snapshots/*/ 2>/dev/null | head -1)
[[ -n "$snapshot" ]] || die "no snapshot found under $(model_cache_dir)"
log "Checkpoint: $snapshot ($(du -shL "$snapshot" | cut -f1))"

# ---------------------------------------------------------------- 4. verify
log "Verifying checkpoint with tensorfold info"
tf_run info "$MODEL_ID"

prepared_state > "$PREPARED_MARKER"
log "Done. Start the server with ./start.sh (port $PORT)."
log "The first start compiles CUDA kernels for GB10 (a few minutes); they are cached in $KERNEL_CACHE."

#!/usr/bin/env bash
# Prepare this Spark to serve Qwen3.8-Flash-Next ($QUANT_LABEL; see CHECKPOINT in scripts/config.sh) with TensorFold's Zig engine (tensorfold-native, one GPU):
#   1. preflight: docker, the GPU, disk
#   2. the image: tensorfold-native built from TensorFold (TF_REF) plus patches/*.patch with Zig, on NVIDIA's
#      PyTorch container (kernels are embedded in the binary), with libjpeg-turbo and libwebp's decoder; pulled prebuilt when a matching tag is reachable (PULL=0 skips
#      that), else built locally
#   3. the checkpoint in the Hugging Face cache (~96 GiB for the NVIDIA FP4 default, ~122 GiB for CHECKPOINT=int4), at its pinned revision
#   4. verify it: every shard its index names is there, and it is a checkpoint format this engine serves (ModelOpt NVFP4 or GPTQ int4)
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it"

log "Preflight checks"
_mem_here=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
if (( _mem_here < MEM_FLOOR_GIB )); then
  _msg="only ${_mem_here} GiB MemAvailable (the floor is MEM_FLOOR_GIB=$MEM_FLOOR_GIB): stop other work first"
  if [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$_msg"; else die "$_msg (MEM_CHECK=0 continues anyway)"; fi
fi
log "Memory: ${_mem_here} GiB MemAvailable (floor $MEM_FLOOR_GIB)"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime; --gpus all may fail"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

HUB_MISSING_PY='
import math, os, sys
from huggingface_hub import HfApi
hub, repo, rev = sys.argv[1], sys.argv[2], sys.argv[3]
blobs = os.path.join(hub, "models--" + repo.replace("/", "--"), "blobs")
missing = 0
for f in HfApi().model_info(repo, revision=rev or None, files_metadata=True).siblings:
    lfs = f.lfs
    name = (getattr(lfs, "sha256", None) or lfs["sha256"]) if lfs else f.blob_id
    size = (getattr(lfs, "size", None) or lfs["size"]) if lfs else (f.size or 0)
    p = os.path.join(blobs, name)
    missing += 0 if os.path.isfile(p) and os.path.getsize(p) == size else size
print(math.ceil(missing / 2**30))'
hub_missing_gb() {
  local py out tried=""
  for py in "$(command -v python3)" "$(sed -n '1s/^#! *\(\/[^ ]*python[0-9.]*\)$/\1/p' "$(command -v hf || echo /dev/null)" 2>/dev/null)"; do
    [[ -n "$py" && -x "$py" && "$py" != "${tried:-}" ]] || continue
    tried=$py
    out=$(timeout 30 "$py" -c "$HUB_MISSING_PY" "$HF_CACHE/hub" "$MODEL_ID" "$MODEL_REVISION" 2>/dev/null) && [[ "$out" =~ ^[0-9]+$ ]] &&
      { echo "$out"; return 0; }
  done
  return 1
}

DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
rev=$(snapshot_rev)
if missing=$(hub_missing_gb); then
  need_ckpt=0; (( missing == 0 )) || need_ckpt=$((missing + 5))
  ckpt_what="the download needs ~${missing} GB that the cache does not hold yet"
else
  need_ckpt=0
  [[ -f "$(model_cache_dir)/snapshots/$rev/config.json" ]] || need_ckpt=$MIN_FREE_GB
  ckpt_what="the checkpoint needs MIN_FREE_GB (the Hub's file list could not be read)"
fi
(( need_ckpt )) || ckpt_what="nothing to download"
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
if [[ "$(stat -c %d "$HF_CACHE")" == "$(stat -c %d "$DOCKER_ROOT" 2>/dev/null)" ]]; then
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt + need_img )) ||
    die "only ${have} GB free under $HF_CACHE (also Docker's root), ~$((need_ckpt + need_img)) GB needed: $ckpt_what; the image needs ${need_img} GB"
else
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt )) || die "only ${have} GB free under $HF_CACHE, ~${need_ckpt} GB needed: $ckpt_what"
  have=$(free_gb "$DOCKER_ROOT"); (( have >= need_img )) ||
    die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB"
fi
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE ($ckpt_what)"

prebuilt=$(prebuilt_image)
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && "$built_hash" != "$PATCHES_HASH" ]]; then
  log "Pulling the prebuilt image $prebuilt (PULL=0 builds instead)"
  if docker pull "$prebuilt" &&
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$PATCHES_HASH" ]]; then
    docker tag "$prebuilt" "$IMAGE"; built_hash=$PATCHES_HASH
    log "Using $prebuilt as $IMAGE"
  else
    warn "could not pull $prebuilt: building it locally"
  fi
fi
if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }
  log "Building $IMAGE (TensorFold ${TF_REF:0:12}, Zig $ZIG_VERSION, patches $PATCHES_HASH, TP=1 kernel set)"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches"
  compgen -G 'patches/*.patch' >/dev/null && cp -- patches/*.patch "$ctx/patches/"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build ${DOCKER_BUILD_ARGS:-} "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg LIBJPEG_TURBO_VERSION="$LIBJPEG_TURBO_VERSION" --build-arg LIBJPEG_TURBO_SHA256="$LIBJPEG_TURBO_SHA256" \
    --build-arg LIBWEBP_VERSION="$LIBWEBP_VERSION" --build-arg LIBWEBP_SHA256="$LIBWEBP_SHA256" \
    --build-arg TF_RELEASE_VERSION="$TF_RELEASE_VERSION" --build-arg TF_RELEASE_AARCH64_SHA256="$TF_RELEASE_AARCH64_SHA256" \
    --build-arg PATCHES_HASH="$PATCHES_HASH" \
    -f - "$ctx" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS codecs
ARG LIBJPEG_TURBO_VERSION
ARG LIBJPEG_TURBO_SHA256
ARG LIBWEBP_VERSION
ARG LIBWEBP_SHA256
RUN mkdir -p /src /opt/codecs && cd /src && \
    curl -fsSL -o jt.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" && \
    echo "${LIBJPEG_TURBO_SHA256}  jt.tar.gz" | sha256sum -c - && \
    tar -xzf jt.tar.gz && cmake -S libjpeg-turbo-${LIBJPEG_TURBO_VERSION} -B jt-build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DCMAKE_INSTALL_PREFIX=/opt/codecs/jpeg -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build jt-build -j"$(nproc)" && cmake --install jt-build && \
    test -e /opt/codecs/jpeg/lib/libturbojpeg.so.0
RUN cd /src && \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VERSION}.tar.gz" && \
    echo "${LIBWEBP_SHA256}  webp.tar.gz" | sha256sum -c - && \
    tar -xzf webp.tar.gz && cd libwebp-${LIBWEBP_VERSION} && \
    ./configure --prefix=/opt/codecs/webp --enable-libwebpdecoder --disable-static --disable-gl --disable-sdl --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic && \
    make -j"$(nproc)" && make install && \
    test -e /opt/codecs/webp/lib/libwebpdecoder.so.3

FROM ${BASE_IMAGE} AS grammar
ARG TF_RELEASE_VERSION
ARG TF_RELEASE_AARCH64_SHA256
RUN mkdir -p /opt/tfrel && cd /tmp && \
    curl -fsSL -o tf-rel.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TF_RELEASE_VERSION}/tensorfold-${TF_RELEASE_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_RELEASE_AARCH64_SHA256}  tf-rel.tar.gz" | sha256sum -c - && \
    tar -xzf tf-rel.tar.gz -C /opt/tfrel --strip-components=1 \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/lib/libtfgrammar.so \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-Apache-2.0.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-NOTICE.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/dlpack-Apache-2.0.txt && \
    test -s /opt/tfrel/lib/libtfgrammar.so

FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# fatbins + native only: the default install also builds tf-cuda-test, which does not compile on 1.1.4 (contract_controls.cu, w4a8_tests.zig)
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/jpeg/lib/libturbojpeg.so* /opt/tensorfold/lib/
COPY --from=codecs /opt/codecs/webp/lib/libwebpdecoder.so* /opt/tensorfold/lib/
# the engine looks for libtfgrammar.so in <binary dir>/../lib, i.e. next to native/bin
COPY --from=grammar /opt/tfrel/lib/libtfgrammar.so /opt/tensorfold/native/lib/libtfgrammar.so
COPY --from=grammar /opt/tfrel/LICENSES /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.kernels=embedded
ENV HF_HOME=/root/.cache/huggingface CUDA_CACHE_PATH=/cache/nv \
    TENSORFOLD_LIBTURBOJPEG=/opt/tensorfold/lib/libturbojpeg.so.0 TENSORFOLD_LIBWEBP=/opt/tensorfold/lib/libwebpdecoder.so.3
WORKDIR /workspace
DOCKERFILE
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built with patches $PATCHES_HASH"
fi
docker run --rm --network none --entrypoint test "$IMAGE" -x /opt/tensorfold/native/bin/tensorfold-native ||
  die "$IMAGE has no tensorfold-native"
log "Image $IMAGE: tensorfold-native, kernel set present"

command -v hf >/dev/null || warn "host 'hf' CLI not found, downloading from inside the container"
download() {
  if command -v hf >/dev/null; then
    hf download "$1" ${2:+--revision "$2"} --cache-dir "$HF_CACHE/hub" >/dev/null
  else
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)' "$1" "$2"
  fi
}
dir=$(model_cache_dir)
pin=$MODEL_REVISION
had=0; [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] && had=1
log "Downloading $MODEL_ID${pin:+ @ ${pin:0:8}} into $HF_CACHE/hub"
if ! download "$MODEL_ID" "$pin"; then
  [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] || die "$MODEL_ID: the download failed"
  warn "$MODEL_ID: could not reach Hugging Face; using the snapshot already here (${pin:0:8})"
fi
(( had )) || [[ -z "$pin" || -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$pin" > "$dir/refs/main"; }
rev=$(snapshot_rev)
[[ -n "$rev" && -d "$dir/snapshots/$rev" ]] || die "$MODEL_ID: no snapshot after the download"
log "Checkpoint: $dir/snapshots/$rev ($(du -shL "$dir/snapshots/$rev" | cut -f1))"

log "Verifying the checkpoint"
CHECK_PY='
import glob, json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "qwen4_exp":
    sys.exit("model_type is %r, not qwen4_exp" % cfg.get("model_type"))
q = cfg.get("quantization_config") or {}
kind = str(q.get("quant_method", "")).lower()
if kind == "modelopt":
    algos = {str(v.get("quant_algo")) for v in (q.get("quantized_layers") or {}).values()}
    if "NVFP4" not in algos:
        sys.exit("ModelOpt checkpoint without NVFP4 layers (quant_algo %s)" % sorted(algos))
    what = "ModelOpt %s (%s)" % (q.get("quant_algo"), "+".join(sorted(algos)))
elif kind == "gptq" and q.get("bits") == 4:
    table = [f for f in glob.glob(os.path.join(d, "ple-table", "*.safetensors")) if os.path.getsize(f) > 0]
    if not table:
        sys.exit("no n-gram table files in ple-table/")
    what = "AutoRound int4 g%s, %d n-gram table files" % (q.get("group_size"), len(table))
else:
    sys.exit("neither a ModelOpt NVFP4 nor a GPTQ-format int4 checkpoint (quant_method %r, bits %r)" % (kind, q.get("bits")))
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
if bad:
    sys.exit("missing or empty shards: " + ", ".join(bad[:5]))
print("qwen4_exp, %s, %d tensors in %d shards" % (what, len(idx["weight_map"]), len(shards)))'
snap="$dir/snapshots/$rev"
info=$(python3 -I -c "$CHECK_PY" "$snap" 2>&1) || die "the checkpoint at $snap is not ready: $info"
log "Checkpoint OK: $info"
prepared_state > "$PREPARED_MARKER"
log "Done. Start the server with ./start.sh (port $PORT)."

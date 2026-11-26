#!/usr/bin/env bash
# Cut the unused ballast out of the image scripts/prepare.sh built, as a separate opt-in layer, and tag the result
# beside it: a vulnerability scanner mostly reports the NGC base image (Nsight, the kernel headers, Debian's pip and
# setuptools with their vendored urllib3 and msgpack), none of which this recipe's server uses.
#
# Nothing here is required to serve, and it changes nothing about how the server runs: it is a layer you can take or
# leave, so the image start.sh uses (`$IMAGE`) is never touched — you point IMAGE at the hardened tag to get it.
#
# Usage: scripts/harden.sh                      # the default cuts (verified unused on this image), no version changes
#        scripts/harden.sh && IMAGE=... ./start.sh restart   # serve the hardened image (see the README)
#        APT_UPGRADE=1 scripts/harden.sh        # plus apt-get upgrade: distro CVE fixes, at the cost of a reproducible build
#        PIP_UPGRADE="urllib3==2.8.0 pillow==13.0.0" scripts/harden.sh   # plus the pip versions a scanner asked for
#        PIP_UNINSTALL="wandb jupyterlab notebook nbconvert jupyter_server flash_attn" scripts/harden.sh
#          (none of those is in the pinned base today; for a base that still carries them)
#
# The script does not call a scanner — nothing is installed here to run one — so it promises no CVE count. What it
# promises is a gate: the new image is kept only if the server binary still runs, the decoders are still where the
# engine looks, and the Python the in-container checkpoint download needs still imports.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
[[ "$(docker image inspect -f '{{index .Config.Labels "tf.kernels"}}' "$IMAGE")" == embedded ]] ||
  die "$IMAGE has no kernel set: it cannot serve, so it is not worth hardening"

# What goes. Every name here was checked against the images TensorFold v1.2.7, v1.2.10, v1.3.0, v1.3.2, v1.3.4 and v1.3.5 build: they are there, and nothing in
# /opt/tensorfold, the engine's dlopen list or the container-side `hf download` path reads them.
APT_PURGE="${APT_PURGE:-linux-libc-dev nsight-compute-* nsight-systems-cli-* python3-pip python3-setuptools python3-pip-whl python3-setuptools-whl}"
APT_UPGRADE="${APT_UPGRADE:-0}"           # 1: apt-get upgrade (unspecified updates: fewer CVEs, less reproducible image)
PIP_UPGRADE="${PIP_UPGRADE:-}"            # specs for pip install --upgrade, e.g. "urllib3==2.8.0"; empty installs nothing
PIP_UNINSTALL="${PIP_UNINSTALL:-}"        # names for pip uninstall; empty removes nothing
NSIGHT_GLOBS="${NSIGHT_GLOBS:-/usr/local/cuda-*/NsightSystems-cli-* /usr/local/cuda-*/nsight-*}"
OUT="${HARDEN_IMAGE:-${IMAGE}-hardened-$(printf '%s|%s|%s|%s' "$APT_PURGE" "$APT_UPGRADE" "$PIP_UPGRADE" "$PIP_UNINSTALL" | sha256sum | cut -c1-12)}"
HARDEN_HASH=$(sha256sum "$(readlink -f "$0")" | cut -c1-12)

# The gate. `tensorfold-native --help` proves the binary still loads its libraries; the imports are what
# scripts/prepare.sh needs when it downloads a checkpoint from inside the container (no `hf` on the host);
# autoremove is what historically took python3-yaml away, and yaml is what the download path imports.
CHECKS='
set -e
test -x /opt/tensorfold/native/bin/tensorfold-native
tensorfold-native --help >/dev/null 2>&1
tensorfold-native serve --help >/dev/null 2>&1
for lib in /opt/tensorfold/lib/libturbojpeg.so.0 /opt/tensorfold/lib/libwebpdecoder.so.3 /opt/tensorfold/native/lib/libtfgrammar.so; do
  test -f "$lib"; done
# the decoders behind the two opt-ins are only in an image built after they landed; say so rather than fail for them
for opt in /opt/tensorfold/lib/libpdfium.so /opt/tensorfold/libexec/tfvideo/libtfvideo.so; do
  [ -f "$opt" ] || echo "  note: $opt is not in this image, so PDF=1/VIDEO=1 need scripts/prepare.sh first"; done
python3 -c "import yaml, requests, filelock, fsspec, packaging, huggingface_hub"
command -v hf >/dev/null
'

log "Hardening $IMAGE as $OUT"
log "  apt purge: $APT_PURGE (upgrade: $APT_UPGRADE)"
log "  pip install: ${PIP_UPGRADE:-nothing}    pip uninstall: ${PIP_UNINSTALL:-nothing}"
docker build -q -t "$OUT" \
  --build-arg BASE_IMAGE="$IMAGE" \
  --build-arg APT_PURGE="$APT_PURGE" --build-arg APT_UPGRADE="$APT_UPGRADE" \
  --build-arg PIP_UPGRADE="$PIP_UPGRADE" --build-arg PIP_UNINSTALL="$PIP_UNINSTALL" --build-arg NSIGHT_GLOBS="$NSIGHT_GLOBS" \
  --label tf.hardened="$HARDEN_HASH" \
  --label tf.base="$IMAGE" --label tf.kernels=embedded \
  -f - . <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG APT_PURGE
ARG APT_UPGRADE
ARG PIP_UPGRADE
ARG PIP_UNINSTALL
ARG NSIGHT_GLOBS
# the profilers also ship outside dpkg, under the CUDA install; unquoted on purpose, these are globs
# shellcheck disable=SC2086
RUN set -eux; \
    apt-get update; \
    if [ "$APT_UPGRADE" = 1 ]; then DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::=--force-confold upgrade; fi; \
    apt-get -y purge $APT_PURGE || true; \
    DEBIAN_FRONTEND=noninteractive apt-get -y autoremove --purge; \
    rm -rf $NSIGHT_GLOBS; \
    if [ -n "$PIP_UNINSTALL" ]; then python3 -m pip uninstall -y $PIP_UNINSTALL || true; fi; \
    if [ -n "$PIP_UPGRADE" ]; then python3 -m pip install --no-cache-dir --upgrade $PIP_UPGRADE; fi; \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --reinstall python3-yaml; \
    rm -rf /var/lib/apt/lists/*
DOCKERFILE

log "Checking the hardened image still boots what this recipe runs"
if ! docker run --rm --network none --entrypoint bash "$OUT" -c "$CHECKS"; then
  docker run --rm --network none --entrypoint bash "$OUT" -x -c "$CHECKS" 2>&1 | tail -6 || true
  docker rmi -f "$OUT" >/dev/null 2>&1 || true
  die "$OUT failed the boot check and was removed; $IMAGE is untouched, so ./start.sh still serves"
fi
log "$OUT passed the boot check"
echo "  serve it with:  IMAGE=$OUT ./start.sh restart"
echo "  or in ./.env:   IMAGE=$OUT"

#!/usr/bin/env bash
# Push the image scripts/prepare.sh built to GitHub Container Registry as $GHCR_IMAGE:<TensorFold version>-<patches hash>
# and :latest (with DRAFT_LANGUAGE set: the language image, as :languages), labelled with this repository (so the
# package takes the repository's visibility and access).
# Needs a GitHub token with write:packages: `gh auth login -s write:packages`, or GHCR_TOKEN / GITHUB_TOKEN set.
# Usage: scripts/publish-image.sh                    # the default image
#        DRAFT_LANGUAGE=zh scripts/publish-image.sh  # the language image (any DRAFT_LANGUAGE selects it)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REPO_URL="${REPO_URL:-https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
[[ "$hash" == "$(image_hash)" ]] || die "$IMAGE was built from other patches; run scripts/prepare.sh first"
[[ "$(docker image inspect -f '{{index .Config.Labels "tf.kernels"}}' "$IMAGE")" == present ]] ||
  die "$IMAGE has no kernel set: it cannot serve, so it is not published"
tag="${TF_TAG}-${hash}"
alias=latest

token="${GHCR_TOKEN:-${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
[[ -n "$token" ]] || die "no GitHub token: run gh auth login -s write:packages, or set GHCR_TOKEN"
user="${GHCR_USER:-$(gh api user --jq .login 2>/dev/null || echo "${GHCR_IMAGE#ghcr.io/}" | cut -d/ -f1)}"

log "Labelling $IMAGE as $GHCR_IMAGE:$tag"
docker build -q -t "$GHCR_IMAGE:$tag" -t "$GHCR_IMAGE:$alias" \
  --label org.opencontainers.image.source="$REPO_URL" \
  --label org.opencontainers.image.description="Qwen3.8 Flash Next INT4-AutoRound on one DGX Spark: TensorFold's Zig engine (${TF_REF:0:12}) with patches $hash" \
  - <<<"FROM $IMAGE" >/dev/null
log "Pushing $GHCR_IMAGE:$tag and :$alias (~24 GB uncompressed; only changed layers upload)"
# log in with a throwaway Docker config, so no registry credential stays in ~/.docker after the push
login_dir=$(mktemp -d)
trap 'rm -rf -- "$login_dir"' EXIT
echo "$token" | DOCKER_CONFIG="$login_dir" docker login ghcr.io -u "$user" --password-stdin >/dev/null
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$tag"
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$alias"
log "Done: docker pull $GHCR_IMAGE:$tag"

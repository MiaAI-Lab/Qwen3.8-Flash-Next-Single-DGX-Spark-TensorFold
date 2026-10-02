#!/usr/bin/env bash
# Serve Qwen3.8 Flash Next (Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) with TensorFold on one DGX Spark, end to end:
# runs scripts/prepare.sh when the image or the checkpoint is not ready yet (first run, or after patches change),
# launches `tensorfold serve` on port 8888, waits until the OpenAI API answers, then runs a smoke test.
# Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold serve args]
#   ./start.sh                         # scripts/config.sh defaults: 5 streams x 262,144 tokens, int8 KV, --ple-on-ssd,
#                                      # image and video input (--vision)
#                                      # (if the server already runs, says so and leaves it alone)
#   ./start.sh restart                 # stop the running server (./stop.sh), then start it again, e.g. to apply
#                                      # changed settings or patches; the new arguments are checked before stopping
#   ./start.sh restart --parallel 8 --context 172000
#   PARALLEL=3 KV_DTYPE=bf16 ./start.sh restart   # 256k at full KV precision
#   VISION=0 ./start.sh restart        # text only
#   echo DRAFT_LANGUAGE=zh >> .env; ./start.sh restart   # replies mostly in Chinese (or ja): the language image
# Extra arguments come after the defaults, so they win (the last value of a flag counts).
# Settings, from the environment or ./.env (KEY=value lines): PARALLEL, CONTEXT, KV_DTYPE, DRAFT_LANGUAGE, PLE_ON_SSD,
#      VISION, VISION_URLS, VISION_MAX_IMAGES, MTP_DRAFTS, MTP_CONFIDENCE, TEMPERATURE, TOP_P, TOP_K, THINKING,
#      MAX_TOKENS, SERVED_NAME, PORT, HOST, CONTAINER_NAME, IMAGE (see scripts/config.sh); TENSORFOLD_* (passed to the server);
#      PREPARE (auto | 1 | 0); FOREGROUND=1 (stay attached, exit with the server's code); WAIT_TIMEOUT (seconds,
#      default 1800); HF_HUB_OFFLINE=0 (let TensorFold reach the Hub; default serves from the local cache only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
check_draft_language

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

# The serve arguments: scripts/config.sh's defaults first, then the command line's (argparse keeps the last value).
SERVE_ARGS=(--name "$SERVED_NAME" --parallel "$PARALLEL" --context "$CONTEXT" --kv-dtype "$KV_DTYPE"
            --mtp-drafts "$MTP_DRAFTS" --mtp-confidence "$MTP_CONFIDENCE"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --max-tokens "$MAX_TOKENS")
[[ "$PLE_ON_SSD" == 1 ]] && SERVE_ARGS+=(--ple-on-ssd)
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
[[ "$VISION" == 1 && "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
[[ "$VISION" == 1 && -n "$VISION_MAX_IMAGES" ]] && SERVE_ARGS+=(--vision-max-images "$VISION_MAX_IMAGES")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
SERVE_ARGS+=("$@")
# The effective value of a flag (its last occurrence, as --flag value or --flag=value).
arg_value() {
  local flag=$1 value="" i
  for (( i = 0; i < ${#SERVE_ARGS[@]}; i++ )); do
    case "${SERVE_ARGS[i]}" in
      "$flag") value="${SERVE_ARGS[i + 1]:-}" ;;
      "$flag="*) value="${SERVE_ARGS[i]#*=}" ;;
    esac
  done
  echo "$value"
}
# Where to reach the server from this machine: a wildcard bind answers on loopback.
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

# ---------------------------------------------------------------- banner and progress
B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner                                             # the TensorFold ribbon and MIA AI LAB (terminals only)
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · %s x %s tokens · %s KV · port %s%s\n\n' "$D" "$MODEL_ID" "$(arg_value --parallel)" \
  "$(arg_value --context)" "$(arg_value --kv-dtype)" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE"
exec 8>"$KERNEL_CACHE/.start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"

# ---------------------------------------------------------------- already running?
if [[ "$MODE" == start ]] && running; then
  log "$CONTAINER_NAME is already running (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it (e.g. with new settings), or ./stop.sh to stop it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
# scripts/prepare.sh (image, checkpoint download and check) runs whenever what it last prepared differs from now:
# the first run, new patches, another model or image. PREPARE=1 forces it, PREPARE=0 skips it.
step 1 "Setup: image and checkpoint"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  log "Not ready yet: running scripts/prepare.sh (the first time this pulls the image and downloads ~106 GiB)"
  ./scripts/prepare.sh
else
  log "Ready: $IMAGE and $MODEL_ID${PREPARE:+ (PREPARE=$PREPARE)}"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: $why build it"
ls -d "$(model_cache_dir)"/snapshots/*/ >/dev/null 2>&1 || die "$MODEL_ID not in $HF_CACHE: $why download it"

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: arguments, previous server, port, memory"
# tensorfold's own parser, in a throwaway container without the GPU: a typo fails here, before anything is stopped
docker run --rm --entrypoint python "$IMAGE" -c \
  'import sys; from tensorfold.cli import build_parser; build_parser().parse_args(sys.argv[1:])' \
  serve "$MODEL_ID" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" >/dev/null ||
  die "tensorfold serve rejects these arguments (see above); nothing was changed"
if [[ "$MODE" == restart ]] && running; then          # after the setup and the checks: down only while restarting
  ./stop.sh
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Removing the previous (stopped) container $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
# TensorFold budgets MemAvailable minus TENSORFOLD_MEMORY_RESERVE_GIB (2 here). The default 5 x 262k int8 needs
# ~85.4 GiB at startup and ~108 GiB with every stream at its full window; with less it refuses the window and names
# one that fits.
avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
if (( avail_gb >= 103 )); then
  log "Arguments OK, port $PORT free, ${avail_gb} GiB memory available"
else
  warn "only ${avail_gb} GiB memory available (the default needs ~103): stop other GPU workloads (docker ps), or lower PARALLEL / CONTEXT"
fi

# TensorFold's own switches from the environment (TENSORFOLD_*, e.g. TENSORFOLD_MTP_COPY) reach the server too.
ENV_ARGS=()
while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name"); done < <(env | grep -E '^TENSORFOLD_[A-Z0-9_]+=' || true)

# ---------------------------------------------------------------- 3. launch
step 3 "Launch: container $CONTAINER_NAME"
log "tensorfold serve $MODEL_ID --host $HOST --port $PORT ${SERVE_ARGS[*]}"
# No token goes into the container: serving reads only the local cache (HF_HUB_OFFLINE=1), and with
# HF_HUB_OFFLINE=0 huggingface_hub finds the token file in the mounted cache.
docker run -d --name "$CONTAINER_NAME" \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "${ENV_ARGS[@]}" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$KERNEL_CACHE":/cache \
  "$IMAGE" \
  tensorfold serve "$MODEL_ID" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" >/dev/null

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap './stop.sh; exit 130' INT TERM
  docker logs -f "$CONTAINER_NAME" || true
  exit "$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)"
fi

# ---------------------------------------------------------------- 4. load, with the server's log and a heartbeat
step 4 "Loading: ~75 GiB of weights (~2.5 min; the very first start also compiles CUDA kernels)"
# NVIDIA's container banner, without its license notice (GOVERNING TERMS ...), which stays visible
NOISE='^\s*$|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|torch/utils/_pytree\.py.*register_constant'
# docker logs is the background job, so killing it ends the whole pipeline (no orphaned `docker logs -f`)
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT

# GPU memory the container's processes hold so far, against the server's own startup estimate (GiB)
loaded_gib() {
  local pids
  pids=$(docker top "$CONTAINER_NAME" -eo pid 2>/dev/null | tail -n +2 | paste -sd'|')
  [[ -n "$pids" ]] || { echo 0; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v re="^($pids)$" '$1 ~ re { s += $2 } END { printf "%.1f", s / 1024 }'
}
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}

start=$SECONDS
next_beat=15
until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
  running || fail "the server exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); it is still running: docker logs -f $CONTAINER_NAME"
  if (( SECONDS - start >= next_beat )); then
    estimate=$(docker logs "$CONTAINER_NAME" 2>&1 | sed -n 's/.*startup estimate \([0-9.]*\) GiB.*/\1/p' | tail -1)
    printf '  %s⋯ %ss elapsed, %s%s GiB on the GPU%s\n' "$D" "$((SECONDS - start))" "$(loaded_gib)" "${estimate:+ of ~$estimate}" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
log "Server answered after $((SECONDS - start))s"

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion"
SERVED=$(served_name || echo "$SERVED_NAME")
if smoke=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED\", \"max_tokens\": 64, \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}]}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); print(r["usage"]["completion_tokens"], "tokens,", r["tensorfold"].get("decode_s"), "s")' 2>/dev/null); then
  log "OK: $smoke"
else
  warn "the smoke test request failed; the server is still running (docker logs -f $CONTAINER_NAME)"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Logs     docker logs -f $CONTAINER_NAME
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF

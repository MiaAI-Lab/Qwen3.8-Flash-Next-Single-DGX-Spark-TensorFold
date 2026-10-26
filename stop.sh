#!/usr/bin/env bash
# Stop the server that ./start.sh started and remove its container, freeing its GPU memory. The server gets
# STOP_TIMEOUT seconds (default 30) to shut down; requests still running are cut off (it does not drain them), so
# stop.sh says when there are any.
# Usage: ./stop.sh      Env: CONTAINER_NAME, PORT (see scripts/config.sh), STOP_TIMEOUT
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "No container named $CONTAINER_NAME: nothing to stop"
  exit 0
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == true ]]; then
  api_host="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && api_host=127.0.0.1
  [[ "$api_host" == *:* ]] && api_host="[$api_host]"
  busy=$(curl -s --max-time 3 "http://$api_host:$PORT/health" 2>/dev/null |
         python3 -c 'import json,sys; h=json.load(sys.stdin); print(h.get("running", h.get("requests_running", 0)))' 2>/dev/null || echo 0)
  (( busy == 0 )) || warn "$busy request(s) still running will be cut off"
  log "Stopping $CONTAINER_NAME (up to ${STOP_TIMEOUT:-30}s)"
  docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null
fi
docker rm -f "$CONTAINER_NAME" >/dev/null
log "Stopped and removed $CONTAINER_NAME; its GPU memory is free again"

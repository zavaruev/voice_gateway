#!/usr/bin/env bash
# Self-healing for the living-room camera.
#
# MEASURED 05.10.2026 — the first version of this script blamed dying hardware
# and was wrong. During an "outage" the camera had 193 kB stuck in the send queue
# of ONE socket (netstat on the box: `tcp 0 193712 192.168.22.241:554
# 192.168.22.102:37544 ESTABLISHED`). It was not dead: our own gateway had stopped
# reading, its single-threaded majestic blocked on the full buffer, and it
# answered HTTP in 14 s while producing no RTSP. `docker restart voice_gateway`
# cleared it completely — 14.34 s -> 0.021 s, audio back to 100 % of real time,
# no power cycle and no camera-side restart.
#
# So the ladder below starts with US. Rebooting the camera to fix our own
# back-pressure is the mistake this file used to make.
#
# What is still true of the camera, all measured:
#   * A `majestic` restart does restore audio completely without a power cycle.
#   * The box is 70-89 % idle with 32 MB free throughout — not CPU or memory.
#   * `ai0_P0_MAIN` sits in D whether the audio works or not. Not a symptom.
#   * `aio_dma` is the PLAYBACK dma, not capture. Not a health signal.
#   * `/proc/loadavg` is a constant here (11.3 while 88.8 % idle), inflated by
#     vendor threads parked in D. Not a symptom.
#   * Restarting majestic kills the go2rtc producer (bare `recv=None`, zero
#     consumers), so re-register the stream after touching the camera.
#   * HTTP latency alone is not a health signal — it measured 0.066 s while RTSP
#     was dead and 14 s while alive-but-blocked. Pair it with go2rtc.
#
# Install:
#   (crontab -l 2>/dev/null; echo '*/2 * * * * .../scripts/camera_watchdog.sh >/dev/null 2>&1') | crontab -
#
# A camera that cannot be reached is never rebooted: no answer means the network,
# and rebooting something you cannot reach costs a boot cycle and fixes nothing.

set -u

CAM_IP="${CAM_IP:-192.168.22.241}"
CAM_STREAM="${CAM_STREAM:-livingroom}"
CAM_SRC_URL="${CAM_SRC_URL:-rtsp://root:2441@192.168.22.241/stream=0#backchannel=1}"
GO2RTC="${GO2RTC:-http://192.168.22.102:1984}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/voice_watchdog_ed25519}"
LOG="${LOG:-/tmp/camera_watchdog.log}"
STAMP_FILE="${STAMP_FILE:-/tmp/.camera_watchdog_last}"
GATEWAY="${GATEWAY:-voice_gateway}"          # our container: the usual culprit
CAMERA_HOST="${CAMERA_HOST:-192.168.22.241}"

WEDGE_S="${WEDGE_S:-2.5}"          # camera must answer faster than this
CONFIRM_N="${CONFIRM_N:-2}"         # consecutive bad samples before acting
CONFIRM_GAP_S="${CONFIRM_GAP_S:-15}"
MIN_GAP_S="${MIN_GAP_S:-300}"       # never retry more often than this
GW_SETTLE_S="${GW_SETTLE_S:-90}"    # settle time after restarting the gateway
SETTLE_S="${SETTLE_S:-30}"          # wait after a restart before re-checking
WAIT_UP_S="${WAIT_UP_S:-150}"       # how long a reboot may take to come back

CAM_LATENCY="?"                     # `set -u` would abort on an unset log value
STATE="?"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG" 2>/dev/null || true; }

cam_ssh() {
  ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      root@"$CAM_IP" "$@" 2>/dev/null
}

# Seconds for the camera to answer, or nothing at all when it did not answer.
#
# The curl EXIT CODE is the whole point here. `-w %{time_total}` prints a number
# even when the connection is REFUSED — measured: `Connection refused` (exit 7)
# prints **0.000616 s**, which is faster than any real answer this camera has ever
# given (0.016-0.031 s healthy). So the first version of this function called a
# dead box the healthiest thing on the network, and its "unreachable is never
# rebooted" rule could never fire. Non-zero exit => print nothing => unreachable.
cam_latency() {
  local out
  out="$(curl -s -m 20 -o /dev/null -w '%{time_total}' "http://$CAM_IP/" 2>/dev/null)" \
    || return 0
  [ -n "$out" ] || return 0
  printf '%s' "$out"
}

# "up" when go2rtc holds a producer that is actually flowing, with a consumer.
go2rtc_up() {
  curl -s -m 10 "$GO2RTC/api/streams" 2>/dev/null | python3 -c '
import json, sys
try:
    s = json.load(sys.stdin).get("'"$CAM_STREAM"'", {})
except Exception:
    sys.exit(1)
prod = s.get("producers") or []
cons = s.get("consumers") or []
# A producer with recv=None is the shape seen during every outage: the entry
# exists but nothing flows.
if not prod or prod[0].get("recv") is None or not cons:
    sys.exit(1)
sys.exit(0)
'
}

# Re-create the stream. Without this the room stays mute after a majestic
# restart even though the camera itself is fine.
reregister() {
  curl -s -m 10 -X DELETE "$GO2RTC/api/streams?src=$CAM_STREAM" -o /dev/null 2>/dev/null
  sleep 2
  curl -s -m 15 -X PUT "$GO2RTC/api/streams?src=$CAM_STREAM" \
       --data-urlencode "$CAM_SRC_URL" -o /dev/null 2>/dev/null
}

healthy() {
  local lat
  lat="$(cam_latency)"
  [ -n "$lat" ] || return 1
  awk -v l="$lat" -v t="$WEDGE_S" 'BEGIN{exit !(l+0 < t+0)}' || return 1
  go2rtc_up || return 1
  CAM_LATENCY="$lat"
  return 0
}

classify() {
  local lat
  lat="$(cam_latency)"
  if [ -z "$lat" ]; then
    STATE="unreachable"
  elif ! awk -v l="$lat" -v t="$WEDGE_S" 'BEGIN{exit !(l+0 < t+0)}'; then
    STATE="wedged"
  elif go2rtc_up; then
    STATE="healthy"
  else
    STATE="dead-stream"
  fi
}

claim() {
  local last t
  last="$([ -f "$STAMP_FILE" ] && cat "$STAMP_FILE" || echo 0)"
  t="$(date +%s)"
  [ $((t - last)) -lt "$MIN_GAP_S" ] && return 1
  echo "$t" >"$STAMP_FILE"
  return 0
}

wait_healthy() {
  local tries="${1:-6}" i
  for i in $(seq 1 "$tries"); do
    sleep "${2:-15}"
    classify
    [ "$STATE" = "healthy" ] && return 0
  done
  return 1
}

# --- confirm before acting -------------------------------------------------
for i in $(seq 1 "$CONFIRM_N"); do
  classify
  [ "$STATE" = "healthy" ] && exit 0
  if [ "$STATE" = "unreachable" ]; then
    log "UNREACHABLE $CAM_IP — network, not the box; not rebooting"
    exit 0
  fi
  [ "$i" -lt "$CONFIRM_N" ] && sleep "$CONFIRM_GAP_S"
done

log "UNHEALTHY stream=$CAM_STREAM state=$STATE http=$(cam_latency)s (${CONFIRM_N}x)"

claim || { log "  skipped: attempted less than ${MIN_GAP_S}s ago"; exit 0; }

# --- step 1: OUR side -------------------------------------------------------
# The reader that stopped draining the camera. This is the step that works, and
# it was measured working: HTTP 14.34 s -> 0.021 s and audio back to 100 % of real
# time, with the camera untouched.
log "  step 1: restart $GATEWAY (the side that stopped reading)"
docker restart "$GATEWAY" >/dev/null 2>&1
# Longer than the camera settle: the container has to import torch-adjacent
# deps, build the vosk model, reconnect the RTSP loop and pull the stream's
# backlog before its audio is real. Sampling too early would call a working
# restart a failure and climb the ladder for nothing.
sleep "$GW_SETTLE_S"

if wait_healthy 8 15; then
  log "  RECOVERED by restarting $GATEWAY (http ${CAM_LATENCY}s) — the camera was never at fault"
  exit 0
fi
log "  still $STATE after restarting $GATEWAY"

# --- step 2: drop go2rtc's sessions, then the camera's service ---------------
# Only now is the camera involved at all: its producer has to be re-created after
# anything on that path is disturbed.
log "  step 2: re-register the go2rtc stream + restart majestic"
reregister
sleep "$SETTLE_S"
cam_ssh '/etc/init.d/S95majestic restart >/dev/null 2>&1'
reregister

if wait_healthy 6 15; then
  log "  RECOVERED by majestic restart + stream re-register (http ${CAM_LATENCY}s)"
  exit 0
fi
log "  still $STATE after majestic restart"

# --- step 3: reboot the box ------------------------------------------------
# Reached only when restarting the service did not help. ONVIF Reboot is not
# implemented on OpenIPC, so this is SSH or nothing.
log "  step 3: reboot the camera over SSH — last resort, and probably wrong"
cam_ssh 'nohup sh -c "sleep 1; reboot" >/dev/null 2>&1 &'

wait_healthy $((WAIT_UP_S / 15)) 15 || { log "  camera did not come back"; exit 1; }
sleep 20
reregister
if wait_healthy 6 15; then
  log "  RECOVERED by reboot + stream re-register (http ${CAM_LATENCY}s)"
  exit 0
fi

log "  NOT RECOVERED — every layer restarted, so this is now worth a look"
exit 1
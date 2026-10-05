#!/usr/bin/env bash
# Self-healing for the living-room camera, which stops producing RTSP audio on
# its own. Written after the measurements of 05.10.2026, all of which changed
# the design:
#
#   * HTTP LATENCY ALONE IS NOT A HEALTH SIGNAL. The camera was measured
#     answering `GET /` in 0.066 s while its RTSP stream was already dead —
#     a fast web server next to a dead media pipeline. The signal is therefore
#     BOTH: the camera must answer quickly AND go2rtc must hold a live producer
#     with a consumer on it.
#   * `aio_dma` in /proc/interrupts is the PLAYBACK dma, not capture. With
#     `audio.outputEnabled: true` it sits near zero while audio streams at
#     100 % of real time. Do not use it.
#   * `/proc/loadavg` is a constant on this box (11.3 while 88.8 % idle),
#     inflated by vendor threads parked in D. Not a symptom.
#   * `ai0_P0_MAIN` sits in D whether the audio works or not. Not a symptom.
#   * A `majestic` restart restores audio COMPLETELY without a power cycle —
#     measured: audio "none" -> "100 % of real time", box uptime unchanged at
#     8 min. So the ladder starts there and reboots only as a last resort.
#   * BUT the fault recurs: a reboot bought 5-15 min and a majestic restart
#     about 3-5 min. This is a hardware/driver fault being mitigated, not fixed.
#   * Restarting majestic KILLS the go2rtc producer too — it then shows a bare
#     `recv=None` with zero consumers — so the recovery must re-register the
#     stream as well, or the room stays mute even though the camera is healthy.
#
# Install:
#   (crontab -l 2>/dev/null; echo '*/2 * * * * .../scripts/camera_watchdog.sh >/dev/null 2>&1') | crontab -
#
# Recovery is idempotent and rate-limited, and it never reboots a camera it
# cannot reach: no answer means the network, and a reboot of something you
# cannot reach costs a boot cycle and fixes nothing.

set -u

CAM_IP="${CAM_IP:-192.168.22.241}"
CAM_STREAM="${CAM_STREAM:-livingroom}"
CAM_SRC_URL="${CAM_SRC_URL:-rtsp://root:2441@192.168.22.241/stream=0#backchannel=1}"
GO2RTC="${GO2RTC:-http://192.168.22.102:1984}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/voice_watchdog_ed25519}"
LOG="${LOG:-/var/log/camera_watchdog.log}"
STAMP_FILE="${STAMP_FILE:-/tmp/.camera_watchdog_last}"

WEDGE_S="${WEDGE_S:-2.5}"          # camera must answer faster than this
CONFIRM_N="${CONFIRM_N:-2}"         # consecutive bad samples before acting
CONFIRM_GAP_S="${CONFIRM_GAP_S:-15}"
MIN_GAP_S="${MIN_GAP_S:-300}"       # never retry more often than this
SETTLE_S="${SETTLE_S:-30}"          # wait after restarting majestic
WAIT_UP_S="${WAIT_UP_S:-150}"       # how long a reboot may take to come back

CAM_LATENCY="?"                     # `set -u` would abort on an unset log value
STATE="?"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG" 2>/dev/null || true; }

cam_ssh() {
  ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      root@"$CAM_IP" "$@" 2>/dev/null
}

cam_latency() {
  curl -s -m 20 -o /dev/null -w '%{time_total}' "http://$CAM_IP/" 2>/dev/null
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

# --- step 1: restart the media service, then re-register the stream --------
log "  step 1: restart majestic + re-register the go2rtc stream"
cam_ssh '/etc/init.d/S95majestic restart >/dev/null 2>&1'
sleep "$SETTLE_S"
reregister

if wait_healthy 6 15; then
  log "  RECOVERED by majestic restart + stream re-register (http ${CAM_LATENCY}s)"
  exit 0
fi
log "  still $STATE after majestic restart"

# --- step 2: reboot the box ------------------------------------------------
# Reached only when restarting the service did not help. ONVIF Reboot is not
# implemented on OpenIPC, so this is SSH or nothing.
log "  step 2: reboot the camera over SSH"
cam_ssh 'nohup sh -c "sleep 1; reboot" >/dev/null 2>&1 &'

wait_healthy $((WAIT_UP_S / 15)) 15 || { log "  camera did not come back"; exit 1; }
sleep 20
reregister
if wait_healthy 6 15; then
  log "  RECOVERED by reboot + stream re-register (http ${CAM_LATENCY}s)"
  exit 0
fi

log "  NOT RECOVERED — the camera needs a manual power cycle"
exit 1
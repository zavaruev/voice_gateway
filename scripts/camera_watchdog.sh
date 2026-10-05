#!/usr/bin/env bash
# Self-healing for the OpenIPC cameras, all of them.
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
# So the ladder starts with US. Rebooting a camera to fix our own back-pressure is
# the mistake this file used to make.
#
# ALL ROOMS ARE HANDLED, and the gateway restart is SHARED: it is one container
# reading both streams, so when two rooms are unhealthy the right move is ONE
# restart, not two. What stays per-camera is the confirmation, the stream
# re-register, the rate limit (so one flapping camera cannot hold the other
# hostage) and everything that needs SSH.
#
# SSH IS PER BOX AND MAY BE MISSING. The watchdog key is authorised on the
# living-room camera only; on a box where it is not, `cam_ssh` returns nothing and
# the camera-side steps are SKIPPED with a log line rather than silently reported
# as "attempted". The gateway step — the one that is measured to work — needs no
# SSH at all, so a camera without the key is still covered.
#
# What is still true of these cameras, all measured:
#   * A `majestic` restart does restore audio completely without a power cycle.
#   * The boxes are 70-89 % idle with ~32 MB free — not CPU or memory pressure.
#   * `ai0_P0_MAIN` sits in D whether the audio works or not. Not a symptom.
#   * `aio_dma` is the PLAYBACK dma, not capture. Not a health signal.
#   * `/proc/loadavg` is a constant here, inflated by vendor threads in D.
#   * Restarting majestic kills the go2rtc producer (bare `recv=None`, zero
#     consumers), so re-register the stream after touching the camera.
#   * HTTP latency alone is NOT a health signal — it measured 0.066 s while RTSP
#     was dead, and 14 s while the box was alive-but-blocked. Pair it with go2rtc.
#   * `curl -w %{time_total}` prints 0.000616 s for a REFUSED connection, faster
#     than any real answer. The exit code is what decides.
#
# Install:
#   (crontab -l 2>/dev/null; echo '*/2 * * * * .../scripts/camera_watchdog.sh >/dev/null 2>&1') | crontab -
set -u

# name:ip[:stream] — the stream name defaults to the room name, and the SOURCE URL
# is read from go2rtc's own registry, which is the single place they are written
# down.
#
# The optional third field exists because writing `name:ip` and getting
# `stream=<the ip>` out of it is the worst possible failure for this script: it
# made `go2rtc_up` look up a stream that does not exist, so BOTH rooms read as
# `dead-stream`, and the ladder rebooted a perfectly healthy camera. It is
# exercised by the tests below for exactly that reason.
CAMERAS="${CAMERAS:-livingroom:192.168.22.241 kitchen:192.168.22.232}"
GO2RTC="${GO2RTC:-http://192.168.22.102:1984}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/voice_watchdog_ed25519}"
LOG="${LOG:-/tmp/camera_watchdog.log}"
STAMP_DIR="${STAMP_DIR:-/tmp}"
GATEWAY="${GATEWAY:-voice_gateway}"

WEDGE_S="${WEDGE_S:-2.5}"          # camera must answer faster than this
CONFIRM_N="${CONFIRM_N:-2}"         # consecutive bad samples before acting
CONFIRM_GAP_S="${CONFIRM_GAP_S:-15}"
MIN_GAP_S="${MIN_GAP_S:-300}"       # per camera: never retry more often
GW_SETTLE_S="${GW_SETTLE_S:-90}"    # settle time after restarting the gateway
SETTLE_S="${SETTLE_S:-30}"
WAIT_UP_S="${WAIT_UP_S:-150}"       # how long a reboot may take to come back

CUR_NAME="?"; CUR_IP="?"; CUR_STREAM="?"
CAM_LATENCY="?"; STATE="?"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG" 2>/dev/null || true; }

# go2rtc's own listing is the source of the source URL. It reports the bare
# `rtsp://<ip>/stream=0` with credentials stripped — lossy, already noted in
# AGENTS.md — so it is used ONLY to recover the registered URL for a re-register,
# and never parsed for anything else.
stream_url() {
  curl -s -m 10 "$GO2RTC/api/streams" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin).get(sys.argv[1], {})
except Exception:
    sys.exit(1)
prod = d.get("producers") or []
for p in prod:
    u = p.get("url")
    if u and "#backchannel" in u:
        print(u)
        break
else:
    for p in prod:
        if p.get("url"):
            print(p["url"])
            break
' "$1"
}

go2rtc_up() {
  curl -s -m 10 "$GO2RTC/api/streams" 2>/dev/null | python3 -c '
import json, sys
try:
    s = json.load(sys.stdin).get(sys.argv[1], {})
except Exception:
    sys.exit(1)
prod = s.get("producers") or []
cons = s.get("consumers") or []
# recv=None is the shape seen during every outage: the entry exists, nothing flows.
if not prod or prod[0].get("recv") is None or not cons:
    sys.exit(1)
sys.exit(0)
' "$1"
}

cam_ssh() {
  ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      root@"$CUR_IP" "$@" 2>/dev/null
}

# Does this box accept our key at all? Checked once, because "the key is not
# authorised here" and "the command failed" must not look the same.
cam_ssh_ok() {
  [ -f "$SSH_KEY" ] || return 1
  cam_ssh 'echo ok' | grep -q ok
}

cam_latency() {
  local out
  out="$(curl -s -m 20 -o /dev/null -w '%{time_total}' "http://$CUR_IP/" 2>/dev/null)" \
    || return 0
  [ -n "$out" ] || return 0
  printf '%s' "$out"
}

reregister() {
  local url
  url="$(stream_url "$CUR_STREAM")"
  if [ -z "$url" ]; then
    log "  $CUR_STREAM: no registered URL in go2rtc to re-register from"
    return 1
  fi
  curl -s -m 10 -X DELETE "$GO2RTC/api/streams?src=$CUR_STREAM" -o /dev/null 2>/dev/null
  sleep 2
  curl -s -m 15 -X PUT "$GO2RTC/api/streams?src=$CUR_STREAM" \
       --data-urlencode "$url" -o /dev/null 2>/dev/null
}

classify() {
  local lat
  lat="$(cam_latency)"
  CAM_LATENCY="${lat:-?}"
  if [ -z "$lat" ]; then
    STATE="unreachable"
  elif ! awk -v l="$lat" -v t="$WEDGE_S" 'BEGIN{exit !(l+0 < t+0)}'; then
    STATE="wedged"
  elif go2rtc_up "$CUR_STREAM"; then
    STATE="healthy"
  else
    STATE="dead-stream"
  fi
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

claim() {
  local f="$STAMP_DIR/.camera_watchdog_$CUR_NAME" last t
  last="$([ -f "$f" ] && cat "$f" || echo 0)"
  t="$(date +%s)"
  [ "$last" -gt 0 ] 2>/dev/null && [ $((t - last)) -lt "$MIN_GAP_S" ] && return 1
  echo "$t" >"$f"
  return 0
}

# --- pass 1: classify every room, collect the sick ones ----------------------
# Names, space separated. The gateway restart is deliberately deferred: it is one
# container for all rooms, so it must be decided once, not per camera.
select_camera() {
  local entry="$1" rest
  CUR_NAME="${entry%%:*}"
  rest="${entry#*:}"
  if [ "$rest" = "$entry" ] || [ -z "$rest" ]; then
    log "BAD CAMERA SPEC \"$entry\" — expected name:ip or name:ip:stream"
    return 1
  fi
  CUR_IP="${rest%%:*}"
  if [ "$rest" = "$CUR_IP" ]; then
    CUR_STREAM="$CUR_NAME"
  else
    CUR_STREAM="${rest#*:}"
  fi
  case "$CUR_STREAM" in
    *[!A-Za-z0-9_-]*|"")
      log "BAD CAMERA SPEC \"$entry\" — stream name \"$CUR_STREAM\" is not a plain name"
      return 1
      ;;
  esac
  return 0
}

SICK=""
for entry in $CAMERAS; do
  select_camera "$entry" || continue
  classify
  case "$STATE" in
    healthy)
      log "$CUR_NAME healthy (http ${CAM_LATENCY}s)"
      ;;
    unreachable)
      # Never rebooted: no answer means the network, and rebooting something you
      # cannot reach costs a boot cycle and fixes nothing.
      log "$CUR_NAME UNREACHABLE $CUR_IP — network, not the box; not rebooting"
      ;;
    *)
      SICK="$SICK $CUR_STREAM"
      log "$CUR_NAME UNHEALTHY state=$STATE http=$(cam_latency)s stream=$CUR_STREAM"
      ;;
  esac
done
[ -n "$SICK" ] || exit 0

# --- confirm before acting ---------------------------------------------------
BAD=""
for entry in $CAMERAS; do
  select_camera "$entry" || continue
  case " $SICK " in *" $CUR_STREAM "*) ;; *) continue ;; esac

  claim || { log "$CUR_NAME skipped: attempted less than ${MIN_GAP_S}s ago"; continue; }
  for i in $(seq 1 "$CONFIRM_N"); do
    classify
    [ "$STATE" = "healthy" ] && break
    if [ "$STATE" = "unreachable" ]; then
      log "$CUR_NAME became UNREACHABLE while confirming — not rebooting"
      break
    fi
    [ "$i" -lt "$CONFIRM_N" ] && sleep "$CONFIRM_GAP_S"
  done
  [ "$STATE" = "healthy" ] && { log "$CUR_NAME recovered on its own"; continue; }
  BAD="$BAD $CUR_NAME:$CUR_IP:$CUR_STREAM"
done
[ -n "$BAD" ] || exit 0

# --- step 1: OUR side, once for all rooms ------------------------------------
# The reader that stopped draining the cameras. Measured working: HTTP 14.34 s ->
# 0.021 s and audio back to 100 % of real time, with the cameras untouched.
log "step 1: restart $GATEWAY once for all rooms —$BAD"
docker restart "$GATEWAY" >/dev/null 2>&1
sleep "$GW_SETTLE_S"

RECOVERED=""; STILL=""
for item in $BAD; do
  IFS=: read -r CUR_NAME CUR_IP CUR_STREAM <<<"$item"
  if wait_healthy 8 15; then
    log "  $CUR_NAME RECOVERED by restarting $GATEWAY (http ${CAM_LATENCY}s) — the camera was never at fault"
    RECOVERED="$RECOVERED $CUR_NAME"
  else
    log "  $CUR_NAME still $STATE after restarting $GATEWAY"
    STILL="$STILL $item"
  fi
done

# --- step 2 and 3: per camera, and only if we can reach it --------------------
for item in $STILL; do
  IFS=: read -r CUR_NAME CUR_IP CUR_STREAM <<<"$item"
  log "$CUR_NAME step 2: re-register the go2rtc stream"
  reregister
  if wait_healthy 4 15; then
    log "  $CUR_NAME RECOVERED by re-registering the stream (http ${CAM_LATENCY}s)"
    continue
  fi

  if ! cam_ssh_ok; then
    log "  $CUR_NAME: no SSH key authorised on $CUR_IP — skipping the camera-side"
    log "  steps (restart majestic, reboot). Authorise $SSH_KEY there to cover it;"
    log "  the gateway step above still applies and is the one that is measured."
    continue
  fi

  log "$CUR_NAME step 3: restart majestic, then re-register"
  reregister
  sleep "$SETTLE_S"
  cam_ssh '/etc/init.d/S95majestic restart >/dev/null 2>&1'
  reregister
  if wait_healthy 6 15; then
    log "  $CUR_NAME RECOVERED by majestic restart + stream re-register (http ${CAM_LATENCY}s)"
    continue
  fi
  log "  $CUR_NAME still $STATE after majestic restart"

  # Step 4: reboot. Reached only when restarting the service did not help.
  # ONVIF Reboot is not implemented on OpenIPC, so this is SSH or nothing.
  log "$CUR_NAME step 4: reboot over SSH — last resort, and probably wrong"
  cam_ssh 'nohup sh -c "sleep 1; reboot" >/dev/null 2>&1 &'
  wait_healthy $((WAIT_UP_S / 15)) 15 || { log "  $CUR_NAME did not come back"; continue; }
  sleep 20
  reregister
  if wait_healthy 6 15; then
    log "  $CUR_NAME RECOVERED by reboot + stream re-register (http ${CAM_LATENCY}s)"
  else
    log "  $CUR_NAME NOT RECOVERED — every layer restarted, worth a look"
  fi
done

log "summary: recovered=[${RECOVERED# }] still_bad=[${STILL# }]"
exit 0

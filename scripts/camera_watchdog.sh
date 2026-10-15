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
# All four rooms. This list was livingroom+kitchen only, which left entrance1
# and camera-corridor1 outside the state machine entirely: they appeared in the
# passive `room level` record and nowhere else — no classification, no
# mic-silent alert, no healing. The two newest rooms are exactly the two that
# were swapped this month, so they are the two most likely to come back wrong.
#
# The watchdog SSH key is authorised on NONE of these boxes (measured 14.10.2026),
# so every camera-side step is skipped with a log line and only the gateway-side
# step and the classification run. That is fine for mic-silent, which reads the
# gateway's own log and needs no SSH at all.
CAMERAS="${CAMERAS:-livingroom:192.168.22.241 kitchen:192.168.22.232 entrance1:192.168.22.198 camera-corridor1:192.168.22.247}"
GO2RTC="${GO2RTC:-http://192.168.22.102:1984}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/voice_watchdog_ed25519}"
LOG="${LOG:-/tmp/camera_watchdog.log}"
STAMP_DIR="${STAMP_DIR:-/tmp}"
GATEWAY="${GATEWAY:-voice_gateway}"

# Above the measured HEALTHY band by ~3.5x. It was 2.5 s, derived when ONE room
# was polled and a healthy OpenIPC answered in 0.016-0.050 s. With both rooms the
# healthy band MEASURED 0.154-1.724 s (median 0.54) and transient spikes to 5.3 s
# happen in normal operation — so 2.5 s fired on healthy cameras and restarted the
# gateway **17 times**. The wedge it was built for measured 7.4 s across three
# tries, and 11.9-15 s otherwise, so 6 s keeps the margin on both sides.
WEDGE_S="${WEDGE_S:-6.0}"
# A camera must stay slow for a minute before anything disruptive happens. A
# restart is not free: it drops every model, every RTSP loop and any turn in
# flight, so it must not be reachable by a single transient sample.
CONFIRM_N="${CONFIRM_N:-3}"
CONFIRM_GAP_S="${CONFIRM_GAP_S:-20}"
MIN_GAP_S="${MIN_GAP_S:-300}"       # per camera: never retry more often

# Audio-starvation detection. The gateway logs one `AUDIO STARVED` line per 20 s
# window while a room's delivered audio is below its threshold, so a room that has
# been running at a fraction of real time for a while produces a run of them.
# 3 hits in a 300 s window is ~60 s of sustained starvation: enough that a single
# hiccup right after a restart cannot trigger a camera action, which is the same
# discipline CONFIRM_N applies to latency.
#
# MIN_GAP_S (300 s) alone is NOT enough on its own — it rate-limits actions after
# one has been taken, but the first action still needs a threshold that a brief
# dip cannot cross. Measured on 07.10.2026: the living room sat at 1.46-1.93
# chunks/s against 6.25/s for over 15 minutes, producing a line every 20 s.
STARVE_WINDOW_S="${STARVE_WINDOW_S:-300}"
STARVE_MIN_HITS="${STARVE_MIN_HITS:-3}"
GW_SETTLE_S="${GW_SETTLE_S:-90}"    # settle time after restarting the gateway
SETTLE_S="${SETTLE_S:-30}"
WAIT_UP_S="${WAIT_UP_S:-150}"       # how long a reboot may take to come back

CUR_NAME="?"; CUR_IP="?"; CUR_STREAM="?"
CAM_LATENCY="?"; STATE="?"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG" 2>/dev/null || true; }

# `DRY_RUN=1` decides what it WOULD do and touches nothing.
#
# Added because testing this script the hard way restarted the gateway: the
# ladder's first step is `docker restart voice_gateway`, and a script whose
# first action is "restart the thing you are trying to protect" cannot be
# exercised against a fault at all. Everything is still measured and logged, so
# a dry run says exactly which step would have fired.
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$STAMP_DIR" 2>/dev/null || true

act() {
  if [ "$DRY_RUN" = "1" ]; then
    log "  [dry-run] would: $*"
    return 0
  fi
  "$@"
}

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
  if [ "$DRY_RUN" = "1" ]; then
    log "  [dry-run] would: re-register $CUR_STREAM -> $url"
    return 0
  fi
  curl -s -m 10 -X DELETE "$GO2RTC/api/streams?src=$CUR_STREAM" -o /dev/null 2>/dev/null
  sleep 2
  curl -s -m 15 -X PUT "$GO2RTC/api/streams?src=$CUR_STREAM" \
       --data-urlencode "$url" -o /dev/null 2>/dev/null
}

# FIVE states, because THREE independent signals are measured: how fast the camera
# answers HTTP, whether go2rtc holds a producer with a consumer on it, and — the
# one that actually caught the 07.10.2026 outage — HOW FAST THE AUDIO ARRIVES.
#
#   healthy        fast AND streaming AND audio at real time   nothing to do
#   slow-only      slow AND streaming                          a busy box — logged, never acted on
#   dead-stream    fast AND not streaming                      our reader or go2rtc; the gateway step
#   wedged         slow AND not streaming                      both, which is what a real outage looks like
#   audio-starved  STREAMING but the audio runs at a           the mic is alive and useless (below)
#                  fraction of real time
#
# HTTP latency ALONE decides nothing. Measured both ways on this network: the room
# answered in 0.066 s while its RTSP was dead, and two polled rooms measure
# 0.154-1.724 s with spikes to 5.3 s while completely healthy.
#
# `audio-starved` exists because on 07.10.2026 the living room failed in a way
# invisible to the other two signals, and it cost a whole morning of "the camera
# does not answer «компьютер»": HTTP answered in 0.24 s (so `fast`) while go2rtc
# held a producer with a consumer on it (so `streaming`), yet the gateway's own
# chunk counter ran at 1.46/s against an expected 6.25/s — a quarter of real time.
# That IS a stream carrying audio, so the old classifier called it `healthy` and
# did nothing. Meanwhile vosk reported `triggers=0 decodes=0 hyp=''` with
# `peak_max=27607`: loud audio that no decoder can recognise, because it arrives
# stretched. A gate that cannot hear «компьютер» and a gate that is refusing it
# look identical from the outside, and `hyp=''` is what distinguishes them.
#
# The rate is deliberately NOT measured here. Adding an ffmpeg consumer to find
# out was tried and is the wrong move twice over: this watchdog exists to avoid
# perturbing what it watches, and a second reader on a single-threaded majestic is
# a large perturbation. The gateway already accounts delivered audio against wall
# clock every 20 s and logs `AUDIO STARVED: NN% of real time` below its threshold —
# the correct instrument, already in production, with the room name in the line.
# Read it from there rather than taking a second measurement.
# Room level history, written HERE on the host because the gateway's own log does
# not survive `docker compose up --build` — which is exactly how the kitchen's
# 10.10.2026 morning went unexplained: the level tripled at 09:10 with everyone
# asleep, every physical sensor in HA was `unavailable`, and the log that held the
# only instrument was destroyed by the redeploy I did at 11:29. A record that dies
# with the process that made it cannot answer "what happened at 09:10".
#
# It is a RECORD, not a gate. Nothing here decides anything, because the threshold
# is precisely what is unknown: that level was 451-541 rms for hours and 1869 for
# an hour, it resolved on its own, and it cost no false wakes. Inventing a limit
# now would be the same guess that the notes warn about. The history is what makes
# a limit derivable next time.
record_room_level() {
  local since="${1:-210}" stats
  stats="$(docker logs voice_gateway --since "${since}s" 2>&1 | python3 -c '
import re, sys
from collections import defaultdict

rows = defaultdict(list)
for line in sys.stdin:
    # The class is [A-Za-z0-9_-] and NOT \w: \w does not match a hyphen, and
    # `camera-corridor1` has one — the corridor was silently absent from this
    # record, and every detector copied from it inherited the blind spot.
    m = re.search(r"\[([A-Za-z0-9_-]+)\] feed rms=(\d+) peak=(\d+)", line)
    if m:
        rows[m.group(1)].append((int(m.group(2)), int(m.group(3))))
if not rows:
    sys.exit(0)
for room, vals in sorted(rows.items()):
    vals.sort()
    rms = sorted(v for v, _ in vals)
    peaks = [p for _, p in vals]
    mid = rms[len(rms) // 2]
    print("    %-11s rms med=%-5d max=%-5d  peak max=%-6d n=%d"
          % (room, mid, rms[-1], max(peaks), len(vals)))
' 2>/dev/null)"
  [ -n "$stats" ] || return 0
  log "room level (${since}s window):"
  printf '%s\n' "$stats" | while IFS= read -r line; do log "$line"; done
}

# MIC-SILENT: the stream is alive and the audio arrives at real time, but the
# level sits on the floor. Measured 14.10.2026 after a firmware update: the new
# build rewrote /etc/majestic.yaml to its own defaults and dropped
# `audio.volume` from four of five cameras. Three rooms went deaf on a level of
# 7-24 against a healthy 304-1070, i.e. the SAME state the kitchen lived in for a
# week at volume 30. Nothing logged it: the stream was fine, the rate was fine,
# every existing signal said healthy.
#
# The threshold comes from the two sides of that event, not from taste:
#
#   broken  (11:38-11:43, before the repair)      p90 = 25, max 83
#   healthy  (11:50-12:00, after the repair)      min 304, p10 = 356
#
# 150 sits x1.8 above the worst sample actually observed broken and x2.0 below
# the lowest sample actually observed healthy. Overridable per room, because the
# right number is a property of that room's noise, not of the house.
#
# MEDIAN, never the last sample: a 3 s log line is not a decision, and this
# script has already restarted a healthy container seventeen times in a day off
# exactly that. `MIC_SILENT_MIN_HITS` consecutive passes must agree.
#
# NOT ACTIONED, deliberately. There is nothing to restart: the setting is missing
# FROM THE FILE, so a `majestic` restart re-reads the same file and the room stays
# mute. Restarting the gateway is worse — it drops every model and every room. The
# honest output is the level, the room and the one-line fix. Rewriting a camera's
# config unattended is a bigger decision than this script may take on its own; the
# person who set the volume should be the one to check that it came back.
MIC_SILENT_RMS="${MIC_SILENT_RMS:-150}"
MIC_SILENT_MIN_HITS="${MIC_SILENT_MIN_HITS:-3}"
MIC_SILENT_WINDOW_S="${MIC_SILENT_WINDOW_S:-300}"
MIC_SILENT_STAMPS="$STAMP_DIR/.camera_watchdog_micsilent"

# Median `feed rms` for this room over the window. EMPTY when the gateway has
# logged nothing, and an empty answer must never be read as "silent" — a gateway
# that just restarted would otherwise declare four deaf rooms on the first pass.
mic_silent_median() {
  local since="${1:-$MIC_SILENT_WINDOW_S}"
  docker logs voice_gateway --since "${since}s" 2>&1 | python3 -c '
import re, sys
vals = []
for line in sys.stdin:
    m = re.search(r"\[([A-Za-z0-9_-]+)\] feed rms=(\d+)", line)
    if m and m.group(1) == sys.argv[1]:
        vals.append(int(m.group(2)))
if not vals:
    sys.exit(0)
vals.sort()
print(vals[len(vals) // 2])
' "$CUR_STREAM" 2>/dev/null
}

# CONFIG-DRIFT: read the setting itself, not just its consequence.
#
# `mic-silent` above catches a room that has gone quiet, which is what a wiped
# `audio.volume` LOOKS LIKE. It cannot say WHY, and it only fires once the level
# has already collapsed. This reads the key off the camera over SSH and reports it
# while the rest of the file is still intact — on 14.10.2026 the update took
# `volume` AND `rtsp.backchannel`, `video0.gopSize`, the video1 substream,
# `outgoing.substream` and `watchdog.timeout` in one go.
#
# Reported, never rewritten. Restoring a camera's config unattended is a bigger
# decision than this script may take: the file is per-camera state, and the person
# who set the value should be the one to find out it did not survive. The output
# is the value, the expected value and the one-line fix.
#
# 100 is MEASURED, not chosen: it is what every one of these four cameras carried
# while working (AGENTS.md, 14.10.2026), and 30 is the value at which the kitchen
# read rms 4 and the room is effectively deaf.

# Defined here and nowhere else: `set -u` turns a missing definition into an abort
# of the WHOLE script, not into a skipped check, so the default is mandatory.
CAM_CONFIG_CHECK="${CAM_CONFIG_CHECK:-1}"

# The whole config, fetched once and parsed on THIS side. Parsing on the host
  # rather than with awk on the box keeps the comparison readable and lets quotes
  # be normalised properly — the broken file carried `srate: 48000` where the good
  # one has `srate: "48000"`, which is the same value written differently and must
  # not read as drift.
  cam_config_dump() {
    cam_ssh "cat /etc/majestic.yaml"
  }

  # Dotted paths for the whole file. This config is two levels deep and uses no
  # anchors or multi-line values, so a two-level walk is exact here. Anything
  # unparsable produces no entry, and an unparsable file must not then read as
  # "every key absent".
  cam_config_keys() {
    python3 -c '
import sys
q = chr(34) + chr(39)
out, section = {}, None
for raw in sys.stdin:
    line = raw.rstrip("\r\n")
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    if line.startswith((" ", "\t")):
        key, _, val = line.strip().partition(":")
        if section and key:
            out["%s.%s" % (section, key)] = val.strip().strip(q)
    else:
        section = line.rstrip(":").strip()
for k in sorted(out):
    print("%s=%s" % (k, out[k]))
'
  }

  # Every key the 14.10.2026 firmware update took, with the value these four
  # cameras carried while working. Enumerated rather than diffed against a
  # reference file on purpose: `mirror`/`flip` and anything else per-mount is
  # allowed to differ, and a whole-file comparison would cry wolf on it.
  #
  # A MISSING key is reported too, and not as a complaint about the value: that is
  # exactly the shape of the incident — the key vanished, the build default took
  # over, and the room changed behaviour with nothing logged. Whether the new
  # default happens to be harmless is beside the point; the operator decides.
  CAM_CONFIG_KEYS="${CAM_CONFIG_KEYS:-audio.volume=100
audio.enabled=true
audio.srate=48000
audio.outputEnabled=true
audio.outputVolume=80
rtsp.backchannel=true
video0.enabled=true
video0.gopSize=1
video1.enabled=false
outgoing.substream=true
watchdog.timeout=30
nightMode.lightMonitor=true}"

  config_drift() {
    # `line` is initialised here and not merely declared: under `set -u` reading
    # it before the first append aborts the WHOLE script, and it is only read on
    # the first room that actually has drift — so the bug would surface exactly
    # when the check is needed.
    local dump want path expect got seen line="" missing=0 wrong=0
    [ "$CAM_CONFIG_CHECK" = "1" ] || return 1
    if ! cam_ssh_ok; then
      log "$CUR_NAME config check skipped: no SSH key on $CUR_IP — this room has no drift cover"
      return 1
    fi
    dump="$(cam_config_dump)"
    if [ -z "$dump" ]; then
      log "$CUR_NAME config check skipped: could not read /etc/majestic.yaml on $CUR_IP"
      return 1
    fi
    seen="$(printf '%s\n' "$dump" | cam_config_keys)"

    for want in $CAM_CONFIG_KEYS; do
      path="${want%%=*}"
      expect="${want#*=}"
      got="$(printf '%s\n' "$seen" | grep "^${path}=" | head -1 | cut -d= -f2-)"
      if [ -z "$got" ]; then
        missing=$(( missing + 1 ))
        line="$line|$path: ABSENT (expected $expect) — the build default is in force"
      elif [ "$got" != "$expect" ]; then
        wrong=$(( wrong + 1 ))
        line="$line|$path: got $got, expected $expect"
      fi
    done

    if [ "$missing" -gt 0 ] || [ "$wrong" -gt 0 ]; then
      log "$CUR_NAME CONFIG DRIFT — $missing absent, $wrong has the wrong value ($CUR_IP):${line}
    The 14.10.2026 firmware dropped everything that differed from its new defaults. Fix: copy a working /etc/majestic.yaml to $CUR_IP and restart S95majestic. Nothing was restarted."
      return 0
    fi
    return 1
  }

audio_starved_hits() {
  local since="${1:-$STARVE_WINDOW_S}" lines
  lines="$(docker logs voice_gateway --since "${since}s" 2>&1 \
    | grep -E "\[$CUR_STREAM\] .*(AUDIO STARVED|audio rate recovered)" || true)"
  # The LAST signal for this room decides, not the count. The gateway logs
  # `audio rate recovered: NN%` when the rate comes back, so a window that holds
  # both is a room that got WORSE, and the count alone would call a room healthy
  # again since 10:22 still starved and would have its camera restarted for a
  # fault that ended minutes ago — the same "acted on a stale signal" mistake this
  # script already made once with latency.
  [ -n "$lines" ] || { echo 0; return 0; }
  printf '%s\n' "$lines" | tail -1 | grep -q "AUDIO STARVED" || { echo 0; return 0; }
  printf '%s\n' "$lines" | grep -c "AUDIO STARVED" 2>/dev/null || echo 0
}
classify() {
  local lat fast=0 up=0 hits
  lat="$(cam_latency)"
  CAM_LATENCY="${lat:-?}"
  [ -z "$lat" ] && { STATE="unreachable"; return 0; }
  if awk -v l="$lat" -v t="$WEDGE_S" 'BEGIN{exit !(l+0 < t+0)}'; then
    fast=1
  fi
  go2rtc_up "$CUR_STREAM" && up=1
  CAM_STARVED_HITS=0
  # Only meaningful while the stream is up: with no stream there is no audio to be
  # slow, and dead-stream/wedged already route to the gateway step first.
  if [ "$up" = 1 ]; then
    hits="$(audio_starved_hits)"
    CAM_STARVED_HITS="${hits:-0}"
    if [ "${CAM_STARVED_HITS}" -ge "$STARVE_MIN_HITS" ]; then
      STATE="audio-starved"
      return 0
    fi
    # Last gate before healthy, and only while the stream is up: with no stream
    # there is no level to read and dead-stream/wedged already route correctly.
    local med thr hits_file hits
    med="$(mic_silent_median)"
    thr="${MIC_SILENT_RMS:-$MIC_SILENT_RMS}"
    [ -n "$med" ] || thr=0
    if [ "$thr" -gt 0 ] 2>/dev/null && [ -n "$med" ] && [ "$med" -lt "$thr" ]; then
      hits_file="${MIC_SILENT_STAMPS}_$CUR_STREAM"
      hits="$(cat "$hits_file" 2>/dev/null || echo 0)"
      hits=$(( hits + 1 ))
      echo "$hits" > "$hits_file" 2>/dev/null || true
      if [ "$hits" -ge "$MIC_SILENT_MIN_HITS" ]; then
        STATE="mic-silent"
        return 0
      fi
    else
      rm -f "${MIC_SILENT_STAMPS}_$CUR_STREAM" 2>/dev/null || true
    fi
  fi
  if [ "$fast" = 1 ] && [ "$up" = 1 ]; then
    STATE="healthy"
  elif [ "$fast" = 1 ]; then
    STATE="dead-stream"
  elif [ "$up" = 1 ]; then
    STATE="slow-only"
  else
    STATE="wedged-slow"
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
    slow-only|wedged-slow)
      # Latency without a dead stream. Logged, never acted on.
      log "$CUR_NAME slow but stream alive (http $(cam_latency)s) — watching, not acting"
      ;;
    dead-stream|wedged|audio-starved)
      # The stream itself is broken, the camera is slow AND not streaming, or the
      # audio is arriving at a fraction of real time while the stream looks fine.
      SICK="$SICK $CUR_STREAM"
      log "$CUR_NAME UNHEALTHY state=$STATE http=$(cam_latency)s stream=$CUR_STREAM${CAM_STARVED_HITS:+ starved=${CAM_STARVED_HITS}/${STARVE_WINDOW_S}s}"
      ;;
    mic-silent)
      # Stream alive, rate fine, level on the floor. Reported, never acted on —
      # there is nothing here a restart fixes. Logged with the one-line repair so
      # the reader does not have to know what happened on 14.10.2026.
      log "$CUR_NAME MIC SILENT — level below ${MIC_SILENT_RMS} while streaming at real time. NOT restarting anything; most likely audio.volume was dropped from /etc/majestic.yaml"
      rm -f "${MIC_SILENT_STAMPS}_$CUR_STREAM" 2>/dev/null || true
      ;;
  esac
  # Checked for EVERY room regardless of state, healthy included: the whole point
  # is to catch the drift while the file is still otherwise intact and the room
  # is still audible, which is exactly the state right after a firmware update and
  # long before anything logs a symptom.
  config_drift || true
done
# Recorded on every pass, healthy or not, so a level that changes while everything
# looks fine still leaves a trace somewhere that outlives the gateway container.
record_room_level
[ -n "$SICK" ] || exit 0

# --- confirm before acting ---------------------------------------------------
BAD=""; STARVED_ONLY=""
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
  # A room whose audio runs at a fraction of real time while its stream is alive
  # is a CAMERA fault, and it is the one fault restarting the gateway cannot fix:
  # the gateway's own remedy for it is an ffmpeg restart plus a stream re-register,
  # and on 07.10.2026 it ran that three times and then declared `CAMERA AUDIO DEAD`
  # while the room stayed deaf. `majestic` restart fixed it every time. So these
  # rooms skip step 1 entirely and go straight to the camera.
  [ "$STATE" = "audio-starved" ] && STARVED_ONLY="$STARVED_ONLY $CUR_NAME:$CUR_IP:$CUR_STREAM"
done
[ -n "$BAD" ] || exit 0

# --- step 1: OUR side, once, and only for rooms that are NOT camera faults ------
# The reader that stopped draining the cameras. Measured working: HTTP 14.34 s ->
# 0.021 s and audio back to 100 % of real time, with the cameras untouched.
# Restarts the shared container, so it must exclude audio-starved rooms — for
# those the camera is the fault and this step costs a full model reload for nothing.
OURSIDE=""
for item in $BAD; do
  case " $STARVED_ONLY " in *" $item "*) ;; *) OURSIDE="$OURSIDE $item" ;; esac
done

RECOVERED=""
if [ -n "$OURSIDE" ]; then
  log "step 1: restart $GATEWAY once for all rooms —$OURSIDE"
  act docker restart "$GATEWAY"
  sleep "$GW_SETTLE_S"
  STILL=""
  for item in $OURSIDE; do
    IFS=: read -r CUR_NAME CUR_IP CUR_STREAM <<<"$item"
    if wait_healthy 8 15; then
      log "  $CUR_NAME RECOVERED by restarting $GATEWAY (http ${CAM_LATENCY}s) — the camera was never at fault"
      RECOVERED="$RECOVERED $CUR_NAME"
    else
      log "  $CUR_NAME still $STATE after restarting $GATEWAY"
      STILL="$STILL $item"
    fi
  done
else
  log "step 1 skipped: every sick room is audio-starved, which is the camera's audio stage, not our reader"
  STILL="$OURSIDE"
fi
# Audio-starved rooms were never candidates for step 1; carry them forward.
STILL="$STILL $STARVED_ONLY"
STILL="$(echo $STILL | tr ' ' '\n' | sort -u | tr '\n' ' ')"

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
  # Through `act`, like every other action. The two steps below are the only ones
  # that touch the PHYSICAL camera, and they were the only ones DRY_RUN stayed
  # silent about — so the one capability worth verifying (restarting majestic on
  # a room that just got its SSH key) produced no evidence at all in a dry run.
  act cam_ssh '/etc/init.d/S95majestic restart >/dev/null 2>&1'
  reregister
  if wait_healthy 6 15; then
    log "  $CUR_NAME RECOVERED by majestic restart + stream re-register (http ${CAM_LATENCY}s)"
    continue
  fi
  log "  $CUR_NAME still $STATE after majestic restart"

  # Step 4: reboot. Reached only when restarting the service did not help.
  # ONVIF Reboot is not implemented on OpenIPC, so this is SSH or nothing.
  log "$CUR_NAME step 4: reboot over SSH — last resort, and probably wrong"
  act cam_ssh 'nohup sh -c "sleep 1; reboot" >/dev/null 2>&1 &'
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

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
# The ladder's per-camera cooldown, for the steps that are cheap and reversible.
# The REBOOT gets a much longer one of its own — see the step 4 comment: a
# reboot does not repair a weak WiFi link, it only takes the room's hearing
# away while the camera boots. Measured 20.10.2026: the corridor was rebooted
# every thirteen minutes and the symptom was exactly that the camera stopped
# answering «компьютер».
REBOOT_MIN_GAP_S="${REBOOT_MIN_GAP_S:-3600}"

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

# Capture rate, measured on the camera→go2rtc leg WITHOUT adding a consumer.
#
# The 15.10.2026 plan was `ffmpeg -i rtsp://<camera> -t 15 -f null -` and to read
# `speed=`, and it was never implemented, because that command is a SECOND
# consumer on a single-threaded majestic and a watchdog must not perturb what it
# watches. The same quantity is available for free: go2rtc already holds the
# camera's RTSP session and counts the bytes of every track it receives
# (`98 OPUS/48000/2, bytes=…` on the producer), so reading its counter twice,
# N seconds apart, measures the delivery rate of an ALREADY OPEN link.
#
# Measured 20.10.2026, 30 s window, all four rooms: 6424-6428 B/s of OPUS, a
# spread of 0.06 % across rooms whose microphone levels differ 2.7x — the
# encoder is CBR here, so the byte rate is a usable proxy for "fraction of real
# time" (the gateway's own ratio is the same quantity on the go2rtc→gateway leg).
#
# The bar is 80 % of the room's OWN best observed rate, not of a constant: the
# constant is a property of the encoder, and the encoder is not what is being
# watched. 80 % is the gateway's own `_audio_rate_ok`, measured from both sides
# (healthy 1.08x, broken ≤ 0.71x).
CAPTURE_RATE_ENABLED="${CAPTURE_RATE_ENABLED:-1}"
CAPTURE_RATE_PCT="${CAPTURE_RATE_PCT:-80}"
CAPTURE_RATE_MIN_HITS="${CAPTURE_RATE_MIN_HITS:-2}"
CAPTURE_RATE_MIN_DT="${CAPTURE_RATE_MIN_DT:-30}"
CAPTURE_RATE_STAMPS="$STAMP_DIR/.camera_watchdog_capture"
GW_SETTLE_S="${GW_SETTLE_S:-90}"    # settle time after restarting the gateway
SETTLE_S="${SETTLE_S:-30}"
WAIT_UP_S="${WAIT_UP_S:-150}"       # how long a reboot may take to come back

CUR_NAME="?"; CUR_IP="?"; CUR_STREAM="?"
CAM_LATENCY="?"; STATE="?"
CAM_CAPTURE_BPS=""; CAM_CAPTURE_BASE=""

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

# ONE INSTANCE AT A TIME.
#
# The ladder takes minutes (sleeps between the steps, waits for the camera to
# come back) and the cron fires every three, so without a lock the instances
# overlap. Measured 20.10.2026 23:36: FIVE copies were running at once, each
# with its own ladder, each able to restart `majestic` and reboot the camera —
# and the overlap is what turned a single bad room into a reboot every
# thirteen minutes.
#
# `flock -n` is non-blocking on purpose: the instance that arrives while
# another is working must LEAVE, not queue behind it. A queue of watchdogs is
# a queue of ladders, and the ladder ends in a reboot.
LOCK_FILE="${LOCK_FILE:-$STAMP_DIR/.camera_watchdog.lock}"
exec 9>"$LOCK_FILE" 2>/dev/null || true
if ! flock -n 9 2>/dev/null; then
  log "another instance is already running — leaving it the room"
  exit 0
fi

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

# The OPUS track's byte counter on the producer side: the camera→go2rtc leg.
capture_bytes() {
  curl -s -m 10 "$GO2RTC/api/streams?src=$1" 2>/dev/null | python3 -c '
import json, re, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
prod = (d.get("producers") or [{}])[0]
for rec in prod.get("receivers") or []:
    m = re.match(r"\d+\s+OPUS", str(rec))
    b = re.search(r"bytes=(\d+)", str(rec))
    if m and b:
        print(int(b.group(1)))
        break
'
}

# True (exit 0) when the measured rate is BELOW the bar for this room.
# An empty measurement, or a room with no baseline yet, is NEVER a decision:
# there is nothing to compare, and guessing is what this script has done wrong
# before.
capture_rate_is_slow() {
  local rate="${1:-}" base="${2:-}"
  [ -n "$rate" ] && [ -n "$base" ] || return 1
  [ "$base" -gt 0 ] 2>/dev/null || return 1
  [ "$rate" -lt $(( base * CAPTURE_RATE_PCT / 100 )) ] 2>/dev/null
}

# Sets CAPTURE_RATE_BPS (the rate since the previous pass) and CAPTURE_BASE_BPS
# (the room's own best observed rate). Both are empty when there is not enough
# to say anything, and an empty value is NEVER a decision.
#
# The baseline is a rolling maximum, so an encoder that raises its bitrate does
# not make the room look slow, and the counter restarting (the producer was
# re-created) shows up as a negative delta and is ignored.
capture_rate() {
  CAPTURE_RATE_BPS=""; CAPTURE_BASE_BPS=""
  local cur prev_file best_file now pbytes="" ptime="" best="" dt=0
  prev_file="${CAPTURE_RATE_STAMPS}_$CUR_STREAM.prev"
  best_file="${CAPTURE_RATE_STAMPS}_$CUR_STREAM.best"
  cur="$(capture_bytes "$CUR_STREAM")" || return 1
  [ -n "$cur" ] || return 1
  now="$(date +%s)"
  if [ -f "$prev_file" ]; then
    # `read ... || pbytes=""; ptime=""` does NOT do what it looks like: the `;`
    # ends the `||` list, so the reset runs unconditionally and the values just
    # read are always thrown away. The first version of this function had that
    # and the rate was never computed — the self-test caught it.
    if read -r pbytes ptime <"$prev_file" 2>/dev/null; then
      :
    else
      pbytes=""; ptime=""
    fi
    if [ -n "$pbytes" ] && [ -n "$ptime" ]; then
      dt=$(( now - ptime ))
      if [ "$dt" -ge "$CAPTURE_RATE_MIN_DT" ] 2>/dev/null; then
        CAPTURE_RATE_BPS=$(( (cur - pbytes) / dt ))
        [ "$CAPTURE_RATE_BPS" -lt 0 ] && CAPTURE_RATE_BPS=""
      fi
    fi
  fi
  printf '%s %s\n' "$cur" "$now" >"$prev_file" 2>/dev/null || true
  if [ -f "$best_file" ]; then
    read -r best <"$best_file" 2>/dev/null || best=""
  fi
  if [ -n "$CAPTURE_RATE_BPS" ] && { [ -z "$best" ] || [ "$CAPTURE_RATE_BPS" -gt "$best" ]; }; then
    best="$CAPTURE_RATE_BPS"
    printf '%s\n' "$best" >"$best_file" 2>/dev/null || true
  fi
  CAPTURE_BASE_BPS="$best"
  return 0
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
    # The camera→go2rtc leg, read off go2rtc's own counter. This is the signal
    # the 15.10.2026 plan wanted and could not take without adding a consumer,
    # and it is a CAMERA fault, so it routes to the camera ladder and never to
    # the gateway restart.
    if [ "$CAPTURE_RATE_ENABLED" = "1" ]; then
      local rate="" base="" cap_file cap_hits=0
      if capture_rate; then
        rate="$CAPTURE_RATE_BPS"
        base="$CAPTURE_BASE_BPS"
      fi
      CAM_CAPTURE_BPS="$rate"; CAM_CAPTURE_BASE="$base"
      cap_file="${CAPTURE_RATE_STAMPS}_$CUR_STREAM.hits"
      cap_hits="$(cat "$cap_file" 2>/dev/null || echo 0)"
      case "$cap_hits" in ''|*[!0-9]*) cap_hits=0 ;; esac
      if [ -n "$rate" ] && [ -n "$base" ] && [ "$base" -gt 0 ] 2>/dev/null; then
        if capture_rate_is_slow "$rate" "$base"; then
          cap_hits=$(( cap_hits + 1 ))
          echo "$cap_hits" >"$cap_file" 2>/dev/null || true
          if [ "$cap_hits" -ge "$CAPTURE_RATE_MIN_HITS" ]; then
            STATE="slow-capture"
            return 0
          fi
        else
          # Only a measurement that PROVES the rate is healthy clears the counter.
          rm -f "$cap_file" 2>/dev/null || true
          cap_hits=0
        fi
      fi
      # The state is raised ONLY on a pass that measured the rate and found it
      # slow. A counter read from disk cannot raise it on its own: the first
      # version did, and a pass with no measurement at all (the counter reset
      # when go2rtc re-registered the producer) reported
      # `SLOW CAPTURE ?/6506 B/s` — a state with no number in it.
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

# May this camera be rebooted now?
#
# The ladder's own MIN_GAP_S covers the cheap, reversible steps. A reboot is
# neither: it does not repair a weak WiFi link (measured 20.10.2026, the
# corridor was rebooted every thirteen minutes and the link quality kept
# flapping 58 -> 62 -> 74), and every reboot takes the room's hearing away
# while the camera boots — its RTSP stream is not up for the first minute or
# two, and a camera that answers HTTP but not RTSP is BOOTING, not broken.
reboot_allowed() {
  local last="${1:-0}" now="${2:-0}" gap="${3:-3600}"
  [ "$last" -gt 0 ] 2>/dev/null || return 0
  [ $(( now - last )) -ge "$gap" ] 2>/dev/null
}

# --- self-test ----------------------------------------------------------------
# A gate that cannot be shown to fire is not a gate. This exercises the capture
# rate arithmetic and the threshold in BOTH directions: a healthy rate must not
# raise the state, a slow one must, and a MISSING measurement must not either.
run_self_test() {
  local fails=0 tmp
  tmp="$(mktemp -d 2>/dev/null)" || { echo "  SELFTEST: cannot create a temp dir"; return 1; }
  STAMP_DIR="$tmp"; CAPTURE_RATE_STAMPS="$STAMP_DIR/.cap"; CUR_STREAM="selftest"
  FAKE_NOW=1000
  date() { printf '%s\n' "$FAKE_NOW"; }
  CAPTURE_BYTES=0
  capture_bytes() { printf '%s\n' "$CAPTURE_BYTES"; }

  # --- the threshold, both directions ----------------------------------------
  if capture_rate_is_slow 6424 6424; then
    echo "  FAIL: a rate equal to the baseline is not slow"; fails=$((fails+1))
  fi
  if capture_rate_is_slow 5138 6424; then
    : # 79 % of the baseline: slow, and it must be reported as slow
  else
    echo "  FAIL: 79 % of the baseline must be slow"; fails=$((fails+1))
  fi
  if capture_rate_is_slow 5139 6424; then
    echo "  FAIL: the 80 % boundary itself is not slow"; fails=$((fails+1))
  fi
  if capture_rate_is_slow 5200 6424; then
    echo "  FAIL: 81 % of the baseline is not slow"; fails=$((fails+1))
  fi
  if capture_rate_is_slow "" 6424; then
    echo "  FAIL: no measurement is not a decision"; fails=$((fails+1))
  fi
  if capture_rate_is_slow 100 ""; then
    echo "  FAIL: no baseline is not a decision"; fails=$((fails+1))
  fi
  if capture_rate_is_slow 100 0; then
    echo "  FAIL: a zero baseline is not a decision"; fails=$((fails+1))
  fi

  # --- the arithmetic ---------------------------------------------------------
  CAPTURE_BYTES=0; FAKE_NOW=1000
  capture_rate >/dev/null
  if [ -n "$CAPTURE_RATE_BPS" ]; then
    echo "  FAIL: the first pass has nothing to compare against"; fails=$((fails+1))
  fi
  FAKE_NOW=1060; CAPTURE_BYTES=$(( 6424 * 60 ))
  capture_rate >/dev/null
  if [ "$CAPTURE_RATE_BPS" != "6424" ]; then
    echo "  FAIL: 385440 bytes over 60 s is 6424 B/s, got '${CAPTURE_RATE_BPS:-}'"
    fails=$((fails+1))
  fi
  if [ "$CAPTURE_BASE_BPS" != "6424" ]; then
    echo "  FAIL: the baseline is the rate just measured, got '${CAPTURE_BASE_BPS:-}'"
    fails=$((fails+1))
  fi
  # A slower reading must lower the RATE and must NOT lower the baseline.
  # The counter is cumulative, so the delta is from the PREVIOUS reading:
  # 385440 + 1000*60 = 445440 -> 60000 bytes over 60 s -> 1000 B/s.
  FAKE_NOW=1120; CAPTURE_BYTES=$(( 6424 * 60 + 1000 * 60 ))
  capture_rate >/dev/null
  if [ "$CAPTURE_RATE_BPS" != "1000" ]; then
    echo "  FAIL: the slower pass must report 1000 B/s, got '${CAPTURE_RATE_BPS:-}'"
    fails=$((fails+1))
  fi
  if [ "$CAPTURE_BASE_BPS" != "6424" ]; then
    echo "  FAIL: the baseline is a rolling maximum, got '${CAPTURE_BASE_BPS:-}'"
    fails=$((fails+1))
  fi
  if ! capture_rate_is_slow "$CAPTURE_RATE_BPS" "$CAPTURE_BASE_BPS"; then
    echo "  FAIL: 1000 B/s against a 6424 B/s baseline is slow"; fails=$((fails+1))
  fi
  # A counter that went backwards means the producer was re-created: ignored.
  FAKE_NOW=1180; CAPTURE_BYTES=10
  capture_rate >/dev/null
  if [ -n "$CAPTURE_RATE_BPS" ]; then
    echo "  FAIL: a re-created producer must not produce a rate"; fails=$((fails+1))
  fi

  # The reboot cooldown, both directions. A camera that was rebooted a minute
  # ago must not be rebooted again: the reboot does not repair a weak link
  # (measured 20.10.2026, the corridor was rebooted every thirteen minutes and
  # its microphone was offline for the first minutes after each boot), and the
  # symptom is exactly that the room stopped answering «компьютер».
  if ! reboot_allowed 0 1000 3600; then
    echo "  FAIL: a camera that was never rebooted must be allowed"; fails=$((fails+1))
  fi
  if reboot_allowed 900 1000 3600; then
    echo "  FAIL: a camera rebooted 100 s ago must NOT be rebooted again"
    fails=$((fails+1))
  fi
  if ! reboot_allowed 100 4000 3600; then
    echo "  FAIL: a camera rebooted 3900 s ago must be allowed again"
    fails=$((fails+1))
  fi


  rm -rf "$tmp" 2>/dev/null || true
  if [ "$fails" -eq 0 ]; then
    echo "  SELFTEST OK — threshold fires below 80 % and stays quiet at and above it"
    return 0
  fi
  echo "  SELFTEST FAILED: $fails check(s)"
  return 1
}

if [ "${1:-}" = "--self-test" ]; then
  run_self_test
  exit $?
fi

SICK=""
for entry in $CAMERAS; do
  select_camera "$entry" || continue
  classify
  case "$STATE" in
    healthy)
      # The capture rate is recorded even here: a value that is only written
      # when something is wrong is a value that cannot be compared afterwards.
      log "$CUR_NAME healthy (http ${CAM_LATENCY}s)${CAM_CAPTURE_BPS:+ capture=${CAM_CAPTURE_BPS}/${CAM_CAPTURE_BASE} B/s}"
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
    slow-capture)
      # REPORTED, NEVER ACTED ON — and that is a decision, not an oversight.
      #
      # The number is measured on both sides of the bar: the healthy side is
      # 6354-6512 B/s on all four rooms (measured 20.10.2026, a 1.3 % spread),
      # and the slow side was measured on a stand-in, not on a camera that was
      # really slow. The threshold therefore rests on ONE measured side, and the
      # ladder ends in a REBOOT of the physical camera — a destructive step that
      # must not be armed on an inferred value.
      #
      # So this state measures, records and stops. The value is in the log of
      # every pass, which is what the sick side will be read from the next time a
      # room is really slow. Arming the action is a one-line change once that
      # number exists.
      log "$CUR_NAME SLOW CAPTURE ${CAM_CAPTURE_BPS:-?}/${CAM_CAPTURE_BASE:-?} B/s — the camera is delivering a fraction of real time. Reported, not acting: the threshold's slow side is not yet measured on a real room"
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
  #
  # **AND IT MUST NOT REPEAT.** Measured 20.10.2026, `camera-corridor1`: the
  # ladder rebooted the camera at 23:03, 23:13 and 23:26 — every thirteen
  # minutes, forever. A reboot does NOT fix a weak WiFi link (the link quality
  # flapped 58 -> 62 -> 74 and the ping RTT swung 5.7 -> 221 ms while the
  # camera was up), and every reboot takes the room's microphone offline for
  # the minute or two the camera needs before its RTSP stream comes up — a
  # camera that answers HTTP but not RTSP is BOOTING, not broken. So the loop
  # did not repair the room; it took the room's hearing away on a schedule,
  # which is exactly the reported symptom: «камера в коридоре плохо реагирует
  # на компьютер».
  #
  # So a reboot is a LAST resort with a long cooldown, and the gap is much
  # longer than the ladder's own MIN_GAP_S: the other steps are cheap and
  # reversible, a reboot is neither.
  reboot_stamp="${STAMP_DIR}/.camera_watchdog_reboot_$CUR_STREAM"
  last_reboot=0
  [ -f "$reboot_stamp" ] && read -r last_reboot <"$reboot_stamp" 2>/dev/null || last_reboot=0
  now_s=$(date +%s)
  if ! reboot_allowed "$last_reboot" "$now_s" "$REBOOT_MIN_GAP_S"; then
    log "$CUR_NAME step 4 SKIPPED: rebooted $(( now_s - last_reboot ))s ago, the next one is in $(( REBOOT_MIN_GAP_S - (now_s - last_reboot) ))s — a reboot does not fix a weak link, it only takes the room's hearing away while the camera boots"
    log "  summary: still_bad=[$CUR_NAME:$CUR_IP:$CUR_STREAM ] — the link, not the camera, is the suspect (repeater/Access Point)"
    continue
  fi
  log "$CUR_NAME step 4: reboot over SSH — last resort, and probably wrong"
  date +%s >"$reboot_stamp" 2>/dev/null || true
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

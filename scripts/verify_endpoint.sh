#!/usr/bin/env bash
# Full verification before enabling level-based endpointing in production.
#
# Run from the repository root:  bash scripts/verify_endpoint.sh
#
# Deploys nothing and changes no configuration. It builds a test image, runs the
# suite in it, checks syntax, and prints the exact commands to enable the flag.

REPO=/mnt/media/docker-compose/ai-prod/voice_gateway
STACK=/mnt/media/docker-compose/ai-prod
IMG=voice_gateway:verify
CTR=vg-verify
FAIL=0

step() { echo; echo "=== $* ==="; }
ok()   { echo "  OK: $*"; }
bad()  { echo "  FAIL: $*"; FAIL=1; }

# 1/7 runs before anything is built: it needs no container and fails fastest.
# `--self-test` first, always — a check that has never been seen failing is the
# failure mode this repository has already paid for several times over.
step "1/7 language: prose is English, Cyrillic is data"
cd "$REPO" || exit 1
if python3 scripts/check_lang.py --self-test >/dev/null 2>&1 \
   && python3 scripts/check_lang.py; then
  ok "check_lang"
else
  bad "check_lang — Russian prose outside quoted data (see above)"
fi
if python3 scripts/check_doc_headings.py --self-test >/dev/null 2>&1 \
   && python3 scripts/check_doc_headings.py; then
  ok "check_doc_headings"
else
  bad "check_doc_headings — spliced headings in AGENTS.md"
fi
# The one number the filesystem cannot answer is the test COUNT, and step 6/7
# has already run the suite by the time this would matter. So it is checked there
# instead, from the pytest output, rather than starting a second run here.

cd "$STACK" || exit 1
step "2/7 build the test image"
if docker build -q -t "$IMG" voice_gateway >/dev/null; then ok "image built"
else bad "building the image"; exit 1; fi

docker rm -f "$CTR" >/dev/null 2>&1
docker run -d --name "$CTR" "$IMG" sleep infinity >/dev/null
docker exec "$CTR" pip install -q pytest httpx pytest-asyncio 2>&1 | grep -viE "notice|warning|^$"
ok "container ready"

step "3/7 lay the sources out"
docker exec "$CTR" mkdir -p /tmp/vg_src
for f in audio_utils.py backends.py camera_client.py engine.py main.py telegram_client.py vosk_wake.py; do
  docker cp "$REPO/$f" "$CTR:/tmp/vg_src/$f" >/dev/null
done
docker cp "$REPO/services"  "$CTR:/tmp/vg_src/services"  >/dev/null
docker cp "$REPO/templates" "$CTR:/tmp/vg_src/templates" >/dev/null
docker cp "$REPO/tests"    "$CTR:/tmp/vg_src/tests"     >/dev/null
ok "sources inside $CTR"

step "4/7 syntax of the changed files"
if docker exec -w /tmp/vg_src "$CTR" python3 -m py_compile \
     camera_client.py main.py \
     services/jev-router/app.py services/jev-router/ha_client.py \
     services/jev-router/resolver.py \
     tests/test_vad_endpoint.py tests/test_camera_client.py; then
  ok "py_compile passed"
else
  bad "py_compile"
fi

step "5/7 the endpoint detector and the audio-cycle changes"
R4=$(docker exec -w /tmp/vg_src "$CTR" python3 -m pytest tests/test_vad_endpoint.py tests/test_camera_client.py -q --tb=short 2>&1 | tail -8)
echo "$R4"
if echo "$R4" | grep -qE "(^|[ ,])failed"; then
  bad "the new tests have failures — see above"
else
  ok "the new tests are green"
fi

step "6/7 the full suite"
R5=$(docker exec -w /tmp/vg_src "$CTR" python3 -m pytest tests -q --tb=line 2>&1 | tail -6)
echo "$R5"
# Gate on CORRECTNESS, not on a count I typed from memory. A hardcoded total
# was wrong twice (421 -> 437 -> 445) and each time it produced a FAIL that had
# nothing to do with the code. "Zero failures" is the invariant; the count is
# printed for information only.
if echo "$R5" | grep -qE "(^|[ ,])failed"; then
  bad "the suite has failures — see above for which test"
elif echo "$R5" | grep -qE "error"; then
  bad "the suite has collection errors"
else
  ok "no failures"
  echo "$R5" | tail -1
  # The README's counts are typed by hand at the end of a task and decayed
  # silently for two releases before anything verified them. The suite size is
  # taken from THIS run — no second pytest invocation.
  TESTS=$(echo "$R5" | grep -oE '[0-9]+ passed' | tail -1 | grep -oE '[0-9]+')
  if [ -n "$TESTS" ] && cd "$REPO" \
     && python3 scripts/check_doc_numbers.py --tests "$TESTS"; then
    ok "check_doc_numbers ($TESTS tests)"
  else
    bad "check_doc_numbers — README counts do not match the repository"
  fi
  cd "$STACK" || exit 1
fi

step "7/7 the flag defaults OFF and the detector counts correctly"
# -i is REQUIRED: without it docker exec does not forward stdin, the
# heredoc never reaches python, it reads empty input and exits 0 — and the
# check reports success without having checked anything.
docker exec -i -w /tmp/vg_src "$CTR" python3 - <<'PY'
from camera_client import CameraConfig, _PauseEndpoint

assert CameraConfig(stream_name="livingroom").pause_endpoint is False, "flag must default OFF"
print("  default flag            :", CameraConfig(stream_name="livingroom").pause_endpoint)

ep = _PauseEndpoint()
speech = [0.020, 0.024, 0.022, 0.026, 0.023, 0.025, 0.021, 0.024]
pause  = [0.008, 0.006, 0.007, 0.005, 0.006, 0.004, 0.005, 0.003]
v = [ep.feed(x)[0] for x in speech]
i = None
for n, x in enumerate(pause):
    verdict = ep.feed(x)[0]
    if verdict == "end":
        i = n
        break
print("  ends on pause frame     :", i, "(expected 5)")
assert i == 5, "endpoint fired on the wrong pause frame"

# silence must not count as speech (the windowed-reference regression)
ep3 = _PauseEndpoint()
[ep3.feed(x) for x in [0.020, 0.024, 0.023, 0.026]]
before = ep3._speech_frames
[ep3.feed(x) for x in [0.008] * 30]
assert ep3._speech_frames == before, (
    "silence counted as speech: %d -> %d" % (before, ep3._speech_frames)
)
print("  silence is not speech   : yes")

ep2 = _PauseEndpoint()
[ep2.feed(x) for x in [0.050] * 20]
quiet = [ep2.feed(x)[0] for x in [0.030] * 30]
print("  quieter than floor->end?:", "end" in quiet, "(expected False)")
assert "end" not in quiet, "endpoint fired with speech below the background"
print("  OK")
PY
[ $? -ne 0 ] && FAIL=1

step "RESULT"
if [ "$FAIL" -eq 0 ]; then
  echo "  ALL GREEN. Next, one step at a time — the flag is still OFF:"
  echo
  echo "  1) confirm it is off in production too:"
  echo "     docker exec voice_gateway python3 -c \"import sys;sys.path.insert(0,'/app');import camera_client;print(camera_client.CameraConfig('x').pause_endpoint)\""
  echo "     expected False"
  echo
  echo "  2) take a baseline (as things stand now):"
  echo "     docker logs voice_gateway --since 2h | grep 'UTTERANCE END' | tail -20"
  echo "     all of them 'via cap 7s' — that is the current 7 seconds"
  echo
  echo "  3) enable it on the LIVING ROOM ONLY and restart:"
  echo "     # in docker-compose.yml, next to WAKE_VOSK_MODEL_LIVINGROOM:"
  echo "     - CAMERA_PAUSE_ENDPOINT_LIVINGROOM=true"
  echo "     docker compose up -d --no-deps --build --force-recreate voice_gateway"
  echo
  echo "  4) say 3-5 commands in the room and look:"
  echo "     docker logs voice_gateway --since 5m | grep 'UTTERANCE END'"
  echo "     expect: via pause (rms=... ref=... run=... speech=...)"
  echo "     and:    awake ends: {'pause': N, 'cap': M}  with M NOT growing"
  echo
  echo "  5) if Whisper starts cutting phrases, turn the threshold in"
  echo "     _PauseEndpoint (ratio=0.55) using the logged ratio/rms, and repeat"
  echo "     from step 3."
  echo "     If EVERYTHING is still capped, the reference level does not catch"
  echo "     the dip while the television is on — that is a result, not a reason"
  echo "     to keep tuning."
else
  echo "  THERE ARE ERRORS — see FAIL above. Do NOT enable the flag in production."
fi

docker rm -f "$CTR" >/dev/null 2>&1
exit "$FAIL"
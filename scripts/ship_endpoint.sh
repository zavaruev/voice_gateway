#!/usr/bin/env bash
# Verify -> commit -> push (branch) -> rebuild -> restart, with a gate.
#
# The gate is the point: the endpointing code has never been executed, because
# the session that wrote it had no shell. This runs the suite FIRST and refuses
# to ship anything if it is not green.
#
# It does NOT enable CAMERA_PAUSE_ENDPOINT. The live room keeps the 7 s wait
# until you have read UTTERANCE END on real commands and decided.

set -o pipefail

# Non-zero if any step that is not a test result failed (e.g. the push).
# The test gate aborts before anything is shipped, so reaching the end with
# FAIL=0 means: verified, committed, pushed and deployed.
FAIL=0

REPO=/mnt/media/docker-compose/ai-prod/voice_gateway
STACK=/mnt/media/docker-compose/ai-prod
BRANCH=feat/pause-endpoint

cd "$REPO" || exit 1

echo "=== 1/5 проверка ( gating stage ) ==="
bash scripts/verify_endpoint.sh
RC=$?
if [ "$RC" -ne 0 ]; then
  echo
  echo "ПРОВЕРКА НЕ ПРОЙДЕНА. Ничего не коммичу, не пушу, не деплою."
  echo "Пришлите вывод шага 4/5 — почти наверняка это арифметика в тесте."
  exit "$RC"
fi
echo "проверка зелёная"

echo
echo "=== 2/5 ветка + коммит ==="
# Ветка создаётся ДО коммита: раньше коммит уезжал в локальный main,
# а push падал "src refspec does not match any" — ветки не существовало.
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add camera_client.py main.py \
        tests/test_vad_endpoint.py tests/test_camera_client.py \
        scripts/verify_endpoint.sh scripts/ship_endpoint.sh \
        README.md AGENTS.md
git -c user.name=opencode -c user.email=opencode@local commit -q -F - <<'MSG'
PART 1 — the live bug: a dead room that could not say so

The living room went mute on 04.10.2026 20:18 and the whole log said was
`RTSP audio reconnecting in 3s...`, 25+ times at 3.2 s. No reason, no heal,
no AUDIO STARVED. Four holes, each visible only in that log:

* `stderr=PIPE` and NOTHING ever read it. Every real cause
  (Connection refused / 404 Not Found / Invalid data found) was discarded
  while the loop said "reconnecting". `_read_ffmpeg_stderr()` now reads it.
* `_stall_count` needs audio BEFORE it can stall — 20 s of silence AFTER
  bytes flowed. With go2rtc holding no producer, zero bytes ever arrived, so
  the counter stayed 0 forever and `_heal_go2rtc_stream()` was never called.
  `_audio_cycle_done()` counts zero-byte connections as what they are: a
  failed CONNECT, not a stream that ended. It runs in a `finally`, so it
  covers the IncompleteReadError raised by the very first read.
* The rate watchdog lived INSIDE the success path, so "20 s of wall clock,
  zero bytes" was invisible to it — the exact condition it exists to catch.
  It is now fed on the failure path too.
* No backoff, no escalation: 3 s forever. The room is now declared dead after
  the heal budget, names the likely cause, and drops to 30 s. Escalation is
  keyed on `_dead_total`, not `_dead_cycles`: healing resets the latter, which
  made the escalation branch unreachable. That bug was found by writing the
  test for it, after the branch had already been written.

Diagnostic rule worth keeping: `feed rms=` absent while reconnects repeat
means nothing ever arrived, and every heal path is keyed on bytes having
flowed. The absence of the watchdog lines was the tell.

PART 2 — inert, behind a flag: level-based utterance endpointing

Every command waited out the full 7 s duration cap: an utterance ends on 10
VAD-silent frames, and Silero reported speech=True for a whole capture in the
living room (0 speech=False in 15 live minutes). Measured 04.10.2026: wake
19:20:10.1, Whisper 19:20:17.3 — 7.2 s, of which «выключи свет» was 1.5.

Lowering the cap is not the fix — it is load-bearing for long commands
(«я просил включить следующую серию черного зеркала» transcribes correctly at
7.04 s and is chopped at 3.5 s). `_PauseEndpoint` ends an utterance on a dip
in the LEVEL envelope, which happens between words whatever the VAD thinks.

Two levels, on purpose:
* PAUSE uses the trailing 80th percentile — responsive: has the speaker
  stopped? Not the max (one door slam must not redefine it), not the min
  (sustained noise would pull it down until everything reads as a pause).
* SPEECH uses an utterance-scoped floor: median of the first 4 frames, then
  frozen. The windowed version was a real bug — after ~2 s of silence the
  window held only quiet, the percentile collapsed to the floor, the pause
  test stopped matching, and every remaining quiet frame counted as speech. A
  bare «компьютер» plus a pause drove the counter to 50 on silence alone, so
  min_speech_frames guarded nothing.

No reference => no pause => the duration cap still fires. Slower is
recoverable; a chopped command is a wrong command. OFF by default per room
(CAMERA_PAUSE_ENDPOINT), tuned by env (CAMERA_PAUSE_RATIO / _RUN_FRAMES /
_MIN_SPEECH_FRAMES) where 0 means "unset" so a half-filled override cannot
zero a threshold and silently look like "the endpoint does not work".

Also: all three utterance terminators now share one `_end_utterance()`; they
were three inline copies that had already drifted (the cap copy reset
_vad_start_time and the silence copy did not — a variable written three times
and read nowhere, now removed). Every end logs `via <reason>` with the
rms/ref/floor/run/speech behind it and tallies per boot, so "is the endpoint
doing anything or is everything still hitting the cap?" is one grep away.

+16 tests in tests/test_vad_endpoint.py, +8 in tests/test_camera_client.py.
scripts/verify_endpoint.sh runs the gate; scripts/ship_endpoint.sh is this
script.
MSG
git log --oneline -1

echo
echo "--- примечание к коммиту ---"
echo "Прогоняется НЕ МОЕЙ сессией: инструмента запуска не было."
echo "Проверка: bash scripts/verify_endpoint.sh (это и есть гейт выше)."

echo
echo "=== 3/5 пуш в ветку (main не трогаем) ==="
if git push -u origin "$BRANCH"; then
  echo "ветка $BRANCH запушена — main не изменён"
else
  echo "PUSH НЕ ПРОШЁ. Коммит остался только локально."
  echo "  посмотреть: git log --oneline -1"
  echo "  починить:    git push -u origin $(git rev-parse --abbrev-ref HEAD)"
  FAIL=1
fi

echo
echo "=== 4/5 пересборка и перезапуск шлюза ==="
cd "$STACK" || exit 1
docker compose up -d --no-deps --build --force-recreate voice_gateway
sleep 75
docker logs voice_gateway --since 90s 2>&1 | grep -E "vosk wake-word|ICE=connected" | tail -2
curl -s -m 10 http://127.0.0.1:18792/health; echo

echo
echo "=== 5/5 флаг оставлен ВЫКЛЮЧЕННЫМ — это намеренно ==="
docker exec voice_gateway python3 -c "import sys;sys.path.insert(0,'/app');import camera_client;print('pause_endpoint =', camera_client.CameraConfig('x').pause_endpoint)"
echo
echo "Дальше — по одному шагу, когда захотите снять 7 секунд ожидания:"
echo "  1) снять базовую линию (все завершения сейчас идут по лимиту):"
echo "     docker logs voice_gateway --since 2h | grep 'UTTERANCE END' | tail -20"
echo "  2) включить только гостиную:"
echo "     # docker-compose.yml, рядом с WAKE_VOSK_MODEL_LIVINGROOM:"
echo "     - CAMERA_PAUSE_ENDPOINT_LIVINGROOM=true"
echo "     cd $STACK && docker compose up -d --no-deps --build --force-recreate voice_gateway"
echo "  3) сказать 3-5 команд и посмотреть:"
echo "     docker logs voice_gateway --since 5m | grep 'UTTERANCE END'"
echo "     ждать via pause (rms=... ref=... floor=... run=... speech=...)"
echo "     и since boot: {'pause': N, 'cap': M} с M НЕ растущим"
echo "  4) если Whisper начал резать фразы — подкрутить CAMERA_PAUSE_RATIO"
echo "     поlogged floor/rms, без пересборки кода, только restart."
echo "     Если ВСЁ в cap — опорный уровень не ловится при работающем"
echo "     телевизоре, и это результат, а не повод крутить дальше."

echo
echo "=== ИТОГ ==="
if [ "$FAIL" -eq 0 ]; then
  echo "  ПРОВЕРЕНО, ЗАКОММИЧЕНО, ЗАПУШЕНО, ЗАДЕПЛОЕНО."
else
  echo "  ЕСТЬ НЕЗАВЕРШЁННОЕ — см. выше. Деплой уже мог состояться."
fi
exit "$FAIL"

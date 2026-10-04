#!/usr/bin/env bash
# Полная проверка перед тем, как включить level-based endpointing в бою.
#
# Запускать из корня репозитория:  bash scripts/verify_endpoint.sh
#
# Ничего не деплоит и не меняет конфигурацию. Только собирает тестовый образ,
# гоняет на��ор, проверяет синтаксис и печатает точные команды включения.

REPO=/mnt/media/docker-compose/ai-prod/voice_gateway
STACK=/mnt/media/docker-compose/ai-prod
IMG=voice_gateway:verify
CTR=vg-verify
FAIL=0

step() { echo; echo "=== $* ==="; }
ok()   { echo "  OK: $*"; }
bad()  { echo "  FAIL: $*"; FAIL=1; }

step "1/6 сборка тестового образа"
cd "$STACK" || exit 1
if docker build -q -t "$IMG" voice_gateway >/dev/null; then ok "образ собран"
else bad "сборка образа"; exit 1; fi

docker rm -f "$CTR" >/dev/null 2>&1
docker run -d --name "$CTR" "$IMG" sleep infinity >/dev/null
docker exec "$CTR" pip install -q pytest httpx pytest-asyncio 2>&1 | grep -viE "notice|warning|^$"
ok "контейнер готов"

step "2/6 раскладка исходников"
docker exec "$CTR" mkdir -p /tmp/vg_src
for f in audio_utils.py backends.py camera_client.py engine.py main.py telegram_client.py vosk_wake.py; do
  docker cp "$REPO/$f" "$CTR:/tmp/vg_src/$f" >/dev/null
done
docker cp "$REPO/services"  "$CTR:/tmp/vg_src/services"  >/dev/null
docker cp "$REPO/templates" "$CTR:/tmp/vg_src/templates" >/dev/null
docker cp "$REPO/tests"    "$CTR:/tmp/vg_src/tests"     >/dev/null
ok "исходники внутри $CTR"

step "3/6 синтаксис изменённых файлов"
if docker exec -w /tmp/vg_src "$CTR" python3 -m py_compile \
     camera_client.py main.py \
     services/jev-router/app.py services/jev-router/ha_client.py \
     services/jev-router/resolver.py \
     tests/test_vad_endpoint.py tests/test_camera_client.py; then
  ok "py_compile прошёл"
else
  bad "py_compile"
fi

step "4/6 детектор endpointing и правки аудио-цикла"
R4=$(docker exec -w /tmp/vg_src "$CTR" python3 -m pytest tests/test_vad_endpoint.py tests/test_camera_client.py -q --tb=short 2>&1 | tail -8)
echo "$R4"
if echo "$R4" | grep -qE "(^|[ ,])failed"; then
  bad "в новых тестах есть падения — смотри выше"
else
  ok "новые тесты зелёные"
fi

step "5/6 полный набор"
R5=$(docker exec -w /tmp/vg_src "$CTR" python3 -m pytest tests -q --tb=line 2>&1 | tail -6)
echo "$R5"
# Gate on CORRECTNESS, not on a count I typed from memory. A hardcoded total
# was wrong twice (421 -> 437 -> 445) and each time it produced a FAIL that had
# nothing to do with the code. "Zero failures" is the invariant; the count is
# printed for information only.
if echo "$R5" | grep -qE "(^|[ ,])failed"; then
  bad "в наборе есть падения — смотри выше, какой тест"
elif echo "$R5" | grep -qE "error"; then
  bad "в наборе есть ошибки сбора"
else
  ok "падений нет"
  echo "$R5" | tail -1
fi

step "6/6 флаг по умолчанию ВЫКЛЮЧЕН и детектор считает верно"
# -i is REQUIRED: without it docker exec does not forward stdin, the
# heredoc never reaches python, it reads empty input and exits 0 — and the
# check reports success without having checked anything.
docker exec -i -w /tmp/vg_src "$CTR" python3 - <<'PY'
from camera_client import CameraConfig, _PauseEndpoint

assert CameraConfig(stream_name="livingroom").pause_endpoint is False, "flag must default OFF"
print("  флаг по умолчанию        :", CameraConfig(stream_name="livingroom").pause_endpoint)

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
print("  конец на кадре паузы    :", i, "(ожидается 5)")
assert i == 5, "endpoint fired on the wrong pause frame"

# silence must not count as speech (the windowed-reference regression)
ep3 = _PauseEndpoint()
[ep3.feed(x) for x in [0.020, 0.024, 0.023, 0.026]]
before = ep3._speech_frames
[ep3.feed(x) for x in [0.008] * 30]
assert ep3._speech_frames == before, (
    "silence counted as speech: %d -> %d" % (before, ep3._speech_frames)
)
print("  тишина не считается речью: да")

ep2 = _PauseEndpoint()
[ep2.feed(x) for x in [0.050] * 20]
quiet = [ep2.feed(x)[0] for x in [0.030] * 30]
print("  тише фона -> конец?      :", "end" in quiet, "(ожидается False)")
assert "end" not in quiet, "endpoint fired with speech below the background"
print("  OK")
PY
[ $? -ne 0 ] && FAIL=1

step "ИТОГ"
if [ "$FAIL" -eq 0 ]; then
  echo "  ВСЁ ЗЕЛЁНОЕ. Дальше — по одному шагу, флаг всё ещё ВЫКЛЮЧЕН:"
  echo
  echo "  1) убедиться, что в бою тоже выключено:"
  echo "     docker exec voice_gateway python3 -c \"import sys;sys.path.insert(0,'/app');import camera_client;print(camera_client.CameraConfig('x').pause_endpoint)\""
  echo "     ожидается False"
  echo
  echo "  2) снять базовую линию (как сейчас):"
  echo "     docker logs voice_gateway --since 2h | grep 'UTTERANCE END' | tail -20"
  echo "     все с 'via cap 7s' — это и есть нынешние 7 секунд"
  echo
  echo "  3) включить ТОЛЬКО на гостиной и перезапустить:"
  echo "     # в docker-compose.yml, рядом с WAKE_VOSK_MODEL_LIVINGROOM:"
  echo "     - CAMERA_PAUSE_ENDPOINT_LIVINGROOM=true"
  echo "     docker compose up -d --no-deps --build --force-recreate voice_gateway"
  echo
  echo "  4) сказать в комнате 3-5 команд и посмотреть:"
  echo "     docker logs voice_gateway --since 5m | grep 'UTTERANCE END'"
  echo "     ждать: via pause (rms=... ref=... run=... speech=...)"
  echo "     и      since boot: {'pause': N, 'cap': M}  с M НЕ растущим"
  echo
  echo "  5) если Whisper начал резать фразы — по logged ratio/rms подкрутить"
  echo "     порог в _PauseEndpoint (ratio=0.55) и повторить с шага 3."
  echo "     Если ВСЁ в cap — опорный уровень не ловится при работающем телевизоре,"
  echo "     и это результат, а не повод крутить дальше."
else
  echo "  ЕСТЬ ОШИБКИ — см. FAIL выше. НЕ включайте флаг в бою."
fi

docker rm -f "$CTR" >/dev/null 2>&1
exit "$FAIL"

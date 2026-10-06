#!/usr/bin/env bash
# One-shot field status: everything to read BEFORE concluding a fault.
#
# Run from the stack root:  bash voice_gateway/scripts/field_status.sh
#
# Read-only. Touches nothing, changes nothing, deploys nothing. It exists because
# the answer to "what happened at HH:MM" is spread over four places — the gateway
# log, the decoder's lifetime counters, the watchdog log and Home Assistant's own
# state — and reading them out of order has produced wrong conclusions repeatedly:
#
#   * `VAD SPEECH START` read as proof of speech when it fires every 6.55 s on the
#     room's own ambient, because this room's VAD never reports silence.
#   * `drop[muted=…]` read as a fault when it is normal during playback.
#   * HTTP latency read as a health signal at all (it is one; see AGENTS.md).
#
# What each section answers:
#   события        — did a wake fire, was one refused, was a command dropped
#   счётчики       — where the audio can die: gate (suppressed), mic (muted),
#                    echo (cross-correlation), and whether the decoder ever fired
#   близкие        — the decoder HEARD the word and the spelling allowlist threw
#                    it away; this is the only line that separates that from a
#                    model that did not hear the word, and it is the first thing to
#                    read after a silent wake
#   реле в HA      — the side effect, read from Home Assistant rather than inferred
#                    from what the gateway said it did

set -u

STACK="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$STACK" || exit 1

date '+сейчас %H:%M:%S MSK'
echo
echo "--- комнаты ---"
docker logs voice_gateway --since 40m 2>&1 \
  | grep -E "CameraSession started" | tail -2 | cut -c1-100 | sed 's/^/  /'
echo
echo "--- счётчики декодера ---"
for R in livingroom kitchen; do
  printf '  %-11s ' "$R"
  docker logs voice_gateway --since 40m 2>&1 \
    | grep "\[$R\] vosk diag" | tail -1 \
    | grep -oP "chunks=\S+ suppressed=\S+ peak_max=\S+ triggers=\S+ decodes=\S+|drop\[[^]]*\]" \
    | tr '\n' ' '
  echo
done
echo
echo "--- события: успех / отказ / потерянная команда ---"
docker logs voice_gateway --since 40m 2>&1 \
  | grep -E "VOSK WAKE|wake heard but not fired|Ignoring '|Wake-only|kept open|attention pip skipped" \
  | cut -c1-165 | sed 's/^/  /'
echo
echo "--- близкие попадания декодера (первое, что читать после тихого пробуждения) ---"
NEAR=$(docker logs voice_gateway 2>&1 | grep -icE "NEAR MISS")
echo "  всего: $NEAR"
docker logs voice_gateway 2>&1 | grep -iE "NEAR MISS" | tail -5 | cut -c1-170 | sed 's/^/  /'
echo
echo "--- watchdog ---"
tail -4 /tmp/camera_watchdog.log 2>/dev/null | sed 's/^/  /'
echo
echo "--- реле света в HA (сторона эффекта, а не что шлюз сказал) ---"
HA_URL=$(grep -oP '^\s+- HA_URL=\K.*' docker-compose.yml | head -1)
HA_TOKEN=$(grep -oP '^\s+- HA_TOKEN=\K.*' docker-compose.yml | head -1)
if [ -n "$HA_URL" ] && [ -n "$HA_TOKEN" ]; then
  curl -s -m 10 -H "Authorization: Bearer $HA_TOKEN" "$HA_URL/api/states" | python3 -c "
import json, sys
try:
    states = json.load(sys.stdin)
except Exception as exc:
    print('  не прочитал states:', exc); raise SystemExit(0)
for s in states:
    e = s['entity_id']
    if e.startswith('switch.') and e.endswith('_relay') and 'light' in e:
        print('  %-46s %s' % (e, s['state']))
"
else
  echo "  HA_URL/HA_TOKEN не найдены в docker-compose.yml"
fi

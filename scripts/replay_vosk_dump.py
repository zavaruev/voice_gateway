"""Прогон боевого пути подачи на реальном куске, который видел декодер.

Именно этот порядок важен и повторяет `_vad_process`: `begin()` один раз,
`feed()` чанками по 2560 байт, БЕЗ `flush()` и без границ высказывания.
Внешняя запись отвечает на другой вопрос: она упирается в рельс 32768,
тогда как шлюз получает неискажённый сигнал (AGENTS.md, 14.10.2026).
"""
import sys
import time

from vosk_wake import VoskWakeMatcher

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/replay.raw"
MODEL = "config/vosk-model-small-ru-0.22"
SR, CHUNK = 16000, 2560

raw = open(PATH, "rb").read()
print("  кусок: %.1f с, модель %s" % (len(raw) / 2 / SR, MODEL))

m = VoskWakeMatcher(MODEL)
m.begin()
fired = []
t0 = time.time()
for i in range(0, len(raw) - CHUNK, CHUNK):
    if m.feed(raw[i : i + CHUNK]):
        fired.append(i / 2 / SR)
print("  прогон занял %.1f с" % (time.time() - t0))
print("  СРАБАТЫВАНИЙ: %d" % len(fired))
for t in fired:
    print("    t=%6.1f с" % t)
nm = m.near_misses
print("  near_misses: %s" % (nm if nm else "— ничего похожего не декодировано"))
print("  chunks: %s   triggers: %s   decodes: %s" % (
    getattr(m, "chunks", "?"), getattr(m, "triggers", "?"), getattr(m, "decodes", "?")))


"""Replay the production feed path over real bytes the decoder actually saw.

The ORDER is the point, and it reproduces `_vad_process`: `begin()` once,
`feed()` in 2560-byte chunks, NO `flush()`, no utterance boundaries.

An external recording answers a different question: it clips at the 32768 rail,
while the gateway received an undistorted signal over the same seconds
(AGENTS.md, 14.10.2026). Two different signals, so every offline reproduction
built on the capture is invalid by construction.
"""
import sys
import time

from vosk_wake import VoskWakeMatcher

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/replay.raw"
MODEL = "config/vosk-model-small-ru-0.22"
SR, CHUNK = 16000, 2560

raw = open(PATH, "rb").read()
print("  clip: %.1f s, model %s" % (len(raw) / 2 / SR, MODEL))

m = VoskWakeMatcher(MODEL)
m.begin()
fired = []
t0 = time.time()
for i in range(0, len(raw) - CHUNK, CHUNK):
    if m.feed(raw[i : i + CHUNK]):
        fired.append(i / 2 / SR)
print("  replay took %.1f s" % (time.time() - t0))
print("  TRIGGERS: %d" % len(fired))
for t in fired:
    print("    t=%6.1f s" % t)
nm = m.near_misses
print("  near_misses: %s" % (nm if nm else "— nothing remotely like it decoded"))
print("  chunks: %s   triggers: %s   decodes: %s" % (
    getattr(m, "chunks", "?"), getattr(m, "triggers", "?"), getattr(m, "decodes", "?")))
"""Streaming STT with stable-prefix commitment (LocalAgreement-2).

Borrowed from ufal/whisper_streaming (MIT, the LocalAgreement-n policy): a
prefix is COMMITTED only when two consecutive overlapping windows agree on it.
A committed prefix is stable evidence that somebody said those words — unlike
a full-buffer Whisper result, which is available only after the utterance has
already ended, and unlike a level threshold, which cannot separate a voice
from a television in these rooms.

This module is deliberately PURE about I/O: `transcribe` is injected, so the
policy can be replayed over recorded wav files and unit-tested without a
microphone, a camera or the Whisper service.

THE HARD RULE: nothing is ever dispatched from a hypothesis. Only the
committed prefix leaves this module. A window that disagrees with the previous
one rolls the prefix BACK to the longest common prefix — Whisper rewriting the
end of a word must never become a spoken command.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Awaitable, Callable

logger = logging.getLogger("streaming_stt")

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


class LocalAgreement:
    """Commit the prefix agreed on by `n` consecutive hypotheses.

    `n=2` is the setting used by whisper_streaming's LocalAgreement-2. On a
    disagreement the stored hypothesis rolls back to the common prefix, so a
    late correction cannot leak into a committed command.
    """

    def __init__(self, n: int = 2):
        self.n = max(2, int(n))
        self.hypotheses: list[list[str]] = []
        self.committed: list[str] = []

    def add(self, text: str) -> tuple[str, bool]:
        """Add one window hypothesis; return (committed text, changed?)."""
        h = words(text)
        self.hypotheses.append(h)
        if len(self.hypotheses) > self.n:
            self.hypotheses.pop(0)
        if len(self.hypotheses) < self.n:
            return self.text, False

        # Longest prefix agreed on by ALL of the last n hypotheses.
        common = self.hypotheses[0]
        for other in self.hypotheses[1:]:
            common = common_prefix(common, other)
        changed = common != self.committed
        self.committed = common
        return self.text, changed

    @property
    def text(self) -> str:
        return " ".join(self.committed)

    def reset(self) -> None:
        self.hypotheses.clear()
        self.committed = []


Transcriber = Callable[[bytes, float], Awaitable[str]]
"""async (wav_bytes, rms_raw) -> transcript text"""


class StreamingTranscriber:
    """Feed overlapping raw PCM windows and expose the stable prefix.

    `feed(pcm_tail, rms_raw)` must be awaited; it calls the injected
    transcriber. Windows are kept bounded (`window_s`) and the cursor advances
    by `step_s` of newly arrived audio, so each word is seen by at least two
    windows before it can be committed — that is the overlap LocalAgreement
    needs.
    """

    def __init__(
        self,
        transcribe: Transcriber,
        *,
        sample_rate: int = 16000,
        window_s: float = 2.0,
        step_s: float = 0.5,
        n: int = 2,
    ):
        if step_s <= 0 or window_s < step_s * n:
            raise ValueError("window_s must cover at least n steps")
        self.transcribe = transcribe
        self.bytes_per_s = sample_rate * 2
        self.window_bytes = int(window_s * self.bytes_per_s)
        self.step_bytes = int(step_s * self.bytes_per_s)
        self.la = LocalAgreement(n=n)
        self._seen = 0                 # total bytes ever fed
        self._next_at = 0              # byte position for the next window
        self._buf = bytearray()
        self.last_latency_s = 0.0
        self.windows = 0
        self.last_changed_at = 0.0     # wall clock of the last prefix growth

    def reset(self) -> None:
        self.la.reset()
        self._seen = 0
        self._next_at = 0
        self._buf.clear()
        self.windows = 0
        self.last_changed_at = 0.0

    @property
    def committed(self) -> str:
        return self.la.text

    async def feed(self, pcm: bytes, rms_raw: float, now: float) -> bool:
        """Add newly arrived audio. Returns True when the committed prefix changed."""
        self._buf.extend(pcm)
        self._seen += len(pcm)
        if self._seen - self._next_at < self.step_bytes:
            return False
        self._next_at = self._seen
        if len(self._buf) < self.window_bytes:
            return False
        window = bytes(self._buf[-self.window_bytes :])
        t0 = time.monotonic()
        text = await self.transcribe(window, rms_raw)
        self.last_latency_s = time.monotonic() - t0
        self.windows += 1
        _, changed_now = self.la.add(text)
        if changed_now:
            self.last_changed_at = now
        logger.debug(
            f"streaming stt window={self.windows} text={text[:60]!r} "
            f"committed={self.committed!r} latency={self.last_latency_s:.2f}s"
        )
        del self._buf[: max(0, len(self._buf) - self.window_bytes)]
        return changed_now

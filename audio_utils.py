"""
Shared audio utilities: Ogg Opus packing, text validation.

Extracted from main.py to be shared between main.py and camera_client.py
without circular imports or code duplication.

Two independent concerns live here:
  1. pack_ogg() — turns raw Opus packets into a standards-compliant Ogg
     stream (what a browser <audio>/media player needs for playback of a
     recorded turn).
  2. is_valid_text() + the hallucination tables — a gate between STT and
     everything downstream (TTS, LLM, Home Assistant actions). Whisper on
     silence/echo invents phrases like «спасибо за просмотр»; speaking or
     acting on those is worse than dropping the turn, so the filter is
     deliberately aggressive (false negatives are acceptable, false
     positives — an accidental device action — are not).

Keep this module dependency-free (stdlib only): it is imported by both
the FastAPI process and camera_client, and tests import it on the host.
"""

import re
import struct
import time
import zlib


# ═══════════════════════════════════════════════════════
# Ogg Opus packer
# ═══════════════════════════════════════════════════════


# Precomputed bit-reversal of every byte: zlib.crc32() and Python's
# int.from_bytes() disagree on bit order (CRC-32 as used by Ogg is defined
# over reflected bytes), so both ends of the CRC are byte-swapped with this
# table instead of an expensive per-bit loop. Built once at import.
_BIT_REVERSE_TABLE = bytes(int("{:08b}".format(i)[::-1], 2) for i in range(256))


def pack_ogg(frames: list, sample_rate=16000) -> bytes:
    """Wrap a list of raw Opus packets into one complete Ogg stream.

    Layout: page 0 = OpusHead (BOS), page 1 = OpusTags (comments), then
    data pages with up to 50 packets each; the last page carries EOS.
    granule position counts 48 kHz samples (Opus always runs at 48 kHz
    internally regardless of `sample_rate`) and one packet is 60 ms ->
    48000 * 0.06 = 2880 samples. `ser` (serial) is a per-file random-ish
    id so a player can tell two concatenated streams apart; starting it
    at the current unix time is enough for local playback.
    """
    ser = int(time.time()) & 0xFFFFFFFF

    def ogg_crc(data: bytes) -> int:
        """Ogg's CRC-32 (reflected variant): bit-reverse before and after
        the standard zlib.crc32 and invert the final mask, per the Ogg
        container spec (RFC-implemented in libogg, no stdlib equivalent)."""
        crc = zlib.crc32(data.translate(_BIT_REVERSE_TABLE), 0xFFFFFFFF) ^ 0xFFFFFFFF
        return int.from_bytes(crc.to_bytes(4, "little").translate(_BIT_REVERSE_TABLE), "big")

    def page(idx: int, gran: int, ser: int, bos: bool, eos: bool, pkts: list) -> bytes:
        """Build one Ogg page: 27-byte header (OggS, version 0, header
        type flags BOS=2/EOS=4, granule, serial, seqno, CRC placeholder)
        + segment table + payload with the CRC field zeroed, then patched
        in place (p[:22] + crc + p[26:]).

        Gotcha: the segment table writes ONE lacing byte per packet, so
        every packet must be < 256 bytes — fine for the 20 ms/60 ms Opus
        frames produced here, but a >255-byte packet would silently
        produce a corrupt page.
        """
        h = struct.pack(
            "<4sBBqIIIB",
            b"OggS",
            0,
            (2 if bos else 0) | (4 if eos else 0),
            gran,
            ser,
            idx,
            0,
            len(pkts),
        )
        p = h + bytearray(len(x) for x in pkts) + b"".join(pkts)
        crc = ogg_crc(p)
        return p[:22] + struct.pack("<I", crc) + p[26:]

    res = page(
        0, 0, ser, True, False,
        [struct.pack("<8sBBHIHB", b"OpusHead", 1, 1, 312, sample_rate, 0, 0)],
    )
    res += page(
        1, 0, ser, False, False,
        [struct.pack("<8sI8sI", b"OpusTags", 8, b"VoiceGW ", 0)],
    )
    for i in range(0, len(frames), 50):
        c = frames[i : i + 50]
        res += page(
            2 + i // 50,
            (i + len(c)) * int(48000 * 0.06),
            ser,
            False,
            (i + 50 >= len(frames)),
            c,
        )
    return res


# ═══════════════════════════════════════════════════════
# Hallucination filters & text validation
# ═══════════════════════════════════════════════════════

# Whisper's known filler phrases on silence/noise (subtitles/YouTube
# clichés, Russian and English). Matched as substrings anywhere in the
# text — they never appear in a genuine voice-assistant request, so a
# substring match costs nothing and catches the "прощение, спасибо за
# просмотр" class of hallucinations.
WHISPER_HALLUCINATIONS = [
    "субтитры подогнал симон",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "аминь",
    "субтитры создавал",
    "редактор субтитров",
    "thank you",
    "thanks for watching",
    "so",
    "dimatorzok",
    "субтитры сделал",
    "dima torzok",
    "продолжение следует",
    "синкинг",
]

WHISPER_HALLUCINATIONS_PATTERN = re.compile(
    "|".join(re.escape(bad) for bad in WHISPER_HALLUCINATIONS)
)

SINGLE_WORD_HALLUCINATIONS = {
    "о", "а", "и", "как", "кх-кх", "ха-ха", "жизнь", "пьютер",
}

# TTS echo feedback loops: camera hears its own response via the mic,
# Whisper hears the mangled echo as short repeated syllables like "а а".
SHORT_ECHO_PATTERNS = re.compile(
    r"^(а|аа|ааа|и|ии|у|уу|э|о|гм|хм|ага|аха)\s*(?:-\s*)?$"
)


def is_valid_text(txt: str) -> bool:
    """True if an STT result is worth acting on / speaking aloud.

    Ordered cheapest-first, every branch rejects one class of Whisper or
    echo hallucination (documented inline). Deliberately biased toward
    false REJECTS: dropping a garbled utterance only loses one turn,
    while a false accept can speak nonsense or trigger a device action.
    The thresholds (0.4 character run ratio, min word length 3, >= 3
    repetitions) came from field recordings of echo feedback loops —
    retune them only with new samples in hand.
    """
    clean = txt.strip(" .,?!-").lower()
    if not clean:
        return False

    # Reject text with long runs of repeated characters (e.g. "кккккккк...")
    if len(clean) >= 5:
        max_run, cur = 1, 1
        for i in range(1, len(clean)):
            cur = cur + 1 if clean[i] == clean[i - 1] else 1
            max_run = max(max_run, cur)
        if max_run / len(clean) > 0.4:
            return False

    # Reject text composed of repeated short substrings (e.g. "каккаккак", "каккак", "атататат")
    for sub_len in (1, 2, 3, 4):
        if len(clean) >= sub_len * 2:
            for start in range(min(sub_len, len(clean) - sub_len + 1)):
                sub = clean[start:start + sub_len]
                reps = len(clean) // len(sub)
                if reps >= 2 and clean == sub * reps:
                    return False

    words = clean.split()
    if len(words) == 1:
        # «о», «как», «жизнь» etc. are Whisper's inventions on silence,
        # and a <= 2 char word can never be a real request from this STT.
        if words[0] in SINGLE_WORD_HALLUCINATIONS:
            return False
        if len(words[0]) <= 2:
            return False

    # Reject TTS echo feedback ("а а", "ага", "аха")
    if SHORT_ECHO_PATTERNS.search(clean):
        return False

    # Reject repeated single words (e.g. "а а а а а", "да да да")
    if len(words) >= 3:
        unique_words = set(words)
        if len(unique_words) == 1 and len(words[0]) <= 4:
            return False

    # Reject 1-2 word texts made only of short vowel sounds ("и и", "э э", "о")
    if len(words) <= 2:
        all_vowels = True
        for w in words:
            if not re.fullmatch(r"[аеиоуэыяю]{1,3}", w):
                all_vowels = False
                break
        if all_vowels:
            return False

    # Reject dash-separated repeated syllables (e.g. "бе-бе-бе-бе-бе")
    stripped = clean.replace(" ", "").replace("-", "")
    if len(stripped) >= 6:
        for sub_len in (1, 2, 3):
            if len(stripped) >= sub_len * 3:
                for start in range(min(sub_len, len(stripped) - sub_len + 1)):
                    sub = stripped[start:start + sub_len]
                    reps = len(stripped) // len(sub)
                    if reps >= 3 and stripped == sub * reps:
                        return False

    if WHISPER_HALLUCINATIONS_PATTERN.search(clean):
        return False
    return True

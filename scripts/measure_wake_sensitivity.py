"""How well does each room's microphone actually hear the wake word?

Run it against the LIVE stack, then run it again after changing whatever makes
noise in a room, and compare the per-band SNR. It exists because of a question
that no log line could answer and that two wrong answers were already given:

  * "make it louder"  — refuted: +6 dB stops the LIVING room from firing too.
  * "filter the hum"   — refuted: a 4th-order high-pass at 80/120/160/200/300 Hz
                         changes the kitchen from not firing to not firing, at
                         every cutoff.
  * peak               — refuted: the kitchen never fires at any level from +0 to
                         −18 dB, because level does not change SNR.

What does move is the ratio of the wake word to the room's own noise **in the
band the word lives in** — «компьютер» needs /k/, /pʲ/ and /tʃ/, all of it in
1-4 kHz. Measured 06.10.2026: +15.4 dB in the living room, +4.4 dB in the
kitchen, and the decoder resolved the word cleanly in one room and as «как
театр» in the other.

Usage
-----
    docker exec voice_gateway python3 scripts/measure_wake_sensitivity.py

    # one room only, or a different phrase
    docker exec voice_gateway python3 scripts/measure_wake_sensitivity.py \\
        --rooms kitchen --phrase "Компьютер, выключи свет."

Needs the admin credentials for POST /api/camera/tts (it answers 401 without
them) and ffmpeg inside the container. It PLAYS the phrase through each camera's
speaker at the normal reply level, so run it when the rooms are quiet and do not
run it at night.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

SR = 16000
CHUNK = 2560  # 160 ms, the hop the pipeline runs at
BANDS = (
    (50, 200, "гул 50-200"),
    (200, 1000, "200-1к"),
    (1000, 4000, "1-4к речь"),
    (4000, 8000, "4-8к"),
)
TTS_URL = os.getenv("TTS_URL", "http://192.168.22.102:5050/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")
GATEWAY = os.getenv("GATEWAY_URL", "http://127.0.0.1:18792")
USER = os.getenv("ADMIN_USERNAME", "voiceadmin")
PASSWORD = os.getenv("ADMIN_PASSWORD", "")
PLAY_USER = os.getenv("CAMERA_PLAY_AUDIO_USER", "root")
PLAY_PASSWORD = os.getenv("CAMERA_PLAY_AUDIO_PASSWORD", "2441")
DEFAULT_ROOMS = ("livingroom", "kitchen")


def rooms_from_streams():
    """Room name -> camera IP, from CAMERA_PLAY_AUDIO_URL_<ROOM>."""
    out = {}
    for name in (x.strip() for x in os.getenv("CAMERA_STREAMS", "").split(",") if x.strip()):
        url = os.getenv(f"CAMERA_PLAY_AUDIO_URL_{name.upper()}", "")
        if "/play_audio" not in url:
            continue
        host = url.split("//", 1)[-1].split("/", 1)[0]
        out[name] = host.split(":")[0]
    return out


def say_through(room: str, phrase: str) -> bool:
    body = json.dumps({"stream_name": room, "text": phrase})
    cmd = ["curl", "-s", "-m", "60", "-u", f"{USER}:{PASSWORD}",
           "-X", "POST", f"{GATEWAY}/api/camera/tts",
           "-H", "Content-Type: application/json", "-d", body,
           "-o", "/dev/null", "-w", "%{http_code}"]
    code = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    if code != "200":
        print(f"  ! {room}: шлюз ответил {code} — проверьте ADMIN_PASSWORD", file=sys.stderr)
        return False
    return True


def capture(ip: str, seconds: int, path: str) -> bytes:
    """Direct from the camera, so nothing the gateway does is in the way."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(seconds),
         "-i", f"rtsp://{PLAY_USER}:{PLAY_PASSWORD}@{ip}/stream=0",
         "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", path, "-y"],
        capture_output=True,
    )
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return b""


def chunk_peaks(a: np.ndarray, n: int = CHUNK) -> np.ndarray:
    usable = len(a) - (len(a) % n)
    blocks = a[:usable].reshape(-1, n)
    return np.sqrt(np.mean(np.square(blocks.astype(np.float64)), axis=1))


def band_energy(x1d: np.ndarray, lo: int, hi: int) -> float:
    from scipy import signal
    f, _t, S = signal.stft(x1d, SR, nperseg=512)
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.abs(S[m]).sum()))


def decode(raw: bytes):
    """Exactly the production path: begin() once, feed() chunk by chunk, no flush."""
    from vosk import SetLogLevel
    SetLogLevel(-1)
    sys.path.insert(0, "/app")
    from vosk_wake import VoskWakeMatcher

    m = VoskWakeMatcher("config/vosk-model-small-ru-0.22")
    m.begin()
    fired, partials = None, []
    for i in range(0, len(raw) - CHUNK * 2 + 1, CHUNK * 2):
        c = raw[i:i + CHUNK * 2]
        if m.feed(c):
            fired = i / 32000.0
        if m.last_partial and (not partials or partials[-1][1] != m.last_partial):
            partials.append((i / 32000.0, m.last_partial))
    return fired, partials, m


def measure(room: str, ip: str, phrase: str, seconds: int, tmpdir: str) -> dict:
    path = os.path.join(tmpdir, f"loop_{room}.raw")
    # Capture in the background so the room's reverberation tail is in there too.
    cap = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(seconds),
         "-i", f"rtsp://{PLAY_USER}:{PLAY_PASSWORD}@{ip}/stream=0",
         "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", path, "-y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    ok = say_through(room, phrase)
    cap.wait()
    if not ok:
        return {}

    raw = open(path, "rb").read()
    if not raw:
        print(f"  {room}: камера не отдала звук")
        return {}
    a = np.frombuffer(raw, dtype=np.int16)

    r = chunk_peaks(a)
    bg = float(np.median(r))
    loud = np.where(r > bg * 2.0)[0]
    if not len(loud):
        print(f"  {room}: фраза не выделилась над фоном (фон rms {bg:.0f})")
        return {}

    n = CHUNK
    blocks = a[: len(a) - (len(a) % n)].reshape(-1, n).astype(np.float64)
    s, e = loud[0], loud[-1]
    seg = blocks[s:e + 1].ravel()
    noise = blocks[: max(s - 3, 1)].ravel()

    snr = {}
    for lo, hi, tag in BANDS:
        sb, nb = band_energy(seg, lo, hi), band_energy(noise, lo, hi)
        snr[tag] = 20 * np.log10(sb / max(nb, 1e-9))

    fired, partials, m = decode(raw)
    return {
        "room": room,
        "seconds": len(a) / SR,
        "bg_rms": bg,
        "phrase_rms": float(r.max()),
        "snr": snr,
        "fired_at": fired,
        "triggers": m.triggers,
        "text": m.last_text,
        "partials": partials,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--phrase", default="Компьютер, включи свет в гостиной.")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1,
                    help="how many times to play and decode per room; "
                         "detection is a coin flip at some SNRs")
    args = ap.parse_args()

    ips = rooms_from_streams()
    if not ips:
        print("нет комнат: проверьте CAMERA_STREAMS и CAMERA_PLAY_AUDIO_URL_<ROOM>",
              file=sys.stderr)
        return 2
    rooms = args.rooms or [r for r in DEFAULT_ROOMS if r in ips] or list(ips)
    unknown = [r for r in rooms if r not in ips]
    if unknown:
        print(f"неизвестные комнаты: {unknown}; есть {list(ips)}", file=sys.stderr)
        return 2

    print(f"  фраза: {args.phrase!r}   повторов: {args.repeat}")
    hits = {r: 0 for r in rooms}
    snrs = {r: [] for r in rooms}
    with tempfile.TemporaryDirectory() as tmpdir:
        for k in range(args.repeat):
            for room in rooms:
                res = measure(room, ips[room], args.phrase, args.seconds, tmpdir)
                if not res:
                    continue
                if res["fired_at"] is not None:
                    hits[room] += 1
                snrs[room].append(res["snr"]["1-4к речь"])
                # Every run, not just the first: a single play of a phrase
                # through a speaker into its own microphone varies by several dB,
                # and one reading is exactly the kind of number this script exists
                # to replace.
                print(f"  {room:<11} прогон {k + 1}: фон rms {res['bg_rms']:>5.0f}  "
                      f"фраза x{res['phrase_rms'] / max(res['bg_rms'], 1):>5.1f}  "
                      f"SNR речи {res['snr']['1-4к речь']:+6.1f} дБ  "
                      f"декодер: {'СРАБОТАЛ ' + repr(res['text']) if res['fired_at'] is not None else 'промах'}")
                if k == 0:
                    for tag in [b[2] for b in BANDS]:
                        print(f"      все полосы {tag:<12} {res['snr'][tag]:+6.1f} дБ")
                    for t, p in res["partials"][:6]:
                        print(f"        гипотеза {t:.1f}с {p!r}")

    print("\n  === итог ===")
    for room in rooms:
        v = snrs[room]
        avg = sum(v) / len(v) if v else float("nan")
        print(f"  {room:<11} сработало {hits[room]}/{args.repeat}   "
              f"SNR речи 1-4 кГц {avg:+.1f} дБ")
    print("\n  Сравните с прошлым прогоном ДО и ПОСЛЕ того, что вы выключили."
          "\n  Порог, который здесь ищется, — между комнатами, а не абсолютный.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

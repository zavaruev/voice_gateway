"""Record the livingroom mic and cut speech bursts into individual WAVs.

Zero manual marking: the room's background sits at peak ~1400 and speech at
14000-24000, so an energy gate separates them by an order of magnitude.
Intended run from the throwaway vg-tr container:

    python3 record_bursts.py <out_dir> <seconds> [rtsp_url]
"""
import os
import subprocess
import sys
import wave

import numpy as np

SR = 16000
CHUNK = 1600  # 100 ms


def capture(url, seconds, path):
    """Pull raw mono s16le from go2rtc until the ffmpeg process exits."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp",
        "-i", url, "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
        "-f", "s16le", "-t", str(seconds), path,
    ]
    subprocess.run(cmd, check=True)
    return np.frombuffer(open(path, "rb").read(), dtype=np.int16)


def bursts(x, gate_frac=0.28, min_chunks=3, gap_chunks=4):
    """Indices of energy chunks above gate, grouped into utterances.

    The gate is relative to the file's own median so it works at whatever
    `audio.volume` the camera happens to be set to, and p99 guards against a
    single transient (a door slam) opening a burst on its own.
    """
    n = len(x) // CHUNK
    rms = np.array([
        float(np.sqrt(np.mean(x[i*CHUNK:(i+1)*CHUNK].astype(np.float64) ** 2)))
        for i in range(n)
    ])
    gate = max(np.median(rms) * gate_frac, np.percentile(rms, 90))
    hot = rms > gate
    groups, cur = [], []
    for i, h in enumerate(hot):
        if h:
            cur.append(i)
        elif cur and i - cur[-1] > gap_chunks:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return [(g, rms, gate) for g in groups if len(g) >= min_chunks]


def save(x, g, out_dir, idx, pad=3):
    """Write one utterance with a little lead-in/out, trimmed to its own peak."""
    lo = max(0, g[0] - pad) * CHUNK
    hi = min(len(x), (g[-1] + pad + 1) * CHUNK)
    seg = x[lo:hi]
    pk = int(np.max(np.abs(seg.astype(np.int32))))
    if pk == 0:
        return None
    # leave headroom: a clipped training sample teaches the model nothing
    seg = (seg.astype(np.float32) * min(0.9 * 32767 / pk, 3.0)).clip(
        -32768, 32767).astype(np.int16)
    path = os.path.join(out_dir, f"burst_{idx:03d}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(seg.tobytes())
    return path, len(seg) / SR, pk


def main():
    out_dir = sys.argv[1]
    # Second arg is either a capture length in seconds or an existing raw
    # s16le file to segment (handy for re-cutting a long take without
    # re-recording, and for a still-running capture snapshot).
    arg = sys.argv[2]
    url = sys.argv[3] if len(sys.argv) > 3 else \
        "rtsp://192.168.22.102:8554/livingroom?audio=copy"
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(arg):
        x = np.frombuffer(open(arg, "rb").read(), dtype=np.int16)
        print(f"читаю {arg}: {len(x)/SR:.1f}s")
    else:
        raw = os.path.join(out_dir, "raw16k.raw")
        x = capture(url, int(arg), raw)
        print(f"записано {len(x)/SR:.1f}s")

    found = bursts(x)
    print(f"найдено одиночных высказываний: {len(found)}")
    kept = []
    for i, (g, rms, gate) in enumerate(found):
        res = save(x, g, out_dir, i)
        if res is None:
            continue
        path, dur, pk = res
        kept.append(path)
        print(f"  {os.path.basename(path)}  {dur:5.2f}s  peak={pk:6d}")
    print(f"порог отсечки rms={gate:.0f}, всего сохранено {len(kept)}")


if __name__ == "__main__":
    main()

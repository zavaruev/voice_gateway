"""Does vosk actually solve the wake-word problem here, and what does it cost?

The openWakeWord route failed three times for one reason: 27 recordings of the
word is not enough to separate it from speech at the same level as the user.
A decoder has no such ceiling — it recognises WORD, so a false activation is
structurally impossible. That claim needs measuring before wiring anything into
the camera loop, and so does the price, because the whole point of dropping
Whisper is CPU.

Three numbers matter:
  1. hit rate on the 36 Whisper-confirmed «компьютер» bursts, clean and at
     several SNRs against the real TV capture;
  2. false accepts on the held-out TV tail — how often a 2.5 s window of
     television decodes as «комп...». This is the number that decides whether
     the thing is usable at all;
  3. wall-clock per attempt, because it runs on the audio path.

Usage (inside the image):
    python3 scripts/probe_vosk_wake.py --model /tmp/vosk-model-small-ru-0.22 \\
        --bursts /data/bursts --labels /data/labels.tsv \\
        --ambient /data/ambience.raw
"""
import argparse
import json
import os
import sys
import time

import numpy as np

RNG = np.random.default_rng(1234)

CHUNK = 3200          # 200 ms — a good vosk feed granularity
WIN_S = 2.5           # how much audio one attempt may consume


def hit(text):
    """Tolerant «компьютер» match on the decoded transcript.

    vosk spells the word inconsistently on short/noisy audio ("комп", "компютер",
    "компьытер"), so a plain equality test would understate the hit rate and make
    vosk look worse than it is. Prefix match on a token, like wake_features.
    """
    t = text.lower().replace("ё", "е")
    return any(w.startswith("комп") for w in t.split())


def decode(model, audio, use_partial=True):
    """Run one recogniser over `audio`; return (text, latency_s).

    `use_partial` reads vosk's interim hypothesis after every chunk and stops at
    the first keyword hit, which is what the live path needs for latency. The
    final result is still read so the transcript is complete either way.
    """
    import vosk
    rec = vosk.KaldiRecognizer(model, 16000)
    text, latency = "", 0.0
    for i in range(0, len(audio), CHUNK):
        blk = audio[i:i + CHUNK]
        if rec.AcceptWaveform(blk.tobytes()):
            text = json.loads(rec.Result()).get("text", "")
            if use_partial and hit(text):
                break
        else:
            if use_partial:
                part = json.loads(rec.PartialResult()).get("partial", "")
                if hit(part):
                    text = part
                    break
    if not text:
        text = json.loads(rec.FinalResult()).get("text", "")
    return text, latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bursts", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--ambient", default="")
    ap.add_argument("--holdout-fraction", type=float, default=0.30)
    ap.add_argument("--snrs", default="30,20,9,5")
    args = ap.parse_args()

    import vosk
    t0 = time.time()
    model = vosk.Model(args.model)
    print(f"модель загружена за {time.time() - t0:.1f}s\n")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wake_features import (FeatureExtractor, is_positive, load_wav,
                               read_labels, runtime_chunks)
    from build_wake_dataset import mix_at_snr

    fx = FeatureExtractor()

    def as_runtime(x):
        """Concatenate the chunks the runtime would actually feed (AGC'd)."""
        return np.concatenate([c for c, _ in runtime_chunks(x)])

    def as_raw(x):
        """The same span, WITHOUT the per-chunk AGC.

        openWakeWord needs peak-normalised chunks, but a decoder does not: every
        independent 200 ms gain jump is distortion to it. vosk should therefore
        get the raw signal, and that is worth measuring rather than assuming.
        """
        return x.astype(np.int16)

    amb = (np.frombuffer(open(args.ambient, "rb").read(), dtype=np.int16)
           if args.ambient else None)

    # ---- 1. hit rate on the real bursts
    if args.bursts and args.labels:
        rows = [(n, t) for n, t in read_labels(args.labels)
                if os.path.exists(os.path.join(args.bursts, n))]
        pos = [(n, t) for n, t in rows if is_positive(t)]
        print("=" * 70)
        print("1. ПОПАДАНИЯ НА 36 ПОДТВЕРЖДЁННЫХ BURST-АХ")
        print("=" * 70)
        for feed_name, feed in (("AGC (как openWakeWord)", as_runtime),
                                ("сырой (как декодеру)", as_raw)):
            print(f" -- подача на вход: {feed_name} --")
            for snr in [None] + [float(s) for s in args.snrs.split(",")]:
                hits, seen, miss = 0, 0, []
                for n, _ in pos:
                    x = load_wav(os.path.join(args.bursts, n))
                    if snr is not None and amb is not None:
                        o = int(RNG.integers(0, max(len(amb) - 30000, 1)))
                        bg = amb[o:o + 30000]
                        x = mix_at_snr(x, bg, snr)
                    audio = feed(x)
                    if len(audio) == 0:
                        continue
                    seen += 1
                    txt, _ = decode(model, audio)
                    if hit(txt):
                        hits += 1
                    elif len(miss) < 3:
                        miss.append(txt[:38])
                lbl = "чисто" if snr is None else f"SNR {snr:g} дБ"
                print(f"    {lbl:10s} {hits}/{seen} ({100*hits/max(seen,1):3.0f}%)"
                      + (f"  промахи: {miss}" if miss else ""))

        # negative speech: vosk must NOT decode the wake word out of it
        neg = [(n, t) for n, t in rows if not is_positive(t)]
        fp, seen = 0, 0
        for n, _ in neg:
            x = load_wav(os.path.join(args.bursts, n))
            audio = as_runtime(x)
            if len(audio) == 0:
                continue
            seen += 1
            txt, _ = decode(model, audio)
            if hit(txt):
                fp += 1
        print(f" речь без слова: ложных {fp}/{seen}")
    else:
        pos = neg = []
        print("(burst-ы не переданы — пропускаю проверку попаданий)")

    # ---- 2. false accepts on held-out television
    if amb is not None:
        cut = int(len(amb) * (1 - args.holdout_fraction))
        train_part, hold = amb[:cut], amb[cut:]
        print()
        print("=" * 70)
        print(f"2. ЛОЖНЫЕ СРАБАТЫВАНИЯ НА ТВ ({len(hold)/16000:.0f}s хвоста, "
              f"которого модель не видела)")
        print("=" * 70)
        win = int(WIN_S * 16000)
        step = win // 2
        for name, part in (("ОБУЧАЮЩАЯ часть ТВ", train_part),
                           ("НЕВИДИМЫЙ хвост ТВ", hold)):
            audio = np.concatenate([c for c, _ in runtime_chunks(part)])
            n_windows = max(0, (len(audio) - win) // step + 1)
            fp, texts = 0, []
            t0 = time.time()
            for i in range(n_windows):
                txt, _ = decode(model, audio[i * step:i * step + win])
                if hit(txt):
                    fp += 1
                    if len(texts) < 4:
                        texts.append(txt[:44])
            dt = time.time() - t0
            hrs = len(part) / 16000 / 3600
            per_s = fp / n_windows if n_windows else 0
            print(f" {name:18s} окон {n_windows:5d}  ложных {fp:3d} "
                  f"({per_s*100:5.1f}% окон, {per_s*n_windows/hrs:6.1f}/час)")
            if texts:
                print(f"                   примеры: {texts}")
            print(f"                   {dt/max(n_windows,1)*1000:.0f} мс на окно, "
                  f"{len(part)/16000/hrs:.0f} окон/час при переборе всего ТВ "
                  f"-> {per_s * dt/max(n_windows,1) * len(part)/16000/hrs * 100:.1f}% "
                  f"ядра")

    # ---- 3. cost
    print()
    print("=" * 70)
    print("3. ЦЕНА ОДНОЙ ПОПЫТКИ")
    print("=" * 70)
    if amb is not None:
        audio = np.concatenate([c for c, _ in runtime_chunks(amb[:16000 * 20])])
        for secs in (1.0, 1.5, 2.5):
            w = audio[:int(secs * 16000)]
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                decode(model, w)
                ts.append(time.perf_counter() - t0)
            print(f" окно {secs:.1f} с: {min(ts)*1000:6.0f} мс "
                  f"(медиана {sorted(ts)[2]*1000:.0f} мс)")

    del fx


if __name__ == "__main__":
    main()
"""Grammar-constrained vosk vs free decoding: hits, false accepts, cost.

vosk can restrict decoding to a word list, which is the textbook way to do
keyword spotting: the decoder is forced towards «компьютер» and everything else
becomes [unk]. Measured free decoding: 94 % hits clean / 39 % at 9 dB SNR,
0 false accepts on speech, ~500 ms per 2.5 s window.

The obvious risk is the flip side of "forced": if the only words available are
«компьютер» and [unk], a television monologue can be forced onto the keyword.
That is precisely the failure mode this whole exercise exists to avoid, so both
numbers are measured, and the filler vocabulary is swept rather than assumed.

Usage (inside the image):
    python3 scripts/probe_vosk_grammar.py --model /tmp/vosk-model-small-ru-0.22 \\
        --bursts /data/bursts --labels /data/labels.tsv --ambient /data/ambience.raw
"""
import argparse
import json
import os
import sys
import time

import numpy as np

RNG = np.random.default_rng(1234)
CHUNK = 3200

WAKE = ["компьютер", "комп"]
FILLERS = ["да", "нет", "не", "это", "как", "что", "где", "когда", "пожалуйста",
           "включи", "выключи", "свет", "телевизор", "комната", "кухня",
           "привет", "окей", "ладно", "так", "ты", "мне", "тебе", "на", "и",
           "в", "я", "мы", "вы", "он", "она", "они", "уже", "ещё", "нету"]


def hit(text):
    t = text.lower().replace("ё", "е")
    return any(w.startswith("комп") for w in t.split())


def decode(model, audio, grammar=None, partial=True):
    import vosk
    rec = (vosk.KaldiRecognizer(model, 16000, grammar) if grammar
           else vosk.KaldiRecognizer(model, 16000))
    rec.SetWords(True)
    text = ""
    for i in range(0, len(audio), CHUNK):
        blk = audio[i:i + CHUNK]
        if rec.AcceptWaveform(blk.tobytes()):
            text = json.loads(rec.Result()).get("text", "")
            if partial and hit(text):
                break
        else:
            if partial:
                p = json.loads(rec.PartialResult()).get("partial", "")
                if hit(p):
                    text = p
                    break
    if not text:
        text = json.loads(rec.FinalResult()).get("text", "")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bursts", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--ambient", default="")
    ap.add_argument("--holdout-fraction", type=float, default=0.30)
    ap.add_argument("--snrs", default="30,20,14,9,5")
    args = ap.parse_args()

    import vosk
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wake_features import is_positive, load_wav, read_labels
    from build_wake_dataset import mix_at_snr

    t0 = time.time()
    model = vosk.Model(args.model)
    print(f"модель за {time.time()-t0:.1f}s\n")

    variants = [
        ("свободное декодирование", None),
        ("словарь: только wake", json.dumps(WAKE + ["[unk]"],
                                            ensure_ascii=False)),
        ("словарь: wake + 32 слова",
         json.dumps(WAKE + FILLERS + ["[unk]"], ensure_ascii=False)),
    ]

    amb = (np.frombuffer(open(args.ambient, "rb").read(), dtype=np.int16)
           if args.ambient else None)
    rows = ([(n, t) for n, t in read_labels(args.labels)
             if os.path.exists(os.path.join(args.bursts, n))]
            if args.labels else [])
    pos = [os.path.join(args.bursts, n) for n, t in rows if is_positive(t)]
    neg = [os.path.join(args.bursts, n) for n, t in rows if not is_positive(t)]

    for name, grammar in variants:
        print("=" * 72)
        print(name)
        print("=" * 72)
        line = "  попадания:"
        for snr in [None] + [float(s) for s in args.snrs.split(",")]:
            hits = seen = 0
            for p in pos:
                x = load_wav(p)
                if snr is not None and amb is not None:
                    o = int(RNG.integers(0, max(len(amb) - 30000, 1)))
                    x = mix_at_snr(x, amb[o:o + 30000], snr)
                seen += 1
                if hit(decode(model, x, grammar)):
                    hits += 1
            lbl = "чисто" if snr is None else f"{snr:g}дБ"
            line += f"  {lbl} {100*hits/max(seen,1):3.0f}%"
        print(line)

        fp = seen = 0
        for p in neg:
            seen += 1
            if hit(decode(model, load_wav(p), grammar)):
                fp += 1
        print(f"  речь без слова: ложных {fp}/{seen}")

        if amb is not None:
            cut = int(len(amb) * (1 - args.holdout_fraction))
            for part_name, part in (("ТВ (обуч.)", amb[:cut]),
                                    ("ТВ (хвост)", amb[cut:])):
                win, step = int(2.5 * 16000), int(1.25 * 16000)
                n = max(0, (len(part) - win) // step + 1)
                f = 0
                t0 = time.time()
                for i in range(n):
                    if hit(decode(model, part[i * step:i * step + win],
                                  grammar)):
                        f += 1
                dt = (time.time() - t0) / max(n, 1)
                print(f"  {part_name:11s} окон {n:4d}  ложных {f:3d} "
                      f"({100*f/max(n,1):5.1f}%)  {dt*1000:5.0f} мс/окно")
        print()


if __name__ == "__main__":
    main()
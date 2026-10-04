"""Build the wake-word training set so that LEVEL CARRIES NO CLASS INFORMATION.

WHY THIS EXISTS
    Both shipped heads failed the same way: clean CV, then they fired on live
    TV. `scripts/diagnose_wake_data.py` found the cause in the data — the
    ambience negatives were written pre-normalised to peak 4000
    (`cut_ambience_negatives.py`), while the positives stayed raw at
    1964..29490. Median rms was 976 for negatives and 2451 for positives, so a
    classifier could reach a perfect CV score by learning LOUDNESS. The word
    was decoration. Raising the threshold to 0.85 did not fix that; it just
    moved where the same shortcut landed.

THE FIX
    Every sample, in every class, goes through the identical runtime chain
    (`wake_features.runtime_chunks`): 160 ms chunks, per-chunk bidirectional AGC
    to peak 4000, chunks under peak 600 dropped. After that, absolute level is
    gone from every example and the head has to use spectral structure.

    Two more things the previous runs never did:

    * COMPOSITE POSITIVES. The user will say «компьютер» over the TV, so the
      training set must contain «компьютер» mixed with the real TV at a range
      of SNRs — otherwise the model has only ever seen the word in silence and
      treats background as an adversarial signal.
    * SNR AS A NON-DISCRIMINATIVE AXIS. Speech negatives get the same mixes at
      the same SNRs, so "noisy" cannot stand in for "negative" either.

    And a strict time-disjoint split: the TV capture is cut into train and
    holdout segments that share no samples, and whole positive BURSTS (not
    windows) are held out, so the acceptance number is about unseen speech.

Usage (inside the image):
    python3 scripts/build_wake_dataset.py \\
        --bursts /data/bursts --labels /data/labels.tsv \\
        --ambient /data/ambience.raw --out /data/ds.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wake_features import (CHUNK, N_FRAMES, PEAK_FLOOR,  # noqa: E402
                           FeatureExtractor, is_positive, load_wav,
                           read_labels, runtime_chunks)

RNG = np.random.default_rng(1234)
SEC = 16000


# --- helpers -------------------------------------------------------------

def speed_perturb(x, factor):
    """Resample by `factor` without pitch shift artifacts worth worrying about.

    A wake word has to fire when the user says it slightly faster or slower;
    36 bursts of one tempo is not enough signal to learn that, and speed is a
    far bigger lever than gain (gain is deleted by the AGC anyway).
    """
    n_out = int(len(x) / factor)
    src = np.arange(len(x))
    dst = np.arange(n_out) * factor
    y = np.interp(dst, src, x.astype(np.float32))
    return y.astype(np.int16)


def mix_at_snr(foreground, background, snr_db):
    """background scaled to `snr_db` below the foreground's RMS, then summed.

    `background` is drawn by the caller; if it is shorter than the foreground it
    is looped (TV speech is not periodic, so looping is harmless here and it
    keeps every example exactly one window long).
    """
    fg = foreground.astype(np.float32)
    rms_fg = float(np.sqrt(np.mean(fg ** 2))) + 1e-6
    need = len(fg)
    if len(background) < need:
        reps = int(np.ceil(need / max(len(background), 1)))
        bg = np.tile(background.astype(np.float32), reps)[:need]
    else:
        start = int(RNG.integers(0, len(background) - need + 1))
        bg = background[start:start + need].astype(np.float32)
    rms_bg = float(np.sqrt(np.mean(bg ** 2))) + 1e-6
    bg *= rms_fg / (rms_bg * (10 ** (snr_db / 20.0)))
    y = fg + bg
    peak = float(np.max(np.abs(y)))
    if peak > 32767:  # hard-clip only what the ADC would have clipped
        y *= 32767.0 / peak
    return y.astype(np.int16)


def windows_from(x, dur=1.4, stride=None):
    """Slice a capture into fixed-length windows on the 160 ms grid.

    `dur` must be long enough to yield one full 1.28 s feature window: the
    extractor emits nothing until N_FRAMES rows of real audio have accumulated,
    which is 8 chunks = 1.28 s, so a shorter slice produces zero rows and
    vanishes without a trace.

    Note the `i * CHUNK + n` end offset. Writing `x[i * CHUNK:(i + n)]` instead
    shrinks every slice by one chunk — 1.28 s, 1.12 s, 0.96 s ... down to empty —
    which is how a 294 s capture produced exactly ONE usable window.
    """
    n = CHUNK * int((dur * SEC) / CHUNK)
    if len(x) < n:
        return []
    step = CHUNK * int((stride * SEC) / CHUNK) if stride else CHUNK
    return [x[i * CHUNK:i * CHUNK + n]
            for i in range(0, (len(x) - n) // step + 1)]


def speech_span(x, floor=PEAK_FLOOR):
    """Locate the speech inside a burst, in samples, via the runtime's Silero VAD.

    Needed because a 1.4 s burst yields exactly ONE 1.28 s feature window, and it
    always lands at the very END of the burst — Whisper's padding sits in front
    and the word is last. 27 training bursts then become 27 examples with an
    identical shape ("silence, silence, ..., word at the end"), and the head
    learns the level jump rather than the phonemes. `frame_std` — the spread of
    magnitude across the 16 frames — separated the classes by 1.38 sigma, more
    than any spectral feature.

    So the word has to be able to sit anywhere in the window, with real room
    audio around it. That is what stream_vectors() does.
    """
    from engine import LocalAudioEngine
    if not hasattr(speech_span, "_eng"):
        speech_span._eng = LocalAudioEngine(vad_threshold=0.03)
        speech_span._eng.initialize_models("")
    eng = speech_span._eng
    on = []
    for i in range(len(x) // CHUNK):
        c = x[i * CHUNK:(i + 1) * CHUNK]
        if int(np.max(np.abs(c.astype(np.int32)))) < floor:
            continue
        if eng._ww_vad_speech(c):
            on.append((i * CHUNK, (i + 1) * CHUNK))
    if not on:
        return 0, len(x)
    return on[0][0], on[-1][1]


def stream_labels(stream, span, n, floor_ratio=0.25):
    """For each 1.28 s window in `stream`, does it contain the wake word?

    Uses the SAME overlap rule stream_vectors applies, so labels and features
    cannot disagree.
    """
    s0, s1 = span
    out = []
    for i in range(0, (len(stream) - n) // CHUNK + 1):
        w0, w1 = i * CHUNK, i * CHUNK + n
        out.append(min(w1, s1) - max(w0, s0) > n * floor_ratio)
    return out


def stream_vectors(x, span, ambient, rng, dur=1.4):
    """-> list of (vector, contains_word) cut from ONE live-looking stream.

    Built as a single pass over the stream rather than per window: each window
    costs a full AudioFeatures reset, and resetting per window turned a 10-minute
    job into half an hour.
    """
    n = CHUNK * int((dur * SEC) / CHUNK)
    # Pad must be LONGER than the burst, or `cut` clamps to zero and the word
    # always lands at the same place in every window — which is the shortcut
    # this whole function exists to break. Two windows of room audio around a
    # 1.3 s burst gives cut a real range.
    pad_n = n + max(len(x), n)
    off = int(rng.integers(0, max(len(ambient) - pad_n, 1)))
    pad = ambient[off:off + pad_n]
    if len(pad) < pad_n:
        pad = np.concatenate(
            [pad, np.tile(pad, int(np.ceil(pad_n / max(len(pad), 1))))])[:pad_n]
    cut = int(rng.integers(0, max(len(pad) - len(x), 1)))
    stream = np.concatenate([pad[:cut], x, pad[cut:]])
    span2 = (span[0] + cut, span[1] + cut)
    labels = stream_labels(stream, span2, n)

    fx = _extractor()
    fx.reset()
    out, idx = [], 0
    for ch, _pk in runtime_chunks(stream):
        f = fx.feed(ch)
        if f is None:
            continue
        out.append((np.asarray(f, dtype=np.float32).reshape(-1),
                    labels[idx] if idx < len(labels) else False))
        idx += 1
    return out


_EXTRACTOR = None


def _extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = FeatureExtractor()
    return _EXTRACTOR


# --- dataset -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ambient", nargs="+", required=True,
                    help="raw 16 kHz mono s16le captures of the room/TV")
    ap.add_argument("--out", required=True, help="npz to write")
    ap.add_argument("--holdout-fraction", type=float, default=0.25,
                    help="fraction of POSITIVE bursts withheld entirely")
    ap.add_argument("--ambient-holdout", type=float, default=0.30,
                    help="fraction of each ambient capture's TAIL withheld")
    ap.add_argument("--snrs", default="30,20,14,9,5",
                    help="dB for composite speech-with-background. The SAME "
                         "list is applied to positives and negatives on purpose: "
                         "if only the negatives are noisy, then 'has background' "
                         "becomes a perfect proxy for the label. Measured leak "
                         "with one-sided mixing: 0.76 sigma on the embedding "
                         "magnitude, i.e. clean speech was separable at a glance.")
    ap.add_argument("--draws", type=int, default=3,
                    help="different background segments per SNR; raw audio is "
                         "the scarce resource (27 training bursts), so "
                         "augmentation multiplicity is what buys sample count")
    args = ap.parse_args()

    snrs = [float(s) for s in args.snrs.split(",")]
    fx = FeatureExtractor()

    # ---- ambient: time-disjoint train / holdout
    amb_train, amb_hold = [], []
    for path in args.ambient:
        raw = np.frombuffer(open(path, "rb").read(), dtype=np.int16)
        cut = int(len(raw) * (1 - args.ambient_holdout))
        amb_train.append(raw[:cut])
        amb_hold.append(raw[cut:])
        print(f"фон {os.path.basename(path)}: {len(raw)/SEC:.0f}s -> "
              f"обучение {cut/SEC:.0f}s, приёмка {len(raw[cut:])/SEC:.0f}s")
    if not any(len(a) for a in amb_train) or not any(len(a) for a in amb_hold):
        print("ФОН СЛИШКОМ КОРОТКИЙ ДЛЯ РАЗДЕЛА — уменьшите --ambient-holdout")
        return 1

    # ---- speech: whole bursts held out, never individual windows
    pos, neg = [], []
    for name, text in read_labels(args.labels):
        p = os.path.join(args.bursts, name)
        if not os.path.exists(p):
            continue
        (pos if is_positive(text) else neg).append(p)
    pos.sort(); neg.sort()
    k = max(1, int(len(pos) * args.holdout_fraction))
    pos_tr, pos_ho = pos[:-k], pos[-k:]
    # The speech negatives need their own holdout. Re-using all 42 of them in the
    # acceptance set looked harmless and was not: the trainer reported max score
    # 0.999 on neg_holdout, i.e. plain memorisation of recordings the model had
    # trained on, reported next to recall as if it were a generalisation number.
    # Now both classes are split by BURST.
    m = max(1, int(len(neg) * args.holdout_fraction))
    neg_tr, neg_ho = neg[:-m], neg[-m:]
    print(f"позитивных burst-ов: обучение {len(pos_tr)}, приёмка {len(pos_ho)} "
          f"(отложено целиком, не окнами)")
    print(f"негативных burst-ов: обучение {len(neg_tr)}, приёмка {len(neg_ho)}")

    X, y, meta, grp = [], [], [], []

    def add(raw, label, tag, group=""):
        v = fx.vectors(raw)
        if not v:
            return 0                      # shorter than one 1.28 s window
        X.extend(v)
        y.extend([label] * len(v))
        meta.extend([tag] * len(v))
        grp.extend([group] * len(v))
        return len(v)

    def addv(vec, label, tag, group=""):
        X.append(vec)
        y.append(label)
        meta.append(tag)
        grp.append(group)
        return 1

    def ambient_pool(pool):
        return [a for a in pool if len(a) > CHUNK * 20]

    tr_pool = ambient_pool(amb_train)
    ho_pool = ambient_pool(amb_hold)
    if not tr_pool or not ho_pool:
        print("ФОН СЛИШКОМ КОРОТКИЙ ДЛЯ РАЗДЕЛА")
        return 1

    def draw(pool):
        return pool[int(RNG.integers(len(pool)))]

    print("\nгенерирую позитивы: слово в случайной позиции окна...")
    for i, path in enumerate(pos_tr, 1):
        x = load_wav(path)
        span = speech_span(x)
        for f in (1.0, 0.92, 1.08):
            xs = x if f == 1.0 else speed_perturb(x, f)
            sp = span if f == 1.0 else (int(span[0] / f), int(span[1] / f))
            for snr in snrs:
                for _ in range(args.draws):
                    bg = draw(tr_pool)
                    mixed = (mix_at_snr(xs, bg, snr) if snr < 25
                             else xs)
                    for v, has_word in stream_vectors(mixed, sp, bg, RNG):
                        addv(v, 1 if has_word else 0,
                             f"pos_s{snr:g}" if has_word else f"stream_neg_s{snr:g}",
                             group=os.path.basename(path))
        if i % 9 == 0:
            print(f"  {i}/{len(pos_tr)} burst-ов, окон {len(y)} "
                  f"(позитивов {sum(y)})")

    print("генерирую речевые негативы (та же конструкция, без слова)...")
    # NO span for negative bursts. speech_span() answers "is there speech here",
    # and on a speech burst that is true almost everywhere — using it as the
    # wake-word span marked ~60 % of the whole set as POSITIVE (neg_label_leak
    # = 12599 windows vs 7297 real positives) and taught the head that ordinary
    # speech is the wake word. For these clips the word span is empty by
    # definition: every window is a negative.
    EMPTY = (0, 0)
    for i, path in enumerate(neg_tr, 1):
        x = load_wav(path)
        for f in (1.0, 0.92, 1.08):
            xs = x if f == 1.0 else speed_perturb(x, f)
            for snr in snrs:
                for _ in range(args.draws):
                    bg = draw(tr_pool)
                    mixed = (mix_at_snr(xs, bg, snr) if snr < 25 else xs)
                    for v, _has in stream_vectors(mixed, EMPTY, bg, RNG):
                        addv(v, 0, f"neg_s{snr:g}",
                             group=os.path.basename(path))
        if i % 14 == 0:
            print(f"  {i}/{len(neg_tr)} burst-ов, окон {len(y)} "
                  f"(позитивов {sum(y)})")

    print("генерирую чистый фон (с большим шагом — приёмку делает живой гейт)...")
    n_amb = 0
    for idx, a in enumerate(tr_pool):
        for w in windows_from(a, stride=0.7):
            n_amb += add(w, 0, "amb_train", group=f"amb{idx}")
    print(f"  окон фона: {n_amb}")

    # ---- acceptance set: unseen bursts + unseen ambient tail
    print("генерирую приёмочный набор...")
    HX, Hy, Hmeta = [], [], []

    def hadd(raw, label, tag):
        v = fx.vectors(raw)
        if not v:
            return
        HX.extend(v)
        Hy.extend([label] * len(v))
        Hmeta.extend([tag] * len(v))

    def haddv(vec, label, tag):
        HX.append(vec)
        Hy.append(label)
        Hmeta.append(tag)

    for path in pos_ho:
        x = load_wav(path)
        span = speech_span(x)
        for f in (1.0, 0.92, 1.08):
            xs = x if f == 1.0 else speed_perturb(x, f)
            sp = span if f == 1.0 else (int(span[0] / f), int(span[1] / f))
            for snr in snrs:
                for _ in range(args.draws):
                    bg = draw(ho_pool)
                    mixed = (mix_at_snr(xs, bg, snr) if snr < 25 else xs)
                    for v, has_word in stream_vectors(mixed, sp, bg, RNG):
                        haddv(v, 1 if has_word else 0,
                              f"pos_holdout_s{snr:g}" if has_word
                              else f"posstream_holdout_s{snr:g}")
    for path in neg_ho:
        x = load_wav(path)
        for snr in snrs:
            for _ in range(args.draws):
                bg = draw(ho_pool)
                mixed = (mix_at_snr(x, bg, snr) if snr < 25 else x)
                # windows that DO contain this clip's speech stay NEGATIVE here:
                # a holdout speech burst is not the wake word, whatever the VAD
                # says about it
                for v, _has in stream_vectors(mixed, (0, 0), bg, RNG):
                    haddv(v, 0, f"neg_holdout_s{snr:g}")
        print(f"  приёмка: {os.path.basename(path)} добавлен")
    for a in ho_pool:
        for w in windows_from(a, stride=0.4):
            hadd(w, 0, "amb_holdout")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    grp = np.asarray(grp)
    HX = np.asarray(HX, dtype=np.float32)
    Hy = np.asarray(Hy, dtype=np.int64)
    np.savez_compressed(args.out, X=X, y=y, meta=np.asarray(meta), grp=grp,
                        HX=HX, Hy=Hy, Hmeta=np.asarray(Hmeta))

    # ---- report
    print(f"\nобучение {X.shape}, позитивов {int(y.sum())} "
          f"({100*y.mean():.1f}%), негативов {int((y==0).sum())}")
    print(f"приёмка {HX.shape}, позитивов {int(Hy.sum())}")
    print(f"групп (burst-ов): {len(set(grp.tolist()))} — CV будет по группам, "
          f"окна одного burst-а почти дубликаты")
    tags = {}
    for t in meta:
        tags[t] = tags.get(t, 0) + 1
    print("состав обучения:", dict(sorted(tags.items())))

    # The leak check that matters. After the per-chunk AGC every example peaks at
    # exactly 4000, so absolute LEVEL is gone by construction — what survives is
    # the crest factor within a chunk (peak/rms), i.e. "is this peaky speech or
    # a dense wall of TV?". That is legitimate acoustic evidence, but if the two
    # classes differ in it systematically the head will lean on it and stop
    # firing whenever the TV gets loud. So both classes are mixed over the SAME
    # SNR list, and this measures whether that worked.
    print("\n--- контроль утечки: crest factor сырых чанков ---")
    cf_pos, cf_neg = [], []
    for paths, bucket in ((pos_tr, cf_pos), (neg, cf_neg)):
        for path in paths:
            x = load_wav(path)
            for i in range(len(x) // CHUNK):
                c = x[i * CHUNK:(i + 1) * CHUNK].astype(np.float32)
                pk = float(np.max(np.abs(c)))
                if pk < PEAK_FLOOR:
                    continue
                bucket.append(pk / (float(np.sqrt(np.mean(c ** 2))) + 1e-6))
    for lbl, v in (("позитивы", cf_pos), ("негативы", cf_neg)):
        v = np.asarray(v)
        print(f" {lbl:9s} crest: медиана {np.median(v):5.2f}  "
              f"p10 {np.percentile(v,10):5.2f}  p90 {np.percentile(v,90):5.2f}")
    if cf_pos and cf_neg:
        a, b = np.median(cf_pos), np.median(cf_neg)
        allv = np.asarray(cf_pos + cf_neg)
        sep = abs(a - b) / (allv.std() + 1e-9)
        print(f" разделение медиан = {sep:.2f} sigma  "
              f"({'ОСТОРОЖНО: crest factor различает классы — модель может '
                 'предпочесть его слову' if sep > 0.35 else 'ok, не различает'})")
    print(f"\nзаписано {args.out} "
          f"({os.path.getsize(args.out)/1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Measure what the training set actually teaches, before training anything.

Run this FIRST on any new labelling round. The two shipped models both looked
perfect in cross-validation and both fired on live TV, so the lesson is that
the questions worth asking are about the DATA, not the classifier:

  1. How many positive windows are there really?  With a 1.28 s window and an
     80 ms hop, a 1.4 s burst yields TWO windows — and both are dominated by the
     silence that Whisper padded around the word. Label quality, not window
     count, sets the ceiling.
  2. What do the positives actually sound like?  If the word only ever appears
     at one distance / one level / one speed, the head learns that
     fingerprint, not the word.
  3. Do the negatives cover the confusion?  The failure was TV speech at the
     same level as the user, so negatives must contain speech, not just room
     tone.
  4. Does the model separate them WITHOUT relying on level?  If the only thing
     separating the classes is loudness, it will not survive a different TV
     volume — which is exactly what happened.

Usage (inside the image):
    python3 scripts/diagnose_wake_data.py --bursts /data/bursts \\
        --labels /data/labels.tsv --ambience /data/amb \\
        --amb-labels /data/amb_labels.tsv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wake_features import (CHUNK, N_FRAMES, PEAK_FLOOR,  # noqa: E402
                           FeatureExtractor, is_positive, load_wav, read_labels)


def burst_report(rows, bursts_dir, fx):
    print("=" * 72)
    print("1. ЧТО РЕАЛЬНО ЕСТЬ В ДАННЫХ")
    print("=" * 72)
    pos, neg = [], []
    for name, text in rows:
        p = os.path.join(bursts_dir, name)
        if os.path.exists(p):
            (pos if is_positive(text) else neg).append((p, text))
    print(f" burst-ов: {len(pos) + len(neg)}   позитивов {len(pos)}   "
          f"негативов {len(neg)}")

    tot_w = 0
    for lbl, group in (("позитивы", pos), ("негативы", neg)):
        if not group:
            continue
        durs = [len(load_wav(p)) / 16000 for p, _ in group]
        win = sum(max(0, int((d - 1.28) / 0.08) + 1) for d in durs)
        tot_w += win
        print(f" {lbl:10s} n={len(group):3d}  длины "
              f"{min(durs):.2f}-{max(durs):.2f}с (медиана "
              f"{np.median(durs):.2f})  окон 16x80мс: {win}")
    print(f" ИТОГО обучающих окон: {tot_w}")
    return pos, neg


def level_report(paths, label):
    """The fingerprint check: how much do levels vary inside one class?"""
    print(f"\n--- уровни: {label} ---")
    pk, rms = [], []
    for p in paths:
        x = load_wav(p)
        for i in range(len(x) // CHUNK):
            c = x[i * CHUNK:(i + 1) * CHUNK]
            peak = int(np.max(np.abs(c.astype(np.int32))))
            if peak < PEAK_FLOOR:
                continue
            pk.append(peak)
            rms.append(float(np.sqrt((c.astype(np.float64) ** 2).mean())))
    if not pk:
        print(" нет чанков выше порога")
        return None
    pk, rms = np.asarray(pk, float), np.asarray(rms, float)
    print(f" peak: медиана {np.median(pk):.0f}  p10 {np.percentile(pk,10):.0f}  "
          f"p90 {np.percentile(pk,90):.0f}  макс {pk.max():.0f}")
    print(f" rms : медиана {np.median(rms):.0f}  p10 {np.percentile(rms,10):.0f}  "
          f"p90 {np.percentile(rms,90):.0f}")
    print(f" разброс peak p90/p10 = {np.percentile(pk,90)/max(np.percentile(pk,10),1):.1f}x"
          f"   (узкий разброс = модель запомнит громкость, а не слово)")
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ambience", default="")
    ap.add_argument("--amb-labels", default="")
    args = ap.parse_args()

    rows = read_labels(args.labels)
    fx = FeatureExtractor()

    pos, neg = burst_report(rows, args.bursts, fx)

    print()
    print("=" * 72)
    print("2. ОТПЕЧАТОК ПОЗИТИВОВ (риск запомнить обстановку, а не слово)")
    print("=" * 72)
    pk_pos = level_report([p for p, _ in pos], "позитивы")
    pk_neg = level_report([p for p, _ in neg], "негативы (речь)")

    if args.ambience and os.path.isdir(args.ambience):
        amb = []
        for name, text in (read_labels(args.amb_labels)
                           if args.amb_labels else
                           [(n, "") for n in sorted(os.listdir(args.ambience))]):
            p = os.path.join(args.ambience, name)
            if os.path.exists(p) and not is_positive(text):
                amb.append(p)
        print()
        print(f"--- уровни: фон ({len(amb)} окон) ---")
        pk_amb = level_report(amb[:200], "фон")

        print()
        print("=" * 72)
        print("3. ПЕРЕКРЫТИЕ КЛАССОВ ПО УРОВНЮ (главный вопрос к данным)")
        print("=" * 72)
        if pk_pos is not None and pk_amb is not None:
            for lbl, a in (("позитивы", pk_pos), ("фон", pk_amb)):
                print(f" {lbl:9s} peak p10-p90 = {np.percentile(a,10):.0f}"
                      f"-{np.percentile(a,90):.0f}")
            lo_pos, hi_amb = np.percentile(pk_pos, 10), np.percentile(pk_amb, 90)
            if lo_pos < hi_amb:
                print(f" ВНИМАНИЕ: {100*(hi_amb-lo_pos)/max(hi_amb,1):.0f}% диапазона "
                      f"позитивов лежит НИЖЕ 90-го перцентиля фона.")
                print(" Модель может отличить классы только по громкости — "
                      "смените громкость ТВ, и она сломается.")
            else:
                print(" диапазоны не перекрываются по 10/90 перцентилям — "
                      "классы различимы не только по громкости")
    else:
        print("\n(фон не передан — проверка перекрытия уровней пропущена)")

    print()
    print("=" * 72)
    print("4. СКОЛЬКО РЕАЛЬНО ПОЗИТИВНЫХ ОКОН (с учётом 1.28 с окна)")
    print("=" * 72)
    npos_win = sum(len(fx.vectors(load_wav(p))) for p, _ in pos)
    print(f" позитивных окон после нормализации как в рантайме: {npos_win}")
    print(" ОКОНА = 1.28 с КОНТЕКСТА. Слово «компьютер» длится ~0.5 с, значит")
    print(" каждое окно содержит его лишь часть — плюс тишину, которую Whisper")
    print(" добавил при нарезке. При 36 burst-ах это очень мало для 1536-мерного")
    print(" входа. Сколько бы ни помогали веса классов, информации просто нет:")
    print(" либо собрать больше позитивов, либо дообучать голову на замороженных")
    print(" фичах с сильной регуляризацией и проверкой на отложенных burst-ах.")


if __name__ == "__main__":
    main()
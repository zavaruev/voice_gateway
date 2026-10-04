"""Label ambience windows with Whisper so only non-wake audio becomes negative.

The livingroom capture is a TV / conversation recording, which is exactly the
speech that must NOT wake the room. Every window is transcribed and kept as a
negative only when Whisper did NOT hear the wake word in it: a stray
"компьютер" from the soundtrack (or the user testing while the mic was open)
would otherwise be trained in as a negative and make the model deaf to the
real thing.

Writes a labels.tsv in the same format train_wake_head.py reads.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

WHISPER = "http://192.168.22.111:8000/v1/audio/transcriptions"
MODEL = "koekaverna/faster-whisper-podlodka-turbo"
WAKE = "комп"


def label(path):
    out = subprocess.run(
        ["curl", "-s", "-m", "20", WHISPER,
         "-F", f"file=@{path}", "-F", f"model={MODEL}", "-F", "language=ru"],
        capture_output=True, text=True, timeout=30).stdout
    try:
        body = out.split('"text":"', 1)[1].split('"', 1)[0]
    except IndexError:
        body = ""
    return os.path.basename(path), body.replace("\t", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir)
        if f.endswith(".wav"))
    print(f"размечаю {len(files)} окон")

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, (name, text) in enumerate(ex.map(label, files), 1):
            rows.append((name, text))
            if i % 50 == 0:
                print(f"  {i}/{len(files)}")

    wake = [r for r in rows if WAKE in r[1].lower()]
    with open(args.out, "w", encoding="utf-8") as w:
        for name, text in rows:
            w.write(f"{name}\t{text}\n")

    print(f"готово: {len(rows)} строк в {args.out}")
    print(f"со словом «комп»: {len(wake)} — они ИСКЛЮЧЕНЫ из негативов")
    for name, text in wake[:8]:
        print(f"  исключено: {name}: {text[:50]}")
    print("\nпример разметки:")
    for name, text in rows[:10]:
        print(f"  {name}: {text[:55]}")


if __name__ == "__main__":
    sys.exit(main())

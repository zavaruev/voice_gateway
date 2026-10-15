"""Patch the Frigate config: 15.10.2026, two causes of stuttering in the kitchen
and the corridor.

1. **A duplicate producer in the corridor.** In the go2rtc block the stream was
   named `corridor1` — a name go2rtc had already refused on 11.10. As a result
   TWO RTSP producers reached camera .247: one for the gateway
   (`camera-corridor1`), one for the NVR (`corridor1`). `majestic` is
   single-threaded, and with two video sessions its output audio channel was
   underfed — `Device 0 is empty`. The name is now the one the gateway reads.

2. **A crash loop of ffmpeg in the kitchen.** In the NVR logs:
       [rtsp] RTP: PT=60: bad cseq 0
       [segment] Timestamps are unseverable
       [segment] Non-monotonous DTS
       kitchen exceeded fps limit. Exiting ffmpeg...
   `PT=60` is the AUDIO payload, while detection only needs a video frame.
   `-an` was added so ffmpeg does not negotiate audio at all.

Every replacement is exact and verified to match exactly once: a silent
replacement that changed nothing would have cost a week of diagnosis here.
"""
import shutil
import sys

PATH = "/config/config.yaml"
BAK = "/config/config.yaml.bak.20261015"

EDITS = [
    # 1. the stream name in go2rtc
    (
        "    corridor1:\n      - rtsp://root:2441@192.168.22.247/stream=0#backchannel=1",
        "    camera-corridor1:\n      - rtsp://root:2441@192.168.22.247/stream=0#backchannel=1",
        "go2rtc: stream corridor1 renamed to camera-corridor1",
    ),
    # 1b. the corridor1 camera in the `cameras` section reads the same stream.
    # The pattern is the path line only: this block has no `input_args` (unlike
    # balcony), and the first version searched for a structure containing one,
    # found nothing, and did NOT write the file — which was the correct
    # behaviour for this script and the reason the mistake was visible.
    (
        "        - path: rtsp://127.0.0.1:8554/corridor1",
        "        - path: rtsp://127.0.0.1:8554/camera-corridor1",
        "camera corridor1 reads camera-corridor1",
    ),
    (
        "      stream_name: corridor1",
        "      stream_name: camera-corridor1",
        "stream_name of camera corridor1",
    ),
    # 2. kitchen: do not negotiate audio
    (
        "  kitchen:\n"
        "    live:\n"
        "      stream_name: kitchen\n"
        "    ffmpeg:\n"
        "      inputs:\n"
        "        - path: rtsp://127.0.0.1:8554/kitchen\n"
        "          roles:\n"
        "            - record\n"
        "            - detect\n"
        "          input_args:\n"
        "            - -rtsp_transport\n"
        "            - tcp\n",
        "  kitchen:\n"
        "    live:\n"
        "      stream_name: kitchen\n"
        "    ffmpeg:\n"
        "      inputs:\n"
        "        - path: rtsp://127.0.0.1:8554/kitchen\n"
        "          roles:\n"
        "            - record\n"
        "            - detect\n"
        "          input_args:\n"
        "            - -rtsp_transport\n"
        "            - tcp\n"
        "            # 15.10.2026: the NVR does not need sound, and negotiating it\n"
        "            # crashed ffmpeg in a loop (PT=60 bad cseq, unseverable\n"
        "            # timestamps), and every reconnect spent camera resources on\n"
        "            # video — the audio channel was underfed and the room stuttered.\n"
        "            - -an\n",
        "kitchen: -an (audio is not negotiated)",
    ),
]

text = open(PATH, encoding="utf-8").read()
shutil.copyfile(PATH, BAK)
print("  backup: %s" % BAK)

failed = 0
for old, new, what in EDITS:
    n = text.count(old)
    if n != 1:
        print("  SKIPPED (%d matches): %s" % (n, what))
        failed += 1
        continue
    text = text.replace(old, new, 1)
    print("  ок: %s" % what)

if failed:
    print("  NOT WRITTEN: %d edits did not find their text" % failed)
    sys.exit(1)

open(PATH, "w", encoding="utf-8").write(text)
print("  written")

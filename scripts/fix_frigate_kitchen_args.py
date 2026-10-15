"""Kitchen NVR: drop the non-standard `input_args` that made the watchdog kill ffmpeg.

Measured 15.10.2026. The kitchen was the **only** one of the NVR's seven cameras
with non-standard arguments:

    kitchen        fps=3   -rtsp_transport tcp -analyzeduration 10000000 -probesize 10000000
    the other six  fps=5   (the default)

`-analyzeduration 10000000` is ten seconds of stream that ffmpeg buffers
**before the first frame**. The NVR watchdog kills a camera that has delivered no
frames for 20 s and then starts it again: `watchdog.kitchen Terminating the
existing ffmpeg process` → `kitchen: exiting subprocess` every ~100 seconds,
continuously.

While that held, the kitchen kept ONE stream consumer (the gateway), because the
NVR would not pull it — and the 11:53 measurement says exactly one consumer
means `play_audio` ×25.7 and stuttering. An unstable NVR capture is therefore
not a side effect, it is a cause of the room stuttering.

One more detail that makes the fix obvious: these arguments address the DIRECT
link to the camera (tuning the RTSP stream), while the NVR pulls from go2rtc,
which normalises the stream itself. The workaround for the old problem stayed in
the config after the path changed.

The END state is what gets checked: after the edits the kitchen must have no
`input_args` left, and the YAML must still parse. The script does not write the
file if that is not true, or if the original block was not found exactly once.
"""
import shutil
import sys

PATH = "/config/config.yaml"
BAK = "/config/config.yaml.bak.20261015.kitchenargs"

OLD = (
    "          input_args:\n"
    "            - -rtsp_transport\n"
    "            - tcp\n"
    "            - -analyzeduration\n"
    '            - "10000000"\n'
    "            - -probesize\n"
    '            - "10000000"\n'
)

text = open(PATH, encoding="utf-8").read()
shutil.copyfile(PATH, BAK)
print("  backup: %s" % BAK)

n = text.count(OLD)
if n != 1:
    print("  SKIPPED: the kitchen input_args block was found %d times (need exactly 1)" % n)
    print("  NOT WRITTEN")
    sys.exit(1)
text = text.replace(OLD, "", 1)

# The end state, not the fact of the replacement. Checked against the PARSED YAML
# and against the SPECIFIC camera: the first version searched for `input_args`
# across the whole file, found one belonging to balcony, and so refused to apply
# a correct edit because of somebody else's key. The check has to answer "does
# the kitchen still have one?", not "is there one anywhere".
try:
    import yaml
    cfg = yaml.safe_load(text)
except Exception as exc:
    print("  the YAML no longer parses: %r — write cancelled" % (exc,))
    sys.exit(1)

kitchen = cfg["cameras"]["kitchen"]["ffmpeg"]["inputs"][0]
if "input_args" in kitchen:
    print("  the kitchen still has input_args=%r — write cancelled"
          % (kitchen["input_args"],))
    sys.exit(1)
others = [n for n, c in cfg["cameras"].items()
          if "input_args" in c["ffmpeg"]["inputs"][0]]
print("  the kitchen has no input_args left; path=%s" % kitchen["path"])
print("  input_args remain on: %s (not touching those)" % (others or "—"))
open(PATH, "w", encoding="utf-8").write(text)
print("  written")
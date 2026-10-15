"""Remove `-an` from the kitchen: it was MY regression and it will not stay.

15.10.2026. The chronology, because the chronology IS the lesson:

  * `-an` was added to stop the ffmpeg crash loop (`PT=60 bad cseq`). The loop
    did stop — zero crashes.
  * But afterwards the kitchen's ffmpeg process **stopped being a consumer of the
    go2rtc stream**: `kitchen` was left with one consumer (the gateway) where
    the neighbouring rooms have two. A win, and no connection.
  * Two attempts to revert it failed because I guessed the indentation: in the
    `input_args` block the lines are indented **12** spaces, and both versions of
    this script used 14. Both reverts printed "NOT WRITTEN" and left the file
    untouched — **which is exactly why the script verifies every replacement
    rather than "just replacing" it.** The text is now read from the file rather
    than from memory: `repr` of the lines before writing, see the print below.

The script checks not only that the text matched but the END RESULT: if `-an`
is still in the kitchen's `input_args` after the edits, it does not write the
file.

Leaving in the config something unverified that I broke myself is not an option.
The correct way to drop sound from the NVR is `#video` in the go2rtc stream URL
(the channel is never created), not `-an` on the consumer; that is a separate
change and is not made here.
"""
import shutil
import sys

PATH = "/config/config.yaml"
BAK = "/config/config.yaml.bak.20261015.noan3"

# The exact text from the file: 12 spaces of indentation, not 14.
BLOCK = (
    "            # 15.10.2026: the NVR does not need sound, and negotiating it\n"
    "            # crashed ffmpeg in a loop (PT=60 bad cseq, unseverable\n"
    "            # timestamps), and every reconnect spent camera resources on\n"
    "            # video — the audio channel was underfed and the room stuttered.\n"
    "            - -an\n"
)

text = open(PATH, encoding="utf-8").read()
shutil.copyfile(PATH, BAK)
print("  backup: %s" % BAK)

n = text.count(BLOCK)
if n != 1:
    print("  SKIPPED (%d matches)" % n)
    print("  NOT WRITTEN")
    sys.exit(1)
text = text.replace(BLOCK, "", 1)

# Check the result, not merely that a replacement happened.
if "            - -an\n" in text:
    print("  `-an` IS STILL THERE AFTER THE EDIT — write cancelled")
    sys.exit(1)

open(PATH, "w", encoding="utf-8").write(text)
print("  ok: -an and its comment are gone")
print("  written")
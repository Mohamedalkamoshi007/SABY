import glob
import os
import sys
import wave

SECONDS = 5.0                      # length of the breath-hold part to cut from the start of each file
OUT_DIR = "hold_segments"

if len(sys.argv) < 2:
    print("Usage: py cut_hold.py file1.wav file2.wav   (wildcards like up-*.wav also work)")
    sys.exit()

os.makedirs(OUT_DIR, exist_ok=True)
for pattern in sys.argv[1:]:
    for f in glob.glob(pattern):
        with wave.open(f, "rb") as w:
            r = w.getframerate()
            data = w.readframes(int(SECONDS * r))
        out = os.path.join(OUT_DIR, "hold_" + os.path.basename(f))
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(r)
            w.writeframes(data)
        print("Saved:", out, f"({len(data) / 2 / r:.1f}s)")
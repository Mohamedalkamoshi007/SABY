import sys
import wave

import numpy as np
from scipy import signal

inp = sys.argv[1]
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 100
hi = float(sys.argv[3]) if len(sys.argv) > 3 else 1500
out = inp[:-4] + "_lung.wav"

with wave.open(inp, "rb") as w:
    r = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0

sos = signal.butter(4, [lo, hi], "bandpass", fs=r, output="sos")
y = signal.sosfiltfilt(sos, x)
peak = float(np.max(np.abs(y)))
y = y / peak * 0.8 if peak > 0 else y

with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(r)
    w.writeframes(np.clip(np.round(y * 32767), -32768, 32767).astype("<i2").tobytes())
print("Saved:", out, f"(band {lo}-{hi} Hz)")
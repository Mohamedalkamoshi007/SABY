import sys
import wave

import numpy as np
from scipy import signal

inp = sys.argv[1]
strength = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
noise_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
out = inp[:-4] + "_clean.wav"

with wave.open(inp, "rb") as w:
    rate = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0

nper, hop = 256, 128
_, _, Z = signal.stft(x, fs=rate, nperseg=nper, noverlap=nper - hop)
mag, phase = np.abs(Z), np.angle(Z)

k = max(2, int(noise_sec * rate / hop))
noise = np.mean(mag[:, :k], axis=1, keepdims=True)
clean = np.maximum(mag - strength * noise, 0.1 * mag)

_, y = signal.istft(clean * np.exp(1j * phase), fs=rate, nperseg=nper, noverlap=nper - hop)
y = y[:len(x)]
peak = float(np.max(np.abs(y)))
if peak > 0:
    y = y / peak * 0.9

pcm = np.clip(np.round(y * 32767.0), -32768, 32767).astype("<i2")
with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(pcm.tobytes())
print("Saved:", out)
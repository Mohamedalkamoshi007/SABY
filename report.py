import sys
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if len(sys.argv) < 2:
    print("Usage: py report.py file.wav [packets] [lost]")
    sys.exit()

f = sys.argv[1]
pkts = sys.argv[2] if len(sys.argv) > 2 else "?"
lost = sys.argv[3] if len(sys.argv) > 3 else "?"

with wave.open(f, "rb") as w:
    r = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def db(v):
    return 20 * np.log10(max(float(v), 1e-6))


dur = len(x) / r
peak = float(np.max(np.abs(x)))
rms = float(np.sqrt(np.mean(x ** 2)))
dc = float(np.mean(x)) * 32768
clipped = int(np.sum(np.abs(x) > 0.98))

step = int(0.25 * r)
n = len(x) // step
lv = [db(np.sqrt(np.mean(x[i * step:(i + 1) * step] ** 2))) for i in range(n)]
tl = (np.arange(n) + 0.5) * 0.25

xh = signal.sosfilt(signal.butter(4, 60, "highpass", fs=r, output="sos"), x)
fr, tt, S = signal.spectrogram(xh, fs=r, nperseg=1024, noverlap=768)
m = fr <= 2000
Sd = 10 * np.log10(S[m] + 1e-14)

fw, P = signal.welch(x, fs=r, nperseg=4096)
mw = fw <= 4000

fig, axs = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1, 2.2, 1.2]})
fig.suptitle(
    f"{f} | {dur:.1f}s | {r}Hz | rms {db(rms):.1f} dBFS | peak {db(peak):.1f} dBFS | "
    f"DC {dc:.0f} | clipped {clipped} | packets {pkts} lost {lost}",
    fontsize=9,
)

axs[0].plot(tl, lv)
axs[0].set_ylabel("level dBFS (0.25s)")
axs[0].set_xlim(0, dur)
axs[0].grid(alpha=0.3)

axs[1].pcolormesh(tt, fr[m], Sd, shading="auto", cmap="magma",
                  vmin=np.percentile(Sd, 20), vmax=np.percentile(Sd, 99.5))
axs[1].set_ylabel("Hz (spectrogram, HP 60Hz)")
axs[1].set_xlim(0, dur)

axs[2].plot(fw[mw], 10 * np.log10(P[mw] + 1e-14))
axs[2].set_ylabel("avg spectrum dB")
axs[2].set_xlabel("Hz")
axs[2].grid(alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.96))
out = f[:-4] + "_report.png"
plt.savefig(out, dpi=90)
plt.close()
print("Saved:", out)
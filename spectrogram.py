import sys
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if len(sys.argv) < 2:
    print("Usage: py spectrogram.py file1.wav [file2.wav ...]")
    sys.exit()

for f in sys.argv[1:]:
    with wave.open(f, "rb") as w:
        r = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    x = signal.sosfilt(signal.butter(4, 60, "highpass", fs=r, output="sos"), x)
    fr, tt, S = signal.spectrogram(x, fs=r, nperseg=1024, noverlap=768)
    m = fr <= 2000
    Sd = 10 * np.log10(S[m] + 1e-14)
    plt.figure(figsize=(11, 4))
    plt.pcolormesh(tt, fr[m], Sd, shading="auto", cmap="magma",
                   vmin=np.percentile(Sd, 20), vmax=np.percentile(Sd, 99.5))
    plt.xlabel("seconds")
    plt.ylabel("Hz")
    plt.title(f)
    plt.tight_layout()
    out = f[:-4] + "_spec.png"
    plt.savefig(out, dpi=90)
    plt.close()
    print("Saved:", out)
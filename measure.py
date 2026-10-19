import sys
import wave

import numpy as np

with wave.open(sys.argv[1], "rb") as w:
    rate = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0

sec = len(x) // rate
db = []
for i in range(sec):
    seg = x[i * rate:(i + 1) * rate]
    db.append(20 * np.log10(max(np.sqrt(np.mean(seg ** 2)), 1e-6)))

print("per-second dBFS:", " ".join(f"{d:.0f}" for d in db))

sil = np.array(db[1:4])
brt = np.array(db[6:])
if len(sil) and len(brt):
    print(f"silence (sec 1-3):   {sil.mean():.1f} dBFS")
    print(f"breathing (sec 6+):  {brt.mean():.1f} dBFS")
    print(f"difference:          {brt.mean() - sil.mean():.1f} dB")
"""
extract_panns_icbhi.py - turn every ICBHI respiratory cycle into a 2048-number PANNs (Cnn14) embedding.
Run once (it can take 15-60 minutes on CPU). It saves progress every 500 cycles and can be resumed:
if you stop it (Ctrl+C) or it crashes, run the same command again and it continues.

Usage:   py -u extract_panns_icbhi.py --data_dir "PATH_TO_ICBHI_FOLDER"
Output:  icbhi_panns_emb.npz  (emb, y, patients, durations)  - labels: 0 Normal, 1 Crackle, 2 Wheeze, 3 Both
"""
import argparse
import glob
import os
import time
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

from panns_inference import AudioTagging

SR = 32000                 # PANNs Cnn14 expects 32 kHz
FMIN, FMAX = 100.0, 2000.0  # same band for ICBHI and for my own recordings (the stethoscope tube cuts the rest)
MIN_CYCLE_S = 0.4
MIN_SAMPLES = SR           # pad every clip to at least 1 s (Cnn14 pools time by 32)
TARGET_RMS = 0.05


def load_audio_32k(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    x = x - np.mean(x)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    sos = butter(4, [FMIN / (SR / 2), FMAX / (SR / 2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def prepare_clip(seg):
    """RMS-normalise (so recording gain does not matter) and zero-pad to at least 1 s."""
    rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
    seg = seg * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(seg)))
    if peak > 0.99:
        seg = seg * (0.99 / peak)
    if len(seg) < MIN_SAMPLES:
        seg = np.pad(seg, (0, MIN_SAMPLES - len(seg)))
    return seg.astype(np.float32)


def label_of(crackle, wheeze):
    return 0 if (crackle == 0 and wheeze == 0) else 1 if (crackle == 1 and wheeze == 0) else 2 if (crackle == 0 and wheeze == 1) else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing the ICBHI .wav and .txt files")
    ap.add_argument("--out", default="icbhi_panns_emb.npz")
    args = ap.parse_args()
    partial = args.out + ".partial.npz"

    txt_files = [f for f in glob.glob(os.path.join(args.data_dir, "**", "*.txt"), recursive=True)
                 if os.path.exists(f[:-4] + ".wav") and "diagnosis" not in os.path.basename(f).lower()]
    txt_files.sort()
    if not txt_files:
        raise FileNotFoundError(f"No wav/txt pairs found in: {args.data_dir}")

    done_keys, emb_list, y_list, pat_list, dur_list = set(), [], [], [], []
    if os.path.exists(partial):
        p = np.load(partial, allow_pickle=True)
        done_keys = set(p["keys"].tolist())
        emb_list, y_list = list(p["emb"]), list(p["y"])
        pat_list, dur_list = list(p["patients"]), list(p["durations"])
        keys_list = p["keys"].tolist()
        print(f"[RESUME] {len(done_keys)} cycles already done")
    else:
        keys_list = []

    at = AudioTagging(checkpoint_path=None, device="cpu")
    t0, n_new = time.time(), 0

    def save_partial():
        np.savez(partial, emb=np.array(emb_list, dtype=np.float32), y=np.array(y_list), patients=np.array(pat_list),
                 durations=np.array(dur_list, dtype=np.float32), keys=np.array(keys_list))

    print(f"[INFO] {len(txt_files)} recordings")
    for fi, txt in enumerate(txt_files):
        wav = txt[:-4] + ".wav"
        base = os.path.basename(txt)
        patient = base.split("_")[0]
        rows = []
        with open(txt, "r", encoding="utf-8", errors="ignore") as f:
            for ci, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    s, e = float(parts[0]), float(parts[1])
                    c, w = int(float(parts[2])), int(float(parts[3]))
                except ValueError:
                    continue
                if e - s >= MIN_CYCLE_S and f"{base}|{ci}" not in done_keys:
                    rows.append((ci, s, e, c, w))
        if not rows:
            continue
        try:
            audio = load_audio_32k(wav)
        except Exception as ex:
            print(f"[WARN] skipping {os.path.basename(wav)}: {ex}")
            continue
        for ci, s, e, c, w in rows:
            seg = audio[max(0, int(s * SR)):min(len(audio), int(e * SR))]
            if len(seg) < int(MIN_CYCLE_S * SR):
                continue
            _, embedding = at.inference(prepare_clip(seg)[None, :])
            emb_list.append(np.asarray(embedding)[0])
            y_list.append(label_of(c, w))
            pat_list.append(patient)
            dur_list.append(e - s)
            keys_list.append(f"{base}|{ci}")
            n_new += 1
            if n_new % 500 == 0:
                save_partial()
        el = time.time() - t0
        print(f"  file {fi + 1}/{len(txt_files)} | cycles {len(y_list)} | {el / 60:.1f} min elapsed", flush=True)

    np.savez_compressed(args.out, emb=np.array(emb_list, dtype=np.float32), y=np.array(y_list),
                        patients=np.array(pat_list), durations=np.array(dur_list, dtype=np.float32))
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[DONE] saved {len(y_list)} cycles to {args.out}")
    print("class counts (Normal, Crackle, Wheeze, Both):", np.bincount(np.array(y_list), minlength=4).tolist())


if __name__ == "__main__":
    main()
"""
predict_lung_v2.py - SABY breathing-gated prediction.

What is new compared with predict_lung.py:
  * Breath gate: windows with no breathing (breath-hold, silence, heart only) are SKIPPED, not averaged.
  * "_clean.wav" files are ignored by default (spectral subtraction leaves tonal residue that looks like wheeze).
  * Result is "Uncertain" when the best class has low confidence, and "No breathing detected" when nothing passes the gate.
  * Per-class window votes, and a timeline PNG per file (<name>_pred.png).
Put this file in the same folder as predict_lung.py and the .pt model.
Probabilities are NOT calibrated; this is an experimental research tool, not a diagnostic tool.
"""
import argparse
import glob
import json
import os
import sys
from math import gcd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import butter, medfilt, resample_poly, sosfilt

from predict_lung import LungCNN, bandpass_filter, create_mel_filterbank

COLORS = {"Normal": "tab:green", "Crackle": "tab:red", "Wheeze": "tab:blue", "Both": "tab:purple"}
FRAME_S = 0.1


def load_audio(path, target_sr, fmin, fmax):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    raw_peak = float(np.max(np.abs(data))) if len(data) else 0.0
    data = data - np.mean(data)
    duration = len(data) / float(sr)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    data = bandpass_filter(data, target_sr, fmin, fmax)
    return np.ascontiguousarray(data, dtype=np.float32), duration, raw_peak


def breath_gate(data, sr, band=(200.0, 1000.0), gate_db=4.0):
    """Frame-wise breathing detector: level in the lung band (median-filtered to remove heart clicks)
    must exceed the quiet baseline (10th percentile) by gate_db."""
    sos = butter(4, [band[0] / (sr / 2), band[1] / (sr / 2)], btype="bandpass", output="sos")
    y = sosfilt(sos, data)
    n = int(FRAME_S * sr)
    m = len(y) // n
    if m < 5:
        return np.zeros(m), np.zeros(m, dtype=bool), 0.0
    rms = np.sqrt(np.mean(y[:m * n].reshape(m, n) ** 2, axis=1))
    rms = medfilt(rms, 5)                               # filter the linear level, not the dB value
    env_db = 20 * np.log10(rms + 1e-9)
    baseline = float(np.percentile(env_db, 10))
    return env_db, env_db > baseline + gate_db, baseline


def make_windows(data, sr, n_samples, step_samples):
    wins = []
    if len(data) < n_samples:
        reps = int(np.ceil(n_samples / max(len(data), 1)))
        wins.append((0.0, len(data) / sr, np.tile(data, reps)[:n_samples]))
    else:
        for s in range(0, len(data) - n_samples + 1, step_samples):
            wins.append((s / sr, (s + n_samples) / sr, data[s:s + n_samples]))
    return wins


def active_fraction(active, t0, t1):
    a, b = int(round(t0 / FRAME_S)), int(round(t1 / FRAME_S))
    seg = active[a:b]
    return float(np.mean(seg)) if len(seg) else 0.0


def log_mel(seg, n_fft, hop, window, mel_fbank):
    n_frames = 1 + (len(seg) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        seg, shape=(n_frames, n_fft), strides=(seg.strides[0] * hop, seg.strides[0])).copy()
    frames *= window
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    lm = np.log10(np.maximum(np.dot(spec, mel_fbank.T), 1e-10)).T
    return ((lm - np.mean(lm)) / (np.std(lm) + 1e-6)).astype(np.float32)


def summarize(probs, keep, classes, min_conf):
    if keep.sum() == 0:
        return "No breathing detected", 0.0, None
    avg = probs[keep].mean(axis=0)
    i = int(np.argmax(avg))
    conf = float(avg[i])
    return (classes[i] if conf >= min_conf else "Uncertain"), conf, avg


def plot_result(title, env_db, active, baseline, gate_db, wins, out_png):
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if len(env_db):
        t = (np.arange(len(env_db)) + 0.5) * FRAME_S
        axs[0].plot(t, env_db)
        axs[0].axhline(baseline + gate_db, ls="--", color="gray")
        axs[0].fill_between(t, env_db.min(), env_db.max(), where=active, alpha=0.15)
    axs[0].set_ylabel("lung-band level (dB)")
    axs[0].set_title(title, fontsize=9)
    for i, w in enumerate(wins):
        row = i % 2
        color = COLORS.get(w["predicted"], "tab:gray") if w["kept"] else "lightgray"
        axs[1].barh(row + 0.5, w["t1"] - w["t0"], left=w["t0"], height=0.9, color=color, alpha=0.75, edgecolor="k")
        label = f'{w["predicted"][0]} {w["confidence"] * 100:.0f}' if w["kept"] else "skip"
        axs[1].text((w["t0"] + w["t1"]) / 2, row + 0.5, label, ha="center", va="center", fontsize=7)
    axs[1].set_ylim(0, 2)
    axs[1].set_yticks([])
    axs[1].set_xlabel("seconds   (gray = skipped by breath gate; N/C/W/B = Normal/Crackle/Wheeze/Both + confidence %)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="a wav file or a folder (default: current folder)")
    ap.add_argument("--model", default="saby_lung_cnn.pt")
    ap.add_argument("--json_out", default="predictions_v2.json")
    ap.add_argument("--gate_db", type=float, default=4.0, help="breathing = level above quiet baseline by this many dB")
    ap.add_argument("--min_active", type=float, default=0.2, help="min fraction of a window that must be active")
    ap.add_argument("--min_contrast", type=float, default=6.0,
                    help="file-level check: 95th percentile level must exceed the quiet baseline by this many dB")
    ap.add_argument("--min_conf", type=float, default=0.6, help="below this the result is 'Uncertain'")
    ap.add_argument("--include_clean", action="store_true", help="also process *_clean.wav files")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = LungCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sr_t, n_fft, hop = ckpt["target_sr"], ckpt["n_fft"], ckpt["hop_length"]
    n_samples = int(sr_t * ckpt["cycle_seconds"])
    mel_fbank = create_mel_filterbank(sr_t, n_fft, ckpt["n_mels"], ckpt["fmin"], ckpt["fmax"])
    window = np.hanning(n_fft).astype(np.float32)

    if args.wav is None:
        files = sorted(glob.glob("*.wav"))
    elif os.path.isdir(args.wav):
        files = sorted(glob.glob(os.path.join(args.wav, "*.wav")))
    else:
        files = [args.wav]
    if not args.include_clean:
        files = [f for f in files if not f.lower().endswith("_clean.wav")]
    if not files:
        print("[WARN] No .wav files found.")
        sys.exit(1)

    results = []
    for path in files:
        data, duration, raw_peak = load_audio(path, sr_t, ckpt["fmin"], ckpt["fmax"])
        env_db, active, baseline = breath_gate(data, sr_t, gate_db=args.gate_db)
        wins = make_windows(data, sr_t, n_samples, int(sr_t * 1.5))
        fracs = np.array([active_fraction(active, t0, t1) for t0, t1, _ in wins])
        keep = fracs >= args.min_active
        contrast = float(np.percentile(env_db, 95) - baseline) if len(env_db) else 0.0
        low_contrast = contrast < args.min_contrast
        if low_contrast:
            keep = np.zeros_like(keep)                   # no clear breathing above the noise in this file

        specs = [log_mel(seg, n_fft, hop, window, mel_fbank) for _, _, seg in wins]
        batch = torch.from_numpy(np.stack(specs, axis=0)).unsqueeze(1)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).numpy()

        label, conf, avg = summarize(probs, keep, classes, args.min_conf)
        best_guess = classes[int(np.argmax(avg))] if avg is not None else None

        win_rows = []
        votes = {c: 0 for c in classes}
        for (t0, t1, _), p, k, fr in zip(wins, probs, keep, fracs):
            wi = int(np.argmax(p))
            if k:
                votes[classes[wi]] += 1
            win_rows.append({"t0": round(t0, 2), "t1": round(t1, 2), "kept": bool(k),
                             "active_fraction": round(float(fr), 2),
                             "predicted": classes[wi], "confidence": round(float(p[wi]), 3)})

        peak_db = 20 * np.log10(max(raw_peak, 1e-6))
        print("=" * 60)
        print(f"File: {os.path.basename(path)} ({duration:.1f}s | peak {peak_db:.1f} dBFS)")
        print(f"  Breathing windows: {int(keep.sum())}/{len(wins)}   (quiet baseline {baseline:.0f} dB, gate +{args.gate_db:g} dB, "
              f"file contrast {contrast:.1f} dB)")
        if avg is None:
            print("  -> No clear breathing above the noise" if low_contrast else "  -> No breathing detected: not classified")
        else:
            print(f"  -> Result: {label}" + (f" (best guess {best_guess} {conf * 100:.0f}%)" if label == "Uncertain"
                                              else f" ({conf * 100:.0f}%, not calibrated)"))
            print("  -> Window votes: " + " | ".join(f"{c} {n}" for c, n in votes.items()))
            print("  -> Mean probabilities: " + " | ".join(f"{c} {avg[i] * 100:.0f}%" for i, c in enumerate(classes)))
        if raw_peak > 0.98:
            print("  [!] recording is clipped: results are unreliable")

        if not args.no_plot:
            png = os.path.splitext(path)[0] + "_pred.png"
            plot_result(f"{os.path.basename(path)} -> {label}", env_db, active, baseline, args.gate_db, win_rows, png)
            print(f"  -> Plot: {png}")

        results.append({"file": os.path.basename(path), "duration_s": round(duration, 2),
                        "peak_dbfs": round(float(peak_db), 1), "baseline_db": round(baseline, 1),
                        "contrast_db": round(contrast, 1), "breathing_windows": int(keep.sum()), "windows_total": len(wins),
                        "result": label, "best_guess": best_guess, "confidence": round(conf, 3),
                        "mean_probabilities": None if avg is None else {c: round(float(avg[i]), 3) for i, c in enumerate(classes)},
                        "votes": votes, "windows": win_rows})

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("=" * 60)
    print(f"[SAVED] {args.json_out}")


if __name__ == "__main__":
    main()
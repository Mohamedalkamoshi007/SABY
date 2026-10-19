"""
predict_panns.py - score recordings with the PANNs embedding + logistic-regression heads (panns_heads.joblib).

Per file: breath gate (same as predict_lung_v2.py) -> 3 s windows (hop 1.5 s) -> PANNs embedding per kept window
-> crackle score, wheeze score and 4-class probabilities.
IMPORTANT: the scores come from class-balanced logistic regressions trained on ICBHI. They are NOT probabilities of disease,
the model is weak (see eval_panns_head.py), and my device/recording conditions differ from ICBHI. Research prototype only.

Usage:  py predict_panns.py --wav final4
        py predict_panns.py --wav hold_segments --no_gate      (heart-only / no-breathing control test)
Needs in the same folder: extract_panns_icbhi.py, predict_lung_v2.py, predict_lung.py, panns_heads.joblib
"""
import argparse
import glob
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_panns_icbhi import SR, load_audio_32k, prepare_clip
from predict_lung_v2 import FRAME_S, active_fraction, breath_gate, load_audio
from panns_inference import AudioTagging

WIN_S, HOP_S = 3.0, 1.5


def plot_file(title, env_db, active, baseline, gate_db, rows, out_png):
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    if len(env_db):
        t = (np.arange(len(env_db)) + 0.5) * FRAME_S
        axs[0].plot(t, env_db)
        axs[0].axhline(baseline + gate_db, ls="--", color="gray")
        axs[0].fill_between(t, env_db.min(), env_db.max(), where=active, alpha=0.15)
    axs[0].set_ylabel("lung-band level (dB)")
    axs[0].set_title(title, fontsize=9)
    for r in rows:
        if r["kept"]:
            axs[1].plot([r["t0"], r["t1"]], [r["crackle"]] * 2, color="tab:red", lw=3, alpha=0.6)
            axs[1].plot([r["t0"], r["t1"]], [r["wheeze"]] * 2, color="tab:blue", lw=3, alpha=0.6)
    axs[1].axhline(0.5, ls="--", color="gray")
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("score")
    axs[1].set_xlabel("seconds   (red = crackle score, blue = wheeze score; only breathing windows are scored)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=".", help="a wav file or a folder")
    ap.add_argument("--heads", default="panns_heads.joblib")
    ap.add_argument("--json_out", default="predictions_panns.json")
    ap.add_argument("--gate_db", type=float, default=4.0)
    ap.add_argument("--min_active", type=float, default=0.2)
    ap.add_argument("--min_contrast", type=float, default=6.0)
    ap.add_argument("--no_gate", action="store_true", help="score every window (control test on heart-only / noise clips)")
    ap.add_argument("--include_clean", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    heads = joblib.load(args.heads)
    classes = heads["classes"]
    at = AudioTagging(checkpoint_path=None, device="cpu")

    if os.path.isdir(args.wav):
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
        data8, duration, raw_peak = load_audio(path, 8000, 150, 2000)
        env_db, active, baseline = breath_gate(data8, 8000, gate_db=args.gate_db)
        contrast = float(np.percentile(env_db, 95) - baseline) if len(env_db) else 0.0
        low_contrast = contrast < args.min_contrast and not args.no_gate

        audio = load_audio_32k(path)
        wl, hl = int(WIN_S * SR), int(HOP_S * SR)
        starts = [0] if len(audio) < wl else list(range(0, len(audio) - wl + 1, hl))
        rows, kept_idx, embs = [], [], []
        for s in starts:
            seg = audio[s:s + wl]
            t0, t1 = s / SR, (s + len(seg)) / SR
            frac = active_fraction(active, t0, t1)
            keep = args.no_gate or (frac >= args.min_active and not low_contrast)
            rows.append({"t0": round(t0, 2), "t1": round(t1, 2), "kept": bool(keep), "active_fraction": round(frac, 2),
                         "crackle": None, "wheeze": None})
            if keep:
                _, e = at.inference(prepare_clip(seg)[None, :])
                embs.append(np.asarray(e)[0])
                kept_idx.append(len(rows) - 1)

        print("=" * 60)
        print(f"File: {os.path.basename(path)} ({duration:.1f}s | file contrast {contrast:.1f} dB)")
        summary = None
        if not embs:
            print("  -> No clear breathing above the noise: not scored")
        else:
            E = np.vstack(embs)
            pc = heads["crackle"].predict_proba(E)[:, 1]
            pw = heads["wheeze"].predict_proba(E)[:, 1]
            p4 = heads["four_class"].predict_proba(E)
            for j, i in enumerate(kept_idx):
                rows[i]["crackle"], rows[i]["wheeze"] = round(float(pc[j]), 3), round(float(pw[j]), 3)
            n = len(kept_idx)
            print(f"  Scored windows: {n}/{len(starts)}")
            print(f"  Crackle score: mean {pc.mean():.2f} | max {pc.max():.2f} | windows > 0.5: {int((pc > 0.5).sum())}/{n}")
            print(f"  Wheeze  score: mean {pw.mean():.2f} | max {pw.max():.2f} | windows > 0.5: {int((pw > 0.5).sum())}/{n}")
            print("  4-class mean: " + " | ".join(f"{c} {p4[:, k].mean() * 100:.0f}%" for k, c in enumerate(classes)))
            summary = {"scored_windows": n, "crackle_mean": round(float(pc.mean()), 3), "crackle_max": round(float(pc.max()), 3),
                       "crackle_over_half": int((pc > 0.5).sum()), "wheeze_mean": round(float(pw.mean()), 3),
                       "wheeze_max": round(float(pw.max()), 3), "wheeze_over_half": int((pw > 0.5).sum()),
                       "four_class_mean": {c: round(float(p4[:, k].mean()), 3) for k, c in enumerate(classes)}}
        if not args.no_plot:
            png = os.path.splitext(path)[0] + "_panns.png"
            plot_file(os.path.basename(path), env_db, active, baseline, args.gate_db, rows, png)
            print(f"  -> Plot: {png}")
        results.append({"file": os.path.basename(path), "duration_s": round(duration, 2), "contrast_db": round(contrast, 1),
                        "summary": summary, "windows": rows})

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("=" * 60)
    print(f"[SAVED] {args.json_out}")


if __name__ == "__main__":
    main()
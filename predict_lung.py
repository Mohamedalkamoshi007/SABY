import os
import sys
import json
import glob
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd
import torch
import torch.nn as nn


class LungCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank


def bandpass_filter(x, sr, fmin, fmax):
    nyq = 0.5 * sr
    low = max(fmin / nyq, 0.001)
    high = min(fmax / nyq, 0.99)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def predict_file(wav_path, ckpt, model, mel_fbank, window):
    target_sr = ckpt["target_sr"]
    cycle_seconds = ckpt["cycle_seconds"]
    fmin = ckpt["fmin"]
    fmax = ckpt["fmax"]
    n_fft = ckpt["n_fft"]
    hop_length = ckpt["hop_length"]
    classes = ckpt["classes"]
    n_samples = int(target_sr * cycle_seconds)

    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    data = data - np.mean(data)
    duration_s = len(data) / float(sr)

    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

    data = bandpass_filter(data, target_sr, fmin, fmax)

    step_samples = int(target_sr * 1.5)
    windows = []
    times = []

    if len(data) < n_samples:
        repeats = int(np.ceil(n_samples / max(len(data), 1)))
        seg = np.tile(data, repeats)[:n_samples]
        windows.append(seg)
        times.append((0.0, duration_s))
    else:
        for start in range(0, len(data) - n_samples + 1, step_samples):
            seg = data[start:start + n_samples]
            windows.append(seg)
            times.append((round(start / target_sr, 2), round((start + n_samples) / target_sr, 2)))

    specs = []
    for seg in windows:
        n_frames = 1 + (len(seg) - n_fft) // hop_length
        frames = np.lib.stride_tricks.as_strided(
            seg,
            shape=(n_frames, n_fft),
            strides=(seg.strides[0] * hop_length, seg.strides[0]),
        ).copy()
        frames *= window
        spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
        mel_spec = np.dot(spec, mel_fbank.T)
        log_mel = np.log10(np.maximum(mel_spec, 1e-10)).T
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-6
        specs.append(((log_mel - mean) / std).astype(np.float32))

    batch = torch.from_numpy(np.stack(specs, axis=0)).unsqueeze(1)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).numpy()

    avg_probs = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    pred_class = classes[pred_idx]

    window_results = []
    for (t0, t1), p in zip(times, probs):
        w_idx = int(np.argmax(p))
        window_results.append({
            "time_range_s": f"{t0:.1f}-{t1:.1f}",
            "predicted": classes[w_idx],
            "confidence": round(float(p[w_idx]), 3),
        })

    return {
        "file": os.path.basename(wav_path),
        "duration_s": round(duration_s, 2),
        "windows_analyzed": len(windows),
        "final_prediction": pred_class,
        "confidence": round(float(avg_probs[pred_idx]), 3),
        "probabilities": {c: round(float(avg_probs[i]), 3) for i, c in enumerate(classes)},
        "windows": window_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="Path to a single wav file or folder")
    parser.add_argument("--model", type=str, default="saby_lung_cnn.pt")
    parser.add_argument("--json_out", type=str, default="predictions.json")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = LungCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mel_fbank = create_mel_filterbank(
        ckpt["target_sr"], ckpt["n_fft"], ckpt["n_mels"], ckpt["fmin"], ckpt["fmax"]
    )
    window = np.hanning(ckpt["n_fft"]).astype(np.float32)

    if args.wav is None:
        wav_files = sorted(glob.glob("*.wav"))
    elif os.path.isdir(args.wav):
        wav_files = sorted(glob.glob(os.path.join(args.wav, "*.wav")))
    else:
        wav_files = [args.wav]

    if not wav_files:
        print("[WARN] No .wav files found in current directory.")
        sys.exit(1)

    all_results = []
    for wpath in wav_files:
        res = predict_file(wpath, ckpt, model, mel_fbank, window)
        all_results.append(res)
        print("=" * 60)
        print(f"File: {res['file']} ({res['duration_s']}s | {res['windows_analyzed']} windows)")
        print(f"  -> Final Prediction : {res['final_prediction']} (Confidence: {res['confidence']*100:.1f}%)")
        probs_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in res["probabilities"].items()])
        print(f"  -> Class Probabilities: {probs_str}")

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("=" * 60)
    print(f"[SAVED] Full JSON report saved to: {args.json_out}")


if __name__ == "__main__":
    main()

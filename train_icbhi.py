import os
import glob
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

TARGET_SR = 8000
CYCLE_SECONDS = 3.0
N_SAMPLES = int(TARGET_SR * CYCLE_SECONDS)
FMIN = 150.0
FMAX = 2000.0
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128

CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]


def bandpass_filter(x, sr, fmin=FMIN, fmax=FMAX):
    nyq = 0.5 * sr
    low = max(fmin / nyq, 0.001)
    high = min(fmax / nyq, 0.99)
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


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


MEL_FBANK = create_mel_filterbank(TARGET_SR, N_FFT, N_MELS, FMIN, FMAX)
WINDOW = np.hanning(N_FFT).astype(np.float32)


def compute_log_mel(x):
    if len(x) < N_SAMPLES:
        repeats = int(np.ceil(N_SAMPLES / max(len(x), 1)))
        x = np.tile(x, repeats)[:N_SAMPLES]
    else:
        x = x[:N_SAMPLES]

    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, N_FFT),
        strides=(x.strides[0] * HOP_LENGTH, x.strides[0]),
    ).copy()
    frames *= WINDOW
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel_spec = np.dot(spec, MEL_FBANK.T)
    log_mel = np.log10(np.maximum(mel_spec, 1e-10)).T

    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    return ((log_mel - mean) / std).astype(np.float32)


def load_wav_resampled(wav_path):
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    data = data - np.mean(data)
    if sr != TARGET_SR:
        g = gcd(int(sr), int(TARGET_SR))
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    data = bandpass_filter(data, TARGET_SR)
    return data


def build_or_load_cache(data_dir, cache_path="icbhi_features_cache.npz"):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached features from {cache_path} ...")
        cached = np.load(cache_path)
        return cached["X"], cached["y"], cached["patients"]

    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    txt_files = [
        f for f in txt_files
        if os.path.exists(f[:-4] + ".wav") and "diagnosis" not in os.path.basename(f).lower()
    ]

    if not txt_files:
        raise FileNotFoundError(f"No wav/txt files found in: {data_dir}")

    print(f"[INFO] Found {len(txt_files)} recordings. Extracting respiratory cycles...")
    X_list, y_list, patient_list = [], [], []

    for idx, txt_path in enumerate(sorted(txt_files)):
        wav_path = txt_path[:-4] + ".wav"
        patient_id = os.path.basename(txt_path).split("_")[0]

        try:
            audio = load_wav_resampled(wav_path)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(wav_path)}: {e}")
            continue

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    start_s, end_s = float(parts[0]), float(parts[1])
                    crackle, wheeze = int(float(parts[2])), int(float(parts[3]))
                except ValueError:
                    continue

                s_idx = max(0, int(start_s * TARGET_SR))
                e_idx = min(len(audio), int(end_s * TARGET_SR))
                if e_idx - s_idx < int(0.4 * TARGET_SR):
                    continue

                segment = audio[s_idx:e_idx]
                feat = compute_log_mel(segment)

                if crackle == 0 and wheeze == 0:
                    label = 0
                elif crackle == 1 and wheeze == 0:
                    label = 1
                elif crackle == 0 and wheeze == 1:
                    label = 2
                else:
                    label = 3

                X_list.append(feat)
                y_list.append(label)
                patient_list.append(patient_id)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(txt_files):
            print(f"  Processed {idx + 1}/{len(txt_files)} files (cycles: {len(X_list)})")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    patients = np.array(patient_list)
    np.savez_compressed(cache_path, X=X, y=y, patients=patients)
    print(f"[INFO] Saved features to {cache_path} | Shape: {X.shape}")
    return X, y, patients


class LungDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = self.X[idx].copy()
        if self.augment:
            f_mask = np.random.randint(0, 8)
            f0 = np.random.randint(0, max(1, spec.shape[0] - f_mask))
            spec[f0:f0 + f_mask, :] = 0.0
        return torch.from_numpy(spec).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_model", type=str, default="saby_lung_cnn.pt")
    args = parser.parse_args()

    X, y, patients = build_or_load_cache(args.data_dir)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=patients))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"\n[SPLIT] Train cycles: {len(y_train)} | Val cycles: {len(y_val)}")
    for c_idx, c_name in enumerate(CLASSES):
        print(f"  {c_name:8s}: Train={np.sum(y_train == c_idx):4d}, Val={np.sum(y_val == c_idx):4d}")

    counts = np.bincount(y_train, minlength=4).astype(np.float32)
    weights = np.sqrt(counts.sum() / (4.0 * np.maximum(counts, 1.0)))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(LungDataset(X_train, y_train, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LungDataset(X_val, y_val, augment=False), batch_size=args.batch_size, shuffle=False)

    model = LungCNN(num_classes=4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    print("\n[TRAINING] Starting CPU training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_preds.extend(out.argmax(dim=1).numpy())
                val_targets.extend(yb.numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "target_sr": TARGET_SR,
                "cycle_seconds": CYCLE_SECONDS,
                "fmin": FMIN,
                "fmax": FMAX,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
            }, args.out_model)

    print(f"\n[DONE] Best Validation Accuracy: {best_val_acc*100:.2f}% (Saved to {args.out_model})")
    print("\n=== Classification Report on Validation Set ===")
    print(classification_report(val_targets, val_preds, target_names=CLASSES, digits=3))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(val_targets, val_preds))


if __name__ == "__main__":
    main()

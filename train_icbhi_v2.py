"""
train_icbhi_v2.py - same data, cache and model as train_icbhi.py, but with honest evaluation:
  * 3-way split by patient (train / validation / test)
  * best epoch chosen by validation macro-recall (not accuracy)
  * final report computed on the TEST set with the BEST checkpoint
  * ICBHI score (Se, Sp), per-class recall, confusion matrix
  * fixed random seed
  * stronger augmentation (time + frequency masks, random smooth spectral colouring, light noise)
The checkpoint keeps the same keys as v1, so predict_lung.py can load it.
Put this file in the same folder as train_icbhi.py.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from train_icbhi import (CLASSES, CYCLE_SECONDS, FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS,
                         TARGET_SR, LungCNN, build_or_load_cache)


def macro_recall(cm):
    cm = np.asarray(cm, dtype=np.float64)
    return float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1)))


def icbhi_scores(cm):
    """Sp = Normal recall; Se = correct abnormal (Crackle/Wheeze/Both) / all abnormal."""
    cm = np.asarray(cm, dtype=np.float64)
    sp = cm[0, 0] / max(cm[0].sum(), 1)
    se = np.trace(cm[1:, 1:]) / max(cm[1:].sum(), 1)
    return float(se), float(sp), float((se + sp) / 2)


def augment_spec(spec, eq_amp=0.3, noise_std=0.05):
    spec = spec.copy()
    n_mels, n_t = spec.shape
    for _ in range(2):                                   # frequency masks
        w = np.random.randint(0, 9)
        if w:
            f0 = np.random.randint(0, n_mels - w + 1)
            spec[f0:f0 + w, :] = 0.0
    for _ in range(2):                                   # time masks
        w = np.random.randint(0, 21)
        if w:
            t0 = np.random.randint(0, n_t - w + 1)
            spec[:, t0:t0 + w] = 0.0
    if eq_amp > 0:                                       # random smooth "device / tube" colouring
        knots = np.random.randn(6) * eq_amp
        prof = np.interp(np.linspace(0, 5, n_mels), np.arange(6), knots)
        spec += prof[:, None].astype(np.float32)
    if noise_std > 0:
        spec += (np.random.randn(n_mels, n_t) * noise_std).astype(np.float32)
    return spec.astype(np.float32)


class AugDataset(Dataset):
    def __init__(self, X, y, augment=False, eq_amp=0.3, noise_std=0.05):
        self.X, self.y = X, y
        self.augment, self.eq_amp, self.noise_std = augment, eq_amp, noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = self.X[idx]
        if self.augment:
            spec = augment_spec(spec, self.eq_amp, self.noise_std)
        return torch.from_numpy(np.ascontiguousarray(spec, dtype=np.float32)).unsqueeze(0), \
            torch.tensor(self.y[idx], dtype=torch.long)


def predict_all(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb).argmax(dim=1).numpy())
            targets.extend(yb.numpy())
    return np.array(targets), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eq_amp", type=float, default=0.3)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--out_model", default="saby_lung_cnn_v2.pt")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, y, patients = build_or_load_cache(args.data_dir)

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    trainval_idx, test_idx = next(outer.split(X, y, groups=patients))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_rel, va_rel = next(inner.split(X[trainval_idx], y[trainval_idx], groups=patients[trainval_idx]))
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]

    print(f"\n[SPLIT] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(patients: {len(set(patients[train_idx]))}/{len(set(patients[val_idx]))}/{len(set(patients[test_idx]))})")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        c = np.bincount(y[idx], minlength=4)
        print(f"  {name:5s}: " + " ".join(f"{CLASSES[i]}={c[i]}" for i in range(4)))

    counts = np.bincount(y[train_idx], minlength=4).astype(np.float32)
    weights = np.sqrt(counts.sum() / (4.0 * np.maximum(counts, 1.0)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=0.1)

    train_loader = DataLoader(AugDataset(X[train_idx], y[train_idx], True, args.eq_amp, args.noise_std),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AugDataset(X[val_idx], y[val_idx]), batch_size=args.batch_size)
    test_loader = DataLoader(AugDataset(X[test_idx], y[test_idx]), batch_size=args.batch_size)

    model = LungCNN(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mr, best_state, best_epoch = -1.0, None, 0
    print("\n[TRAINING]")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(dim=1) == yb).sum().item()
            tn += len(yb)
        scheduler.step()

        vt, vp = predict_all(model, val_loader)
        cm = confusion_matrix(vt, vp, labels=[0, 1, 2, 3])
        mr = macro_recall(cm)
        se, sp, sc = icbhi_scores(cm)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {tl / tn:.3f} | train acc {tc / tn * 100:.1f}% | "
              f"val acc {np.mean(vt == vp) * 100:.1f}% | val macro-recall {mr * 100:.1f}% | "
              f"val ICBHI {sc * 100:.1f}% (Se {se * 100:.1f} / Sp {sp * 100:.1f})")
        if mr > best_mr:
            best_mr, best_epoch = mr, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state, "classes": CLASSES, "target_sr": TARGET_SR,
        "cycle_seconds": CYCLE_SECONDS, "fmin": FMIN, "fmax": FMAX, "n_mels": N_MELS,
        "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "best_epoch": best_epoch, "val_macro_recall": best_mr,
    }, args.out_model)

    tt, tp = predict_all(model, test_loader)
    cm = confusion_matrix(tt, tp, labels=[0, 1, 2, 3])
    se, sp, sc = icbhi_scores(cm)
    print(f"\n[DONE] best epoch {best_epoch} (val macro-recall {best_mr * 100:.1f}%) -> saved to {args.out_model}")
    print("\n=== TEST set (never used for choosing the epoch), best checkpoint ===")
    print(classification_report(tt, tp, labels=[0, 1, 2, 3], target_names=CLASSES, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted; Normal, Crackle, Wheeze, Both):")
    print(cm)
    print(f"Test accuracy {np.mean(tt == tp) * 100:.1f}% | macro-recall {macro_recall(cm) * 100:.1f}% | "
          f"ICBHI score {sc * 100:.1f}% (Se {se * 100:.1f}% / Sp {sp * 100:.1f}%)")
    print("Note: the test set contains few patients, so these numbers vary a lot with the split/seed.")


if __name__ == "__main__":
    main()
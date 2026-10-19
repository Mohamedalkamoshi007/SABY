"""
eval_panns_head.py - patient-grouped cross-validation of simple classifiers on the PANNs embeddings.
Tasks: 4-class (Normal/Crackle/Wheeze/Both), plus two independent binary tasks: crackle present? wheeze present?
Reports pooled out-of-fold results (every patient is tested only by a model that never saw that patient).
Usage: py eval_panns_head.py [--emb icbhi_panns_emb.npz] [--folds 5] [--seeds 3] [--save]
"""
import argparse
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]


def macro_recall(cm):
    cm = np.asarray(cm, dtype=np.float64)
    return float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1)))


def icbhi_scores(cm):
    cm = np.asarray(cm, dtype=np.float64)
    sp = cm[0, 0] / max(cm[0].sum(), 1)
    se = np.trace(cm[1:, 1:]) / max(cm[1:].sum(), 1)
    return float(se), float(sp), float((se + sp) / 2)


def make_model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=500))


def cv_oof(X, y, groups, C, seed, folds, proba=False):
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    n_cls = len(np.unique(y))
    oof = np.zeros((len(y), n_cls), dtype=np.float64) if proba else np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y, groups):
        m = make_model(C).fit(X[tr], y[tr])
        if proba:
            oof[te] = m.predict_proba(X[te])
        else:
            oof[te] = m.predict(X[te])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="icbhi_panns_emb.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    ap.add_argument("--save", action="store_true", help="fit on all data with the best C and save panns_heads.joblib")
    args = ap.parse_args()

    d = np.load(args.emb, allow_pickle=True)
    X, y, groups = d["emb"].astype(np.float32), d["y"].astype(int), d["patients"]
    crackle = np.isin(y, [1, 3]).astype(int)
    wheeze = np.isin(y, [2, 3]).astype(int)
    print(f"cycles={len(y)} patients={len(set(groups))} classes={np.bincount(y, minlength=4).tolist()} "
          f"crackle+={crackle.sum()} wheeze+={wheeze.sum()}")

    best = {"score": -1, "C": None}
    print("\n=== 4-class (pooled over folds, mean over seeds) ===")
    print("   C    | macro-recall | ICBHI score (Se / Sp) | accuracy")
    for C in args.Cs:
        mr, sc, se_, sp_, acc = [], [], [], [], []
        for seed in range(args.seeds):
            pred = cv_oof(X, y, groups, C, seed, args.folds)
            cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
            se, sp, s = icbhi_scores(cm)
            mr.append(macro_recall(cm)); sc.append(s); se_.append(se); sp_.append(sp); acc.append(np.mean(pred == y))
        print(f" {C:<6g} | {np.mean(mr) * 100:5.1f} +-{np.std(mr) * 100:.1f} | {np.mean(sc) * 100:5.1f} "
              f"({np.mean(se_) * 100:.1f} / {np.mean(sp_) * 100:.1f}) | {np.mean(acc) * 100:.1f}")
        if np.mean(mr) > best["score"]:
            best = {"score": float(np.mean(mr)), "C": C, "cm": cm}
    print(f"\nconfusion matrix of the last seed for the best C={best['C']:g} (rows=true; Normal, Crackle, Wheeze, Both):")
    print(best["cm"])

    print("\n=== binary tasks with the best C: AUC (pooled out-of-fold, mean +- std over seeds) ===")
    for name, lab in (("crackle present", crackle), ("wheeze present", wheeze), ("any abnormality", (y > 0).astype(int))):
        aucs = []
        for seed in range(args.seeds):
            p = cv_oof(X, lab, groups, best["C"], seed, args.folds, proba=True)[:, 1]
            aucs.append(roc_auc_score(lab, p))
        print(f"  {name:16s}: AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}   (0.5 = chance)")

    print("\nNote: C was chosen on the same cross-validation, so the best-C numbers are slightly optimistic.")
    print("Reference: the small CNN from scratch had macro-recall 36.1% / ICBHI 33.3% on its own test split.")

    if args.save:
        heads = {"C": best["C"], "classes": CLASSES,
                 "four_class": make_model(best["C"]).fit(X, y),
                 "crackle": make_model(best["C"]).fit(X, crackle),
                 "wheeze": make_model(best["C"]).fit(X, wheeze)}
        joblib.dump(heads, "panns_heads.joblib")
        print("saved panns_heads.joblib")


if __name__ == "__main__":
    main()
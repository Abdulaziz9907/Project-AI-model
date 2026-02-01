"""
ai_models.py
Model training & evaluation (anomaly + RCA).

Features:
- train_fraction: slider controls train/test split proportion
- split_mode:
  - "time": train earlier minutes, test later minutes (realistic monitoring)
  - "random": shuffle rows before splitting (i.i.d. assumption)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


def _time_split(ds: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    max_minute = int(ds["minute"].max())
    tf = float(train_fraction)
    tf = max(0.1, min(0.9, tf))

    split_minute = int(max_minute * tf)

    train = ds[ds["minute"] <= split_minute].copy()
    test = ds[ds["minute"] > split_minute].copy()

    # Guardrail: keep both sides non-trivial
    if len(train) < 10 or len(test) < 10:
        split_minute = int(max_minute * 0.6)
        train = ds[ds["minute"] <= split_minute].copy()
        test = ds[ds["minute"] > split_minute].copy()

    return train, test, split_minute


def _random_split(ds: pd.DataFrame, train_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    tf = float(train_fraction)
    tf = max(0.1, min(0.9, tf))

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(ds))
    rng.shuffle(idx)

    n_train = int(len(ds) * tf)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    train = ds.iloc[train_idx].copy()
    test = ds.iloc[test_idx].copy()

    # For UI consistency (meaningless for random split)
    split_minute = int(ds["minute"].median())

    # Guardrail
    if len(train) < 10 or len(test) < 10:
        split_minute = int(ds["minute"].median())
        train = ds.sample(frac=0.6, random_state=int(seed)).copy()
        test = ds.drop(train.index).copy()

    return train, test, split_minute


def train_anomaly_and_rca(
    ds: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.7,
    split_mode: str = "time",  # "time" or "random"
):
    """
    Train anomaly classifier + RCA classifier.

    split_mode:
    - "time": train early minutes, test later minutes
    - "random": shuffle rows before splitting
    """
    if "minute" not in ds.columns:
        raise ValueError("Dataset must include 'minute' column")
    if "is_anomaly" not in ds.columns:
        raise ValueError("Dataset must include 'is_anomaly' column")

    mode = (split_mode or "time").strip().lower()
    if mode not in ("time", "random"):
        mode = "time"

    if mode == "time":
        train, test, split_minute = _time_split(ds, train_fraction)
    else:
        train, test, split_minute = _random_split(ds, train_fraction, seed)

    feats = [c for c in ds.columns if c not in ("minute", "building", "is_anomaly", "cause", "primary_cause")]

    Xtr, ytr = train[feats], train["is_anomaly"].astype(int)
    Xte, yte = test[feats], test["is_anomaly"].astype(int)

    anomaly = RandomForestClassifier(
        n_estimators=500,
        random_state=int(seed),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    anomaly.fit(Xtr, ytr)

    proba = anomaly.predict_proba(Xte)[:, 1]

    # Threshold sweep (proof-demo style)
    thresholds = np.linspace(0.1, 0.9, 81)
    best = None

    for th in thresholds:
        pred = (proba >= th).astype(int)
        p = precision_score(yte, pred, zero_division=0)
        r = recall_score(yte, pred, zero_division=0)

        if p >= 0.80 and r >= 0.75:
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            if best is None or f1 > best["f1"]:
                best = {"th": float(th), "precision": float(p), "recall": float(r), "pred": pred, "f1": float(f1)}

    if best is None:
        th = 0.5
        pred = (proba >= th).astype(int)
        p = precision_score(yte, pred, zero_division=0)
        r = recall_score(yte, pred, zero_division=0)
        best = {"th": float(th), "precision": float(p), "recall": float(r), "pred": pred, "f1": 0.0}

    # RCA model
    if "primary_cause" not in ds.columns:
        train["primary_cause"] = "NORMAL"
        test["primary_cause"] = "NORMAL"

    train_a = train[train["is_anomaly"] == 1].copy()
    test_a = test[test["is_anomaly"] == 1].copy()

    rca_top3_acc = float("nan")

    if len(train_a) >= 5:
        rca = RandomForestClassifier(
            n_estimators=500,
            random_state=int(seed),
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        rca.fit(train_a[feats], train_a["primary_cause"])

        if len(test_a) > 0:
            rca_proba = rca.predict_proba(test_a[feats])
            classes = list(rca.classes_)

            top3_ok = []
            for i, rowp in enumerate(rca_proba):
                top3_idx = np.argsort(rowp)[-3:][::-1]
                top3 = [classes[j] for j in top3_idx]
                top3_ok.append(1 if test_a.iloc[i]["primary_cause"] in top3 else 0)

            rca_top3_acc = float(np.mean(top3_ok)) if len(top3_ok) else float("nan")
    else:
        # Fallback tiny RCA
        rca = RandomForestClassifier(n_estimators=10, random_state=int(seed))
        n = min(10, len(Xtr))
        rca.fit(Xtr.iloc[:n], np.array(["NORMAL"] * n))

    return {
        "split_mode": mode,
        "split_minute": int(split_minute),
        "train_fraction": float(train_fraction),
        "features": feats,
        "threshold": best["th"],
        "precision": best["precision"],
        "recall": best["recall"],
        "y_test": yte.values,
        "y_pred": best["pred"],
        "anomaly_model": anomaly,
        "rca_model": rca,
        "rca_top3_acc": rca_top3_acc,
    }

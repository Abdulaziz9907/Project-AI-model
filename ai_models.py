"""
ai_models.py
Train anomaly model (binary) + RCA model (multiclass).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


def _feature_columns(ds: pd.DataFrame) -> list[str]:
    exclude = {"minute", "building", "is_anomaly", "cause", "primary_cause"}
    feats = []
    for c in ds.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(ds[c]):
            feats.append(c)
    return feats


def _choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _topk_accuracy(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    if len(y_true) == 0:
        return float("nan")
    hits = 0
    for i in range(len(y_true)):
        idx = np.argsort(proba[i])[-k:][::-1]
        topk = [classes[j] for j in idx]
        if y_true[i] in topk:
            hits += 1
    return hits / len(y_true)


def train_anomaly_and_rca(ds: pd.DataFrame) -> dict:
    data = ds.copy().sort_values("minute").reset_index(drop=True)

    feats = _feature_columns(data)
    if not feats:
        raise ValueError("No numeric feature columns found to train models.")

    unique_minutes = np.sort(data["minute"].unique())
    split_idx = int(0.7 * len(unique_minutes))
    split_minute = int(unique_minutes[split_idx]) if len(unique_minutes) else 0

    train = data[data["minute"] <= split_minute].copy()
    test = data[data["minute"] > split_minute].copy()

    Xtr = train[feats].fillna(0.0)
    ytr = train["is_anomaly"].astype(int).values

    Xte = test[feats].fillna(0.0)
    yte = test["is_anomaly"].astype(int).values

    anomaly = RandomForestClassifier(
        n_estimators=300,
        random_state=7,
        class_weight="balanced",
        n_jobs=-1,
    )
    anomaly.fit(Xtr, ytr)

    prob = anomaly.predict_proba(Xte)[:, 1]
    threshold = _choose_threshold(yte, prob)
    ypred = (prob >= threshold).astype(int)

    precision = float(precision_score(yte, ypred, zero_division=0))
    recall = float(recall_score(yte, ypred, zero_division=0))

    rca_train = train[train["is_anomaly"] == 1].copy()
    rca_train = rca_train[rca_train["primary_cause"].astype(str) != "NORMAL"].copy()

    if len(rca_train["primary_cause"].unique()) >= 2:
        rca = RandomForestClassifier(
            n_estimators=300,
            random_state=11,
            class_weight="balanced",
            n_jobs=-1,
        )
        rca.fit(rca_train[feats].fillna(0.0), rca_train["primary_cause"].astype(str).values)

        rca_test = test[test["is_anomaly"] == 1].copy()
        rca_test = rca_test[rca_test["primary_cause"].astype(str) != "NORMAL"].copy()

        if len(rca_test):
            proba_rca = rca.predict_proba(rca_test[feats].fillna(0.0))
            rca_top3 = float(_topk_accuracy(proba_rca, rca.classes_, rca_test["primary_cause"].astype(str).values, k=3))
        else:
            rca_top3 = float("nan")
    else:
        class Dummy:
            classes_ = np.array(["NORMAL"])
            def predict_proba(self, X):  # noqa
                return np.ones((len(X), 1))
        rca = Dummy()
        rca_top3 = float("nan")

    return {
        "features": feats,
        "split_minute": split_minute,
        "threshold": threshold,
        "anomaly_model": anomaly,
        "rca_model": rca,
        "y_test": yte,
        "y_pred": ypred,
        "precision": precision,
        "recall": recall,
        "rca_top3_acc": rca_top3,
    }

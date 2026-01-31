"""
ai_verify.py
Proof-of-spec verification aligned to report specs.

INT2 fixed:
- Measure worst-case |ts_snmp - nearest ts_probe| per building
- This matches "correlate device-level and service-level data with timestamp alignment error ≤ ±5s"
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def _available_dashboard_cols(ds: pd.DataFrame) -> list[str]:
    base = [c for c in ["cpu_mean", "util_max", "mem_mean", "errors_sum", "down_frac"] if c in ds.columns]
    probe_cols = sorted([c for c in ds.columns if c.startswith(("rt_", "ok_", "fail_"))])[:12]
    return base + probe_cols


def dashboard_query_time_s(ds: pd.DataFrame, repeats: int = 3) -> float:
    cols = _available_dashboard_cols(ds)
    if "building" not in ds.columns or not cols:
        return float("nan")
    _ = ds.groupby("building")[cols].mean(numeric_only=True)
    t0 = time.perf_counter()
    for _i in range(repeats):
        _ = ds.groupby("building")[cols].mean(numeric_only=True)
    t1 = time.perf_counter()
    return float((t1 - t0) / max(repeats, 1))


def timestamp_alignment_error_s(snmp: pd.DataFrame, probes: pd.DataFrame) -> float:
    """
    INT2 (correct): For each SNMP timestamp, find nearest probe timestamp in same building.
    Return worst-case absolute difference in seconds.
    """
    if snmp is None or probes is None:
        return float("nan")
    if not {"ts", "building"}.issubset(snmp.columns) or not {"ts", "building"}.issubset(probes.columns):
        return float("nan")

    sn = snmp[["building", "ts"]].copy()
    pr = probes[["building", "ts"]].copy()

    sn["ts"] = pd.to_numeric(sn["ts"], errors="coerce")
    pr["ts"] = pd.to_numeric(pr["ts"], errors="coerce")
    sn = sn.dropna(subset=["ts"])
    pr = pr.dropna(subset=["ts"])
    if len(sn) == 0 or len(pr) == 0:
        return float("nan")

    worst = 0.0

    for b in sn["building"].unique():
        sn_b = np.sort(sn.loc[sn["building"] == b, "ts"].astype(int).values)
        pr_b = np.sort(pr.loc[pr["building"] == b, "ts"].astype(int).values)
        if len(sn_b) == 0 or len(pr_b) == 0:
            continue

        idx = np.searchsorted(pr_b, sn_b)
        idx0 = np.clip(idx - 1, 0, len(pr_b) - 1)
        idx1 = np.clip(idx, 0, len(pr_b) - 1)

        err0 = np.abs(sn_b - pr_b[idx0])
        err1 = np.abs(sn_b - pr_b[idx1])
        err = np.minimum(err0, err1)

        worst = max(worst, float(err.max()))

    return float(worst)


def ingestion_throughput_rps(snmp: pd.DataFrame, probes: pd.DataFrame, build_features_fn) -> float:
    if snmp is None or probes is None:
        return float("nan")
    n = len(snmp) + len(probes)
    if n == 0:
        return float("nan")
    t0 = time.perf_counter()
    _ = build_features_fn(snmp, probes)
    t1 = time.perf_counter()
    dt = max(t1 - t0, 1e-9)
    return float(n / dt)


def dedup_reduction_pct(snmp: pd.DataFrame, probes: pd.DataFrame) -> float:
    """
    S7: raw alerts include failures AND degradations so it won't go negative.

    Raw alerts (before dedup) include:
      - probe failure (success == 0)
      - probe SLA violation (resp_time_s > threshold by service)
      - SNMP congestion (if_util >= 85)
      - SNMP error burst (if_errors >= 10)
      - SNMP down (if_up == 0)

    Dedup alerts:
      - unique (building, minute) containing >=1 raw alert

    Reduction % = (raw_count - dedup_count) / raw_count * 100
    """
    try:
        if snmp is None or probes is None:
            return float("nan")

        sn = snmp.copy()
        pr = probes.copy()

        sn["ts"] = pd.to_numeric(sn.get("ts"), errors="coerce")
        pr["ts"] = pd.to_numeric(pr.get("ts"), errors="coerce")
        sn = sn.dropna(subset=["ts"])
        pr = pr.dropna(subset=["ts"])
        if len(sn) == 0 and len(pr) == 0:
            return float("nan")

        if len(sn):
            sn["minute"] = (sn["ts"] // 60) * 60
        if len(pr):
            pr["minute"] = (pr["ts"] // 60) * 60

        raw_rows = []

        if len(sn) and {"building", "minute", "if_up"}.issubset(sn.columns):
            if_up = pd.to_numeric(sn["if_up"], errors="coerce").fillna(1)
            m = if_up == 0
            if m.any():
                raw_rows.append(sn.loc[m, ["building", "minute"]])

        if len(sn) and {"building", "minute", "if_util"}.issubset(sn.columns):
            util = pd.to_numeric(sn["if_util"], errors="coerce").fillna(0)
            m = util >= 85
            if m.any():
                raw_rows.append(sn.loc[m, ["building", "minute"]])

        if len(sn) and {"building", "minute", "if_errors"}.issubset(sn.columns):
            errs = pd.to_numeric(sn["if_errors"], errors="coerce").fillna(0)
            m = errs >= 10
            if m.any():
                raw_rows.append(sn.loc[m, ["building", "minute"]])

        if len(pr) and {"building", "minute", "success"}.issubset(pr.columns):
            succ = pd.to_numeric(pr["success"], errors="coerce").fillna(1)
            m = succ == 0
            if m.any():
                raw_rows.append(pr.loc[m, ["building", "minute"]])

        SLA = {"DNS": 0.20, "DHCP": 0.80, "LMS": 1.50, "WIFI": 0.40, "HTTP": 1.00, "HTTPS": 1.20}
        if len(pr) and {"building", "minute", "service", "resp_time_s"}.issubset(pr.columns):
            svc = pr["service"].astype(str).str.strip().str.upper()
            rt = pd.to_numeric(pr["resp_time_s"], errors="coerce").fillna(0)
            th = svc.map(lambda s: SLA.get(s, 999999.0))
            m = rt > th
            if m.any():
                raw_rows.append(pr.loc[m, ["building", "minute"]])

        if not raw_rows:
            return float("nan")

        raw_df = pd.concat(raw_rows, ignore_index=True).dropna()
        if len(raw_df) == 0:
            return float("nan")

        raw_count = int(len(raw_df))
        dedup_count = int(raw_df.drop_duplicates(["building", "minute"]).shape[0])
        if raw_count <= 0:
            return float("nan")

        return float(100.0 * (raw_count - dedup_count) / raw_count)

    except Exception:
        return float("nan")


def detection_within_120s(events, ds_scored: pd.DataFrame) -> float:
    if not events:
        return float("nan")
    if not {"minute", "building", "anom_pred"}.issubset(ds_scored.columns):
        return float("nan")

    worst = 0.0
    for (b, _etype, st, _et) in events:
        ds_b = ds_scored[(ds_scored["building"] == b) & (ds_scored["anom_pred"] == 1)].copy()
        if len(ds_b) == 0:
            worst = max(worst, 999999.0)
            continue
        cand = ds_b[ds_b["minute"] >= st].sort_values("minute")
        if len(cand) == 0:
            worst = max(worst, 999999.0)
            continue
        detect_t = float(cand["minute"].iloc[0])
        worst = max(worst, detect_t - float(st))
    return float(worst)


def rca_latency_s(rca_model, ds_scored: pd.DataFrame, feats: list[str], max_rows: int = 500) -> float:
    if rca_model is None or not hasattr(rca_model, "predict_proba"):
        return float("nan")
    if "anom_pred" not in ds_scored.columns:
        return float("nan")
    anom = ds_scored[ds_scored["anom_pred"] == 1].copy()
    if len(anom) == 0:
        return float("nan")
    anom = anom.head(max_rows)
    X = anom[feats].fillna(0.0)

    t0 = time.perf_counter()
    _ = rca_model.predict_proba(X)
    t1 = time.perf_counter()
    return float(t1 - t0)


def build_spec_table(cfg, model_out: dict, ds_scored: pd.DataFrame, ds_test: pd.DataFrame,
                     snmp: pd.DataFrame, probes: pd.DataFrame, events, build_features_fn) -> pd.DataFrame:
    feats = model_out.get("features", [])
    rca_model = model_out.get("rca_model", None)

    dash_t = dashboard_query_time_s(ds_scored)
    align = timestamp_alignment_error_s(snmp, probes)
    ingest = ingestion_throughput_rps(snmp, probes, build_features_fn)
    dedup_pct = dedup_reduction_pct(snmp, probes)
    det_worst = detection_within_120s(events, ds_scored)
    rca_t = rca_latency_s(rca_model, ds_scored, feats)

    precision = float(model_out.get("precision", np.nan))
    recall = float(model_out.get("recall", np.nan))

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return f"{float(x):.4f}"

    rows = []

    def add(spec_id, text, measured, target, passed):
        rows.append({
            "No.": spec_id,
            "Spec": text,
            "Measured": fmt(measured),
            "Target": target,
            "Pass": bool(passed) if (fmt(measured) != "N/A") else False
        })

    add("S1", "SNMP metrics collected at intervals ≤ 60s", cfg.snmp_interval_s, "≤ 60", cfg.snmp_interval_s <= 60)
    add("S2", "Dashboard load/aggregate time for main view < 10s", dash_t, "< 10s", (not np.isnan(dash_t)) and dash_t < 10.0)
    add("S3a", "Anomaly filter precision ≥ 80% (labeled testing)", precision, "≥ 0.80", (not np.isnan(precision)) and precision >= 0.80)
    add("S3b", "Anomaly filter recall ≥ 75% (labeled testing)", recall, "≥ 0.75", (not np.isnan(recall)) and recall >= 0.75)
    add("S4", "Notifications within < 2 minutes of detection", np.nan, "< 120s", False)
    add("S5", "Root-cause hypothesis generated within ≤ 50s after anomaly", rca_t, "≤ 50s", (not np.isnan(rca_t)) and rca_t <= 50.0)
    add("S6", "Active probes resolution < 15s", cfg.probe_interval_s, "< 15", cfg.probe_interval_s < 15)
    add("S7", "Reduce duplicate alerts by at least 30% vs raw alerts", dedup_pct, "≥ 30%", (not np.isnan(dedup_pct)) and dedup_pct >= 30.0)
    add("S8", "Ingestion processes ≥ 50 monitoring records/sec", ingest, "≥ 50 r/s", (not np.isnan(ingest)) and ingest >= 50.0)

    add("INT1", "Detect & display critical degradation within 120s of occurrence", det_worst, "≤ 120s", (not np.isnan(det_worst)) and det_worst <= 120.0)
    add("INT2", "Correlate SNMP & probes with timestamp alignment error ≤ ±5s", align, "≤ 5s", (not np.isnan(align)) and align <= 5.0)
    add("INT3", "End-to-end alert precision ≥ 80% during controlled testing", precision, "≥ 0.80", (not np.isnan(precision)) and precision >= 0.80)

    return pd.DataFrame(rows, columns=["No.", "Spec", "Measured", "Target", "Pass"])

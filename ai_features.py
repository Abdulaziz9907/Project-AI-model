"""
ai_features.py
Minute-level feature engineering for SNMP + probe telemetry.

- SNMP aggregated per (building, minute)
- Probes aggregated per (building, minute, service)
- Dynamic services => columns:
    rt_{SERVICE}, ok_{SERVICE}, fail_{SERVICE}
Optional:
    pl_{SERVICE} if packet loss column exists
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REPORT_SERVICES = ["DNS", "DHCP", "LMS", "WIFI", "HTTP", "HTTPS"]


def _minute_bucket(ts: pd.Series) -> pd.Series:
    return (ts // 60) * 60


def _normalize_service(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "WI-FI": "WIFI",
                "WLAN": "WIFI",
                "WIRELESS": "WIFI",
                "LEARNING MANAGEMENT SYSTEM": "LMS",
                "WEB": "HTTP",
            }
        )
    )


def primary_cause(cause_str: str) -> str:
    if not isinstance(cause_str, str) or not cause_str.strip():
        return "NORMAL"
    parts = [p.strip().upper() for p in cause_str.split(";") if p.strip()]
    if not parts:
        return "NORMAL"

    priority = [
        "WLC_DOWN",
        "WIFI_DOWN",
        "UPLINK_CONGEST",
        "DNS_OUTAGE",
        "DHCP_ISSUE",
        "LMS_OUTAGE",
        "HTTP_OUTAGE",
        "HTTPS_OUTAGE",
        "LMS_SLOW",
        "HTTP_SLOW",
        "HTTPS_SLOW",
    ]
    for p in priority:
        if p in parts:
            return p
    return parts[0]


def build_minute_features(snmp: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    sn = snmp.copy()
    pr = probes.copy()

    # -------- SNMP --------
    sn["ts"] = pd.to_numeric(sn.get("ts"), errors="coerce").fillna(0).astype(int)
    sn["minute"] = _minute_bucket(sn["ts"])

    for c in ["cpu", "mem", "if_util", "if_errors", "if_up"]:
        if c in sn.columns:
            sn[c] = pd.to_numeric(sn[c], errors="coerce")

    sn_agg = (
        sn.groupby(["building", "minute"], as_index=False)
        .agg(
            cpu_mean=("cpu", "mean"),
            cpu_max=("cpu", "max"),
            mem_mean=("mem", "mean"),
            util_mean=("if_util", "mean"),
            util_max=("if_util", "max"),
            errors_sum=("if_errors", "sum"),
            down_frac=("if_up", lambda x: float((x.fillna(1) == 0).mean())),
        )
    )

    if "dtype" in sn.columns:
        ap = sn[sn["dtype"].astype(str).str.upper() == "AP"].copy()
        if len(ap):
            ap_agg = ap.groupby(["building", "minute"], as_index=False).agg(
                ap_down_frac=("if_up", lambda x: float((x.fillna(1) == 0).mean()))
            )
            sn_agg = sn_agg.merge(ap_agg, on=["building", "minute"], how="left")
        else:
            sn_agg["ap_down_frac"] = 0.0

        wlc = sn[sn["dtype"].astype(str).str.upper() == "WLC"].copy()
        if len(wlc):
            wlc_agg = wlc.groupby(["building", "minute"], as_index=False).agg(
                wlc_down_frac=("if_up", lambda x: float((x.fillna(1) == 0).mean()))
            )
            sn_agg = sn_agg.merge(wlc_agg, on=["building", "minute"], how="left")
        else:
            sn_agg["wlc_down_frac"] = 0.0
    else:
        sn_agg["ap_down_frac"] = 0.0
        sn_agg["wlc_down_frac"] = 0.0

    sn_agg["ap_down_frac"] = sn_agg["ap_down_frac"].fillna(0.0)
    sn_agg["wlc_down_frac"] = sn_agg["wlc_down_frac"].fillna(0.0)

    # -------- PROBES --------
    if pr is None or len(pr) == 0 or ("service" not in pr.columns):
        return sn_agg.sort_values(["building", "minute"]).reset_index(drop=True)

    pr["ts"] = pd.to_numeric(pr.get("ts"), errors="coerce").fillna(0).astype(int)
    pr["minute"] = _minute_bucket(pr["ts"])
    pr["service"] = _normalize_service(pr["service"])

    if "resp_time_s" not in pr.columns:
        if "latency_s" in pr.columns:
            pr["resp_time_s"] = pr["latency_s"]
        elif "rtt_ms" in pr.columns:
            pr["resp_time_s"] = pd.to_numeric(pr["rtt_ms"], errors="coerce") / 1000.0
        else:
            pr["resp_time_s"] = np.nan

    pr["resp_time_s"] = pd.to_numeric(pr["resp_time_s"], errors="coerce")
    pr["success"] = pd.to_numeric(pr.get("success", 1), errors="coerce").fillna(1).astype(int)
    pr["fail"] = (1 - pr["success"]).clip(0, 1)

    pl_col = None
    for cand in ["packet_loss", "pkt_loss", "loss", "loss_rate"]:
        if cand in pr.columns:
            pl_col = cand
            pr[cand] = pd.to_numeric(pr[cand], errors="coerce")
            break

    grp_cols = ["building", "minute", "service"]
    aggs = {"resp_time_s": "mean", "success": "mean", "fail": "sum"}
    if pl_col:
        aggs[pl_col] = "mean"

    pr_svc = pr.groupby(grp_cols, as_index=False).agg(aggs)

    wide_parts = []
    all_svcs = sorted(set(pr_svc["service"].unique()).union(REPORT_SERVICES))
    for svc in all_svcs:
        sub = pr_svc[pr_svc["service"] == svc].copy()
        if len(sub) == 0:
            continue
        sub = sub.rename(
            columns={
                "resp_time_s": f"rt_{svc}",
                "success": f"ok_{svc}",
                "fail": f"fail_{svc}",
            }
        )
        keep = ["building", "minute", f"rt_{svc}", f"ok_{svc}", f"fail_{svc}"]
        if pl_col:
            sub = sub.rename(columns={pl_col: f"pl_{svc}"})
            keep.append(f"pl_{svc}")
        wide_parts.append(sub[keep])

    if wide_parts:
        pr_wide = wide_parts[0]
        for w in wide_parts[1:]:
            pr_wide = pr_wide.merge(w, on=["building", "minute"], how="outer")
    else:
        pr_wide = pd.DataFrame(columns=["building", "minute"])

    out = sn_agg.merge(pr_wide, on=["building", "minute"], how="left")

    for c in out.columns:
        if c.startswith("ok_"):
            out[c] = out[c].fillna(1.0)
        elif c.startswith(("rt_", "pl_")):
            out[c] = out[c].fillna(0.0)
        elif c.startswith("fail_"):
            out[c] = out[c].fillna(0.0)

    return out.sort_values(["building", "minute"]).reset_index(drop=True)

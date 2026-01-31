"""
ai_features.py
Feature engineering module (data -> model-ready features).

Works with simulated SNMP/probes OR real data later if you load it
into the same column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_minute_features(snmp: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    sn = snmp.copy()
    pr = probes.copy()

    sn["minute"] = (sn.ts // 60) * 60
    pr["minute"] = (pr.ts // 60) * 60

    snmp_agg = sn.groupby(["minute", "building"]).agg(
        cpu_mean=("cpu", "mean"),
        cpu_max=("cpu", "max"),
        mem_mean=("mem", "mean"),
        util_mean=("if_util", "mean"),
        util_max=("if_util", "max"),
        errors_sum=("if_errors", "sum"),
        down_frac=("if_up", lambda x: 1 - np.mean(x)),
    ).reset_index()

    probe_agg = pr.groupby(["minute", "building", "service"]).agg(
        rt_mean=("resp_time_s", "mean"),
        ok_rate=("success", "mean"),
    ).reset_index()

    rt = (
        probe_agg.pivot_table(index=["minute", "building"], columns="service", values="rt_mean")
        .add_prefix("rt_")
        .reset_index()
    )
    ok = (
        probe_agg.pivot_table(index=["minute", "building"], columns="service", values="ok_rate")
        .add_prefix("ok_")
        .reset_index()
    )

    return snmp_agg.merge(rt, on=["minute", "building"]).merge(ok, on=["minute", "building"])


def primary_cause(c: str) -> str:
    if not isinstance(c, str) or c.strip() == "":
        return "NORMAL"
    if "WLC_DOWN" in c:
        return "WLC_DOWN"
    if "DNS_OUTAGE" in c:
        return "DNS_OUTAGE"
    if "UPLINK_CONGEST" in c:
        return "UPLINK_CONGEST"
    return c.split(";")[0]

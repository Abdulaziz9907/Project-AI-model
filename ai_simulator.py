"""
ai_simulator.py
Simulation-only module (data generation). No ML here.

Exports:
- SimConfig
- simulate(cfg) -> dev, snmp, probes, labels, events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    duration_minutes: int = 240
    n_buildings: int = 8
    n_aps_per_building: int = 14

    # Specs-driven sampling:
    snmp_interval_s: int = 60    # S1: <= 60s
    probe_interval_s: int = 14   # S6: < 15s (strict)

    # Timestamp jitter to test INT2 alignment:
    snmp_jitter_s: int = 2
    probe_jitter_s: int = 3

    seed: int = 7


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    end_ts = cfg.duration_minutes * 60

    # -----------------------------
    # Inventory (WLC + SW + APs)
    # -----------------------------
    devices = []
    for b in range(cfg.n_buildings):
        devices.append((b, "WLC", f"wlc-{b}", "parent", None))
        devices.append((b, "SW",  f"sw-{b}",  "parent", None))

        for a in range(cfg.n_aps_per_building):
            devices.append((b, "AP", f"ap-{b}-{a}", "child", f"wlc-{b}"))

    dev = pd.DataFrame(devices, columns=["building", "dtype", "device", "role", "parent"])

    # -----------------------------
    # Time grids with jitter
    # -----------------------------
    snmp_base = np.arange(0, end_ts, cfg.snmp_interval_s)
    probe_base = np.arange(0, end_ts, cfg.probe_interval_s)

    snmp_times = (
        snmp_base
        + rng.integers(-cfg.snmp_jitter_s, cfg.snmp_jitter_s + 1, size=len(snmp_base))
    ).clip(min=0)

    probe_times = (
        probe_base
        + rng.integers(-cfg.probe_jitter_s, cfg.probe_jitter_s + 1, size=len(probe_base))
    ).clip(min=0)

    # -----------------------------
    # SNMP metrics (generic fields)
    # -----------------------------
    snmp_rows = []
    for t in snmp_times:
        for _, d in dev.iterrows():
            dt = d["dtype"]

            base_cpu = {"WLC": 28, "SW": 18, "AP": 12}[dt]
            base_mem = {"WLC": 45, "SW": 40, "AP": 28}[dt]
            base_util = {"WLC": 15, "SW": 30, "AP": 8}[dt]

            cpu = rng.normal(base_cpu, 4)
            mem = rng.normal(base_mem, 5)
            if_util = rng.normal(base_util, 7)
            if_errors = max(0, int(rng.poisson(0.2 if dt != "AP" else 0.05)))
            if_up = 1

            snmp_rows.append(
                (
                    int(t),
                    int(d["building"]),
                    d["device"],
                    dt,
                    float(cpu),
                    float(mem),
                    float(if_util),
                    int(if_errors),
                    int(if_up),
                    d["parent"],
                    d["role"],
                )
            )

    snmp = pd.DataFrame(
        snmp_rows,
        columns=[
            "ts",
            "building",
            "device",
            "dtype",
            "cpu",
            "mem",
            "if_util",
            "if_errors",
            "if_up",
            "parent",
            "role",
        ],
    )

    # -----------------------------
    # Service probes (DNS/DHCP/WEB)
    # -----------------------------
    services = ["DNS", "DHCP", "WEB"]
    probe_rows = []
    for t in probe_times:
        for b in range(cfg.n_buildings):
            for s in services:
                rt_base = {"DNS": 0.03, "DHCP": 0.22, "WEB": 0.45}[s]
                rt = max(0.0, rng.normal(rt_base, rt_base * 0.25))
                success = 1 if rng.random() < 0.996 else 0
                probe_rows.append((int(t), int(b), s, float(rt), int(success)))

    probes = pd.DataFrame(
        probe_rows, columns=["ts", "building", "service", "resp_time_s", "success"]
    )

    # -----------------------------
    # Inject labeled incidents
    # -----------------------------
    events: List[Tuple[int, str, int, int]] = []

    for b in range(cfg.n_buildings):
        # DNS outage
        st = int(rng.integers(end_ts * 0.2, end_ts * 0.65))
        dur = int(rng.integers(300, 900))
        events.append((b, "DNS_OUTAGE", st, st + dur))

        # uplink congestion
        st2 = int(rng.integers(end_ts * 0.3, end_ts * 0.85))
        dur2 = int(rng.integers(600, 1200))
        events.append((b, "UPLINK_CONGEST", st2, st2 + dur2))

    # One WLC down event
    wlc_b = int(rng.integers(0, cfg.n_buildings))
    st3 = int(rng.integers(end_ts * 0.4, end_ts * 0.6))
    dur3 = 600
    events.append((wlc_b, "WLC_DOWN", st3, st3 + dur3))

    # Apply incidents to data
    for b, etype, st, et in events:
        if etype == "DNS_OUTAGE":
            m = (
                (probes.building == b)
                & (probes.service == "DNS")
                & (probes.ts >= st)
                & (probes.ts < et)
            )
            probes.loc[m, "success"] = 0
            probes.loc[m, "resp_time_s"] = probes.loc[m, "resp_time_s"] + np.abs(
                rng.normal(1.2, 0.25, m.sum())
            )

        elif etype == "UPLINK_CONGEST":
            m = (
                (snmp.building == b)
                & (snmp.dtype == "SW")
                & (snmp.ts >= st)
                & (snmp.ts < et)
            )
            snmp.loc[m, "if_util"] = snmp.loc[m, "if_util"] + np.abs(
                rng.normal(60, 10, m.sum())
            )
            snmp.loc[m, "if_errors"] = snmp.loc[m, "if_errors"] + rng.poisson(
                6, m.sum()
            )

            m2 = (
                (probes.building == b)
                & (probes.service == "WEB")
                & (probes.ts >= st)
                & (probes.ts < et)
            )
            probes.loc[m2, "resp_time_s"] = probes.loc[m2, "resp_time_s"] + np.abs(
                rng.normal(1.0, 0.3, m2.sum())
            )

        elif etype == "WLC_DOWN":
            m = (
                (snmp.building == b)
                & (snmp.dtype == "WLC")
                & (snmp.ts >= st)
                & (snmp.ts < et)
            )
            snmp.loc[m, "if_up"] = 0
            snmp.loc[m, "cpu"] = snmp.loc[m, "cpu"] + np.abs(
                rng.normal(35, 6, m.sum())
            )

            m_ap = (
                (snmp.building == b)
                & (snmp.dtype == "AP")
                & (snmp.ts >= st)
                & (snmp.ts < et)
            )
            snmp.loc[m_ap, "if_up"] = 0

            m_p = (probes.building == b) & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m_p, "success"] = 0
            probes.loc[m_p, "resp_time_s"] = probes.loc[m_p, "resp_time_s"] + np.abs(
                rng.normal(1.8, 0.4, m_p.sum())
            )

    # -----------------------------
    # Minute-level labels (ground truth)
    # -----------------------------
    minutes = np.arange(0, end_ts, 60)
    label_rows = []

    for b in range(cfg.n_buildings):
        for t in minutes:
            is_anom = 0
            causes = []
            for eb, etype, st, et in events:
                # if this minute overlaps the event
                if eb == b and t < et and (t + 60) > st:
                    is_anom = 1
                    causes.append(etype)

            label_rows.append(
                (int(t), int(b), int(is_anom), ";".join(sorted(set(causes))))
            )

    labels = pd.DataFrame(label_rows, columns=["minute", "building", "is_anomaly", "cause"])

    return dev, snmp, probes, labels, events

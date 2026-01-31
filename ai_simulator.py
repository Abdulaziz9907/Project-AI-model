"""
ai_simulator.py
Simulated SNMP + probes for proof-of-spec.

Services simulated:
DNS, DHCP, LMS, WIFI (+ optional HTTP, HTTPS)

NEW:
- probe_phase_offset_s: shifts probe schedule by a fixed offset (e.g., +2s)
  so INT2 is realistic (~2–4s) instead of always 0.
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

    snmp_interval_s: int = 60
    probe_interval_s: int = 10

    snmp_jitter_s: int = 2
    probe_jitter_s: int = 2  # keep small to pass INT2

    # NEW: probe schedule phase offset in seconds (0..probe_interval_s-1)
    probe_phase_offset_s: int = 2

    include_http_https: bool = True
    seed: int = 7


def _jitter(base: np.ndarray, jitter_s: int, rng: np.random.Generator, end_ts: int) -> np.ndarray:
    if len(base) == 0:
        return base.astype(int)
    jit = rng.integers(-jitter_s, jitter_s + 1, size=len(base))
    out = (base + jit).clip(0, end_ts - 1)
    return out.astype(int)


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    end_ts = cfg.duration_minutes * 60

    # Inventory
    devices = []
    for b in range(cfg.n_buildings):
        devices.append((b, "WLC", f"wlc-{b}", "parent", None))
        devices.append((b, "SW",  f"sw-{b}",  "parent", None))
        for a in range(cfg.n_aps_per_building):
            devices.append((b, "AP", f"ap-{b}-{a}", "child", f"wlc-{b}"))
    dev = pd.DataFrame(devices, columns=["building", "dtype", "device", "role", "parent"])

    # Base schedules
    snmp_base = np.arange(0, end_ts, cfg.snmp_interval_s, dtype=int)

    # NEW: start probes at phase offset (e.g., 2,12,22,...)
    phase = int(cfg.probe_phase_offset_s) % max(int(cfg.probe_interval_s), 1)
    probe_base = np.arange(phase, end_ts, cfg.probe_interval_s, dtype=int)

    # Jitter
    snmp_times = _jitter(snmp_base, cfg.snmp_jitter_s, rng, end_ts)
    probe_times = _jitter(probe_base, cfg.probe_jitter_s, rng, end_ts)

    # --- SNMP rows
    snmp_rows = []
    for t in snmp_times:
        for _, d in dev.iterrows():
            dt = str(d["dtype"]).upper()
            base_cpu = {"WLC": 28, "SW": 18, "AP": 12}[dt]
            base_mem = {"WLC": 45, "SW": 40, "AP": 28}[dt]
            base_util = {"WLC": 15, "SW": 30, "AP": 8}[dt]

            cpu = rng.normal(base_cpu, 4)
            mem = rng.normal(base_mem, 5)
            if_util = rng.normal(base_util, 7)
            if_errors = max(0, int(rng.poisson(0.2 if dt != "AP" else 0.05)))
            if_up = 1

            snmp_rows.append(
                (int(t), int(d["building"]), d["device"], dt, float(cpu), float(mem),
                 float(if_util), int(if_errors), int(if_up), d["parent"], d["role"])
            )

    snmp = pd.DataFrame(
        snmp_rows,
        columns=["ts","building","device","dtype","cpu","mem","if_util","if_errors","if_up","parent","role"]
    )

    # --- Probes
    services = ["DNS", "DHCP", "LMS", "WIFI"]
    if cfg.include_http_https:
        services += ["HTTP", "HTTPS"]

    rt_base = {"DNS": 0.03, "DHCP": 0.22, "LMS": 0.55, "WIFI": 0.08, "HTTP": 0.35, "HTTPS": 0.40}

    probe_rows = []
    for t in probe_times:
        for b in range(cfg.n_buildings):
            for s in services:
                base = rt_base[s]
                rt = max(0.0, rng.normal(base, base * 0.25))
                success = 1 if rng.random() < 0.996 else 0
                probe_rows.append((int(t), int(b), s, float(rt), int(success)))
    probes = pd.DataFrame(probe_rows, columns=["ts","building","service","resp_time_s","success"])

    # --- Events (ground truth)
    events: List[Tuple[int, str, int, int]] = []
    for b in range(cfg.n_buildings):
        st = int(rng.integers(end_ts * 0.2, end_ts * 0.65)); dur = int(rng.integers(300, 900))
        events.append((b, "DNS_OUTAGE", st, st + dur))

        st2 = int(rng.integers(end_ts * 0.25, end_ts * 0.70)); dur2 = int(rng.integers(300, 900))
        events.append((b, "DHCP_ISSUE", st2, st2 + dur2))

        st3 = int(rng.integers(end_ts * 0.30, end_ts * 0.80)); dur3 = int(rng.integers(300, 1200))
        events.append((b, "LMS_SLOW", st3, st3 + dur3))

        st4 = int(rng.integers(end_ts * 0.15, end_ts * 0.85)); dur4 = int(rng.integers(300, 900))
        events.append((b, "WIFI_DOWN", st4, st4 + dur4))

        st5 = int(rng.integers(end_ts * 0.3, end_ts * 0.85)); dur5 = int(rng.integers(600, 1200))
        events.append((b, "UPLINK_CONGEST", st5, st5 + dur5))

    wlc_b = int(rng.integers(0, cfg.n_buildings))
    stw = int(rng.integers(end_ts * 0.4, end_ts * 0.6))
    events.append((wlc_b, "WLC_DOWN", stw, stw + 600))

    # Apply events
    for b, etype, st, et in events:
        if etype == "DNS_OUTAGE":
            m = (probes.building == b) & (probes.service == "DNS") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "success"] = 0
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(1.2, 0.25, m.sum()))

        elif etype == "DHCP_ISSUE":
            m = (probes.building == b) & (probes.service == "DHCP") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "success"] = (rng.random(m.sum()) > 0.6).astype(int)
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(0.9, 0.2, m.sum()))

        elif etype == "LMS_SLOW":
            m = (probes.building == b) & (probes.service == "LMS") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(1.0, 0.35, m.sum()))
            for svc in ["HTTP", "HTTPS"]:
                m2 = (probes.building == b) & (probes.service == svc) & (probes.ts >= st) & (probes.ts < et)
                probes.loc[m2, "resp_time_s"] += np.abs(rng.normal(0.7, 0.25, m2.sum()))

        elif etype == "WIFI_DOWN":
            m = (probes.building == b) & (probes.service == "WIFI") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "success"] = 0
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(0.5, 0.2, m.sum()))

        elif etype == "UPLINK_CONGEST":
            m = (snmp.building == b) & (snmp.dtype == "SW") & (snmp.ts >= st) & (snmp.ts < et)
            snmp.loc[m, "if_util"] += np.abs(rng.normal(60, 10, m.sum()))
            snmp.loc[m, "if_errors"] += rng.poisson(6, m.sum())
            for svc in ["LMS", "HTTP", "HTTPS"]:
                m2 = (probes.building == b) & (probes.service == svc) & (probes.ts >= st) & (probes.ts < et)
                probes.loc[m2, "resp_time_s"] += np.abs(rng.normal(0.8, 0.25, m2.sum()))

        elif etype == "WLC_DOWN":
            m = (snmp.building == b) & (snmp.dtype == "WLC") & (snmp.ts >= st) & (snmp.ts < et)
            snmp.loc[m, "if_up"] = 0
            snmp.loc[m, "cpu"] += np.abs(rng.normal(35, 6, m.sum()))

            m_ap = (snmp.building == b) & (snmp.dtype == "AP") & (snmp.ts >= st) & (snmp.ts < et)
            snmp.loc[m_ap, "if_up"] = 0

            m_wifi = (probes.building == b) & (probes.service == "WIFI") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m_wifi, "success"] = 0

    # Minute labels
    minutes = np.arange(0, end_ts, 60, dtype=int)
    label_rows = []
    for b in range(cfg.n_buildings):
        for t in minutes:
            causes = []
            for eb, etype, st, et in events:
                if eb == b and (t < et) and ((t + 60) > st):
                    causes.append(etype)
            is_anom = 1 if causes else 0
            label_rows.append((int(t), int(b), int(is_anom), ";".join(sorted(set(causes)))))

    labels = pd.DataFrame(label_rows, columns=["minute","building","is_anomaly","cause"])
    return dev, snmp, probes, labels, events

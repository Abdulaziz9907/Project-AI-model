"""
ai_simulator.py
Simulated SNMP + probes for proof-of-spec.

Services simulated:
DNS, DHCP, LMS, WIFI (+ optional HTTP, HTTPS)

Option B (implemented):
- anomaly_target_pct: target % of (minute x building) buckets labeled anomalous.
  The simulator keeps adding events until the coverage reaches the target.

Also:
- probe_phase_offset_s: shifts probe schedule by a fixed offset
  so INT2 is realistic (~2–4s) instead of always 0.

Progress (optional):
- simulate(..., progress_cb=callable) where progress_cb(pct_0_to_1, message)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np
import pandas as pd


ProgressCB = Callable[[float, str], None]


@dataclass
class SimConfig:
    duration_minutes: int = 240
    n_buildings: int = 8
    n_aps_per_building: int = 14

    snmp_interval_s: int = 60
    probe_interval_s: int = 10

    snmp_jitter_s: int = 2
    probe_jitter_s: int = 2  # keep small to pass INT2

    probe_phase_offset_s: int = 2
    include_http_https: bool = True
    seed: int = 7

    # NEW (Option B): target anomaly coverage (% of minute-buckets)
    anomaly_target_pct: float = 5.0


def _jitter(base: np.ndarray, jitter_s: int, rng: np.random.Generator, end_ts: int) -> np.ndarray:
    if len(base) == 0:
        return base.astype(int)
    if int(jitter_s) <= 0:
        return base.clip(0, end_ts - 1).astype(int)
    jit = rng.integers(-int(jitter_s), int(jitter_s) + 1, size=len(base))
    out = (base + jit).clip(0, end_ts - 1)
    return out.astype(int)


def _format_event_duration(etype: str, rng: np.random.Generator, scale: float) -> int:
    """
    Duration in seconds, scaled by anomaly_target_pct.
    """
    # base ranges (seconds)
    if etype in ("DNS_OUTAGE", "WIFI_DOWN"):
        lo, hi = 300, 900
    elif etype in ("DHCP_ISSUE",):
        lo, hi = 300, 900
    elif etype in ("LMS_SLOW",):
        lo, hi = 300, 1200
    elif etype in ("UPLINK_CONGEST",):
        lo, hi = 600, 1200
    elif etype in ("WLC_DOWN",):
        lo, hi = 600, 900
    else:
        lo, hi = 300, 900

    dur = int(rng.integers(lo, hi + 1))
    dur = int(max(60, dur * scale))
    return dur


def _add_event_and_update_coverage(
    events: List[Tuple[int, str, int, int]],
    covered: set[tuple[int, int]],
    b: int,
    etype: str,
    st: int,
    et: int,
):
    events.append((int(b), str(etype), int(st), int(et)))
    # mark which minute buckets are anomalous for this building
    m0 = (int(st) // 60) * 60
    m1 = ((max(int(et) - 1, int(st)) // 60) * 60)
    for m in range(m0, m1 + 1, 60):
        covered.add((int(b), int(m)))


def _generate_events_option_b(cfg: SimConfig, rng: np.random.Generator, end_ts: int) -> List[Tuple[int, str, int, int]]:
    """
    Option B: keep adding events until we hit the target anomaly coverage.
    Coverage is measured in (building, minute_bucket_start).
    """
    target_pct = float(max(0.0, min(80.0, cfg.anomaly_target_pct)))
    total_buckets = int(cfg.n_buildings) * int(cfg.duration_minutes)
    target_buckets = int(round((target_pct / 100.0) * total_buckets))

    # If target is 0, still generate a tiny baseline so the pipeline works (can be empty if you want)
    if target_buckets <= 0:
        return []

    # Scale durations a bit when target_pct is high
    # 5% -> ~1.0 scale, 20% -> ~1.8 scale, 50% -> ~3.2 scale
    scale = 1.0 + (target_pct / 20.0)

    event_types = ["DNS_OUTAGE", "DHCP_ISSUE", "LMS_SLOW", "WIFI_DOWN", "UPLINK_CONGEST"]
    events: List[Tuple[int, str, int, int]] = []
    covered: set[tuple[int, int]] = set()

    # Seed with at least one event per building (random type), so all buildings can have anomalies
    for b in range(cfg.n_buildings):
        etype = str(rng.choice(event_types))
        dur = _format_event_duration(etype, rng, scale)
        st = int(rng.integers(0, max(1, end_ts - dur)))
        et = int(min(end_ts, st + dur))
        _add_event_and_update_coverage(events, covered, b, etype, st, et)

    # Add one WLC_DOWN for realism
    wlc_b = int(rng.integers(0, cfg.n_buildings))
    dur = _format_event_duration("WLC_DOWN", rng, scale)
    st = int(rng.integers(0, max(1, end_ts - dur)))
    et = int(min(end_ts, st + dur))
    _add_event_and_update_coverage(events, covered, wlc_b, "WLC_DOWN", st, et)

    # Keep adding until we reach target coverage or hit a sane limit
    max_events = max(200, int(target_buckets // 2) + 50)
    tries = 0
    while len(covered) < target_buckets and len(events) < max_events and tries < max_events * 3:
        tries += 1
        b = int(rng.integers(0, cfg.n_buildings))
        etype = str(rng.choice(event_types + ["WLC_DOWN"]))
        dur = _format_event_duration(etype, rng, scale)
        st = int(rng.integers(0, max(1, end_ts - dur)))
        et = int(min(end_ts, st + dur))
        _add_event_and_update_coverage(events, covered, b, etype, st, et)

    return events


def simulate(cfg: SimConfig, progress_cb: Optional[ProgressCB] = None):
    rng = np.random.default_rng(int(cfg.seed))
    end_ts = int(cfg.duration_minutes) * 60

    def prog(p: float, msg: str):
        if progress_cb is not None:
            progress_cb(float(max(0.0, min(1.0, p))), msg)

    prog(0.02, "Building inventory...")

    # Inventory
    devices = []
    for b in range(cfg.n_buildings):
        devices.append((b, "WLC", f"wlc-{b}", "parent", None))
        devices.append((b, "SW",  f"sw-{b}",  "parent", None))
        for a in range(cfg.n_aps_per_building):
            devices.append((b, "AP", f"ap-{b}-{a}", "child", f"wlc-{b}"))
    dev = pd.DataFrame(devices, columns=["building", "dtype", "device", "role", "parent"])

    # Base schedules
    snmp_base = np.arange(0, end_ts, int(cfg.snmp_interval_s), dtype=int)

    phase = int(cfg.probe_phase_offset_s) % max(int(cfg.probe_interval_s), 1)
    probe_base = np.arange(phase, end_ts, int(cfg.probe_interval_s), dtype=int)

    # Jitter
    prog(0.05, "Creating schedules...")
    snmp_times = _jitter(snmp_base, int(cfg.snmp_jitter_s), rng, end_ts)
    probe_times = _jitter(probe_base, int(cfg.probe_jitter_s), rng, end_ts)

    # --- SNMP rows
    prog(0.10, "Generating SNMP telemetry...")
    snmp_rows = []
    # progress updates without killing performance
    n_snmp = len(snmp_times)
    step = max(1, n_snmp // 30)

    for idx_t, t in enumerate(snmp_times):
        if idx_t % step == 0:
            prog(0.10 + 0.35 * (idx_t / max(1, n_snmp)), f"Generating SNMP telemetry... ({idx_t:,}/{n_snmp:,})")

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
    prog(0.45, "Generating probe telemetry...")
    services = ["DNS", "DHCP", "LMS", "WIFI"]
    if bool(cfg.include_http_https):
        services += ["HTTP", "HTTPS"]

    rt_base = {"DNS": 0.03, "DHCP": 0.22, "LMS": 0.55, "WIFI": 0.08, "HTTP": 0.35, "HTTPS": 0.40}

    probe_rows = []
    n_probe = len(probe_times)
    step2 = max(1, n_probe // 30)

    for idx_t, t in enumerate(probe_times):
        if idx_t % step2 == 0:
            prog(0.45 + 0.25 * (idx_t / max(1, n_probe)), f"Generating probe telemetry... ({idx_t:,}/{n_probe:,})")

        for b in range(cfg.n_buildings):
            for s in services:
                base = rt_base[s]
                rt = max(0.0, rng.normal(base, base * 0.25))
                success = 1 if rng.random() < 0.996 else 0
                probe_rows.append((int(t), int(b), s, float(rt), int(success)))

    probes = pd.DataFrame(probe_rows, columns=["ts","building","service","resp_time_s","success"])

    # --- Events (ground truth) — Option B
    prog(0.72, "Generating ground-truth events (Option B)...")
    events: List[Tuple[int, str, int, int]] = _generate_events_option_b(cfg, rng, end_ts)

    # Apply events
    prog(0.78, "Applying events to telemetry...")
    for b, etype, st, et in events:
        if etype == "DNS_OUTAGE":
            m = (probes.building == b) & (probes.service == "DNS") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "success"] = 0
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(1.2, 0.25, int(m.sum())))

        elif etype == "DHCP_ISSUE":
            m = (probes.building == b) & (probes.service == "DHCP") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "success"] = (rng.random(int(m.sum())) > 0.6).astype(int)
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(0.9, 0.2, int(m.sum())))

        elif etype == "LMS_SLOW":
            m = (probes.building == b) & (probes.service == "LMS") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(1.0, 0.35, int(m.sum())))
            for svc in ["HTTP", "HTTPS"]:
                if svc in probes["service"].unique():
                    m2 = (probes.building == b) & (probes.service == svc) & (probes.ts >= st) & (probes.ts < et)
                    probes.loc[m2, "resp_time_s"] += np.abs(rng.normal(0.7, 0.25, int(m2.sum())))

        elif etype == "WIFI_DOWN":
            m = (probes.building == b) & (probes.service == "WIFI") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m, "success"] = 0
            probes.loc[m, "resp_time_s"] += np.abs(rng.normal(0.5, 0.2, int(m.sum())))

        elif etype == "UPLINK_CONGEST":
            m = (snmp.building == b) & (snmp.dtype == "SW") & (snmp.ts >= st) & (snmp.ts < et)
            snmp.loc[m, "if_util"] += np.abs(rng.normal(60, 10, int(m.sum())))
            snmp.loc[m, "if_errors"] += rng.poisson(6, int(m.sum()))
            for svc in ["LMS", "HTTP", "HTTPS"]:
                if svc in probes["service"].unique():
                    m2 = (probes.building == b) & (probes.service == svc) & (probes.ts >= st) & (probes.ts < et)
                    probes.loc[m2, "resp_time_s"] += np.abs(rng.normal(0.8, 0.25, int(m2.sum())))

        elif etype == "WLC_DOWN":
            m = (snmp.building == b) & (snmp.dtype == "WLC") & (snmp.ts >= st) & (snmp.ts < et)
            snmp.loc[m, "if_up"] = 0
            snmp.loc[m, "cpu"] += np.abs(rng.normal(35, 6, int(m.sum())))

            m_ap = (snmp.building == b) & (snmp.dtype == "AP") & (snmp.ts >= st) & (snmp.ts < et)
            snmp.loc[m_ap, "if_up"] = 0

            m_wifi = (probes.building == b) & (probes.service == "WIFI") & (probes.ts >= st) & (probes.ts < et)
            probes.loc[m_wifi, "success"] = 0

    # Minute labels
    prog(0.92, "Building labels...")
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

    prog(1.0, "Done.")
    return dev, snmp, probes, labels, events

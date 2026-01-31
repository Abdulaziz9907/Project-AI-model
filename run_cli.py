#!/usr/bin/env python3
"""
run_cli.py — terminal-first runner (no files required)

Usage:
  python run_cli.py
  python run_cli.py --show-plots
  python run_cli.py --export-dir outputs   (optional: write CSV + PNG)
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai_simulator import SimConfig, simulate
from ai_features import build_minute_features, primary_cause
from ai_models import train_anomaly_and_rca
from ai_verify import build_spec_table, confusion


def print_spec_table(spec_df: pd.DataFrame):
    # Nice terminal output without extra deps
    df = spec_df.copy()

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    df["Measured"] = df["Measured"].map(fmt)
    df["PASS"] = df["PASS"].map(lambda x: "PASS ✅" if x else "FAIL ❌")

    print("\n=== Specification Verification (terminal) ===")
    print(df.to_string(index=False))


def export_outputs(export_dir: str,
                   snmp: pd.DataFrame,
                   probes: pd.DataFrame,
                   labels: pd.DataFrame,
                   events,
                   spec_df: pd.DataFrame,
                   cm: np.ndarray):
    os.makedirs(export_dir, exist_ok=True)

    # Save core data
    snmp.to_csv(os.path.join(export_dir, "snmp_sim.csv"), index=False)
    probes.to_csv(os.path.join(export_dir, "probes_sim.csv"), index=False)
    labels.to_csv(os.path.join(export_dir, "labels_minute.csv"), index=False)
    pd.DataFrame(events, columns=["building", "event", "start_ts", "end_ts"]).to_csv(
        os.path.join(export_dir, "events.csv"), index=False
    )
    spec_df.to_csv(os.path.join(export_dir, "spec_verification.csv"), index=False)

    # Confusion matrix image
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = plt.gca()
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=11)

    plt.title("Anomaly Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(export_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()


def show_example_plot(snmp: pd.DataFrame, probes: pd.DataFrame, building: int = 0):
    sn_b = snmp[(snmp.building == building) & (snmp.dtype == "SW")].sort_values("ts")
    pr_b = probes[(probes.building == building)].sort_values("ts")

    fig = plt.figure(figsize=(12, 4))
    plt.plot(sn_b.ts, sn_b.if_util, label="SW if_util")

    web = pr_b[pr_b.service == "WEB"]
    dns = pr_b[pr_b.service == "DNS"]

    if len(web):
        plt.plot(web.ts, web.resp_time_s, label="WEB resp_time_s")
    if len(dns):
        plt.plot(dns.ts, 1 - dns.success, label="DNS failure (1=fail)")

    plt.title(f"Simulated Telemetry Example (Building {building})")
    plt.xlabel("Timestamp (s)")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-plots", action="store_true", help="Show plots in a window (matplotlib).")
    ap.add_argument("--export-dir", default="", help="Optional: write CSV/PNG outputs to this folder.")
    args = ap.parse_args()

    # 1) simulate data
    cfg = SimConfig()
    dev, snmp, probes, labels, events = simulate(cfg)

    # 2) build minute features + join labels
    feat = build_minute_features(snmp, probes)
    ds = feat.merge(labels, on=["minute", "building"], how="left")
    ds["is_anomaly"] = ds["is_anomaly"].fillna(0).astype(int)
    ds["cause"] = ds["cause"].fillna("")
    ds["primary_cause"] = ds["cause"].apply(primary_cause)

    # 3) train feasibility models
    model_out = train_anomaly_and_rca(ds)
    split = model_out["split_minute"]
    ds_test = ds[ds.minute > split].copy().reset_index(drop=True)

    # 4) verify specs + print table
    spec_df, extra = build_spec_table(cfg, model_out, ds, ds_test, snmp, probes, events)
    print_spec_table(spec_df)

    # confusion matrix for optional export
    cm = confusion(model_out["y_test"], model_out["y_pred"])

    # 5) optional export
    if args.export_dir:
        export_outputs(args.export_dir, snmp, probes, labels, events, spec_df, cm)
        print(f"\nWrote outputs to: {os.path.abspath(args.export_dir)}")

    # 6) optional plots
    if args.show_plots:
        show_example_plot(snmp, probes, building=0)


if __name__ == "__main__":
    main()

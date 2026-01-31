"""
streamlit_dashboard.py — Proof-of-Spec Dashboard (no file outputs)

Fixes:
- Show all buildings at once (overview + anomalies list)
- Avoid recomputation on RCA widget changes (form + session_state)
- Avoid Streamlit cache hashing errors with sklearn models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from ai_simulator import SimConfig, simulate
from ai_features import build_minute_features, primary_cause
from ai_models import train_anomaly_and_rca
from ai_verify import build_spec_table, confusion

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AI Proof-of-Spec Dashboard", layout="wide")
st.title("Simulation-based Verification (Proof-of-Spec)")
st.caption(
    "Feasibility demo using simulated SNMP + probes (NOT the full prototype). "
    "Streamlit reruns the script on widget changes, so we use caching + session_state."
)

# -----------------------------
# Cached compute blocks
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_simulate(cfg_dict: dict):
    cfg = SimConfig(**cfg_dict)
    return simulate(cfg)  # dev, snmp, probes, labels, events

@st.cache_data(show_spinner=False)
def cached_features(snmp: pd.DataFrame, probes: pd.DataFrame):
    return build_minute_features(snmp, probes)

@st.cache_resource(show_spinner=False)
def cached_train(ds: pd.DataFrame):
    # cache_resource is appropriate for model objects
    return train_anomaly_and_rca(ds)

# NOTE: we do NOT cache scoring with a model object to avoid hashing issues
def score_all(ds: pd.DataFrame, feats: list[str], anomaly_model, threshold: float) -> pd.DataFrame:
    out = ds.copy()
    out["anom_prob"] = anomaly_model.predict_proba(out[feats])[:, 1]
    out["anom_pred"] = (out["anom_prob"] >= threshold).astype(int)
    return out

# -----------------------------
# Plot helpers
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.imshow(cm)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Anomaly Confusion Matrix")
    return fig

def plot_example_telemetry(snmp: pd.DataFrame, probes: pd.DataFrame, building: int):
    sn_b = snmp[(snmp.building == building) & (snmp.dtype == "SW")].sort_values("ts")
    pr_b = probes[probes.building == building].sort_values("ts")

    fig = plt.figure(figsize=(12, 4))
    plt.plot(sn_b.ts, sn_b.if_util, label="SW if_util")

    web = pr_b[pr_b.service == "WEB"]
    dns = pr_b[pr_b.service == "DNS"]

    if len(web):
        plt.plot(web.ts, web.resp_time_s, label="WEB resp_time_s")
    if len(dns):
        plt.plot(dns.ts, 1 - dns.success, label="DNS failure (1=fail)")

    plt.title(f"Example Telemetry (Building {building})")
    plt.xlabel("Timestamp (s)")
    plt.legend()
    return fig

def rca_topk(rca_model, row_feats: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    probs = rca_model.predict_proba(row_feats)[0]
    classes = list(rca_model.classes_)
    top_idx = np.argsort(probs)[-k:][::-1]
    return pd.DataFrame(
        {"Likely cause": [classes[i] for i in top_idx],
         "Probability": [float(probs[i]) for i in top_idx]}
    )

# -----------------------------
# Sidebar FORM (only runs heavy compute on submit)
# -----------------------------
with st.sidebar:
    st.header("Simulation Settings (press Run)")
    with st.form("sim_form"):
        duration = st.slider("Duration (minutes)", 60, 720, 240, 60)
        buildings = st.slider("Buildings", 2, 20, 8, 1)
        aps = st.slider("APs per building", 5, 50, 14, 1)
        seed = st.number_input("Random seed", value=7, step=1)

        st.divider()
        st.write("Sampling (spec-driven):")
        snmp_interval = st.selectbox("SNMP interval (S1)", [60, 30, 15], index=0)
        probe_interval = st.selectbox("Probe interval (S6, must be < 15)", [14, 10, 5], index=0)

        st.divider()
        run_btn = st.form_submit_button("Run simulation + train")

# -----------------------------
# Run pipeline only when button pressed
# -----------------------------
if run_btn:
    cfg_dict = dict(
        duration_minutes=int(duration),
        n_buildings=int(buildings),
        n_aps_per_building=int(aps),
        seed=int(seed),
        snmp_interval_s=int(snmp_interval),
        probe_interval_s=int(probe_interval),
    )

    with st.spinner("Simulating data + training models..."):
        dev, snmp, probes, labels, events = cached_simulate(cfg_dict)

        feat = cached_features(snmp, probes)
        ds = feat.merge(labels, on=["minute", "building"], how="left")
        ds["is_anomaly"] = ds["is_anomaly"].fillna(0).astype(int)
        ds["cause"] = ds["cause"].fillna("")
        ds["primary_cause"] = ds["cause"].apply(primary_cause)

        model_out = cached_train(ds)

        # score full dataset (no caching needed)
        feats = model_out["features"]
        ds_scored = score_all(ds, feats, model_out["anomaly_model"], model_out["threshold"])

        # spec verification
        split = model_out["split_minute"]
        ds_test = ds_scored[ds_scored.minute > split].copy().reset_index(drop=True)
        spec_df, extra = build_spec_table(SimConfig(**cfg_dict), model_out, ds_scored, ds_test, snmp, probes, events)

        st.session_state["payload"] = {
            "cfg_dict": cfg_dict,
            "snmp": snmp,
            "probes": probes,
            "events": events,
            "ds": ds_scored,
            "model_out": model_out,
            "spec_df": spec_df,
            "extra": extra,
        }

# If no results computed yet
if "payload" not in st.session_state:
    st.info("Use the sidebar and click **Run simulation + train**.")
    st.stop()

P = st.session_state["payload"]
cfg_dict = P["cfg_dict"]
snmp = P["snmp"]
probes = P["probes"]
events = P["events"]
ds = P["ds"]
model_out = P["model_out"]
spec_df = P["spec_df"]
extra = P["extra"]
feats = model_out["features"]
rca_model = model_out["rca_model"]

# -----------------------------
# Headline metrics
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precision (S3)", f"{model_out['precision']:.3f}")
c2.metric("Recall (S3)", f"{model_out['recall']:.3f}")
c3.metric("RCA top-3 (S5-ext)", f"{model_out['rca_top3_acc']:.3f}")
c4.metric("Dedup reduction (S7)", f"{extra['dedup_reduction']*100:.1f}%")
c5.metric("Max align err (INT2)", f"{extra['align_max']:.1f}s")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_rca, tab_plots, tab_data = st.tabs(
    ["Overview (All Buildings)", "RCA (Likely Cause)", "Plots", "Raw Data"]
)

# ===== Overview =====
with tab_overview:
    st.subheader("All Buildings Overview (latest minute per building)")

    latest = ds.sort_values("minute").groupby("building").tail(1).copy()

    def top1_cause_for_row(row) -> str:
        if int(row["anom_pred"]) != 1:
            return "NORMAL"
        probs = rca_model.predict_proba(row[feats].to_frame().T)[0]
        cls = list(rca_model.classes_)
        return str(cls[int(np.argmax(probs))])

    latest["likely_cause_top1"] = latest.apply(top1_cause_for_row, axis=1)

    st.dataframe(
        latest[
            ["building", "minute", "anom_prob", "anom_pred", "likely_cause_top1",
             "rt_DNS", "ok_DNS", "rt_WEB", "ok_WEB", "util_max", "errors_sum", "down_frac"]
        ].sort_values("building"),
        use_container_width=True,
    )

    st.subheader("Current Anomalies (across all buildings)")
    lookback_minutes = st.slider("Lookback window (minutes)", 10, 120, 30, 5)
    max_minute = int(ds["minute"].max())
    cutoff = max_minute - lookback_minutes * 60

    recent_anom = ds[(ds["minute"] >= cutoff) & (ds["anom_pred"] == 1)].copy()
    recent_anom = recent_anom.sort_values(["minute", "building"], ascending=[False, True]).reset_index(drop=True)

    if len(recent_anom) == 0:
        st.info("No predicted anomalies in the selected lookback window.")
    else:
        # Add top-1 RCA for each anomaly row
        likely = []
        conf = []
        classes = list(rca_model.classes_)
        for _, r in recent_anom.iterrows():
            probs = rca_model.predict_proba(r[feats].to_frame().T)[0]
            idx = int(np.argmax(probs))
            likely.append(classes[idx])
            conf.append(float(probs[idx]))

        recent_anom["likely_cause_top1"] = likely
        recent_anom["cause_conf"] = conf

        st.dataframe(
            recent_anom[
                ["minute", "building", "anom_prob", "likely_cause_top1", "cause_conf",
                 "rt_DNS", "ok_DNS", "rt_WEB", "ok_WEB", "util_max"]
            ],
            use_container_width=True,
        )

    st.subheader("Specification Verification (PASS/FAIL)")
    st.dataframe(spec_df, use_container_width=True)

# ===== RCA tab =====
with tab_rca:
    st.subheader("RCA (Likely Cause) — interactive")

    rca_mode = st.radio(
        "Pick rows for RCA",
        ["Pick any minute", "Only predicted anomaly minutes"],
        horizontal=True,
    )

    b_sel = st.selectbox("Building", sorted(ds["building"].unique().tolist()), index=0)
    ds_b = ds[ds["building"] == b_sel].sort_values("minute").reset_index(drop=True)

    if rca_mode == "Pick any minute":
        idx = st.slider("Minute index (within building)", 0, len(ds_b) - 1, min(10, len(ds_b) - 1))
        row = ds_b.iloc[idx:idx+1].copy()
    else:
        anom_rows = ds_b[ds_b["anom_pred"] == 1].copy().reset_index(drop=True)
        if len(anom_rows) == 0:
            st.warning("No predicted anomaly minutes for this building. Try another building/seed.")
            st.stop()
        idx = st.slider("Anomaly minute index", 0, len(anom_rows) - 1, 0)
        row = anom_rows.iloc[idx:idx+1].copy()

    minute_val = int(row["minute"].iloc[0])
    anom_prob = float(row["anom_prob"].iloc[0])
    anom_pred = int(row["anom_pred"].iloc[0])

    st.write(f"Minute: **{minute_val}s**")
    st.write(
        f"Anomaly probability: **{anom_prob:.3f}** | predicted anomaly: **{anom_pred}** "
        f"(threshold={model_out['threshold']:.2f})"
    )

    if anom_pred == 1:
        topk = rca_topk(rca_model, row[feats], k=3)
        st.write("Top likely causes (Top-3):")
        st.dataframe(topk, use_container_width=True)

        # Ground truth shown only because simulation provides it
        gt = str(row["primary_cause"].iloc[0])
        st.caption(f"(Simulation ground truth for this minute: {gt})")
    else:
        st.info("This minute is not predicted anomalous. Switch to anomaly-only mode or pick another minute.")

# ===== Plots =====
with tab_plots:
    st.subheader("Anomaly Confusion Matrix")
    cm = confusion(model_out["y_test"], model_out["y_pred"])
    st.pyplot(plot_confusion_matrix(cm))

    st.subheader("Example Telemetry")
    b_plot = st.slider("Building (telemetry plot)", 0, int(cfg_dict["n_buildings"]) - 1, 0, 1)
    st.pyplot(plot_example_telemetry(snmp, probes, building=int(b_plot)))

# ===== Raw Data =====
with tab_data:
    st.subheader("Raw data previews")
    st.write("SNMP sample")
    st.dataframe(snmp.head(50), use_container_width=True)
    st.write("Probes sample")
    st.dataframe(probes.head(50), use_container_width=True)
    st.write("Events (ground truth)")
    st.dataframe(pd.DataFrame(events, columns=["building", "event", "start_ts", "end_ts"]), use_container_width=True)

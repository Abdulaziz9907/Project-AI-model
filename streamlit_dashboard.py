"""
streamlit_dashboard.py — Proof-of-Spec Dashboard (UPDATED)

Updates requested:
- Top metrics displayed at the very top of the page (above tabs).
- "How each specification is calculated" is in the Overview tab and collapsed by default.
- Sidebar settings include small explanations (via help=...).
- Fix: jitter/phase settings do NOT affect INT2 when using uploaded data (correct behavior).
  Added optional "Uploaded data testing knobs" to perturb uploaded probe timestamps so INT2 changes for demos.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from ai_simulator import SimConfig, simulate
from ai_features import build_minute_features, primary_cause
from ai_models import train_anomaly_and_rca
from ai_verify import build_spec_table, confusion


EXPECTED_CAUSES = [
    "NORMAL",
    "UPLINK_CONGEST",
    "DNS_OUTAGE",
    "DHCP_ISSUE",
    "LMS_SLOW",
    "LMS_OUTAGE",
    "WIFI_DOWN",
    "WLC_DOWN",
    "HTTP_SLOW",
    "HTTP_OUTAGE",
    "HTTPS_SLOW",
    "HTTPS_OUTAGE",
]

st.set_page_config(page_title="AI Proof-of-Spec Dashboard", layout="wide")
st.title("Simulation-based Verification (Proof-of-Spec)")
st.caption("Feasibility demo using simulated SNMP + probe telemetry (NOT the full prototype).")

# Big table CSS
st.markdown(
    """
    <style>
    .big-spec-table table { font-size: 18px !important; }
    .big-spec-table th, .big-spec-table td { padding: 10px 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Cached helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_simulate(cfg_dict: dict):
    return simulate(SimConfig(**cfg_dict))


@st.cache_data(show_spinner=False)
def cached_features(snmp: pd.DataFrame, probes: pd.DataFrame):
    return build_minute_features(snmp, probes)


@st.cache_resource(show_spinner=False)
def cached_train(ds: pd.DataFrame):
    return train_anomaly_and_rca(ds)


def ensure_feature_columns(df: pd.DataFrame, required_feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    missing = []
    for c in required_feats:
        if c not in out.columns:
            missing.append(c)
            if c.startswith("ok_"):
                out[c] = 1.0
            elif c.startswith("rt_"):
                out[c] = 0.0
            elif c.startswith("fail_"):
                out[c] = 0.0
            else:
                out[c] = 0.0
    return out, missing


def score_all(ds: pd.DataFrame, feats: list[str], anomaly_model, threshold: float) -> pd.DataFrame:
    out = ds.copy()
    out["anom_prob"] = anomaly_model.predict_proba(out[feats])[:, 1]
    out["anom_pred"] = (out["anom_prob"] >= threshold).astype(int)
    return out


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def make_templates():
    snmp_t = pd.DataFrame([{
        "ts": 0, "building": 0, "device": "sw-0", "dtype": "SW",
        "cpu": 18.0, "mem": 40.0, "if_util": 30.0, "if_errors": 0, "if_up": 1,
        "parent": "", "role": "parent"
    }])

    probes_t = pd.DataFrame([
        {"ts": 2, "building": 0, "service": "DNS",  "resp_time_s": 0.03, "success": 1},
        {"ts": 2, "building": 0, "service": "DHCP", "resp_time_s": 0.22, "success": 1},
        {"ts": 2, "building": 0, "service": "LMS",  "resp_time_s": 0.55, "success": 1},
        {"ts": 2, "building": 0, "service": "WIFI", "resp_time_s": 0.08, "success": 1},
        {"ts": 2, "building": 0, "service": "HTTP",  "resp_time_s": 0.35, "success": 1},
        {"ts": 2, "building": 0, "service": "HTTPS", "resp_time_s": 0.40, "success": 1},
    ])

    labels_t = pd.DataFrame([{
        "minute": 0, "building": 0, "is_anomaly": 0, "cause": "NORMAL"
    }])

    return snmp_t, probes_t, labels_t


def validate_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{name} missing columns: {missing}")
        st.stop()


def normalize_services(probes: pd.DataFrame) -> pd.DataFrame:
    p = probes.copy()
    p["service"] = (
        p["service"].astype(str).str.strip().str.upper()
        .replace({"WI-FI": "WIFI", "WLAN": "WIFI", "WIRELESS": "WIFI", "WEB": "HTTP"})
    )
    return p


def apply_uploaded_probe_perturbation(
    probes: pd.DataFrame,
    enable: bool,
    shift_s: int,
    jitter_s: int,
    seed: int,
) -> pd.DataFrame:
    """
    Optional: modify uploaded probe timestamps so INT2 changes for demonstration/sensitivity testing.
    - shift_s: constant shift applied to all probe ts
    - jitter_s: random +/- jitter in seconds
    """
    if not enable:
        return probes

    p = probes.copy()
    p["ts"] = pd.to_numeric(p["ts"], errors="coerce")
    p = p.dropna(subset=["ts"]).copy()
    p["ts"] = p["ts"].astype(int) + int(shift_s)

    if int(jitter_s) > 0:
        rng = np.random.default_rng(int(seed))
        jit = rng.integers(-int(jitter_s), int(jitter_s) + 1, size=len(p))
        p["ts"] = p["ts"].astype(int) + jit.astype(int)

    # keep timestamps non-negative
    p["ts"] = p["ts"].clip(lower=0).astype(int)
    return p


def plot_telemetry_window(snmp: pd.DataFrame, probes: pd.DataFrame, building: int, center_minute: int, window_s: int = 300):
    start = max(0, int(center_minute - window_s))
    end = int(center_minute + window_s)

    sn_b = snmp[(snmp.building == building) & (snmp.ts >= start) & (snmp.ts <= end)].copy()
    pr_b = probes[(probes.building == building) & (probes.ts >= start) & (probes.ts <= end)].copy()
    if "service" in pr_b.columns:
        pr_b = normalize_services(pr_b)

    sn_sw = sn_b[sn_b.dtype.astype(str).str.upper() == "SW"].sort_values("ts")
    pr_b = pr_b.sort_values("ts")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    if len(sn_sw):
        ax1.plot(sn_sw.ts, sn_sw.if_util, label="SW if_util")
        ax1.plot(sn_sw.ts, sn_sw.if_errors, label="SW if_errors")
    ax1.axvline(center_minute, linestyle="--", label=f"minute={center_minute}")
    ax1.axvline(center_minute + 60, linestyle=":", label=f"end={center_minute+60}")
    ax1.set_title(f"Telemetry window — Building {building} — minute {center_minute}")
    ax1.legend(loc="upper left")

    ax2_rt = ax2
    ax2_fail = ax2.twinx()

    if "service" in pr_b.columns:
        for svc in sorted(pr_b["service"].unique()):
            sub = pr_b[pr_b.service == svc]
            ax2_rt.plot(sub.ts, sub.resp_time_s, label=f"{svc} resp_time_s")
        for svc in sorted(pr_b["service"].unique()):
            sub = pr_b[pr_b.service == svc]
            ax2_fail.plot(sub.ts, 1 - sub.success, marker="o", linestyle="--", label=f"{svc} fail(1=fail)")
        ax2_fail.set_ylim(-0.05, 1.05)

    ax2_rt.axvline(center_minute, linestyle="--")
    ax2_rt.axvline(center_minute + 60, linestyle=":")

    ax2_rt.set_xlabel("Timestamp (s)")
    ax2_rt.set_ylabel("Resp time (s)")
    ax2_fail.set_ylabel("Failure (0/1)")

    h1, l1 = ax2_rt.get_legend_handles_labels()
    h2, l2 = ax2_fail.get_legend_handles_labels()
    ax2_rt.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    return fig


def rca_topk(rca_model, row_feats: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    probs = rca_model.predict_proba(row_feats)[0]
    classes = list(rca_model.classes_)
    top_idx = np.argsort(probs)[-k:][::-1]
    return pd.DataFrame({"Likely cause": [classes[i] for i in top_idx], "Probability": [float(probs[i]) for i in top_idx]})


# -----------------------------
# Sidebar
# -----------------------------
snmp_t, probes_t, labels_t = make_templates()

with st.sidebar:
    st.header("Simulation Settings (train model)")
    with st.form("sim_form"):
        duration = st.slider(
            "Duration (minutes)",
            60, 720, 240, 60,
            help="Length of simulated timeline used for training/verification in 'Simulated' mode."
        )
        buildings = st.slider(
            "Buildings",
            2, 20, 8, 1,
            help="Number of buildings (sites) in simulated training data."
        )
        aps = st.slider(
            "APs per building",
            5, 50, 14, 1,
            help="Number of access points per building in simulated SNMP inventory."
        )
        seed = st.number_input(
            "Random seed",
            value=7, step=1,
            help="Controls simulation randomness. Same seed → same dataset."
        )

        snmp_interval = st.selectbox(
            "SNMP interval (S1)",
            [60, 30, 15], index=0,
            help="SNMP polling interval used in simulation (seconds). S1 expects ≤ 60s."
        )
        probe_interval = st.selectbox(
            "Probe interval (S6 < 15s)",
            [14, 12, 10, 5], index=2,
            help="Probe execution interval (seconds). S6 expects < 15s."
        )

        probe_phase = st.slider(
            "Probe phase offset (INT2 realism)",
            0, 9, 2, 1,
            help="Shifts probe schedule relative to SNMP polls to avoid a perfect 0s INT2. Only affects SIMULATED data."
        )
        snmp_jitter = st.slider(
            "SNMP jitter (seconds)",
            0, 5, 2, 1,
            help="Random +/- jitter added to simulated SNMP timestamps. Only affects SIMULATED data."
        )
        probe_jitter = st.slider(
            "Probe jitter (seconds)",
            0, 5, 2, 1,
            help="Random +/- jitter added to simulated probe timestamps. Only affects SIMULATED data."
        )

        include_http = st.checkbox(
            "Simulate HTTP/HTTPS probes",
            value=True,
            help="Adds HTTP/HTTPS services to simulated probe data (and features if present)."
        )

        st.divider()
        run_btn = st.form_submit_button("Run (train model)")

    st.divider()
    st.header("Data Source (scoring/verification)")
    data_source = st.radio(
        "Choose source",
        ["Simulated", "Uploaded CSV"],
        help="Simulated uses the generated dataset. Uploaded uses your CSV data and scores it with the trained model."
    )

    strict_services = st.checkbox(
        "Strict: require DNS, DHCP, LMS, WIFI in probes",
        value=True if data_source == "Uploaded CSV" else False,
        help="If enabled, uploaded probes must include these core services."
    )

    snmp_upload = probes_upload = labels_upload = None
    if data_source == "Uploaded CSV":
        snmp_upload = st.file_uploader("Upload SNMP CSV", type=["csv"], key="snmp_csv")
        probes_upload = st.file_uploader("Upload Probes CSV", type=["csv"], key="probes_csv")
        labels_upload = st.file_uploader("Upload Labels CSV (optional)", type=["csv"], key="labels_csv")

        st.divider()
        st.subheader("Uploaded data testing knobs (optional)")
        st.caption(
            "These DO NOT represent real monitoring. They are only for sensitivity testing "
            "to show INT2 changes when probe timestamps shift."
        )
        perturb_enable = st.checkbox("Apply timestamp shift/jitter to uploaded probes", value=False)
        perturb_shift = st.slider("Probe time shift (seconds)", -10, 10, 0, 1)
        perturb_jitter = st.slider("Probe time jitter (seconds)", 0, 5, 0, 1)
        perturb_seed = st.number_input("Perturb seed", value=123, step=1)
    else:
        perturb_enable = False
        perturb_shift = 0
        perturb_jitter = 0
        perturb_seed = 123


# -----------------------------
# Tabs (Templates always available)
# -----------------------------
tab_overview, tab_rca, tab_plots, tab_templates = st.tabs(
    ["Overview (All Buildings)", "RCA + Linked Plot", "Plots", "Templates & Upload"]
)

with tab_templates:
    st.subheader("Template downloads (always available)")
    st.download_button("Download SNMP template (CSV)", df_to_csv_bytes(snmp_t), "snmp_template.csv", "text/csv")
    st.download_button("Download Probes template (CSV)", df_to_csv_bytes(probes_t), "probes_template.csv", "text/csv")
    st.download_button("Download Labels template (optional CSV)", df_to_csv_bytes(labels_t), "labels_template.csv", "text/csv")

    st.divider()
    st.subheader("Expected labels")
    st.markdown(
        """
**Labels CSV columns**
- `minute`: start of minute bucket (0, 60, 120, ...)
- `building`: building index
- `is_anomaly`: 0 or 1
- `cause`: one or more causes separated by semicolon `;`
"""
    )
    st.write("Report-aligned expected causes:")
    st.code("\n".join(EXPECTED_CAUSES))

    st.divider()
    st.subheader("Template previews")
    st.write("SNMP template"); st.dataframe(snmp_t, use_container_width=True)
    st.write("Probes template"); st.dataframe(probes_t, use_container_width=True)
    st.write("Labels template"); st.dataframe(labels_t, use_container_width=True)


# -----------------------------
# Training
# -----------------------------
if run_btn:
    cfg_dict = dict(
        duration_minutes=int(duration),
        n_buildings=int(buildings),
        n_aps_per_building=int(aps),
        seed=int(seed),
        snmp_interval_s=int(snmp_interval),
        probe_interval_s=int(probe_interval),
        probe_phase_offset_s=int(probe_phase),
        snmp_jitter_s=int(snmp_jitter),
        probe_jitter_s=int(probe_jitter),
        include_http_https=bool(include_http),
    )

    with st.spinner("Simulating training data + training models..."):
        dev_tr, snmp_tr, probes_tr, labels_tr, events_tr = cached_simulate(cfg_dict)
        feat_tr = cached_features(snmp_tr, probes_tr)

        ds_tr = feat_tr.merge(labels_tr, on=["minute", "building"], how="left")
        ds_tr["is_anomaly"] = ds_tr["is_anomaly"].fillna(0).astype(int)
        ds_tr["cause"] = ds_tr["cause"].fillna("")
        ds_tr["primary_cause"] = ds_tr["cause"].apply(primary_cause)

        model_out = cached_train(ds_tr)

        st.session_state["trained"] = {
            "cfg_dict": cfg_dict,
            "model_out": model_out,
            "train_data": (snmp_tr, probes_tr, labels_tr, events_tr, ds_tr),
        }

# If not trained yet: templates are still usable; other tabs show guidance
if "trained" not in st.session_state:
    # Put top metrics placeholders at top
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomaly threshold", "—")
    c2.metric("Precision", "—")
    c3.metric("Recall", "—")
    c4.metric("Rows (minute x building)", "—")

    with tab_overview:
        st.info("Press **Run (train model)** in the sidebar first.")
    with tab_rca:
        st.info("Press **Run (train model)** first.")
    with tab_plots:
        st.info("Press **Run (train model)** first.")
    st.stop()

trained = st.session_state["trained"]
cfg_dict = trained["cfg_dict"]
model_out = trained["model_out"]
feats = model_out["features"]
anom_model = model_out["anomaly_model"]
rca_model = model_out["rca_model"]


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


using_labels = False
events = []

# -----------------------------
# Build ds_base from chosen source
# -----------------------------
if data_source == "Simulated":
    snmp, probes, labels, events, ds_base = trained["train_data"]
    using_labels = True
else:
    if snmp_upload is None or probes_upload is None:
        # Still show top metrics (trained model exists), but warn in Overview
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anomaly threshold", f"{model_out['threshold']:.2f}")
        c2.metric("Precision", f"{model_out['precision']:.3f}")
        c3.metric("Recall", f"{model_out['recall']:.3f}")
        c4.metric("Rows (minute x building)", "—")

        with tab_overview:
            st.warning("Upload both SNMP and Probes CSV in the sidebar to score uploaded data.")
        st.stop()

    snmp = load_uploaded_csv(snmp_upload)
    probes = load_uploaded_csv(probes_upload)

    validate_columns(snmp, ["ts","building","device","dtype","cpu","mem","if_util","if_errors","if_up","parent","role"], "SNMP CSV")
    validate_columns(probes, ["ts","building","service","resp_time_s","success"], "Probes CSV")

    probes = normalize_services(probes)

    if strict_services:
        required = {"DNS", "DHCP", "LMS", "WIFI"}
        present = set(probes["service"].unique())
        if not required.issubset(present):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anomaly threshold", f"{model_out['threshold']:.2f}")
            c2.metric("Precision", f"{model_out['precision']:.3f}")
            c3.metric("Recall", f"{model_out['recall']:.3f}")
            c4.metric("Rows (minute x building)", "—")

            with tab_overview:
                st.error(f"Probes must include services {sorted(required)}. Found: {sorted(present)}")
            st.stop()

    # OPTIONAL: perturb probe timestamps (uploaded data sensitivity testing)
    probes_for_scoring = apply_uploaded_probe_perturbation(
        probes,
        enable=perturb_enable,
        shift_s=int(perturb_shift),
        jitter_s=int(perturb_jitter),
        seed=int(perturb_seed),
    )

    feat_up = cached_features(snmp, probes_for_scoring)
    ds_base = feat_up.copy()

    if labels_upload is not None:
        labels = load_uploaded_csv(labels_upload)
        validate_columns(labels, ["minute","building","is_anomaly","cause"], "Labels CSV")
        ds_base = ds_base.merge(labels, on=["minute","building"], how="left")
        ds_base["is_anomaly"] = ds_base["is_anomaly"].fillna(0).astype(int)
        ds_base["cause"] = ds_base["cause"].fillna("")
        ds_base["primary_cause"] = ds_base["cause"].apply(primary_cause)
        using_labels = True

# Ensure features exist, then score
ds_base, missing_feats = ensure_feature_columns(ds_base, feats)
if missing_feats:
    st.warning(
        "Scoring data is missing some features expected by the trained model. "
        "They were filled with defaults (accuracy may drop): "
        + ", ".join(missing_feats)
    )

ds = score_all(ds_base.fillna(0.0), feats, anom_model, model_out["threshold"])

if "cause" not in ds.columns:
    ds["cause"] = ""
if "primary_cause" not in ds.columns:
    ds["primary_cause"] = ""

split = model_out["split_minute"]
ds_test = ds[ds.minute > split].copy().reset_index(drop=True)

# For spec verification: INT2 must use the *same probes* used for scoring when in Uploaded CSV mode
if data_source == "Uploaded CSV":
    probes_for_spec = probes_for_scoring
else:
    probes_for_spec = probes

spec_df = build_spec_table(
    SimConfig(**cfg_dict),
    model_out,
    ds,
    ds_test,
    snmp,
    probes_for_spec,
    events,
    build_features_fn=build_minute_features,
)

# -----------------------------
# TOP METRICS (always at top)
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Anomaly threshold", f"{model_out['threshold']:.2f}")
c2.metric("Precision", f"{model_out['precision']:.3f}")
c3.metric("Recall", f"{model_out['recall']:.3f}")
c4.metric("Rows (minute x building)", f"{len(ds):,}")


# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    st.subheader("All Buildings Overview (latest minute per building)")
    latest = ds.sort_values("minute").groupby("building").tail(1).copy()

    def top1_cause(row):
        if int(row["anom_pred"]) != 1:
            return "NORMAL"
        probs = rca_model.predict_proba(row[feats].to_frame().T)[0]
        cls = list(rca_model.classes_)
        return str(cls[int(np.argmax(probs))])

    latest["likely_cause_top1"] = latest.apply(top1_cause, axis=1)
    st.dataframe(
        latest[["building","minute","anom_prob","anom_pred","likely_cause_top1"]].sort_values("building"),
        use_container_width=True
    )

    st.subheader("Specification verification (Report S1–S8, INT1–INT3)")

    # Collapsed by default
    with st.expander("How each specification is calculated (proof rules)", expanded=False):
        st.markdown(
            """
### Correlation rule (INT2)
For each SNMP poll timestamp `t_SNMP` (per building), we choose the **nearest probe timestamp** `t_probe` (same building).
We compute:
- `alignment_error = |t_SNMP - t_probe|`

**Measured INT2 = max(alignment_error) across all SNMP samples and buildings.**  
This matches: “correlate device-level (SNMP) and service-level (probe) data with timestamp alignment error ≤ ±5 seconds”.

> Note: In **Uploaded CSV** mode, simulation jitter/phase settings do not change your uploaded timestamps.  
> If you want to demonstrate INT2 changing, enable “Uploaded data testing knobs”.

### S1
Measured = `snmp_interval_s` (simulation setting). PASS if ≤ 60s.

### S2
Measured = average time for a groupby/mean “dashboard-like” query over common columns. PASS if < 10s.

### S3a / S3b
Measured = anomaly model precision / recall on the labeled test split of the simulated dataset.

### S5
Measured = time to run RCA `predict_proba()` over a small batch of anomaly rows. PASS if ≤ 50s.

### S6
Measured = `probe_interval_s` (simulation setting). PASS if < 15s.

### S7
Raw alerts include: probe failures + probe SLA violations + SNMP congestion/errors/down.  
Dedup alerts count unique `(building, minute)` tickets.  
Reduction% = (raw - dedup) / raw × 100. PASS if ≥ 30%.

### S8
Measured = records/sec to build minute features once from SNMP+probes. PASS if ≥ 50 r/s.

### INT1
Measured = worst-case time between an injected event start and first predicted anomaly minute (simulated only). PASS if ≤ 120s.
"""
        )

    spec_show = spec_df.copy()
    spec_show["Pass"] = spec_show["Pass"].apply(lambda x: "✅ PASS" if x else "❌ FAIL")
    st.markdown('<div class="big-spec-table">', unsafe_allow_html=True)
    st.table(spec_show)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# RCA + linked plot tab
# -----------------------------
with tab_rca:
    st.subheader("RCA (Likely Cause) + Linked Plot")

    rca_mode = st.radio("Pick rows for RCA", ["Pick any minute", "Only predicted anomaly minutes"], horizontal=True)

    b_sel = st.selectbox("Building", sorted(ds["building"].unique().tolist()), index=0)
    ds_b = ds[ds["building"] == b_sel].sort_values("minute").reset_index(drop=True)

    if rca_mode == "Pick any minute":
        idx = st.slider("Minute index", 0, len(ds_b) - 1, min(10, len(ds_b) - 1))
        row = ds_b.iloc[idx:idx+1].copy()
    else:
        anom_rows = ds_b[ds_b["anom_pred"] == 1].reset_index(drop=True)
        if len(anom_rows) == 0:
            st.warning("No predicted anomaly minutes for this building.")
            st.stop()
        idx = st.slider("Anomaly minute index", 0, len(anom_rows) - 1, 0)
        row = anom_rows.iloc[idx:idx+1].copy()

    minute_val = int(row["minute"].iloc[0])
    st.write(f"Minute bucket: **[{minute_val}, {minute_val+60})**")
    st.write(f"Anomaly probability: **{float(row['anom_prob'].iloc[0]):.3f}** | pred: **{int(row['anom_pred'].iloc[0])}**")

    if int(row["anom_pred"].iloc[0]) == 1:
        topk = rca_topk(rca_model, row[feats], k=3)
        st.dataframe(topk, use_container_width=True)
        if using_labels:
            st.caption(f"Ground truth causes: {row['cause'].iloc[0]}")
    else:
        st.info("Not predicted anomalous. Switch to anomaly-only mode for RCA minutes.")

    st.subheader("Telemetry around selected minute (linked)")
    window_s = st.slider("Plot window (seconds)", 120, 900, 300, 60)
    # Use probes_for_spec for consistency when uploaded perturbation is enabled
    st.pyplot(plot_telemetry_window(snmp, probes_for_spec, int(b_sel), minute_val, int(window_s)))


# -----------------------------
# Plots tab
# -----------------------------
with tab_plots:
    st.subheader("Confusion matrix (labels only)")
    if using_labels and ("y_test" in model_out) and ("y_pred" in model_out):
        cm = confusion(model_out["y_test"], model_out["y_pred"])
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
        ax.imshow(cm)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    else:
        st.info("Upload Labels CSV to compute confusion matrix (or use Simulated mode).")

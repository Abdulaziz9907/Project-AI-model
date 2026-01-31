Proof-of-Spec — Refactored (data separated + easier output)

This is NOT the full prototype.
It is a simulation-based verification runner that shows feasibility vs specs.

Files:
- ai_simulator.py : data generation only (SNMP + probes + labels)
- ai_features.py : feature engineering only
- ai_models.py : anomaly filter + RCA top-3 evaluation
- ai_verify.py : spec checks, dedup, alignment, performance benchmarks
- run_cli.py : terminal-first runner (NO files required)
- streamlit_dashboard.py : simple dashboard (NO files required)

1) Setup venv (Windows PowerShell)
 py -3.12 -m venv venv
 .\venv\Scripts\Activate.ps1

2) Install deps
 python -m pip install -U pip
 python -m pip install numpy pandas scikit-learn matplotlib

3) Run in terminal (no file outputs)
 python run_cli.py

Optional:
- show plots in a window:
 python run_cli.py --show-plots

- export CSV/PNG to a folder:
 python run_cli.py --export-dir outputs

Simple dashboard (recommended for clean demo)
 python -m pip install streamlit
 streamlit run streamlit_dashboard.py

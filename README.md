# Sentinel — Real-Time Server Failure Predictor

A premium-looking local prototype for monitoring server health and estimating failure risk.

## What it does

- Reads live CPU utilization
- Reads RAM utilization
- Reads disk utilization
- Attempts to read hardware temperature
- Counts running processes
- Calculates an explainable 0–100 failure-risk score
- Shows a LOW / MEDIUM / HIGH status
- Stores recent telemetry in memory
- Displays live charts
- Auto-refreshes like a monitoring dashboard

## Run

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Important

This version is a real monitoring application with an explainable risk engine. It is NOT yet a trained machine-learning failure predictor.

For the final project, collect historical telemetry with labels such as:

- 0 = server healthy
- 1 = server failure within a defined prediction window

Then train an ML model and expose its calibrated probability through the dashboard.

Recommended next architecture:

Telemetry Agent -> Time-series storage -> Feature engineering -> ML model -> Risk API -> Dashboard -> Alerts

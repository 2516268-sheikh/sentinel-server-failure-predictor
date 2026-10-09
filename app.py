
import time
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import psutil
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Sentinel • Server Failure Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium dark UI ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(80, 90, 255, .13), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(0, 220, 190, .09), transparent 25%),
        #070912;
    color: #eef1f8;
}
.block-container { padding-top: 2rem; max-width: 1500px; }

.hero {
    padding: 28px 30px;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
    box-shadow: 0 18px 60px rgba(0,0,0,.30);
    margin-bottom: 20px;
}
.eyebrow { color:#8e9aff; font-size:12px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { margin: 5px 0 8px; font-size: 38px; letter-spacing:-1.5px; }
.hero p { color:#9da5b7; margin:0; font-size:15px; }

.metric-card {
    padding: 20px;
    min-height: 132px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(16,19,31,.78);
    box-shadow: inset 0 1px rgba(255,255,255,.03);
}
.metric-label { color:#8f98aa; font-size:12px; text-transform:uppercase; letter-spacing:1.1px; }
.metric-value { font-size:30px; font-weight:800; margin-top:7px; }
.metric-sub { color:#70798c; font-size:12px; margin-top:5px; }

.risk {
    padding: 24px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.08);
    background: linear-gradient(135deg, rgba(20,24,39,.95), rgba(12,15,25,.9));
}
.risk-score { font-size:58px; line-height:1; font-weight:800; }
.risk-low { color:#56e39f; }
.risk-med { color:#ffd166; }
.risk-high { color:#ff6b7a; }
.pill {
    display:inline-block; padding:6px 11px; border-radius:999px;
    font-size:11px; font-weight:800; letter-spacing:.7px;
    background:rgba(255,255,255,.07); color:#cfd5e1;
}
.small-muted { color:#777f91; font-size:12px; }
.section-title { margin: 24px 0 10px; font-size:18px; font-weight:700; }
div[data-testid="stSidebar"] {
    background: #090c15;
    border-right: 1px solid rgba(255,255,255,.06);
}
button[kind="secondary"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=180)

def temperature_c():
    """Best-effort CPU temperature. Many Windows systems do not expose it via psutil."""
    try:
        sensors = psutil.sensors_temperatures()
        if sensors:
            preferred = []
            for key, entries in sensors.items():
                for item in entries:
                    if item.current is not None:
                        preferred.append(float(item.current))
            if preferred:
                return max(preferred)
    except Exception:
        pass
    return None

def collect_metrics():
    cpu = psutil.cpu_percent(interval=0.35)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    load = (os_load := getattr(psutil, "getloadavg", lambda: (0,0,0))())
    processes = len(psutil.pids())
    temp = temperature_c()
    return {
        "time": datetime.now(),
        "cpu": cpu,
        "memory": mem,
        "disk": disk,
        "load1": float(load[0]) if load else 0,
        "processes": processes,
        "temperature": temp,
    }

def risk_engine(m):
    """
    Explainable baseline risk engine.
    This is deliberately transparent rather than pretending it is a trained ML model.
    """
    signals = []

    cpu_score = np.clip((m["cpu"] - 55) / 45 * 100, 0, 100)
    mem_score = np.clip((m["memory"] - 55) / 45 * 100, 0, 100)
    disk_score = np.clip((m["disk"] - 70) / 30 * 100, 0, 100)

    signals += [
        ("CPU pressure", float(cpu_score), m["cpu"], "%"),
        ("Memory pressure", float(mem_score), m["memory"], "%"),
        ("Disk pressure", float(disk_score), m["disk"], "%"),
    ]

    if m["temperature"] is not None:
        temp_score = np.clip((m["temperature"] - 55) / 35 * 100, 0, 100)
        signals.append(("Temperature", float(temp_score), m["temperature"], "°C"))

    # Weighted risk. Temperature is only included when real sensor data exists.
    weights = [0.40, 0.30, 0.20] + ([0.10] if m["temperature"] is not None else [])
    score = sum(s[1] * w for s, w in zip(signals, weights))

    if score >= 70:
        level = "HIGH"
        cls = "risk-high"
    elif score >= 40:
        level = "MEDIUM"
        cls = "risk-med"
    else:
        level = "LOW"
        cls = "risk-low"

    return float(score), level, cls, signals

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ◈ SENTINEL")
    st.caption("Real-Time Server Intelligence")
    st.divider()

    auto = st.toggle("Live monitoring", value=True)
    interval = st.slider("Refresh interval", 2, 15, 5, 1)

    st.markdown("**Monitoring scope**")
    st.caption("CPU • Memory • Disk • Temperature • Process count")

    if st.button("Clear history", use_container_width=True):
        st.session_state.history.clear()
        st.rerun()

    st.divider()
    st.caption("Baseline engine v1.0")
    st.caption("ML model can be plugged into the prediction layer later.")

# ---------- Collect ----------
metrics = collect_metrics()
score, level, level_cls, signals = risk_engine(metrics)
st.session_state.history.append({**metrics, "risk": score})

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Infrastructure Intelligence</div>
  <h1>Sentinel <span style="color:#8e9aff">/</span> Server Failure Predictor</h1>
  <p>Detect degradation early. Quantify failure risk. Act before the server goes down.</p>
</div>
""", unsafe_allow_html=True)

# ---------- KPI row ----------
temp_text = f"{metrics['temperature']:.1f}°C" if metrics["temperature"] is not None else "N/A"

cols = st.columns(4)
cards = [
    ("CPU LOAD", f"{metrics['cpu']:.1f}%", "Real-time utilization"),
    ("MEMORY", f"{metrics['memory']:.1f}%", "RAM utilization"),
    ("DISK", f"{metrics['disk']:.1f}%", "Root volume usage"),
    ("CPU TEMP", temp_text, "Hardware sensor" if metrics["temperature"] is not None else "Sensor unavailable"),
]
for c, (label, value, sub) in zip(cols, cards):
    with c:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

# ---------- Risk ----------
st.markdown('<div class="section-title">Failure Risk</div>', unsafe_allow_html=True)
r1, r2 = st.columns([1, 2])

with r1:
    st.markdown(
        f'<div class="risk"><div class="small-muted">CURRENT RISK SCORE</div>'
        f'<div class="risk-score {level_cls}">{score:.0f}<span style="font-size:22px">/100</span></div>'
        f'<div style="margin-top:12px"><span class="pill">{level} RISK</span></div>'
        f'<div class="small-muted" style="margin-top:14px">'
        f'Based on explainable pressure signals from the live host.</div></div>',
        unsafe_allow_html=True
    )

with r2:
    names = [s[0] for s in signals]
    vals = [s[1] for s in signals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        text=[f"{v:.0f}" for v in vals], textposition="outside",
        marker=dict(line=dict(width=0))
    ))
    fig.update_layout(
        height=235, margin=dict(l=10,r=45,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfd5e1"),
        xaxis=dict(range=[0,110], showgrid=False, title="Pressure score"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- History ----------
st.markdown('<div class="section-title">Live Telemetry</div>', unsafe_allow_html=True)
hist = pd.DataFrame(list(st.session_state.history))

if not hist.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["cpu"], name="CPU %", mode="lines"))
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["memory"], name="Memory %", mode="lines"))
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["disk"], name="Disk %", mode="lines"))
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["risk"], name="Risk", mode="lines", line=dict(width=3)), secondary_y=True)

    fig.update_layout(
        height=390, hovermode="x unified",
        margin=dict(l=10,r=10,t=15,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.02)",
        font=dict(color="#cfd5e1"),
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_yaxes(title_text="Telemetry (%)", range=[0,105], secondary_y=False, gridcolor="rgba(255,255,255,.06)")
    fig.update_yaxes(title_text="Risk", range=[0,105], secondary_y=True, gridcolor="rgba(255,255,255,.03)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- Process intelligence ----------
st.markdown('<div class="section-title">Host Intelligence</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)
p1.metric("Running Processes", f"{metrics['processes']:,}")
p2.metric("1-min Load", f"{metrics['load1']:.2f}")
p3.metric("Last Sample", metrics["time"].strftime("%H:%M:%S"))

st.markdown("""
<div style="margin-top:18px; padding:16px 18px; border-radius:16px;
background:rgba(142,154,255,.06); border:1px solid rgba(142,154,255,.12);
color:#aab2c3; font-size:13px;">
<b style="color:#e9ecf4">Next upgrade:</b> train an anomaly/failure classifier on historical server telemetry,
then replace the explainable baseline score with a calibrated probability such as
“87% failure risk in the next 30 minutes.”
</div>
""", unsafe_allow_html=True)

if auto:
    time.sleep(interval)
    st.rerun()

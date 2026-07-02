"""Streamlit dashboard for interactive backtest inspection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Forecast Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --fl-bg:#0f1216; --fl-panel:#171b21; --fl-panel2:#1e232b; --fl-border:#2a313b;
  --fl-text:#e6ebf0; --fl-muted:#8b96a4; --fl-signal:#b6c73f; --fl-signal-dk:#94a52f;
}

/* Base */
[data-testid="stAppViewContainer"] { background: var(--fl-bg); }
[data-testid="stHeader"] { background: transparent; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif; }
[data-testid="stSidebar"] { background: #12161d; border-right: 1px solid var(--fl-border); }
[data-testid="stSidebar"] * { color: #c3ccd6 !important; }

/* Sidebar wordmark */
.fl-mark { display:flex; align-items:center; gap:10px; margin:2px 0; }
.fl-mark .sq { width:20px; height:20px; background:var(--fl-signal); border-radius:3px;
               box-shadow: inset 0 0 0 5px #12161d; flex-shrink:0; }
.fl-mark .txt { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem;
                letter-spacing:.03em; color:var(--fl-text) !important; text-transform:uppercase; }

/* Header */
.fl-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.fl-header .hmark { width:22px; height:22px; background:var(--fl-signal); border-radius:3px;
                    box-shadow: inset 0 0 0 6px var(--fl-bg); flex-shrink:0; }
.fl-header h1 { font-family:'Space Grotesk',sans-serif; font-size:1.85rem; font-weight:700;
                letter-spacing:-.01em; color:var(--fl-text); margin:0; text-transform:uppercase; }
.fl-badge { font-family:'IBM Plex Mono',monospace; font-size:.7rem; background:var(--fl-panel2);
            color:var(--fl-signal); border:1px solid var(--fl-border); border-radius:3px;
            padding:3px 9px; letter-spacing:.03em; }

/* Metric cards */
.metric-card { background:var(--fl-panel); border:1px solid var(--fl-border);
               border-left:3px solid var(--fl-signal); border-radius:4px;
               padding:14px 18px; }
.metric-label { font-size:.68rem; color:var(--fl-muted); text-transform:uppercase;
                letter-spacing:.09em; margin-bottom:6px; font-weight:600; }
.metric-value { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:600;
                color:var(--fl-text); }

/* Section headers */
.section-header { font-family:'Space Grotesk',sans-serif; font-size:1.02rem; font-weight:600;
                  color:var(--fl-text); border-left:3px solid var(--fl-signal); padding-left:11px;
                  margin:24px 0 12px; text-transform:uppercase; letter-spacing:.03em; }

/* Dataframe */
[data-testid="stDataFrame"] { border:1px solid var(--fl-border); border-radius:4px;
                               overflow:hidden; }

/* Tabs */
[data-testid="stTabs"] button { color:var(--fl-muted) !important; font-weight:600;
    font-family:'Space Grotesk',sans-serif; letter-spacing:.04em; text-transform:uppercase;
    font-size:.82rem; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:var(--fl-text) !important; border-bottom:2px solid var(--fl-signal) !important; }

/* Buttons */
[data-testid="stDownloadButton"] button {
    background:var(--fl-signal) !important; color:#0f1216 !important; border:none !important;
    border-radius:4px !important; font-weight:600 !important;
    font-family:'Space Grotesk',sans-serif !important; letter-spacing:.02em; }
[data-testid="stDownloadButton"] button:hover { background:var(--fl-signal-dk) !important; }

/* st.metric */
[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace; color:var(--fl-text); }
[data-testid="stMetricLabel"] p { color:var(--fl-muted) !important; text-transform:uppercase;
    letter-spacing:.06em; font-size:.7rem; }

/* Captions */
[data-testid="stCaptionContainer"] { color:var(--fl-muted) !important; }

/* Selectbox label */
[data-testid="stSelectbox"] label { color:var(--fl-muted) !important; font-size:.8rem !important;
    text-transform:uppercase; letter-spacing:.05em; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="fl-mark"><span class="sq"></span>'
        '<span class="txt">Forecast Lab</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    runs_root = Path("runs")
    available = sorted([
        d.name for d in runs_root.iterdir()
        if d.is_dir() and (d / "metrics.csv").exists()
    ]) if runs_root.exists() else []

    if not available:
        st.error("No completed runs found in `runs/`.")
        st.stop()

    run_name = st.selectbox("Select run", available, index=len(available) - 1)
    p = runs_root / run_name
    st.markdown(f"**Path:** `{p}`")
    st.markdown("---")
    st.markdown("#### About")
    st.caption("Walk-forward backtest explorer with probabilistic metrics, "
               "calibration diagnostics, and decision artifacts.")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(p / "metrics.csv", index_col=0)
except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"Could not read metrics for run `{run_name}`: {exc}")
    st.stop()

if df.empty:
    st.warning(f"Run `{run_name}` has no metric rows to display.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="fl-header">'
    f'<span class="hmark"></span>'
    f'<h1>Forecast Lab</h1>'
    f'<span class="fl-badge">run: {run_name}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption(f"Backtest results · {len(df)} models · {p}")
st.markdown("---")

# ── KPI cards (best model) ────────────────────────────────────────────────────
best = df.index[0]
best_row = df.iloc[0]

kpi_cols = st.columns(4)
kpi_map = [
    ("Best Model", best, None),
    ("MAE", f"{best_row['MAE']:.3f}" if "MAE" in df.columns else "—", None),
    ("Coverage", f"{best_row['coverage']:.1%}" if "coverage" in df.columns else "—", None),
    ("MASE", f"{best_row['MASE']:.3f}" if "MASE" in df.columns else "—", None),
]
for col, (label, value, delta) in zip(kpi_cols, kpi_map):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Leaderboard", "Forecasts", "Diagnostics", "Decisions"])

with tab1:
    st.markdown('<div class="section-header">Model Leaderboard</div>', unsafe_allow_html=True)

    grad_cols = [c for c in ["MAE", "RMSE", "MASE", "sMAPE"] if c in df.columns]
    highlight_cols = [c for c in ["coverage", "skill_vs_naive_%"] if c in df.columns]

    styled = df.style
    if grad_cols:
        styled = styled.background_gradient(subset=grad_cols, cmap="RdYlGn_r", axis=0)
    if highlight_cols:
        styled = styled.background_gradient(subset=highlight_cols, cmap="RdYlGn", axis=0)
    styled = styled.format(precision=4).set_properties(**{
        "font-size": "13px",
    })

    st.dataframe(styled, use_container_width=True, height=350)

    csv = df.to_csv().encode()
    st.download_button("Download metrics.csv", csv, "metrics.csv", "text/csv")

with tab2:
    st.markdown('<div class="section-header">Per-model Forecast Plots</div>', unsafe_allow_html=True)
    plots = sorted((p / "plots").glob("*.png")) if (p / "plots").exists() else []
    if not plots:
        st.info("No plot images found. Run the backtest first.")
    else:
        n_cols = 2
        rows = [plots[i:i+n_cols] for i in range(0, len(plots), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for col, plot in zip(cols, row):
                col.image(str(plot), caption=plot.stem, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Calibration Diagnostics</div>', unsafe_allow_html=True)
    diag = p / "diagnostics"
    imgs = sorted(diag.glob("*.png")) if diag.exists() else []
    if not imgs:
        st.info("No diagnostic images found.")
    else:
        n_cols = 2
        rows = [imgs[i:i+n_cols] for i in range(0, len(imgs), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for col, img in zip(cols, row):
                col.image(str(img), caption=img.stem, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Decision Artifacts</div>', unsafe_allow_html=True)
    dec = p / "decisions.json"
    if not dec.exists():
        st.info("No decisions.json found for this run. "
                "Add a `decisions:` block to your config to generate them.")
    else:
        try:
            data = json.loads(dec.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not parse decisions.json: {exc}")
            data = {}
        for key, val in data.items():
            with st.expander(f"**{key}**", expanded=True):
                if isinstance(val, list):
                    arr = np.array(val, dtype=float)
                    if arr.size == 0:
                        st.info("No values recorded for this artifact.")
                        continue
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean", f"{arr.mean():.3f}")
                    c2.metric("Min", f"{arr.min():.3f}")
                    c3.metric("Max", f"{arr.max():.3f}")
                    st.line_chart(pd.Series(arr, name=key))
                else:
                    st.metric(key, f"{val:.3f}" if isinstance(val, float) else str(val))
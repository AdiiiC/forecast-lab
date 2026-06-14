"""Streamlit dashboard for interactive backtest inspection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Forecast Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Header */
.fl-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.fl-header h1 { font-size:2rem; font-weight:700; color:#58a6ff;
                background: linear-gradient(90deg,#58a6ff,#a371f7);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.fl-badge { font-size:.7rem; background:#21262d; color:#8b949e;
            border:1px solid #30363d; border-radius:6px; padding:2px 8px; }

/* Metric cards */
.metric-card { background:#161b27; border:1px solid #30363d; border-radius:12px;
               padding:16px 20px; text-align:center; }
.metric-label { font-size:.75rem; color:#8b949e; text-transform:uppercase;
                letter-spacing:.06em; margin-bottom:4px; }
.metric-value { font-size:1.6rem; font-weight:700; color:#f0f6fc; }
.metric-delta-pos { font-size:.8rem; color:#3fb950; }
.metric-delta-neg { font-size:.8rem; color:#f85149; }

/* Section headers */
.section-header { font-size:1.1rem; font-weight:600; color:#e6edf3;
                  border-left:3px solid #58a6ff; padding-left:10px;
                  margin:24px 0 12px; }

/* Dataframe */
[data-testid="stDataFrame"] { border:1px solid #30363d; border-radius:10px;
                               overflow:hidden; }

/* Tabs */
[data-testid="stTabs"] button { color:#8b949e !important; font-weight:500; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#58a6ff !important; border-bottom:2px solid #58a6ff !important; }

/* Image captions */
[data-testid="caption"] { color:#8b949e !important; font-size:.75rem !important; }

/* Sidebar select */
[data-testid="stSelectbox"] label { color:#8b949e !important; font-size:.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Forecast Lab")
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
df = pd.read_csv(p / "metrics.csv", index_col=0)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="fl-header">'
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Leaderboard", "🔭 Forecasts", "📐 Diagnostics", "🎯 Decisions"])

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
    st.download_button("⬇ Download metrics.csv", csv, "metrics.csv", "text/csv")

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
        data = json.loads(dec.read_text())
        for key, val in data.items():
            with st.expander(f"**{key}**", expanded=True):
                if isinstance(val, list):
                    arr = np.array(val)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean", f"{arr.mean():.3f}")
                    c2.metric("Min", f"{arr.min():.3f}")
                    c3.metric("Max", f"{arr.max():.3f}")
                    st.line_chart(pd.Series(arr, name=key))
                else:
                    st.metric(key, f"{val:.3f}" if isinstance(val, float) else str(val))
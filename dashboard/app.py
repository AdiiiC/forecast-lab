"""Streamlit dashboard for interactive backtest inspection."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="forecast-lab", layout="wide")
st.title("forecast-lab — backtest explorer")

run_dir = st.sidebar.text_input("run dir", "runs/energy_v2")
p = Path(run_dir)
if not (p / "metrics.csv").exists():
    st.warning(f"no metrics.csv at {p}"); st.stop()

df = pd.read_csv(p / "metrics.csv")
st.subheader("Leaderboard")
st.dataframe(df.style.background_gradient(subset=["MAE", "RMSE", "MASE"],
                                          cmap="RdYlGn_r"))

st.subheader("Per-model last-fold forecast")
plots = sorted((p / "plots").glob("*.png"))
cols = st.columns(min(3, len(plots)) or 1)
for i, plot in enumerate(plots):
    cols[i % len(cols)].image(str(plot), caption=plot.stem, use_column_width=True)

diag = p / "diagnostics"
if diag.exists():
    st.subheader("Calibration diagnostics")
    for img in diag.glob("*.png"):
        st.image(str(img), caption=img.stem, use_column_width=True)

dec = p / "decisions.json"
if dec.exists():
    import json
    st.subheader("Decision artifacts (best model)")
    st.json(json.loads(dec.read_text()))
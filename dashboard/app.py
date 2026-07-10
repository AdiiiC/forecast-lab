"""Streamlit dashboard — Forecast Lab (high-end, layman-friendly)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Forecast Lab · Results",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
SIG = "#b6c73f"        # citron signal
SIG_DK = "#94a52f"
BG = "#0f1216"
PANEL = "#171b21"
PANEL2 = "#1e232b"
BORDER = "#2a313b"
TEXT = "#e6ebf0"
MUTED = "#8b96a4"
RED = "#e05c6b"
AMBER = "#e8a94d"
BLUE = "#6bb5ff"
PURPLE = "#c084fc"

PLOTLY_BASE = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL2,
    font=dict(color=TEXT, family="IBM Plex Sans, system-ui, sans-serif"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    margin=dict(t=50, b=40, l=20, r=20),
    hoverlabel=dict(bgcolor=PANEL2, bordercolor=BORDER, font_color=TEXT),
)

MODEL_COLORS = [SIG, AMBER, BLUE, PURPLE, RED, "#34d399", "#fb923c", "#f472b6"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --bg:#0f1216; --panel:#171b21; --panel2:#1e232b; --border:#2a313b;
  --text:#e6ebf0; --muted:#8b96a4;
  --sig:#b6c73f; --sig-dk:#94a52f;
  --red:#e05c6b; --amber:#e8a94d; --blue:#6bb5ff;
}

/* ---- Layout ---- */
[data-testid="stAppViewContainer"]          { background: var(--bg); }
[data-testid="stHeader"]                    { background: transparent; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"]                   { font-family:'IBM Plex Sans',system-ui,sans-serif; }
[data-testid="stSidebar"]                   { background:#12161d; border-right:1px solid var(--border); }
[data-testid="stSidebar"] *                 { color:#c3ccd6 !important; }

/* ---- Wordmark ---- */
.fl-mark                      { display:flex;align-items:center;gap:10px;margin:2px 0; }
.fl-mark .sq                  { width:20px;height:20px;background:var(--sig);border-radius:3px;
                                 box-shadow:inset 0 0 0 5px #12161d;flex-shrink:0; }
.fl-mark .txt                 { font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.1rem;
                                 letter-spacing:.03em;color:var(--text) !important;text-transform:uppercase; }

/* ---- Page header ---- */
.fl-header                    { display:flex;align-items:center;gap:12px;margin-bottom:6px; }
.fl-header .hmark             { width:22px;height:22px;background:var(--sig);border-radius:3px;
                                 box-shadow:inset 0 0 0 6px var(--bg);flex-shrink:0; }
.fl-header h1                 { font-family:'Space Grotesk',sans-serif;font-size:1.85rem;font-weight:700;
                                 letter-spacing:-.01em;color:var(--text);margin:0;text-transform:uppercase; }
.fl-badge                     { font-family:'IBM Plex Mono',monospace;font-size:.7rem;
                                 background:var(--panel2);color:var(--sig);
                                 border:1px solid var(--border);border-radius:3px;
                                 padding:3px 9px;letter-spacing:.03em; }

/* ---- Cards ---- */
.metric-card                  { background:var(--panel);border:1px solid var(--border);
                                 border-left:3px solid var(--sig);border-radius:4px;
                                 padding:14px 18px;height:100%; }
.metric-card.red              { border-left-color:var(--red); }
.metric-card.amber            { border-left-color:var(--amber); }
.metric-card.blue             { border-left-color:var(--blue); }
.metric-label                 { font-size:.68rem;color:var(--muted);text-transform:uppercase;
                                 letter-spacing:.09em;margin-bottom:4px;font-weight:600; }
.metric-value                 { font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                 font-weight:600;color:var(--text); }
.metric-sub                   { font-size:.72rem;color:var(--muted);margin-top:4px;line-height:1.4; }

/* ---- Champion card ---- */
.champion-card                { background:linear-gradient(135deg,#1e232b 0%,#1f2b1a 100%);
                                 border:1px solid var(--sig);border-radius:8px;
                                 padding:24px 28px;text-align:center; }
.champion-name                { font-family:'Space Grotesk',sans-serif;font-size:2rem;
                                 font-weight:700;color:var(--sig);text-transform:uppercase;
                                 letter-spacing:.05em;margin:8px 0 4px; }
.champion-sub                 { color:var(--muted);font-size:.88rem;margin-top:4px; }

/* ---- Section headers ---- */
.section-header               { font-family:'Space Grotesk',sans-serif;font-size:1rem;
                                 font-weight:600;color:var(--text);
                                 border-left:3px solid var(--sig);padding-left:11px;
                                 margin:28px 0 10px;text-transform:uppercase;letter-spacing:.04em; }

/* ---- Insight / callout ---- */
.insight-box                  { background:var(--panel);border:1px solid var(--border);
                                 border-left:3px solid var(--amber);border-radius:4px;
                                 padding:14px 18px;margin:14px 0;font-size:.9rem;
                                 color:var(--text);line-height:1.6; }
.insight-box strong           { color:var(--sig); }
.info-box                     { background:var(--panel);border:1px solid var(--border);
                                 border-left:3px solid var(--blue);border-radius:4px;
                                 padding:14px 18px;margin:14px 0;font-size:.88rem;
                                 color:var(--muted);line-height:1.6; }

/* ---- Pills ---- */
.pill                         { display:inline-block;padding:3px 11px;border-radius:12px;
                                 font-size:.72rem;font-weight:600;
                                 font-family:'IBM Plex Mono',monospace; }
.pill-green                   { background:#1e2e10;color:var(--sig);border:1px solid #3d5a1c; }
.pill-red                     { background:#2e1015;color:var(--red);border:1px solid #5a1c22; }
.pill-amber                   { background:#2e2010;color:var(--amber);border:1px solid #5a3e1c; }
.pill-blue                    { background:#10202e;color:var(--blue);border:1px solid #1c3e5a; }

/* ---- Progress bar ---- */
.prog-wrap                    { background:var(--panel2);border-radius:4px;height:8px;
                                 margin:6px 0 2px;overflow:hidden; }
.prog-fill                    { height:100%;border-radius:4px;transition:width .4s; }

/* ---- Tables ---- */
[data-testid="stDataFrame"]   { border:1px solid var(--border);border-radius:4px;overflow:hidden; }

/* ---- Tabs ---- */
[data-testid="stTabs"] button { color:var(--muted) !important;font-weight:600;
    font-family:'Space Grotesk',sans-serif;letter-spacing:.04em;
    text-transform:uppercase;font-size:.79rem; }
[data-testid="stTabs"] button[aria-selected="true"]
                               { color:var(--text) !important;
                                 border-bottom:2px solid var(--sig) !important; }

/* ---- Buttons ---- */
[data-testid="stDownloadButton"] button
                               { background:var(--sig) !important;color:#0f1216 !important;
                                 border:none !important;border-radius:4px !important;
                                 font-weight:600 !important;
                                 font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stDownloadButton"] button:hover { background:var(--sig-dk) !important; }

/* ---- Native metrics ---- */
[data-testid="stMetricValue"]  { font-family:'IBM Plex Mono',monospace;color:var(--text); }
[data-testid="stMetricLabel"] p{ color:var(--muted) !important;text-transform:uppercase;
                                  letter-spacing:.06em;font-size:.7rem; }
[data-testid="stCaptionContainer"] { color:var(--muted) !important; }

/* ---- Selectbox ---- */
[data-testid="stSelectbox"] label { color:var(--muted) !important;font-size:.8rem !important;
    text-transform:uppercase;letter-spacing:.05em; }

/* ---- Expander ---- */
[data-testid="stExpander"]     { border:1px solid var(--border) !important;border-radius:4px !important; }
</style>
""", unsafe_allow_html=True)

# ── Plain-English metric glossary ─────────────────────────────────────────────
METRIC_PLAIN = {
    "MAE":              ("Average Error",              "On average, how many units off each prediction is. Lower = more accurate."),
    "RMSE":             ("Error (Penalises Outliers)", "Like average error but large mistakes count more. Lower = better."),
    "sMAPE":            ("% Error",                   "Average percentage off. Lower = more accurate."),
    "MASE":             ("Relative Accuracy",          "How much better vs simply repeating yesterday's value. Below 1.0 = beats the naive baseline."),
    "coverage":         ("Prediction Band Accuracy",   "% of real values that landed inside the predicted range. Target: ~90%."),
    "PI_width":         ("Confidence Band Width",      "How wide the uncertainty band is. Narrower = more precise, but harder to achieve."),
    "Winkler":          ("Overall Quality Score",      "Combined accuracy + confidence quality. Lower = better."),
    "skill_vs_naive_%": ("Improvement vs Guessing",   "How much better than just repeating last period's value. Higher = better."),
    "NV_cost(3:1)":     ("Business Cost",              "Estimated supply-chain cost (3x penalty for under-stocking). Lower = better."),
    "CRPS":             ("Probabilistic Score",        "How well the full probability distribution matches reality. Lower = better."),
    "energy":           ("Energy Score",               "A comprehensive score for probabilistic forecasts. Lower = better."),
    "QLoss":            ("Quantile Loss",              "How well the model estimates specific percentiles. Lower = better."),
    "DM_p_vs_naive":    ("Statistical Confidence",     "p-value: how confident we are this model truly beats the baseline. < 0.05 = statistically significant."),
    "sig":              ("Significance Stars",         "*** = very strong evidence of improvement, ** = strong, * = moderate, — = not significant."),
}

METRIC_BETTER = {
    "MAE": "lower", "RMSE": "lower", "sMAPE": "lower", "MASE": "lower",
    "coverage": "~90%", "PI_width": "lower", "Winkler": "lower",
    "skill_vs_naive_%": "higher", "NV_cost(3:1)": "lower", "CRPS": "lower",
    "energy": "lower", "QLoss": "lower", "DM_p_vs_naive": "lower",
}

RUN_DESCRIPTIONS = {
    "energy":           ("Hourly Energy Demand", "Standard ML/statistical models forecasting electricity consumption 24 hours ahead."),
    "energy_v2":        ("Hourly Energy Demand v2", "Extended model set including deep learning (TFT, DeepAR) on the same energy task."),
    "energy_adaptive":  ("Adaptive Conformal Energy", "Same energy task but with adaptive conformal prediction — intervals that self-calibrate over time."),
    "energy_cov":       ("Energy + Covariates", "Energy forecasting enriched with external features. Includes foundation models (Chronos, PatchTST)."),
    "intermittent":     ("Intermittent Demand", "Sporadic/lumpy demand — items that are mostly zero with occasional large spikes."),
    "retail_hier":      ("Retail Hierarchy", "Multi-level retail forecasts (total store → category → SKU) with reconciliation methods."),
}


# ── Helper: apply theme to plotly fig ────────────────────────────────────────
def _theme(fig: go.Figure, title: str | None = None, height: int = 400) -> go.Figure:
    upd = dict(**PLOTLY_BASE, height=height)
    if title:
        upd["title"] = dict(text=title, font=dict(size=13, family="Space Grotesk, sans-serif"),
                            x=0, xanchor="left", pad=dict(l=4))
    fig.update_layout(**upd)
    return fig


def _pill(label: str, kind: str = "green") -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def _card(label: str, value: str, sub: str = "", color: str = "") -> str:
    cls = f"metric-card{' ' + color if color else ''}"
    return (
        f'<div class="{cls}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{"<div class=metric-sub>" + sub + "</div>" if sub else ""}'
        f"</div>"
    )


def _section(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def _insight(html: str) -> None:
    st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)


def _info(html: str) -> None:
    st.markdown(f'<div class="info-box">{html}</div>', unsafe_allow_html=True)


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

    run_name = st.selectbox(
        "Select experiment",
        available,
        index=len(available) - 1,
        format_func=lambda x: RUN_DESCRIPTIONS.get(x, (x.replace("_", " ").title(), ""))[0],
    )

    p = runs_root / run_name
    rd = RUN_DESCRIPTIONS.get(run_name, (run_name, ""))
    st.caption(rd[1])

    st.markdown("---")
    compare_runs = st.multiselect(
        "Compare with other experiments",
        [r for r in available if r != run_name],
        default=[],
        format_func=lambda x: RUN_DESCRIPTIONS.get(x, (x, ""))[0],
    )

    st.markdown("---")
    st.markdown("#### How to read this dashboard")
    st.caption(
        "Every chart has a plain-English caption. "
        "Green = good. Red = bad. "
        "Hover over any chart for details. "
        "Use the tabs above to explore different aspects of the results."
    )


# ── Load primary run data ─────────────────────────────────────────────────────
try:
    df = pd.read_csv(p / "metrics.csv", index_col=0)
except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"Could not read metrics for run `{run_name}`: {exc}")
    st.stop()

if df.empty:
    st.warning(f"Run `{run_name}` has no rows to display.")
    st.stop()

# Drop non-numeric junk columns
drop_cols = [c for c in df.columns if df[c].dtype == object and c not in ("sig",)]
df = df.drop(columns=drop_cols, errors="ignore")

# Load reliability.csv
rel_df: pd.DataFrame | None = None
if (p / "reliability.csv").exists():
    try:
        rel_df = pd.read_csv(p / "reliability.csv")
    except Exception:
        pass

# Load forecasts.json
forecasts: dict = {}
if (p / "forecasts.json").exists():
    try:
        raw = json.loads((p / "forecasts.json").read_text())
        forecasts = raw.get("series", {})
    except Exception:
        pass

# Load decisions.json
decisions: dict = {}
if (p / "decisions.json").exists():
    try:
        raw_d = json.loads((p / "decisions.json").read_text())
        if isinstance(raw_d, dict):
            decisions = raw_d
    except Exception:
        pass

# ── Page header ───────────────────────────────────────────────────────────────
run_label = RUN_DESCRIPTIONS.get(run_name, (run_name.replace("_", " ").upper(), ""))[0]
st.markdown(
    f'<div class="fl-header">'
    f'<span class="hmark"></span>'
    f'<h1>Forecast Lab</h1>'
    f'<span class="fl-badge">{run_label}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption(f"{rd[1]}  ·  {len(df)} models evaluated  ·  All results on held-out test data (no peeking)")
st.markdown("---")

# ── Determine best model ──────────────────────────────────────────────────────
best_model = df.index[0]
best_row = df.iloc[0]

num_cols = df.select_dtypes(include=np.number).columns.tolist()

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Overview",
    "Model Race",
    "Forecasts",
    "Reliability",
    "Hierarchy / Decisions",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1  ·  OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    # ── Champion + KPI row ────────────────────────────────────────────────────
    champ_col, kpi_col = st.columns([1, 2.4], gap="large")

    with champ_col:
        sig_val = str(best_row.get("sig", "")).strip()
        sig_pill = _pill("Statistically significant win", "green") if sig_val in ("***", "**", "*") else _pill("Marginal edge", "amber")
        skill_val = best_row.get("skill_vs_naive_%", None)
        skill_str = f"+{skill_val:.1f}% better than just guessing" if pd.notna(skill_val) else ""
        mae_str = f"{best_row['MAE']:.2f} units average error" if "MAE" in df.columns else ""

        st.markdown(
            f'<div class="champion-card">'
            f'<div style="font-size:2.2rem">🏆</div>'
            f'<div style="font-size:.72rem;color:{MUTED};text-transform:uppercase;letter-spacing:.1em;margin-top:4px">Best Performing Model</div>'
            f'<div class="champion-name">{best_model}</div>'
            f'<div class="champion-sub">{mae_str}</div>'
            f'<div class="champion-sub">{skill_str}</div>'
            f'<div style="margin-top:12px">{sig_pill}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with kpi_col:
        kpis = []

        if "MAE" in df.columns:
            kpis.append((
                "Average Prediction Error",
                f"{best_row['MAE']:.2f}",
                "units off on average — lower is more accurate",
                "green",
            ))

        if "coverage" in df.columns:
            cov = best_row["coverage"]
            cov_color = "green" if 0.85 <= cov <= 0.97 else ("amber" if 0.70 <= cov < 0.85 else "red")
            kpis.append((
                "Prediction Band Accuracy",
                f"{cov:.0%}",
                "of real values fell inside the predicted range (target ≈ 90%)",
                cov_color,
            ))

        if "skill_vs_naive_%" in df.columns and pd.notna(best_row.get("skill_vs_naive_%")):
            sk = best_row["skill_vs_naive_%"]
            kpis.append((
                "Improvement over Guessing",
                f"{sk:+.1f}%",
                "vs simply repeating last period's value",
                "green" if sk > 10 else ("amber" if sk > 0 else "red"),
            ))

        if "MASE" in df.columns:
            mase = best_row["MASE"]
            kpis.append((
                "Relative Accuracy (MASE)",
                f"{mase:.3f}",
                "below 1.0 = beats the naive baseline — lower is better",
                "green" if mase < 0.9 else ("amber" if mase < 1.0 else "red"),
            ))

        if "Winkler" in df.columns:
            kpis.append((
                "Overall Quality Score",
                f"{best_row['Winkler']:.2f}",
                "combined accuracy + confidence quality (lower = better)",
                "blue",
            ))

        cols = st.columns(min(len(kpis), 3))
        for col, (lbl, val, sub, color) in zip(cols, kpis[:3]):
            col.markdown(_card(lbl, val, sub, color), unsafe_allow_html=True)

        if len(kpis) > 3:
            st.markdown("<br>", unsafe_allow_html=True)
            cols2 = st.columns(len(kpis) - 3)
            for col, (lbl, val, sub, color) in zip(cols2, kpis[3:]):
                col.markdown(_card(lbl, val, sub, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ranking bar chart ────────────────────────────────────────────────────
    _section("Which Model Is the Most Accurate?")
    st.caption("Shorter bar = smaller average prediction error = more accurate. The winning model is highlighted.")

    if "MAE" in df.columns:
        df_sorted = df.sort_values("MAE")
        bar_colors = [SIG if m == best_model else MUTED for m in df_sorted.index]

        fig_rank = go.Figure(go.Bar(
            y=df_sorted.index,
            x=df_sorted["MAE"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in df_sorted["MAE"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12, color=TEXT),
            hovertemplate="<b>%{y}</b><br>Average Error: %{x:.3f} units<extra></extra>",
        ))
        _theme(fig_rank, "Average Prediction Error by Model  (MAE — lower is better)",
               height=max(200, len(df) * 60))
        fig_rank.update_layout(showlegend=False, xaxis_title="Average Error (units)")
        st.plotly_chart(fig_rank, use_container_width=True)

    # ── Insight: models worse than naive ─────────────────────────────────────
    if "skill_vs_naive_%" in df.columns:
        losers = df[df["skill_vs_naive_%"] < 0].index.tolist()
        if losers:
            names = ", ".join(f"<b>{m}</b>" for m in losers)
            _insight(
                f"<strong>Watch out:</strong> {names} "
                f"{'is' if len(losers)==1 else 'are'} <em>worse</em> than simply repeating "
                f"last period's value. A layman's guess would beat {'it' if len(losers)==1 else 'them'}. "
                f"Consider dropping {'it' if len(losers)==1 else 'them'} or tuning the parameters."
            )

    # ── Accuracy vs Confidence scatter ───────────────────────────────────────
    if "MAE" in df.columns and "coverage" in df.columns and len(df) >= 2:
        _section("Accuracy vs Confidence — The Sweet Spot")
        st.caption(
            "A great model is both accurate (small error, left) AND well-calibrated (~90% coverage, up). "
            "The shaded band marks the target coverage zone. You want to be in the top-left."
        )

        sc_df = df[["MAE", "coverage"]].dropna().copy()
        sc_df["model"] = sc_df.index
        sc_df["size"] = 14

        fig_sc = go.Figure()
        for i, row in sc_df.iterrows():
            is_best = i == best_model
            fig_sc.add_trace(go.Scatter(
                x=[row["MAE"]], y=[row["coverage"]],
                mode="markers+text",
                marker=dict(size=16 if is_best else 12, color=SIG if is_best else MUTED,
                            line=dict(width=2 if is_best else 1, color=SIG if is_best else BORDER),
                            symbol="star" if is_best else "circle"),
                text=[row["model"]],
                textposition="top center",
                textfont=dict(size=11, color=SIG if is_best else TEXT),
                name=row["model"],
                showlegend=False,
                hovertemplate=f"<b>{row['model']}</b><br>Error: {row['MAE']:.3f}<br>Coverage: {row['coverage']:.1%}<extra></extra>",
            ))

        fig_sc.add_hrect(y0=0.85, y1=0.95, fillcolor=SIG, opacity=0.07,
                         line_width=0, annotation_text="Target zone (85–95% coverage)",
                         annotation_position="top right",
                         annotation_font=dict(color=SIG, size=11))
        _theme(fig_sc, "Accuracy vs Confidence Calibration", height=400)
        fig_sc.update_layout(
            xaxis_title="Average Error — lower means more accurate →",
            yaxis=dict(title="Coverage — target ≈ 90%", tickformat=".0%"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Skill vs cost scatter ────────────────────────────────────────────────
    if "skill_vs_naive_%" in df.columns and "NV_cost(3:1)" in df.columns:
        _section("Business Impact — Accuracy vs Cost")
        st.caption(
            "Lower cost AND higher improvement over guessing = the best business outcome. "
            "Aim for the bottom-right corner."
        )
        biz_df = df[["skill_vs_naive_%", "NV_cost(3:1)"]].dropna().copy()
        biz_df["model"] = biz_df.index

        fig_biz = go.Figure()
        for _, row in biz_df.iterrows():
            fig_biz.add_trace(go.Scatter(
                x=[row["skill_vs_naive_%"]], y=[row["NV_cost(3:1)"]],
                mode="markers+text",
                marker=dict(size=14, color=SIG if row["model"] == best_model else MUTED,
                            line=dict(width=1, color=BORDER)),
                text=[row["model"]], textposition="top center",
                textfont=dict(size=10),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['model']}</b><br>"
                    f"Improvement: {row['skill_vs_naive_%']:+.1f}%<br>"
                    f"Business Cost: {row['NV_cost(3:1)']:.2f}<extra></extra>"
                ),
            ))

        _theme(fig_biz, "Better Forecasts, Lower Costs", height=380)
        fig_biz.update_layout(
            xaxis_title="Improvement over guessing (%) — higher is better →",
            yaxis_title="Business cost — lower is better ↓",
        )
        st.plotly_chart(fig_biz, use_container_width=True)

    # ── Summary table (all models, key stats) ────────────────────────────────
    _section("Quick Summary — All Models at a Glance")
    show_cols = [c for c in ["MAE", "coverage", "skill_vs_naive_%", "MASE", "Winkler", "sig"] if c in df.columns]
    summary_df = df[show_cols].copy()
    rename = {c: METRIC_PLAIN.get(c, (c,))[0] for c in show_cols}
    summary_df = summary_df.rename(columns=rename)

    grad_lower = [rename.get(c) for c in ["MAE", "MASE", "Winkler"] if c in df.columns]
    grad_higher = [rename.get(c) for c in ["skill_vs_naive_%", "coverage"] if c in df.columns]

    sty = summary_df.style
    if grad_lower:
        sty = sty.background_gradient(subset=[c for c in grad_lower if c in summary_df.columns],
                                       cmap="RdYlGn_r", axis=0)
    if grad_higher:
        sty = sty.background_gradient(subset=[c for c in grad_higher if c in summary_df.columns],
                                       cmap="RdYlGn", axis=0)
    sty = sty.format({c: "{:.3f}" for c in summary_df.columns if summary_df[c].dtype == float},
                     na_rep="—")

    st.dataframe(sty, use_container_width=True, height=300)
    st.download_button("Download full metrics", df.to_csv().encode(), "metrics.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2  ·  MODEL RACE
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    # ── Radar chart ───────────────────────────────────────────────────────────
    radar_cols = [c for c in ["MAE", "RMSE", "MASE", "sMAPE", "Winkler", "PI_width"] if c in df.columns]

    if len(radar_cols) >= 3:
        _section("Overall Model Footprint")
        st.caption(
            "Each spoke represents one performance dimension. Scores are normalised so "
            "the outermost edge = best on that metric. A model that fills the chart evenly "
            "is well-balanced. Overlapping shapes reveal where models differ."
        )

        # Normalise: 1 = best on that spoke
        rdf = df[radar_cols].dropna(how="all").copy()
        norm = rdf.copy()
        for col in radar_cols:
            cmax, cmin = rdf[col].max(), rdf[col].min()
            if cmax > cmin:
                norm[col] = 1 - (rdf[col] - cmin) / (cmax - cmin)
            else:
                norm[col] = 1.0

        fig_radar = go.Figure()
        for i, model in enumerate(norm.index):
            vals = norm.loc[model].tolist() + [norm.loc[model, radar_cols[0]]]
            cats = radar_cols + [radar_cols[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself", name=model,
                line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2),
                fillcolor=MODEL_COLORS[i % len(MODEL_COLORS)],
                opacity=0.4 if model == best_model else 0.18,
                hovertemplate=f"<b>{model}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor=PANEL2,
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=BORDER),
                angularaxis=dict(gridcolor=BORDER, color=TEXT, tickfont=dict(size=11)),
            ),
            paper_bgcolor=PANEL,
            font=dict(color=TEXT),
            legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(size=11)),
            height=460,
            title=dict(text="Model Performance Radar — outer edge = best on that metric",
                       font=dict(size=13, family="Space Grotesk"), x=0, xanchor="left"),
            margin=dict(t=60, b=20, l=60, r=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Selectable metric bar ─────────────────────────────────────────────────
    _section("Pick Any Metric — See the Full Ranking")
    cmp_options = [c for c in ["MAE", "RMSE", "MASE", "sMAPE", "coverage", "skill_vs_naive_%",
                                "Winkler", "PI_width", "NV_cost(3:1)", "CRPS", "energy", "QLoss"]
                   if c in df.columns]

    cmp_metric = st.selectbox(
        "Metric",
        cmp_options,
        format_func=lambda c: f"{c}  ·  {METRIC_PLAIN.get(c, (c,))[0]}",
    )
    plain_nm, plain_desc = METRIC_PLAIN.get(cmp_metric, (cmp_metric, ""))
    better = METRIC_BETTER.get(cmp_metric, "lower")

    st.caption(f"{plain_desc}  ·  Better direction: **{better}**")

    ascending = better == "lower"
    cdf = df[[cmp_metric]].dropna().sort_values(cmp_metric, ascending=ascending)
    winner_idx = cdf.index[0] if ascending else cdf.index[-1]
    bar_c = [SIG if m == winner_idx else MUTED for m in cdf.index]

    fig_cmp = go.Figure(go.Bar(
        y=cdf.index, x=cdf[cmp_metric], orientation="h",
        marker_color=bar_c,
        text=[f"{v:.3f}" for v in cdf[cmp_metric]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=12, color=TEXT),
        hovertemplate="<b>%{y}</b><br>" + plain_nm + ": %{x:.4f}<extra></extra>",
    ))
    _theme(fig_cmp, f"{plain_nm} ({cmp_metric}) — {'lower' if ascending else 'higher'} is better",
           height=max(180, len(cdf) * 58))
    fig_cmp.update_layout(showlegend=False, xaxis_title=cmp_metric)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Grouped multi-metric comparison ───────────────────────────────────────
    if len(num_cols) >= 2:
        _section("All Key Metrics Side by Side")
        st.caption(
            "Each column is a metric; each row is a model. "
            "Green = performing well on that metric. Red = poor. "
            "A model with all green cells is the clear winner."
        )

        display_cols = [c for c in ["MAE", "RMSE", "MASE", "coverage", "skill_vs_naive_%",
                                     "Winkler", "NV_cost(3:1)"] if c in df.columns]
        disp_df = df[display_cols].copy()
        rename_m = {c: METRIC_PLAIN.get(c, (c,))[0] for c in display_cols}
        disp_df = disp_df.rename(columns=rename_m)

        g_lower = [rename_m[c] for c in ["MAE", "RMSE", "MASE", "Winkler", "NV_cost(3:1)"] if c in display_cols]
        g_higher = [rename_m[c] for c in ["coverage", "skill_vs_naive_%"] if c in display_cols]

        sty2 = disp_df.style
        if g_lower:
            sty2 = sty2.background_gradient(subset=[c for c in g_lower if c in disp_df.columns],
                                              cmap="RdYlGn_r", axis=0)
        if g_higher:
            sty2 = sty2.background_gradient(subset=[c for c in g_higher if c in disp_df.columns],
                                              cmap="RdYlGn", axis=0)
        sty2 = sty2.format(precision=3, na_rep="—")
        st.dataframe(sty2, use_container_width=True, height=320)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    corr_cols = [c for c in num_cols if c not in ("DM_p_vs_naive",) and df[c].notna().sum() >= 2]

    if len(corr_cols) >= 3 and len(df) >= 3:
        _section("How Do Metrics Move Together?")
        st.caption(
            "When two metrics are strongly correlated, improving one tends to improve the other. "
            "Green = move together; Red = trade-off (improving one worsens the other)."
        )

        corr = df[corr_cols].corr()
        plain_labels = [METRIC_PLAIN.get(c, (c,))[0] for c in corr.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=plain_labels, y=plain_labels,
            colorscale=[[0, RED], [0.5, PANEL2], [1, SIG]],
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(size=10, family="IBM Plex Mono"),
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        ))
        _theme(fig_heat, "Metric Correlation Map — green = move together, red = trade-off", height=430)
        fig_heat.update_layout(xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                                yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("What does this mean for my models?"):
            st.markdown("""
- **MAE ↔ RMSE**: Almost always correlated — a model that reduces average error also reduces penalised error.  
- **Coverage ↔ PI_width**: Wide prediction bands naturally cover more — but that's not useful if they're too vague.  
- **MAE ↔ Winkler**: The overall quality score heavily weights accuracy, so they tend to move together.  
- **skill_vs_naive_%**: If this is *negatively* correlated with error metrics, the model that improves most is also the most accurate — a healthy sign.
            """)

    # ── Metric glossary ───────────────────────────────────────────────────────
    with st.expander("Metric dictionary — what does each number actually mean?"):
        for metric, (plain, desc) in METRIC_PLAIN.items():
            if metric in df.columns:
                better_dir = METRIC_BETTER.get(metric, "—")
                st.markdown(f"**{metric}** · *{plain}*  \n{desc}  \n*Better: {better_dir}*\n")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3  ·  FORECASTS
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    plots_dir = p / "plots"
    plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

    if forecasts:
        # ── Single model deep-dive ────────────────────────────────────────────
        _section("Zoom into Any Model's Predictions")
        st.caption(
            "The solid white line = what actually happened. "
            "The green line = what the model predicted. "
            "The shaded area = the model's uncertainty band (it expected the truth to land here 90% of the time)."
        )

        model_list = list(forecasts.keys())
        sel_model = st.selectbox(
            "Choose a model",
            model_list,
            format_func=str.upper,
            key="fc_model_sel",
        )

        fc = forecasts[sel_model]
        actual = fc.get("actual", [])
        forecast_vals = fc.get("forecast", [])
        lo = fc.get("lo", [])
        hi = fc.get("hi", [])
        n_pts = max(len(actual), len(forecast_vals))
        x_ax = list(range(1, n_pts + 1))

        fig_fc = go.Figure()

        if hi and lo:
            fig_fc.add_trace(go.Scatter(
                x=x_ax[:len(hi)] + x_ax[:len(lo)][::-1],
                y=hi + lo[::-1],
                fill="toself", fillcolor="rgba(182,199,63,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="90% Prediction Band", hoverinfo="skip",
            ))
            for bound, label, dash in [(hi, "Upper bound", "dot"), (lo, "Lower bound", "dot")]:
                fig_fc.add_trace(go.Scatter(
                    x=x_ax[:len(bound)], y=bound,
                    line=dict(color=SIG, width=1, dash=dash),
                    name=label,
                    hovertemplate="Hour %{x}<br>Bound: %{y:.2f}<extra></extra>",
                ))

        if forecast_vals:
            fig_fc.add_trace(go.Scatter(
                x=x_ax[:len(forecast_vals)], y=forecast_vals,
                line=dict(color=SIG, width=2.5),
                name="Predicted",
                hovertemplate="Hour %{x}<br>Predicted: %{y:.2f}<extra></extra>",
            ))

        if actual:
            fig_fc.add_trace(go.Scatter(
                x=x_ax[:len(actual)], y=actual,
                line=dict(color=TEXT, width=2.5),
                name="Actual (what really happened)",
                hovertemplate="Hour %{x}<br>Actual: %{y:.2f}<extra></extra>",
            ))

        _theme(fig_fc, f"{sel_model.upper()} — Predicted vs Actual", height=430)
        fig_fc.update_layout(
            xaxis_title="Hours ahead in the forecast horizon",
            yaxis_title="Demand (units)",
            legend=dict(bgcolor=PANEL, bordercolor=BORDER, orientation="h",
                        yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Per-hour error bar ────────────────────────────────────────────────
        if actual and forecast_vals:
            errors = [abs(a - f) for a, f in zip(actual, forecast_vals)]
            avg_err = np.mean(errors)

            col_err, col_stat = st.columns([3, 1])
            with col_err:
                fig_err = go.Figure(go.Bar(
                    x=list(range(1, len(errors) + 1)), y=errors,
                    marker=dict(
                        color=errors,
                        colorscale=[[0, SIG], [0.5, AMBER], [1, RED]],
                        showscale=True,
                        colorbar=dict(title="Error size", tickfont=dict(size=9)),
                    ),
                    hovertemplate="Hour %{x}<br>Error: %{y:.2f} units<extra></extra>",
                ))
                fig_err.add_hline(y=avg_err, line_color=MUTED, line_dash="dash",
                                   annotation_text=f"Avg: {avg_err:.2f}",
                                   annotation_font_color=MUTED)
                _theme(fig_err, f"Where Does {sel_model.upper()} Struggle? (Error by Hour)", height=280)
                fig_err.update_layout(showlegend=False,
                                       xaxis_title="Hour ahead", yaxis_title="Error (units)")
                st.plotly_chart(fig_err, use_container_width=True)
                st.caption("Green bars = small error (model is confident here). Red bars = large error (model struggles here).")

            with col_stat:
                max_err = max(errors)
                max_hr = errors.index(max_err) + 1
                min_err = min(errors)
                min_hr = errors.index(min_err) + 1

                st.markdown(_card("Average Error", f"{avg_err:.2f}", "units"), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(_card("Worst Hour", f"Hour {max_hr}", f"error: {max_err:.2f} units", "red"), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(_card("Best Hour", f"Hour {min_hr}", f"error: {min_err:.2f} units", ""), unsafe_allow_html=True)

        # ── All models overlay ────────────────────────────────────────────────
        if len(forecasts) > 1:
            _section("All Models vs Actual — Side-by-Side Comparison")
            st.caption(
                "Every model's prediction plotted on the same chart. "
                "The thick white line is the truth. Models closer to it performed better."
            )

            fig_all = go.Figure()
            first_actual = list(forecasts.values())[0].get("actual", [])
            if first_actual:
                fig_all.add_trace(go.Scatter(
                    x=list(range(1, len(first_actual) + 1)), y=first_actual,
                    line=dict(color=TEXT, width=3),
                    name="ACTUAL",
                    hovertemplate="<b>ACTUAL</b><br>Hour %{x}: %{y:.2f}<extra></extra>",
                ))

            for i, (mname, mdata) in enumerate(forecasts.items()):
                mfc = mdata.get("forecast", [])
                if mfc:
                    fig_all.add_trace(go.Scatter(
                        x=list(range(1, len(mfc) + 1)), y=mfc,
                        line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2,
                                  dash="dot" if mname != best_model else "solid"),
                        name=mname.upper(),
                        hovertemplate=f"<b>{mname.upper()}</b><br>Hour %{{x}}: %{{y:.2f}}<extra></extra>",
                    ))

            _theme(fig_all, "All Model Predictions vs Actual Values", height=440)
            fig_all.update_layout(
                xaxis_title="Hours ahead", yaxis_title="Demand (units)",
                legend=dict(bgcolor=PANEL, bordercolor=BORDER, orientation="h",
                            yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_all, use_container_width=True)

            # ── Error distribution violin ─────────────────────────────────────
            _section("How Errors Are Distributed — Signed (Over- vs Under-Prediction)")
            st.caption(
                "Errors above zero = model over-predicted (too high). "
                "Below zero = model under-predicted (too low). "
                "A centred, narrow shape means the model is accurate and unbiased."
            )
            fig_violin = go.Figure()
            for i, (mname, mdata) in enumerate(forecasts.items()):
                act = mdata.get("actual", [])
                prd = mdata.get("forecast", [])
                if act and prd:
                    signed_errs = [p - a for a, p in zip(act, prd)]
                    fig_violin.add_trace(go.Violin(
                        y=signed_errs, name=mname.upper(),
                        line_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                        fillcolor=MODEL_COLORS[i % len(MODEL_COLORS)],
                        opacity=0.5, box_visible=True, meanline_visible=True,
                        hovertemplate=f"<b>{mname.upper()}</b><br>%{{y:.2f}}<extra></extra>",
                    ))

            fig_violin.add_hline(y=0, line_color=MUTED, line_dash="dash", line_width=1.5,
                                  annotation_text="Perfect (no bias)", annotation_font_color=MUTED)
            _theme(fig_violin, "Signed Prediction Errors by Model (centred near zero = unbiased)", height=380)
            fig_violin.update_layout(yaxis_title="Signed Error (predicted − actual)",
                                      legend=dict(bgcolor=PANEL, bordercolor=BORDER))
            st.plotly_chart(fig_violin, use_container_width=True)

    elif plot_files:
        _section("Per-Model Forecast Plots")
        st.caption("Static forecast images from the backtest run. Interactive charts require `forecasts.json`.")
        n_cols = 2
        for row_imgs in [plot_files[i:i+n_cols] for i in range(0, len(plot_files), n_cols)]:
            cols = st.columns(n_cols)
            for col, img in zip(cols, row_imgs):
                col.image(str(img), caption=img.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("No forecast data found for this run. Run the backtest pipeline first.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4  ·  RELIABILITY
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    _section("Are the Uncertainty Estimates Trustworthy?")
    _info(
        "A forecast model does two things: it predicts a value, and it says how confident it is. "
        "If a model claims '90% of the time the real value will fall in this band', "
        "we check: does that actually happen 90% of the time? "
        "A model that is overconfident (too narrow bands) will miss more than it should."
    )

    # ── Coverage gauges ───────────────────────────────────────────────────────
    if "coverage" in df.columns:
        _section("Coverage Check — Does the Model's Confidence Match Reality?")
        st.caption("Target is 90%. Green = well-calibrated. Amber = slightly off. Red = poorly calibrated.")

        cov_cols = st.columns(min(len(df), 4))
        for col, model in zip(cov_cols * (len(df) // 4 + 1), df.index):
            if model not in df.index:
                continue
            cov = df.loc[model, "coverage"]
            if pd.isna(cov):
                continue
            cov_pct = cov * 100
            diff = cov_pct - 90
            color = SIG if abs(diff) < 5 else (AMBER if abs(diff) < 15 else RED)
            arrow = "↑" if diff > 1 else ("↓" if diff < -1 else "≈")
            sub = f"Target 90% · Diff {diff:+.1f}%"
            card_color = "" if abs(diff) < 5 else ("amber" if abs(diff) < 15 else "red")
            col.markdown(_card(model.upper(), f"{cov_pct:.1f}% {arrow}", sub, card_color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if rel_df is not None and not rel_df.empty:
        rl_col, rw_col = st.columns(2)

        with rl_col:
            # Reliability scatter: claimed vs empirical
            _section("Claimed vs Actual Coverage")
            st.caption(
                "Dots on the dashed line = perfectly calibrated. "
                "Below the line = overconfident (misses more than claimed). "
                "Above = conservative (wider bands than necessary)."
            )

            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color=MUTED, dash="dash", width=1.5),
                name="Perfect calibration", hoverinfo="skip",
            ))
            fig_rel.add_annotation(x=0.55, y=0.44, text="← Overconfident",
                                    showarrow=False, font=dict(color=RED, size=11))
            fig_rel.add_annotation(x=0.42, y=0.57, text="Conservative →",
                                    showarrow=False, font=dict(color=AMBER, size=11))

            for _, row in rel_df.iterrows():
                diff = abs(row.get("empirical", 0) - row.get("nominal", 0))
                dot_color = SIG if diff < 0.05 else (AMBER if diff < 0.1 else RED)
                fig_rel.add_trace(go.Scatter(
                    x=[row["nominal"]], y=[row["empirical"]],
                    mode="markers+text",
                    marker=dict(size=15, color=dot_color, line=dict(width=1, color=BORDER)),
                    text=[row.get("model", "")], textposition="top center",
                    textfont=dict(size=10),
                    name=row.get("model", ""), showlegend=False,
                    hovertemplate=(
                        f"<b>{row.get('model','')}</b><br>"
                        f"Claimed: {row['nominal']:.0%}<br>"
                        f"Actual: {row.get('empirical',0):.0%}<extra></extra>"
                    ),
                ))

            _theme(fig_rel, "Calibration Diagram", height=370)
            fig_rel.update_layout(
                xaxis=dict(title="Claimed coverage", tickformat=".0%", range=[0, 1.05]),
                yaxis=dict(title="Actual coverage on test data", tickformat=".0%", range=[0, 1.05]),
                showlegend=False,
            )
            st.plotly_chart(fig_rel, use_container_width=True)

        with rw_col:
            # Width vs coverage
            _section("Confidence vs Precision Tradeoff")
            st.caption(
                "Narrow band + high coverage = the dream. "
                "Wide bands are easy to hit — they're just not very useful. "
                "Aim for the top-left corner."
            )

            if "mean_width" in rel_df.columns:
                fig_tw = go.Figure()
                for _, row in rel_df.iterrows():
                    fig_tw.add_trace(go.Scatter(
                        x=[row["mean_width"]], y=[row.get("empirical", 0)],
                        mode="markers+text",
                        marker=dict(size=15, color=SIG, line=dict(width=1, color=BORDER)),
                        text=[row.get("model", "")], textposition="top center",
                        textfont=dict(size=10),
                        name=row.get("model", ""), showlegend=False,
                        hovertemplate=(
                            f"<b>{row.get('model','')}</b><br>"
                            f"Band Width: {row['mean_width']:.2f}<br>"
                            f"Coverage: {row.get('empirical',0):.0%}<extra></extra>"
                        ),
                    ))

                fig_tw.add_hrect(y0=0.85, y1=1.01, fillcolor=SIG, opacity=0.05, line_width=0,
                                  annotation_text="Good coverage zone",
                                  annotation_font=dict(color=SIG, size=11))
                _theme(fig_tw, "Band Width vs Coverage (aim: narrow + high)", height=370)
                fig_tw.update_layout(
                    xaxis_title="Average band width (narrower = more precise)",
                    yaxis=dict(title="Coverage (higher = more reliable)", tickformat=".0%"),
                    showlegend=False,
                )
                st.plotly_chart(fig_tw, use_container_width=True)

    # ── Calibration diagnostic images ─────────────────────────────────────────
    diag_dir = p / "diagnostics"
    if diag_dir.exists():
        diag_imgs = sorted(diag_dir.glob("*.png"))
        if diag_imgs:
            _section("Deep Calibration Diagnostics")
            dcols = st.columns(len(diag_imgs))
            for col, img in zip(dcols, diag_imgs):
                col.image(str(img), caption=img.stem.replace("_", " ").title(), use_container_width=True)

            with st.expander("What am I looking at? (Plain English guide)"):
                st.markdown("""
### PIT Histogram (Probability Integral Transform)
This tells us if the model's probability estimates are accurate.

| Shape | What it means |
|-------|---------------|
| **Flat bar chart** | Well-calibrated — the model knows what it doesn't know |
| **Spike on the left** | Model systematically over-forecasts |
| **Spike on the right** | Model systematically under-forecasts |
| **U-shape** | Bands too narrow — model is overconfident |
| **Hump in centre** | Bands too wide — model is too cautious |

### Sharpness / Coverage Chart
Shows the tradeoff between how often the model is right (coverage) and how precise it is (sharpness = narrow bands).  
A good model has **high coverage AND narrow bands**.
                """)

    # ── Probabilistic metrics breakdown ──────────────────────────────────────
    prob_cols = [c for c in ["CRPS", "energy", "QLoss", "Winkler"] if c in df.columns]
    if prob_cols:
        _section("Probabilistic Quality Scores")
        st.caption(
            "These scores go beyond point accuracy — they measure how well the model "
            "describes the full range of possible outcomes. Lower is better."
        )

        prob_df = df[prob_cols].dropna(how="all")
        fig_prob = go.Figure()
        for i, col in enumerate(prob_cols):
            col_data = prob_df[col].dropna()
            if col_data.empty:
                continue
            plain_c = METRIC_PLAIN.get(col, (col,))[0]
            fig_prob.add_trace(go.Bar(
                name=plain_c,
                x=col_data.index,
                y=col_data.values,
                marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                hovertemplate=f"<b>%{{x}}</b><br>{plain_c}: %{{y:.3f}}<extra></extra>",
            ))

        _theme(fig_prob, "Probabilistic Metric Comparison (lower = better)", height=360)
        fig_prob.update_layout(
            barmode="group",
            xaxis_title="Model", yaxis_title="Score (lower = better)",
            legend=dict(bgcolor=PANEL, bordercolor=BORDER),
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5  ·  HIERARCHY / DECISIONS
# ════════════════════════════════════════════════════════════════════════════════
with t5:
    RECON_FILES = {
        "recon_bu":         ("Bottom Up",   "Add individual SKU forecasts up to get category & store totals. Simple and consistent."),
        "recon_mint_shrink":("MinT Shrink",  "Smart statistical method — finds the optimal linear combination. Usually best."),
        "recon_ols":        ("OLS",          "Ordinary least-squares regression combines forecasts across levels."),
        "recon_wls":        ("WLS",          "Like OLS but weights series by their variance."),
        "recon_td":         ("Top Down",     "Split the total forecast down to SKU level. Loses granular signal at the bottom."),
    }

    hier_data: dict[str, pd.DataFrame] = {}
    for fname, (method_name, _) in RECON_FILES.items():
        fpath = p / f"{fname}.csv"
        if fpath.exists():
            try:
                hdf = pd.read_csv(fpath)
                hdf["method"] = method_name
                hier_data[method_name] = hdf
            except Exception:
                pass

    if hier_data:
        _section("Retail Hierarchy Reconciliation")
        _info(
            "When forecasting a product hierarchy (store total → categories → individual SKUs), "
            "the numbers often don't add up consistently. "
            "<b>Reconciliation</b> fixes this by adjusting all levels simultaneously. "
            "A positive improvement means the reconciled forecast is <em>more accurate</em> than the raw forecast."
        )

        method_colors = [SIG, AMBER, BLUE, PURPLE, RED]
        all_recon = pd.concat(hier_data.values(), ignore_index=True)
        methods = list(hier_data.keys())

        # ── Improvement grouped bar ───────────────────────────────────────────
        _section("How Much Does Reconciliation Help at Each Level?")
        st.caption(
            "Positive bars (above zero) = reconciliation improved accuracy at that level. "
            "Negative = it made things worse. Higher = better."
        )

        fig_recon = go.Figure()
        for i, method in enumerate(methods):
            mdf = all_recon[all_recon["method"] == method]
            fig_recon.add_trace(go.Bar(
                name=method,
                x=mdf["level"], y=mdf["delta_%"],
                marker_color=method_colors[i % len(method_colors)],
                hovertemplate=(
                    f"<b>{method}</b><br>Level: %{{x}}<br>"
                    "Improvement: %{y:+.2f}%<extra></extra>"
                ),
            ))

        fig_recon.add_hline(y=0, line_color=MUTED, line_dash="dash", line_width=1.5)
        _theme(fig_recon, "% Improvement in Forecast Accuracy After Reconciliation  (positive = better)", height=420)
        fig_recon.update_layout(
            barmode="group",
            xaxis_title="Hierarchy Level (TOP = total store, BOTTOM = individual SKU)",
            yaxis_title="% Improvement in Accuracy",
            legend=dict(bgcolor=PANEL, bordercolor=BORDER),
        )
        st.plotly_chart(fig_recon, use_container_width=True)

        # ── Best method per level ─────────────────────────────────────────────
        _section("Winner at Each Level")
        level_order = ["TOP", "L1", "L2", "BOTTOM"]
        best_per_level = (
            all_recon.groupby("level")
            .apply(lambda g: g.loc[g["delta_%"].idxmax()], include_groups=False)
            .reset_index()
        )

        # Reorder
        best_per_level["_ord"] = best_per_level["level"].apply(
            lambda x: level_order.index(x) if x in level_order else 99
        )
        best_per_level = best_per_level.sort_values("_ord").drop(columns="_ord")

        lvl_cols = st.columns(len(best_per_level))
        for col, (_, row) in zip(lvl_cols, best_per_level.iterrows()):
            ok = row["delta_%"] > 0
            col.markdown(
                _card(
                    f"Level: {row['level']}",
                    row["method"],
                    f"{'↑' if ok else '↓'} {abs(row['delta_%']):.2f}% {'improvement' if ok else 'degradation'}",
                    "" if ok else "red",
                ),
                unsafe_allow_html=True,
            )

        # ── MAE line chart: base vs reconciled ───────────────────────────────
        _section("Forecast Error Before vs After Reconciliation")
        st.caption("Lower = more accurate. See if reconciliation reduces error at each level of the hierarchy.")

        fig_mae = go.Figure()
        base_df = list(hier_data.values())[0]
        fig_mae.add_trace(go.Scatter(
            x=base_df["level"], y=base_df["MAE_base"],
            mode="lines+markers", name="Before Reconciliation (base)",
            line=dict(color=TEXT, width=2.5, dash="dot"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="<b>Base</b><br>Level: %{x}<br>MAE: %{y:.3f}<extra></extra>",
        ))
        for i, method in enumerate(methods):
            mdf = all_recon[all_recon["method"] == method]
            fig_mae.add_trace(go.Scatter(
                x=mdf["level"], y=mdf["MAE_recon"],
                mode="lines+markers", name=method,
                line=dict(color=method_colors[i % len(method_colors)], width=2),
                marker=dict(size=8),
                hovertemplate=f"<b>{method}</b><br>Level: %{{x}}<br>MAE: %{{y:.3f}}<extra></extra>",
            ))

        _theme(fig_mae, "Mean Absolute Error by Hierarchy Level  (lower = better)", height=400)
        fig_mae.update_layout(
            xaxis_title="Hierarchy Level", yaxis_title="Mean Absolute Error (lower = more accurate)",
            legend=dict(bgcolor=PANEL, bordercolor=BORDER),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

        # ── Method description cards ──────────────────────────────────────────
        _section("Reconciliation Methods Explained")
        desc_cols = st.columns(min(len(RECON_FILES), 3))
        for col, (fname, (method, desc)) in zip(desc_cols * 2, RECON_FILES.items()):
            if method in hier_data:
                col.markdown(_card(method, "", desc), unsafe_allow_html=True)

    elif decisions:
        # ── Intermittent demand decisions ─────────────────────────────────────
        _section("Supply Chain Decisions from the Forecast")
        _info(
            "For products with sporadic, lumpy demand (mostly zero with occasional large spikes), "
            "the model recommends <b>how much to order</b> and <b>when to dispatch</b>. "
            "The newsvendor order quantity balances the cost of over-stocking vs running out."
        )

        nv_orders = decisions.get("newsvendor_order", [])
        dispatch_steps = decisions.get("dispatch_triggered_steps", [])
        dispatch_overage = decisions.get("dispatch_expected_overage", [])
        safety_stock = decisions.get("safety_stock", None)

        if safety_stock is not None:
            st.markdown(
                _card("Recommended Safety Stock", f"{safety_stock:.0f} units",
                      "Keep at least this many units on hand to handle demand uncertainty", "blue"),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

        if nv_orders:
            x_d = list(range(1, len(nv_orders) + 1))

            fig_nv = go.Figure()
            fig_nv.add_trace(go.Bar(
                x=x_d, y=nv_orders,
                marker_color=SIG, name="Order Quantity",
                hovertemplate="Period %{x}<br>Order: %{y:.1f} units<extra></extra>",
            ))

            if dispatch_steps:
                dispatch_x = [i + 1 for i, v in enumerate(dispatch_steps) if v > 0]
                dispatch_y = [nv_orders[i] for i in range(len(nv_orders)) if dispatch_steps[i] > 0]
                if dispatch_x:
                    fig_nv.add_trace(go.Scatter(
                        x=dispatch_x, y=dispatch_y,
                        mode="markers", name="Dispatch Triggered",
                        marker=dict(size=14, color=RED, symbol="triangle-up",
                                    line=dict(width=1, color=BORDER)),
                        hovertemplate="Period %{x}<br>Dispatch triggered<extra></extra>",
                    ))

            _theme(fig_nv, "Recommended Order Quantities by Period", height=380)
            fig_nv.update_layout(
                xaxis_title="Forecast Period",
                yaxis_title="Units to Order",
                legend=dict(bgcolor=PANEL, bordercolor=BORDER),
            )
            st.plotly_chart(fig_nv, use_container_width=True)

        if dispatch_overage:
            fig_ov = go.Figure(go.Scatter(
                x=list(range(1, len(dispatch_overage) + 1)), y=dispatch_overage,
                mode="lines+markers", line=dict(color=AMBER, width=2),
                marker=dict(size=7),
                fill="tozeroy", fillcolor="rgba(232,169,77,0.1)",
                hovertemplate="Period %{x}<br>Expected Overage: %{y:.2f}<extra></extra>",
            ))
            _theme(fig_ov, "Expected Overage by Period (excess inventory cost)", height=280)
            fig_ov.update_layout(
                xaxis_title="Period", yaxis_title="Expected Overage (units)",
                showlegend=False,
            )
            st.plotly_chart(fig_ov, use_container_width=True)
            st.caption("Lower overage = less wasted inventory. Spikes here signal periods where the model suggests stocking up but demand may not materialise.")

    else:
        _info(
            "This tab shows <b>hierarchy reconciliation</b> (when the <code>retail_hier</code> experiment is selected) "
            "or <b>supply chain decisions</b> (for the <code>intermittent</code> experiment). "
            "Select one of those experiments from the sidebar to explore this data."
        )

    # ── Cross-run comparison ───────────────────────────────────────────────────
    if compare_runs:
        _section("Cross-Experiment Comparison")
        st.caption("Compare how models perform across different experiments. Each box shows the spread of model scores within that experiment.")

        all_run_data: dict[str, pd.DataFrame] = {run_name: df}
        for crun in compare_runs:
            cpath = runs_root / crun / "metrics.csv"
            if cpath.exists():
                try:
                    all_run_data[crun] = pd.read_csv(cpath, index_col=0)
                except Exception:
                    pass

        if len(all_run_data) > 1:
            xrun_metric = st.selectbox(
                "Metric to compare across experiments",
                [c for c in ["MAE", "RMSE", "MASE", "skill_vs_naive_%"] if c in df.columns],
                format_func=lambda c: f"{c}  ·  {METRIC_PLAIN.get(c, (c,))[0]}",
                key="xrun_sel",
            )

            fig_xrun = go.Figure()
            for i, (rname, rdf) in enumerate(all_run_data.items()):
                if xrun_metric in rdf.columns:
                    vals = rdf[xrun_metric].dropna().values
                    run_label_xr = RUN_DESCRIPTIONS.get(rname, (rname, ""))[0]
                    fig_xrun.add_trace(go.Box(
                        y=vals, name=run_label_xr,
                        marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                        boxpoints="all", jitter=0.35, pointpos=-1.5,
                        hovertemplate=f"<b>{run_label_xr}</b><br>%{{y:.3f}}<extra></extra>",
                    ))

            plain_xr = METRIC_PLAIN.get(xrun_metric, (xrun_metric,))[0]
            _theme(fig_xrun, f"{plain_xr} ({xrun_metric}) Distribution Across Experiments", height=400)
            fig_xrun.update_layout(
                yaxis_title=f"{xrun_metric} (lower = better)" if METRIC_BETTER.get(xrun_metric) == "lower" else xrun_metric,
                legend=dict(bgcolor=PANEL, bordercolor=BORDER),
            )
            st.plotly_chart(fig_xrun, use_container_width=True)

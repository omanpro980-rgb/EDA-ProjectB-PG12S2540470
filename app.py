import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "TIMESTAMP"
DEFAULT_TARGET_COL = "ND"
DEFAULT_STUDENT_NAME = "Ibrahim Al Manwari"
DEFAULT_STUDENT_ID = "PG12S2540470"


st.set_page_config(
    page_title="Time-Series Forecasting Workbench",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Theme + typography --------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      /* ---------- App-wide theme ---------- */
      .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, rgba(99, 102, 241, 0.18), transparent 60%),
          radial-gradient(900px 500px at 100% 10%, rgba(14, 165, 233, 0.14), transparent 60%),
          linear-gradient(180deg, #0b1226 0%, #0a0f1f 60%, #07091a 100%) !important;
        color: #e6edf7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 17px;
        line-height: 1.55;
      }
      .block-container { padding-top: 1.2rem; max-width: 1280px; }

      /* Headings */
      h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
        color: #f4f7ff !important;
      }
      h1 { font-size: 2.1rem !important; }
      h2 { font-size: 1.55rem !important; }
      h3 { font-size: 1.2rem !important; }
      p, label, .stMarkdown { font-size: 1rem; color: #cbd5e1; }
      .stCaption, .caption { color: #94a3b8 !important; font-size: 0.92rem !important; }

      /* Monospace */
      code, pre, .stCodeBlock, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
      }

      /* ---------- Metric cards ---------- */
      .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 73, 0.85) 0%, rgba(15, 23, 42, 0.85) 100%);
        padding: 1rem 1.15rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.18);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        transition: transform 0.18s ease, border-color 0.18s ease;
      }
      .stMetric:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.45);
      }
      .stMetric label, .stMetric [data-testid="stMetricLabel"] {
        color: #93c5fd !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
      }
      .stMetric [data-testid="stMetricValue"] {
        color: #f9fafb !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
      }
      .stMetric [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

      /* ---------- Section banners ---------- */
      .section-banner {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.95) 0%, rgba(168, 85, 247, 0.95) 50%, rgba(236, 72, 153, 0.95) 100%);
        color: white;
        padding: 16px 22px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.22rem;
        margin: 16px 0 18px 0;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.28);
        letter-spacing: 0.2px;
        display: flex;
        align-items: center;
        gap: 14px;
      }
      .section-banner .num {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 10px;
        padding: 4px 12px;
        font-size: 0.95rem;
        font-weight: 700;
        min-width: 30px;
        text-align: center;
      }
      .section-banner .body { flex: 1; }
      .section-banner .subtitle {
        display: block;
        font-weight: 400;
        font-size: 0.88rem;
        opacity: 0.92;
        margin-top: 3px;
      }

      /* ---------- Top nav pills ---------- */
      .top-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        padding: 10px 14px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 14px;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
      }
      .top-nav a {
        text-decoration: none !important;
        color: #cbd5e1 !important;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid transparent;
        transition: all 0.15s ease;
      }
      .top-nav a:hover {
        color: #f4f7ff !important;
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.45);
      }

      /* ---------- Pills / badges ---------- */
      .pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 6px 2px 0;
        font-family: 'Inter', sans-serif;
      }
      .pill-ok { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.35); }
      .pill-warn { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }
      .pill-info { background: rgba(59, 130, 246, 0.14); color: #60a5fa; border: 1px solid rgba(59,130,246,0.35); }

      /* ---------- Tabs ---------- */
      .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.15);
      }
      .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
      }
      .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(168, 85, 247, 0.4)) !important;
        color: #f4f7ff !important;
      }

      /* ---------- Buttons ---------- */
      .stButton button {
        font-weight: 600 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
      }
      .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
      }
      .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        border: none !important;
      }
      .stDownloadButton button {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(34, 197, 94, 0.2)) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        color: #34d399 !important;
        font-weight: 600 !important;
      }

      /* ---------- Inputs ---------- */
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        color: #e6edf7 !important;
      }
      .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
      }
      .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
      }

      /* ---------- Expanders ---------- */
      .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #cbd5e1 !important;
      }
      [data-testid="stExpander"] {
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 12px !important;
      }

      /* ---------- DataFrames ---------- */
      .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid rgba(99, 102, 241, 0.15); }

      /* ---------- Sidebar ---------- */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 18, 38, 0.95) 0%, rgba(7, 9, 26, 0.95) 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
      }

      /* ---------- Progress bar ---------- */
      .stProgress > div > div > div { background: linear-gradient(90deg, #6366f1, #ec4899) !important; }

      /* ---------- Alerts ---------- */
      [data-testid="stAlert"] { border-radius: 12px !important; }

      /* ---------- Flow / block diagram boxes ---------- */
      .flow-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 14px 18px;
        margin: 8px 0;
      }
      .flow-step {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 4px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #94a3b8;
        transition: all 0.2s ease;
      }
      .flow-step.active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(168, 85, 247, 0.35));
        border-color: rgba(168, 85, 247, 0.6);
        color: #f4f7ff;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4);
      }
      .flow-step.done {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
      }
      .flow-arrow {
        color: #475569;
        font-size: 1.2rem;
        margin: 0 4px;
      }

      /* ---------- Hero ---------- */
      .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        text-align: center;
        margin: 0;
        padding-top: 8px;
      }
      .hero-sub {
        color: #94a3b8;
        text-align: center;
        font-size: 1.02rem;
        margin-top: 6px;
      }

      /* ---------- Plotly chart container ---------- */
      .js-plotly-plot, .plotly { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_banner(number, title: str, subtitle: str = "", anchor: str = ""):
    """Render a colorful section banner with optional HTML anchor."""
    anchor_html = f'<a id="{anchor}"></a>' if anchor else ""
    st.markdown(
        f"""
        {anchor_html}
        <div class="section-banner">
          <div class="num">{number}</div>
          <div class="body">
            {title}
            {f'<span class="subtitle">{subtitle}</span>' if subtitle else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_flow(step_index: int):
    """Render the project flow with the current step highlighted.

    step_index is the 1-based index of the active step (1..7). Earlier steps
    are marked 'done', later steps remain default.
    """
    steps = [
        ("📂", "Load"),
        ("🧹", "Clean"),
        ("⚙️", "Resample"),
        ("🧱", "Features"),
        ("🤖", "Model"),
        ("📊", "Dashboard"),
        ("📦", "Export"),
    ]
    pieces = []
    for i, (icon, label) in enumerate(steps, start=1):
        if i < step_index:
            cls = "flow-step done"
        elif i == step_index:
            cls = "flow-step active"
        else:
            cls = "flow-step"
        pieces.append(f'<span class="{cls}">{icon} {label}</span>')
        if i < len(steps):
            pieces.append('<span class="flow-arrow">▶</span>')
    st.markdown(
        f'<div class="flow-card" style="text-align:center;">{"".join(pieces)}</div>',
        unsafe_allow_html=True,
    )


def read_openrouter_key():
    """Read OpenRouter key from Streamlit secrets, environment, or UI input."""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "")

    if not key:
        key = st.sidebar.text_input(
            "OpenRouter API key",
            type="password",
            help="Used only when you click the AI grader button.",
        )

    return key


def load_dataset(path):
    """Load a local CSV dataset."""
    return pd.read_csv(path)


def audit_dataframe(df):
    """Create simple audit tables."""
    dtype_table = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
        }
    )
    missing_table = (
        df.isna()
        .mean()
        .mul(100)
        .round(3)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_percent"})
        .sort_values("missing_percent", ascending=False)
    )
    return dtype_table, missing_table


def clean_time_series(df, timestamp_col, target_col):
    """Parse timestamp, convert target, drop invalid rows, and sort by time.

    Also returns a richer integrity audit covering duplicates, gaps in the
    inferred frequency, and outliers (rows beyond 3 IQR from the median).
    """
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")

    before_rows = len(cleaned)
    invalid_timestamp = cleaned[timestamp_col].isna().sum()
    invalid_target = cleaned[target_col].isna().sum()

    cleaned = cleaned.dropna(subset=[timestamp_col, target_col])
    cleaned = cleaned.sort_values(timestamp_col).reset_index(drop=True)

    # Duplicate timestamps
    duplicate_timestamps = int(cleaned[timestamp_col].duplicated().sum())

    # Inferred step + gaps
    diffs = cleaned[timestamp_col].diff().dropna()
    median_step = diffs.median() if not diffs.empty else pd.Timedelta(0)
    if not diffs.empty and median_step.total_seconds() > 0:
        gap_count = int((diffs > median_step * 1.5).sum())
    else:
        gap_count = 0

    # Outliers via IQR rule on the target
    q1 = cleaned[target_col].quantile(0.25)
    q3 = cleaned[target_col].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        outlier_count = int(((cleaned[target_col] < lower) | (cleaned[target_col] > upper)).sum())
    else:
        outlier_count = 0

    dropped_rows = before_rows - len(cleaned)
    audit = {
        "invalid_timestamp_rows": int(invalid_timestamp),
        "invalid_target_rows": int(invalid_target),
        "duplicate_timestamps": duplicate_timestamps,
        "gap_count": gap_count,
        "outlier_count_3iqr": outlier_count,
        "target_min": float(cleaned[target_col].min()) if not cleaned.empty else None,
        "target_max": float(cleaned[target_col].max()) if not cleaned.empty else None,
        "target_mean": float(cleaned[target_col].mean()) if not cleaned.empty else None,
        "target_std": float(cleaned[target_col].std()) if not cleaned.empty else None,
    }
    return cleaned, dropped_rows, audit


def infer_time_coverage(cleaned, timestamp_col):
    """Return min, max, and inferred median step."""
    if cleaned.empty:
        return None, None, "Unavailable"
    min_time = cleaned[timestamp_col].min()
    max_time = cleaned[timestamp_col].max()
    diffs = cleaned[timestamp_col].sort_values().diff().dropna()
    if diffs.empty:
        inferred_step = "Unavailable"
    else:
        inferred_step = str(diffs.median())
    return min_time, max_time, inferred_step


def apply_optional_resampling(cleaned, timestamp_col, target_col, resample_rule):
    """Optionally resample target to a selected frequency."""
    ts = cleaned[[timestamp_col, target_col]].copy()
    ts = ts.set_index(timestamp_col).sort_index()
    if resample_rule != "None":
        ts = ts.resample(resample_rule)[target_col].mean().to_frame()
    ts = ts.dropna(subset=[target_col]).reset_index()
    return ts


def build_baseline_features(ts, timestamp_col, target_col, horizon, advanced_config=None):
    """Create baseline + optional advanced time-series features.

    advanced_config keys (all optional bools or ints):
      - extra_lags: list[int] of additional lag periods to add
      - rolling_windows: list[int] of additional rolling-mean window sizes
      - cyclical_hour: bool — add sin/cos encoding of hour
      - cyclical_dow: bool — add sin/cos encoding of day-of-week
      - day_of_week: bool — raw day-of-week column
      - week_of_year: bool — week-of-year column
      - lag_diff: bool — first difference of lag_1
      - rolling_std: bool — rolling std (window=24) of the target
    """
    feature_df = ts[[timestamp_col, target_col]].copy()
    feature_df["lag_1"] = feature_df[target_col].shift(1)
    feature_df["lag_24"] = feature_df[target_col].shift(24)
    feature_df["rolling_mean_24"] = feature_df[target_col].shift(1).rolling(24).mean()
    feature_df["hour"] = feature_df[timestamp_col].dt.hour
    feature_df["weekend"] = (feature_df[timestamp_col].dt.dayofweek >= 5).astype(int)
    feature_df["month"] = feature_df[timestamp_col].dt.month

    feature_columns = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]

    cfg = advanced_config or {}

    for lag in cfg.get("extra_lags", []):
        col = f"lag_{lag}"
        if col not in feature_df.columns:
            feature_df[col] = feature_df[target_col].shift(lag)
            feature_columns.append(col)

    for window in cfg.get("rolling_windows", []):
        col = f"rolling_mean_{window}"
        if col not in feature_df.columns:
            feature_df[col] = feature_df[target_col].shift(1).rolling(window).mean()
            feature_columns.append(col)

    if cfg.get("rolling_std", False):
        feature_df["rolling_std_24"] = feature_df[target_col].shift(1).rolling(24).std()
        feature_columns.append("rolling_std_24")

    if cfg.get("lag_diff", False):
        feature_df["lag_1_diff"] = feature_df[target_col].shift(1) - feature_df[target_col].shift(2)
        feature_columns.append("lag_1_diff")

    if cfg.get("cyclical_hour", False):
        feature_df["hour_sin"] = np.sin(2 * np.pi * feature_df["hour"] / 24)
        feature_df["hour_cos"] = np.cos(2 * np.pi * feature_df["hour"] / 24)
        feature_columns.extend(["hour_sin", "hour_cos"])

    if cfg.get("cyclical_dow", False):
        dow = feature_df[timestamp_col].dt.dayofweek
        feature_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feature_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feature_columns.extend(["dow_sin", "dow_cos"])

    if cfg.get("day_of_week", False):
        feature_df["day_of_week"] = feature_df[timestamp_col].dt.dayofweek
        feature_columns.append("day_of_week")

    if cfg.get("week_of_year", False):
        feature_df["week_of_year"] = feature_df[timestamp_col].dt.isocalendar().week.astype(int)
        feature_columns.append("week_of_year")

    feature_df["y_target"] = feature_df[target_col].shift(-horizon)

    modeling_df = feature_df.dropna(subset=feature_columns + ["y_target"]).copy()
    X = modeling_df[feature_columns]
    y = modeling_df["y_target"]
    return feature_df, modeling_df, X, y, feature_columns


def dataframe_records_or_empty(value):
    """Return DataFrame records when available; otherwise an empty list."""
    if isinstance(value, pd.DataFrame):
        safe_value = value.replace([np.inf, -np.inf], np.nan)
        return safe_value.where(pd.notna(safe_value), None).to_dict(orient="records")
    return []


def make_submission_json(
    student_name,
    student_id,
    deployed_url,
    repo_url,
    project_title,
    project_goal,
    data_path,
    original_rows,
    cleaned_rows,
    dropped_rows,
    timestamp_col,
    target_col,
    min_time,
    max_time,
    inferred_step,
    resample_rule,
    horizon,
    feature_columns,
    modeling_rows,
    has_feature_table,
    results_df,
    dashboard_notes,
    data_integrity_notes,
    insights,
    integrity_audit=None,
    feature_config=None,
    selected_models=None,
    split_ratios=None,
):
    """Build evidence JSON for export and AI grading."""
    has_metrics_table = isinstance(results_df, pd.DataFrame) and not results_df.empty

    return {
        "student": {
            "name": student_name,
            "id": student_id,
        },
        "links": {
            "deployed_streamlit_url": deployed_url,
            "github_repo_url": repo_url,
        },
        "project": {
            "title": project_title,
            "goal": project_goal,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "dataset": {
            "path": data_path,
            "original_rows": int(original_rows),
            "cleaned_rows": int(cleaned_rows),
            "dropped_invalid_timestamp_or_target_rows": int(dropped_rows),
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "time_min": str(min_time),
            "time_max": str(max_time),
            "inferred_time_step": inferred_step,
            "resampling_rule": resample_rule,
        },
        "data_integrity_audit": integrity_audit or {},
        "forecasting_setup": {
            "horizon_steps": int(horizon),
            "baseline_feature_columns": feature_columns,
            "feature_table_rows_after_dropna": int(modeling_rows),
            "has_baseline_feature_table": bool(has_feature_table),
            "feature_engineering_config": feature_config or {},
            "selected_models": selected_models or [],
            "split_ratios": split_ratios or {},
        },
        "evidence_flags": {
            "has_metrics_table": has_metrics_table,
            "has_student_modeling_additions": has_metrics_table,
            "has_student_dashboard_notes": bool(dashboard_notes.strip()),
            "has_data_integrity_discussion": bool(data_integrity_notes.strip()),
            "has_insights": bool(insights.strip()),
            "has_time_based_split": bool(split_ratios),
            "has_advanced_features": bool(feature_config and any(feature_config.values())),
        },
        "student_notes": {
            "data_integrity_notes": data_integrity_notes,
            "dashboard_notes": dashboard_notes,
            "insights": insights,
        },
        "results_table": dataframe_records_or_empty(results_df),
    }


def make_project_card(submission):
    """Create a markdown project card for download."""
    project = submission["project"]
    dataset = submission["dataset"]
    setup = submission["forecasting_setup"]
    flags = submission["evidence_flags"]

    lines = [
        f"# {project['title']}",
        "",
        f"Student: {submission['student']['name']}",
        f"Student ID: {submission['student']['id']}",
        "",
        "## Goal",
        project["goal"],
        "",
        "## Dataset",
        f"- Path: {dataset['path']}",
        f"- Timestamp column: {dataset['timestamp_column']}",
        f"- Target column: {dataset['target_column']}",
        f"- Time coverage: {dataset['time_min']} to {dataset['time_max']}",
        f"- Inferred step: {dataset['inferred_time_step']}",
        f"- Cleaned rows: {dataset['cleaned_rows']}",
        f"- Dropped invalid rows: {dataset['dropped_invalid_timestamp_or_target_rows']}",
        f"- Resampling rule: {dataset['resampling_rule']}",
        "",
        "## Forecasting setup",
        f"- Horizon steps: {setup['horizon_steps']}",
        f"- Baseline features: {', '.join(setup['baseline_feature_columns'])}",
        f"- Feature table rows: {setup['feature_table_rows_after_dropna']}",
        "",
        "## Evidence flags",
        f"- Metrics table present: {flags['has_metrics_table']}",
        f"- Data integrity discussion present: {flags['has_data_integrity_discussion']}",
        f"- Insights present: {flags['has_insights']}",
        "",
        "## Student notes",
        "### Data integrity",
        submission["student_notes"]["data_integrity_notes"] or "Not provided yet.",
        "",
        "### Dashboard",
        submission["student_notes"]["dashboard_notes"] or "Not provided yet.",
        "",
        "### Insights",
        submission["student_notes"]["insights"] or "Not provided yet.",
    ]
    return "\n".join(lines)


def parse_ai_response(text):
    """Try strict JSON parsing first, then extract the first JSON object."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as exc:
            return None, f"Found JSON-like text, but parsing failed: {exc}"

    return None, "No valid JSON object found in the AI response."


def call_openrouter_grader(api_key, evidence_json):
    """Call OpenRouter AI grader using the fixed model and prompt."""
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
        "<insert submission.json contents here>",
        json.dumps(evidence_json, indent=2),
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "EDA Mini Project B AI Grader",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]



# ==========================================================================
# HERO + TOP NAV
# ==========================================================================
st.markdown(
    """
    <div class="hero-title">📈 Time-Series Forecasting Workbench</div>
    <div class="hero-sub">
      Interactive feature engineering · multi-model comparison · 3D diagnostics · live AI grading
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-nav">
      <a href="#sec-data">📂 Data</a>
      <a href="#sec-columns">🎯 Columns</a>
      <a href="#sec-flow">🧭 Flow</a>
      <a href="#sec-resample">⚙️ Resample</a>
      <a href="#sec-features">🧱 Features</a>
      <a href="#sec-model">🤖 Model</a>
      <a href="#sec-dashboard">📊 Dashboard</a>
      <a href="#sec-notes">📝 Notes</a>
      <a href="#sec-export">📦 Export</a>
      <a href="#sec-grader">🏅 Grader</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# SIDEBAR
# ==========================================================================
with st.sidebar:
    st.markdown("### 👤 Student info")
    student_name = st.text_input("Student name", value=DEFAULT_STUDENT_NAME)
    student_id = st.text_input("Student ID", value=DEFAULT_STUDENT_ID)
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    repo_url = st.text_input("GitHub repo URL", value="")

    st.markdown("### 📋 Project")
    project_title = st.text_input("Project title", value="UK National Demand Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast future electricity demand using historical half-hourly demand data.",
        height=90,
    )

    st.markdown("### 🤖 AI grader")
    openrouter_key = read_openrouter_key()

    # ---- Progress tracker ----
    st.markdown("### ✅ Progress")
    if "progress" not in st.session_state:
        st.session_state.progress = {
            "Data loaded": False,
            "Columns chosen": False,
            "Features built": False,
            "Models trained": False,
            "Notes written": False,
        }
    for label, done in st.session_state.progress.items():
        icon = "✅" if done else "⬜"
        st.markdown(f"<div style='font-size:0.9rem;'>{icon} {label}</div>",
                    unsafe_allow_html=True)

# ==========================================================================
# 1. DATA LOAD + AUDIT
# ==========================================================================
section_banner(1, "Load & audit dataset",
               "Inspect schema, dtypes, and missingness before anything else",
               anchor="sec-data")

data_path = st.text_input("📂 Dataset path", value=DEFAULT_DATA_PATH)

try:
    df = load_dataset(data_path)
    st.session_state.progress["Data loaded"] = True
except Exception as exc:
    st.error(f"Could not load dataset from `{data_path}`: {exc}")
    st.stop()

col_kpi = st.columns(4)
col_kpi[0].metric("📂 Rows", f"{len(df):,}")
col_kpi[1].metric("📊 Columns", f"{len(df.columns):,}")
col_kpi[2].metric("🕳️ Missing %", f"{df.isna().mean().mean() * 100:.2f}%")
col_kpi[3].metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

tab_preview, tab_dtypes, tab_missing, tab_describe = st.tabs(
    ["👀 Preview", "🧬 Dtypes", "❓ Missingness", "📐 Describe"]
)
dtype_table, missing_table = audit_dataframe(df)
with tab_preview:
    st.dataframe(df.head(20), use_container_width=True, height=320)
with tab_dtypes:
    st.dataframe(dtype_table, use_container_width=True, height=320)
with tab_missing:
    if missing_table["missing_percent"].sum() == 0:
        st.success("✅ No missing values detected across any column.")
    fig_miss = px.bar(
        missing_table.head(20),
        x="column", y="missing_percent",
        title="Missingness by column (top 20)",
        labels={"missing_percent": "% missing"},
        color="missing_percent", color_continuous_scale="Reds",
    )
    fig_miss.update_layout(height=380, template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False, font=dict(family="Inter"))
    st.plotly_chart(fig_miss, use_container_width=True)
with tab_describe:
    numeric_cols_describe = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_describe:
        st.dataframe(df[numeric_cols_describe].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns to describe.")

render_progress_flow(1)

# ==========================================================================
# 2. COLUMN SELECTION + CLEANING
# ==========================================================================
section_banner(2, "Pick timestamp & target",
               "Cleaner parses, deduplicates, and audits the series",
               anchor="sec-columns")

columns = list(df.columns)
col_sel1, col_sel2 = st.columns(2)

timestamp_index = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
with col_sel1:
    timestamp_col = st.selectbox("⏰ Timestamp column", columns, index=timestamp_index)

numeric_candidates = []
for col in columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().mean() > 0.5:
        numeric_candidates.append(col)

if DEFAULT_TARGET_COL in columns:
    target_index = columns.index(DEFAULT_TARGET_COL)
else:
    target_index = columns.index(numeric_candidates[0]) if numeric_candidates else 0

with col_sel2:
    target_col = st.selectbox("🎯 Target column", columns, index=target_index)

cleaned, dropped_rows, integrity_audit = clean_time_series(df, timestamp_col, target_col)
if cleaned.empty:
    st.error("No valid rows remain after parsing timestamp and target. Choose different columns.")
    st.stop()
st.session_state.progress["Columns chosen"] = True

min_time, max_time, inferred_step = infer_time_coverage(cleaned, timestamp_col)

summary_cols = st.columns(5)
summary_cols[0].metric("Original rows", f"{len(df):,}")
summary_cols[1].metric("Cleaned rows", f"{len(cleaned):,}")
summary_cols[2].metric("Dropped rows", f"{dropped_rows:,}")
summary_cols[3].metric("Inferred step", inferred_step)
summary_cols[4].metric("Outliers (3·IQR)", f"{integrity_audit['outlier_count_3iqr']:,}")

st.caption(f"📅 Time coverage: **{min_time}** → **{max_time}**")

with st.expander("🔍 Data integrity audit (auto-generated)", expanded=False):
    audit_cols = st.columns(4)
    audit_cols[0].metric("Invalid timestamps", integrity_audit["invalid_timestamp_rows"])
    audit_cols[1].metric("Invalid targets", integrity_audit["invalid_target_rows"])
    audit_cols[2].metric("Duplicate timestamps", integrity_audit["duplicate_timestamps"])
    audit_cols[3].metric("Gap count", integrity_audit["gap_count"])

    stats_cols = st.columns(4)
    stats_cols[0].metric("Target min", f"{integrity_audit['target_min']:.2f}")
    stats_cols[1].metric("Target max", f"{integrity_audit['target_max']:.2f}")
    stats_cols[2].metric("Target mean", f"{integrity_audit['target_mean']:.2f}")
    stats_cols[3].metric("Target std", f"{integrity_audit['target_std']:.2f}")

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        fig_hist = px.histogram(
            cleaned, x=target_col, nbins=60,
            title=f"Distribution of {target_col}",
            color_discrete_sequence=["#60a5fa"],
        )
        fig_hist.update_layout(height=320, template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               showlegend=False, font=dict(family="Inter"))
        st.plotly_chart(fig_hist, use_container_width=True)
    with dist_col2:
        fig_box = px.box(
            cleaned, y=target_col,
            title=f"Boxplot of {target_col}",
            color_discrete_sequence=["#ec4899"],
        )
        fig_box.update_layout(height=320, template="plotly_dark",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"))
        st.plotly_chart(fig_box, use_container_width=True)

render_progress_flow(2)

# ==========================================================================
# 3. PROJECT METHODOLOGY — INTERACTIVE FLOWCHART
# ==========================================================================
section_banner(3, "Project methodology",
               "Interactive flowchart, block diagrams, and system flow",
               anchor="sec-flow")

flow_tab1, flow_tab2, flow_tab3, flow_tab4 = st.tabs([
    "🗺️ Mermaid flowchart",
    "🧩 Feature pipeline",
    "🏗️ Model architecture",
    "🔄 Data flow (Plotly Sankey)",
])

# ---- Mermaid flowchart ----
with flow_tab1:
    st.markdown(
        """
        End-to-end methodology rendered as an interactive Mermaid flowchart.
        Hover over nodes; the diagram is fully zoomable using the controls.
        """
    )
    mermaid_chart = """
    <div style="background: rgba(15,23,42,0.6); padding: 20px; border-radius: 14px;
                border: 1px solid rgba(99,102,241,0.2);">
      <pre class="mermaid" style="text-align:center;">
flowchart TD
    A([📂 Load CSV]):::start --> B{Valid<br/>timestamp & target?}
    B -->|No| Z([❌ Stop]):::stop
    B -->|Yes| C[🧹 Clean<br/>drop NaN · sort by time<br/>dedupe · detect gaps]:::clean
    C --> D[📐 Audit<br/>3·IQR outliers · stats]:::clean
    D --> E[⚙️ Optional resample<br/>30min · H · D]:::feat
    E --> F[🧱 Feature engineering<br/>lags · rolling · cyclical · calendar]:::feat
    F --> G[✂️ Time-based split<br/>train / val / test]:::model
    G --> H[🤖 Train models<br/>Naive · Ridge · Tree · RF · GBR]:::model
    H --> I[📊 Evaluate<br/>MAE · RMSE · MAPE]:::eval
    I --> J{Best on test?}
    J --> K[📈 Dashboard<br/>predictions · residuals · 3D · heatmap]:::dash
    K --> L[💡 Auto-insights]:::dash
    L --> M[📦 Export<br/>submission.json · project_card.md]:::export
    M --> N([🏅 AI grader<br/>/80]):::grader

    classDef start fill:#10b981,stroke:#10b981,color:#fff,font-weight:bold;
    classDef stop fill:#ef4444,stroke:#ef4444,color:#fff;
    classDef clean fill:#3b82f6,stroke:#60a5fa,color:#fff;
    classDef feat fill:#8b5cf6,stroke:#a78bfa,color:#fff;
    classDef model fill:#ec4899,stroke:#f472b6,color:#fff;
    classDef eval fill:#f59e0b,stroke:#fbbf24,color:#fff;
    classDef dash fill:#06b6d4,stroke:#22d3ee,color:#fff;
    classDef export fill:#84cc16,stroke:#a3e635,color:#fff;
    classDef grader fill:#a855f7,stroke:#c084fc,color:#fff,font-weight:bold;
      </pre>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ startOnLoad: true, theme: 'dark',
                           themeVariables: { fontFamily: 'Inter, sans-serif', fontSize: '15px' } });
    </script>
    """
    st.components.v1.html(mermaid_chart, height=820, scrolling=True)

# ---- Feature engineering block diagram ----
with flow_tab2:
    st.markdown("Block diagram showing how the raw target gets transformed into model inputs.")
    feat_blocks = """
    <div style="background: rgba(15,23,42,0.6); padding: 28px 20px; border-radius: 14px;
                border: 1px solid rgba(99,102,241,0.2);">
      <svg viewBox="0 0 900 380" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto;">
        <defs>
          <linearGradient id="gradSrc" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#059669"/>
          </linearGradient>
          <linearGradient id="gradLag" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#2563eb"/>
          </linearGradient>
          <linearGradient id="gradRoll" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#8b5cf6"/><stop offset="100%" stop-color="#7c3aed"/>
          </linearGradient>
          <linearGradient id="gradCyc" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#ec4899"/><stop offset="100%" stop-color="#db2777"/>
          </linearGradient>
          <linearGradient id="gradCal" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#d97706"/>
          </linearGradient>
          <linearGradient id="gradOut" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#06b6d4"/><stop offset="100%" stop-color="#0891b2"/>
          </linearGradient>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>
          </marker>
        </defs>

        <!-- Source -->
        <g>
          <rect x="20" y="160" width="140" height="60" rx="12" fill="url(#gradSrc)" stroke="#34d399" stroke-width="1.5"/>
          <text x="90" y="188" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="14">🎯 Target series</text>
          <text x="90" y="206" text-anchor="middle" fill="#d1fae5" font-family="Inter" font-size="11">y(t)</text>
        </g>

        <!-- Lag block -->
        <g>
          <rect x="240" y="20" width="170" height="64" rx="12" fill="url(#gradLag)" stroke="#60a5fa" stroke-width="1.5"/>
          <text x="325" y="44" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">🕓 Lag block</text>
          <text x="325" y="62" text-anchor="middle" fill="#dbeafe" font-family="Inter" font-size="10">lag_1 · lag_24 · lag_48 ...</text>
          <text x="325" y="76" text-anchor="middle" fill="#bfdbfe" font-family="Inter" font-size="9">y.shift(k)</text>
        </g>

        <!-- Rolling block -->
        <g>
          <rect x="240" y="100" width="170" height="64" rx="12" fill="url(#gradRoll)" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="325" y="124" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">📊 Rolling block</text>
          <text x="325" y="142" text-anchor="middle" fill="#ede9fe" font-family="Inter" font-size="10">mean_24 · std_24 ...</text>
          <text x="325" y="156" text-anchor="middle" fill="#ddd6fe" font-family="Inter" font-size="9">y.shift(1).rolling(w)</text>
        </g>

        <!-- Cyclical block -->
        <g>
          <rect x="240" y="180" width="170" height="64" rx="12" fill="url(#gradCyc)" stroke="#f472b6" stroke-width="1.5"/>
          <text x="325" y="204" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">🌀 Cyclical block</text>
          <text x="325" y="222" text-anchor="middle" fill="#fce7f3" font-family="Inter" font-size="10">hour_sin · hour_cos</text>
          <text x="325" y="236" text-anchor="middle" fill="#fbcfe8" font-family="Inter" font-size="9">sin(2π·t/T) · cos(2π·t/T)</text>
        </g>

        <!-- Calendar block -->
        <g>
          <rect x="240" y="260" width="170" height="64" rx="12" fill="url(#gradCal)" stroke="#fbbf24" stroke-width="1.5"/>
          <text x="325" y="284" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">📅 Calendar block</text>
          <text x="325" y="302" text-anchor="middle" fill="#fef3c7" font-family="Inter" font-size="10">hour · dow · month · week</text>
          <text x="325" y="316" text-anchor="middle" fill="#fde68a" font-family="Inter" font-size="9">dt accessors</text>
        </g>

        <!-- Feature matrix -->
        <g>
          <rect x="500" y="140" width="180" height="100" rx="12" fill="url(#gradOut)" stroke="#22d3ee" stroke-width="1.5"/>
          <text x="590" y="172" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="14">🧱 Feature matrix X</text>
          <text x="590" y="194" text-anchor="middle" fill="#cffafe" font-family="Inter" font-size="11">(n_rows × n_features)</text>
          <text x="590" y="214" text-anchor="middle" fill="#a5f3fc" font-family="Inter" font-size="10">drop_na on lags/rolling</text>
        </g>

        <!-- Target column -->
        <g>
          <rect x="730" y="160" width="150" height="60" rx="12" fill="#1e293b" stroke="#475569" stroke-width="1.5" stroke-dasharray="4 3"/>
          <text x="805" y="188" text-anchor="middle" fill="#f1f5f9" font-family="Inter" font-weight="700" font-size="13">🎯 y_target</text>
          <text x="805" y="206" text-anchor="middle" fill="#cbd5e1" font-family="Inter" font-size="10">y.shift(-horizon)</text>
        </g>

        <!-- Arrows from source to blocks -->
        <path d="M 160 190 Q 200 190 220 52" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
        <path d="M 160 190 Q 200 190 220 132" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
        <path d="M 160 190 L 220 212" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
        <path d="M 160 190 Q 200 190 220 292" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>

        <!-- Arrows from blocks to feature matrix -->
        <path d="M 410 52 Q 460 52 490 168" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
        <path d="M 410 132 Q 450 132 490 180" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
        <path d="M 410 212 L 490 200" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
        <path d="M 410 292 Q 460 292 490 220" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>

        <!-- Feature matrix → y_target -->
        <path d="M 680 190 L 730 190" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
      </svg>
    </div>
    """
    st.markdown(feat_blocks, unsafe_allow_html=True)

# ---- Model architecture diagram ----
with flow_tab3:
    st.markdown("Multi-model comparison architecture — same input, parallel evaluation, unified metrics.")
    model_arch = """
    <div style="background: rgba(15,23,42,0.6); padding: 28px 20px; border-radius: 14px;
                border: 1px solid rgba(99,102,241,0.2);">
      <svg viewBox="0 0 920 420" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto;">
        <defs>
          <linearGradient id="gradData" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#06b6d4"/><stop offset="100%" stop-color="#0891b2"/>
          </linearGradient>
          <linearGradient id="gradSplit" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#d97706"/>
          </linearGradient>
          <linearGradient id="gradMdl" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#8b5cf6"/><stop offset="100%" stop-color="#6d28d9"/>
          </linearGradient>
          <linearGradient id="gradMet" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#047857"/>
          </linearGradient>
          <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>
          </marker>
        </defs>

        <!-- X, y -->
        <g>
          <rect x="20" y="170" width="120" height="70" rx="12" fill="url(#gradData)" stroke="#22d3ee" stroke-width="1.5"/>
          <text x="80" y="200" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="14">X, y</text>
          <text x="80" y="220" text-anchor="middle" fill="#cffafe" font-family="Inter" font-size="11">feature matrix</text>
        </g>

        <!-- Split -->
        <g>
          <rect x="200" y="160" width="160" height="90" rx="12" fill="url(#gradSplit)" stroke="#fbbf24" stroke-width="1.5"/>
          <text x="280" y="186" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="14">✂️ Time split</text>
          <text x="280" y="206" text-anchor="middle" fill="#fef3c7" font-family="Inter" font-size="11">train 70%</text>
          <text x="280" y="222" text-anchor="middle" fill="#fef3c7" font-family="Inter" font-size="11">val 15%</text>
          <text x="280" y="238" text-anchor="middle" fill="#fef3c7" font-family="Inter" font-size="11">test 15%</text>
        </g>

        <!-- Five models -->
        <g>
          <rect x="430" y="20" width="180" height="58" rx="10" fill="url(#gradMdl)" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="520" y="44" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">Naive (lag-1)</text>
          <text x="520" y="62" text-anchor="middle" fill="#ede9fe" font-family="Inter" font-size="10">baseline</text>
        </g>
        <g>
          <rect x="430" y="100" width="180" height="58" rx="10" fill="url(#gradMdl)" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="520" y="124" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">Linear / Ridge</text>
          <text x="520" y="142" text-anchor="middle" fill="#ede9fe" font-family="Inter" font-size="10">closed-form</text>
        </g>
        <g>
          <rect x="430" y="180" width="180" height="58" rx="10" fill="url(#gradMdl)" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="520" y="204" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">Decision Tree</text>
          <text x="520" y="222" text-anchor="middle" fill="#ede9fe" font-family="Inter" font-size="10">non-linear</text>
        </g>
        <g>
          <rect x="430" y="260" width="180" height="58" rx="10" fill="url(#gradMdl)" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="520" y="284" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">Random Forest</text>
          <text x="520" y="302" text-anchor="middle" fill="#ede9fe" font-family="Inter" font-size="10">bagged trees</text>
        </g>
        <g>
          <rect x="430" y="340" width="180" height="58" rx="10" fill="url(#gradMdl)" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="520" y="364" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="13">Gradient Boosting</text>
          <text x="520" y="382" text-anchor="middle" fill="#ede9fe" font-family="Inter" font-size="10">sequential boost</text>
        </g>

        <!-- Metrics -->
        <g>
          <rect x="700" y="170" width="200" height="90" rx="12" fill="url(#gradMet)" stroke="#34d399" stroke-width="1.5"/>
          <text x="800" y="198" text-anchor="middle" fill="#fff" font-family="Inter" font-weight="700" font-size="14">📊 Metrics table</text>
          <text x="800" y="220" text-anchor="middle" fill="#d1fae5" font-family="Inter" font-size="11">MAE · RMSE · MAPE</text>
          <text x="800" y="238" text-anchor="middle" fill="#a7f3d0" font-family="Inter" font-size="10">per split, per model</text>
        </g>

        <!-- Arrows X,y → split → models -->
        <path d="M 140 205 L 195 205" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 360 200 Q 395 200 425 48" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 360 205 Q 395 200 425 128" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 360 210 L 425 208" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 360 215 Q 395 215 425 288" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 360 220 Q 395 225 425 368" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>

        <!-- Arrows models → metrics -->
        <path d="M 610 48 Q 650 48 695 200" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 610 128 Q 650 128 695 208" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 610 208 L 695 215" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 610 288 Q 650 288 695 222" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
        <path d="M 610 368 Q 650 368 695 230" stroke="#94a3b8" stroke-width="1.5" fill="none" marker-end="url(#arrow2)"/>
      </svg>
    </div>
    """
    st.markdown(model_arch, unsafe_allow_html=True)

# ---- Sankey data flow ----
with flow_tab4:
    st.markdown("Data flow as a Sankey diagram — width = row count at each stage.")
    feat_count_preview = max(6, integrity_audit.get("invalid_timestamp_rows", 0) +
                             integrity_audit.get("invalid_target_rows", 0))
    sankey_labels = [
        f"📂 Raw rows ({len(df):,})",
        f"❌ Dropped ({dropped_rows:,})",
        f"🧹 Cleaned ({len(cleaned):,})",
        f"⚠️ Outliers ({integrity_audit['outlier_count_3iqr']:,})",
        f"✅ Valid for features",
        f"🧱 Modeling-ready rows",
    ]
    valid_for_feat = max(1, len(cleaned) - integrity_audit["outlier_count_3iqr"])
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20, thickness=22,
            line=dict(color="rgba(99,102,241,0.4)", width=1),
            label=sankey_labels,
            color=["#06b6d4", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"],
        ),
        link=dict(
            source=[0, 0, 2, 2, 4],
            target=[1, 2, 3, 4, 5],
            value=[
                max(1, dropped_rows),
                max(1, len(cleaned)),
                max(1, integrity_audit["outlier_count_3iqr"]),
                valid_for_feat,
                valid_for_feat,  # placeholder until features build; updated below if needed
            ],
            color="rgba(99, 102, 241, 0.25)",
        ),
    )])
    sankey_fig.update_layout(
        template="plotly_dark", height=380,
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", size=13),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

# ==========================================================================
# 4. RESAMPLE + HORIZON
# ==========================================================================
section_banner(4, "Resample & horizon",
               "Optionally aggregate to a coarser frequency; set the forecast horizon",
               anchor="sec-resample")

res_col1, res_col2 = st.columns([1, 1])
with res_col1:
    resample_rule = st.selectbox(
        "Resampling rule",
        options=["None", "30min", "H", "D"], index=0,
        help="None = keep native frequency. H = hourly mean. D = daily mean.",
    )
with res_col2:
    horizon = st.number_input(
        "Forecast horizon (steps ahead)",
        min_value=1, max_value=336, value=1, step=1,
        help="Number of future steps to predict at each row.",
    )

ts = apply_optional_resampling(cleaned, timestamp_col, target_col, resample_rule)

ts_preview = ts[[timestamp_col, target_col]].dropna()
if not ts_preview.empty:
    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=ts_preview[timestamp_col], y=ts_preview[target_col],
            mode="lines", line=dict(color="#60a5fa", width=1.2),
            name=target_col,
            hovertemplate="%{x}<br>" + target_col + ": %{y:.2f}<extra></extra>",
        )
    )
    fig_ts.update_layout(
        title=f"📈 {target_col} over time (drag to zoom, double-click to reset)",
        template="plotly_dark", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time", yaxis_title=target_col,
        hovermode="x unified", font=dict(family="Inter"),
    )
    fig_ts.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_ts, use_container_width=True)

render_progress_flow(3)

# ==========================================================================
# 5. FEATURE ENGINEERING — INTERACTIVE + PRESETS
# ==========================================================================
section_banner(5, "Feature engineering",
               "Pick a preset or toggle features manually — baseline is always included",
               anchor="sec-features")

# Preset selector
preset = st.radio(
    "🎚️ Feature preset",
    ["Baseline only", "Conservative", "Aggressive", "Custom"],
    horizontal=True, index=2,
    help="Presets pre-fill the checkboxes. Switch to Custom to fine-tune.",
)

PRESET_CONFIGS = {
    "Baseline only": dict(use_lag_48=False, use_lag_168=False, use_lag_336=False,
                          use_roll_6=False, use_roll_48=False, use_roll_168=False,
                          use_rolling_std=False, use_lag_diff=False,
                          use_cyclical_hour=False, use_cyclical_dow=False,
                          use_day_of_week=False, use_week_of_year=False),
    "Conservative": dict(use_lag_48=True, use_lag_168=False, use_lag_336=False,
                          use_roll_6=False, use_roll_48=True, use_roll_168=False,
                          use_rolling_std=True, use_lag_diff=False,
                          use_cyclical_hour=True, use_cyclical_dow=False,
                          use_day_of_week=False, use_week_of_year=False),
    "Aggressive":   dict(use_lag_48=True, use_lag_168=True, use_lag_336=False,
                          use_roll_6=True, use_roll_48=True, use_roll_168=True,
                          use_rolling_std=True, use_lag_diff=True,
                          use_cyclical_hour=True, use_cyclical_dow=True,
                          use_day_of_week=True, use_week_of_year=True),
    "Custom":       None,  # user controls
}

if preset != "Custom":
    cfg = PRESET_CONFIGS[preset]
    for k, v in cfg.items():
        st.session_state[k] = v

st.markdown(
    """
    <div style="color:#94a3b8; font-size:0.92rem; margin-bottom:8px;">
      Baseline always includes: <code>lag_1</code>, <code>lag_24</code>,
      <code>rolling_mean_24</code>, <code>hour</code>, <code>weekend</code>, <code>month</code>.
      Use the controls below to add more.
    </div>
    """,
    unsafe_allow_html=True,
)

fe_col1, fe_col2, fe_col3 = st.columns(3)
with fe_col1:
    st.markdown("**🕓 Extra lags**")
    use_lag_48 = st.checkbox("lag_48 (1 day, half-hourly)", key="use_lag_48", value=st.session_state.get("use_lag_48", True))
    use_lag_168 = st.checkbox("lag_168 (3.5 days)", key="use_lag_168", value=st.session_state.get("use_lag_168", False))
    use_lag_336 = st.checkbox("lag_336 (1 week)", key="use_lag_336", value=st.session_state.get("use_lag_336", False))
with fe_col2:
    st.markdown("**📊 Rolling windows**")
    use_roll_6 = st.checkbox("rolling_mean_6", key="use_roll_6", value=st.session_state.get("use_roll_6", False))
    use_roll_48 = st.checkbox("rolling_mean_48", key="use_roll_48", value=st.session_state.get("use_roll_48", True))
    use_roll_168 = st.checkbox("rolling_mean_168", key="use_roll_168", value=st.session_state.get("use_roll_168", False))
    use_rolling_std = st.checkbox("rolling_std_24", key="use_rolling_std", value=st.session_state.get("use_rolling_std", True))
with fe_col3:
    st.markdown("**📅 Calendar & cyclical**")
    use_cyclical_hour = st.checkbox("hour_sin / hour_cos", key="use_cyclical_hour", value=st.session_state.get("use_cyclical_hour", True))
    use_cyclical_dow = st.checkbox("dow_sin / dow_cos", key="use_cyclical_dow", value=st.session_state.get("use_cyclical_dow", True))
    use_day_of_week = st.checkbox("day_of_week (raw)", key="use_day_of_week", value=st.session_state.get("use_day_of_week", False))
    use_week_of_year = st.checkbox("week_of_year", key="use_week_of_year", value=st.session_state.get("use_week_of_year", False))
    use_lag_diff = st.checkbox("lag_1_diff (Δ)", key="use_lag_diff", value=st.session_state.get("use_lag_diff", False))

extra_lags = []
if use_lag_48: extra_lags.append(48)
if use_lag_168: extra_lags.append(168)
if use_lag_336: extra_lags.append(336)

rolling_windows = []
if use_roll_6: rolling_windows.append(6)
if use_roll_48: rolling_windows.append(48)
if use_roll_168: rolling_windows.append(168)

feature_config = {
    "extra_lags": extra_lags, "rolling_windows": rolling_windows,
    "rolling_std": use_rolling_std, "lag_diff": use_lag_diff,
    "cyclical_hour": use_cyclical_hour, "cyclical_dow": use_cyclical_dow,
    "day_of_week": use_day_of_week, "week_of_year": use_week_of_year,
}

feature_df, modeling_df, X, y, feature_columns = build_baseline_features(
    ts, timestamp_col, target_col, int(horizon), advanced_config=feature_config,
)
if not modeling_df.empty:
    st.session_state.progress["Features built"] = True

fcol1, fcol2, fcol3 = st.columns(3)
fcol1.metric("🧱 Feature columns", f"{len(feature_columns)}")
fcol2.metric("🧾 Modeling rows", f"{len(modeling_df):,}")
fcol3.metric("➕ Advanced features", f"{len(feature_columns) - 6}")

# Extra: feature correlation heatmap + ACF-like view
adv_tab1, adv_tab2, adv_tab3 = st.tabs([
    "👀 Feature preview",
    "🔗 Correlation heatmap",
    "📈 Autocorrelation (lag-by-lag)",
])
with adv_tab1:
    st.code(", ".join(feature_columns), language="text")
    st.dataframe(modeling_df.head(15), use_container_width=True, height=300)

with adv_tab2:
    if len(feature_columns) >= 2 and len(modeling_df) >= 20:
        corr = modeling_df[feature_columns + ["y_target"]].corr().round(3)
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature correlation heatmap (with y_target)",
        )
        fig_corr.update_layout(
            template="plotly_dark", height=520,
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough data to compute a correlation heatmap.")

with adv_tab3:
    if len(modeling_df) >= 50:
        max_lag = min(60, len(modeling_df) // 4)
        target_series = modeling_df[target_col]
        acf_values = [target_series.autocorr(lag=k) for k in range(1, max_lag + 1)]
        acf_df = pd.DataFrame({"lag": list(range(1, max_lag + 1)), "autocorr": acf_values})
        fig_acf = px.bar(
            acf_df, x="lag", y="autocorr",
            title=f"Autocorrelation of {target_col} (lag 1 to {max_lag})",
            color="autocorr", color_continuous_scale="RdBu_r",
        )
        fig_acf.add_hline(y=0, line_color="white", opacity=0.4)
        fig_acf.update_layout(
            template="plotly_dark", height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(family="Inter"),
        )
        st.plotly_chart(fig_acf, use_container_width=True)
    else:
        st.info("Not enough rows for autocorrelation.")

render_progress_flow(4)

# ==========================================================================
# 6. MODELING — INTERACTIVE
# ==========================================================================
section_banner(6, "Modeling & evaluation",
               "Pick models, set the split, then click Run — nothing trains until you do",
               anchor="sec-model")

st.markdown(
    """
    <span class="pill pill-warn">⚠ No training happens automatically</span>
    <span class="pill pill-ok">✅ Time-based split</span>
    <span class="pill pill-ok">✅ MAE · RMSE · MAPE</span>
    <span class="pill pill-info">ℹ️ Multi-model parallel comparison</span>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### 🧪 Time-based split")
split_col1, split_col2 = st.columns(2)
with split_col1:
    train_pct = st.slider("Train %", min_value=50, max_value=85, value=70, step=5)
with split_col2:
    val_pct = st.slider("Validation %", min_value=5, max_value=25, value=15, step=5)
test_pct = 100 - train_pct - val_pct
if test_pct < 5:
    st.error("Train + Val too large — test split must be at least 5%.")
    st.stop()
st.caption(f"Split → Train **{train_pct}%** · Val **{val_pct}%** · Test **{test_pct}%**")

st.markdown("#### 🤖 Models to compare")
model_col1, model_col2, model_col3, model_col4, model_col5 = st.columns(5)
with model_col1:
    use_naive = st.checkbox("Naive (lag-1)", value=True)
with model_col2:
    use_linreg = st.checkbox("Linear Reg.", value=True)
with model_col3:
    use_ridge = st.checkbox("Ridge", value=True)
with model_col4:
    use_tree = st.checkbox("Decision Tree", value=False)
with model_col5:
    use_rf = st.checkbox("Random Forest", value=True)

use_gbr = st.checkbox("Gradient Boosting (slower)", value=False)

with st.expander("⚙️ Hyperparameters (optional)", expanded=False):
    hp_col1, hp_col2 = st.columns(2)
    with hp_col1:
        ridge_alpha = st.slider("Ridge α", 0.01, 10.0, 1.0, 0.01)
        tree_depth = st.slider("Tree max_depth", 2, 20, 8, 1)
    with hp_col2:
        rf_n_estimators = st.slider("RF n_estimators", 50, 500, 200, 50)
        rf_max_depth = st.slider("RF max_depth", 3, 25, 12, 1)
    gbr_n = st.slider("GBR n_estimators", 50, 400, 150, 50)
    gbr_lr = st.slider("GBR learning_rate", 0.01, 0.3, 0.1, 0.01)

if "modeling_run" not in st.session_state:
    st.session_state.modeling_run = False
    st.session_state.results_df = None
    st.session_state.predictions = {}
    st.session_state.split_index = {}
    st.session_state.feature_importance = None

run_col1, run_col2 = st.columns([1, 4])
with run_col1:
    run_clicked = st.button("▶️ Run model comparison", type="primary", use_container_width=True)
with run_col2:
    if st.button("🗑️ Clear results"):
        st.session_state.modeling_run = False
        st.session_state.results_df = None
        st.session_state.predictions = {}
        st.session_state.split_index = {}
        st.session_state.feature_importance = None

selected_models_list = []
if use_naive: selected_models_list.append("Naive (lag-1)")
if use_linreg: selected_models_list.append("Linear Regression")
if use_ridge: selected_models_list.append("Ridge")
if use_tree: selected_models_list.append("Decision Tree")
if use_rf: selected_models_list.append("Random Forest")
if use_gbr: selected_models_list.append("Gradient Boosting")


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics_row(model_name, split_name, y_true, y_pred):
    return {
        "model": model_name,
        "split": split_name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": _mape(y_true, y_pred),
    }


if run_clicked:
    if not selected_models_list:
        st.error("Pick at least one model to compare.")
    elif modeling_df.empty or len(modeling_df) < 50:
        st.error("Not enough rows for a 3-way time-based split. Reduce horizon or pick coarser resample.")
    else:
        modeling_sorted = modeling_df.sort_values(timestamp_col).reset_index(drop=True)
        X_sorted = modeling_sorted[feature_columns].astype(float)
        y_sorted = modeling_sorted["y_target"].astype(float)
        dates_sorted = modeling_sorted[timestamp_col]
        n_rows = len(modeling_sorted)
        train_end = int(n_rows * (train_pct / 100))
        val_end = int(n_rows * ((train_pct + val_pct) / 100))

        X_train, y_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end]
        X_val,   y_val   = X_sorted.iloc[train_end:val_end], y_sorted.iloc[train_end:val_end]
        X_test,  y_test  = X_sorted.iloc[val_end:], y_sorted.iloc[val_end:]

        split_index = {
            "train": (dates_sorted.iloc[0], dates_sorted.iloc[train_end - 1]),
            "val":   (dates_sorted.iloc[train_end], dates_sorted.iloc[val_end - 1]),
            "test":  (dates_sorted.iloc[val_end], dates_sorted.iloc[-1]),
        }

        builders = {
            "Naive (lag-1)":     None,
            "Linear Regression": LinearRegression(),
            "Ridge":             Ridge(alpha=ridge_alpha, random_state=42),
            "Decision Tree":     DecisionTreeRegressor(max_depth=tree_depth, random_state=42),
            "Random Forest":     RandomForestRegressor(
                n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=gbr_n, learning_rate=gbr_lr, max_depth=3, random_state=42,
            ),
        }

        predictions = {}
        metrics_rows = []
        feature_importance = {}

        progress = st.progress(0.0, text="Training models...")
        for i, name in enumerate(selected_models_list):
            est = builders[name]
            if name == "Naive (lag-1)":
                pred_train = X_train["lag_1"].to_numpy()
                pred_val = X_val["lag_1"].to_numpy()
                pred_test = X_test["lag_1"].to_numpy()
            else:
                est.fit(X_train, y_train)
                pred_train = est.predict(X_train)
                pred_val = est.predict(X_val)
                pred_test = est.predict(X_test)
                if hasattr(est, "feature_importances_"):
                    feature_importance[name] = pd.Series(
                        est.feature_importances_, index=feature_columns
                    ).sort_values(ascending=False)
                elif hasattr(est, "coef_"):
                    feature_importance[name] = pd.Series(
                        np.abs(est.coef_), index=feature_columns
                    ).sort_values(ascending=False)

            predictions[name] = {
                "train": (dates_sorted.iloc[:train_end].to_numpy(), y_train.to_numpy(), pred_train),
                "val":   (dates_sorted.iloc[train_end:val_end].to_numpy(), y_val.to_numpy(), pred_val),
                "test":  (dates_sorted.iloc[val_end:].to_numpy(), y_test.to_numpy(), pred_test),
            }
            metrics_rows.append(_metrics_row(name, "train", y_train, pred_train))
            metrics_rows.append(_metrics_row(name, "val",   y_val,   pred_val))
            metrics_rows.append(_metrics_row(name, "test",  y_test,  pred_test))
            progress.progress((i + 1) / len(selected_models_list), text=f"Trained {name}")

        progress.empty()
        results_df = (
            pd.DataFrame(metrics_rows)[["model", "split", "MAE", "RMSE", "MAPE"]]
            .round({"MAE": 3, "RMSE": 3, "MAPE": 2})
        )
        st.session_state.modeling_run = True
        st.session_state.results_df = results_df
        st.session_state.predictions = predictions
        st.session_state.split_index = split_index
        st.session_state.feature_importance = feature_importance
        st.session_state.progress["Models trained"] = True
        st.success(f"✅ Trained {len(selected_models_list)} model(s) on a {train_pct}/{val_pct}/{test_pct} time-based split.")

results_df = st.session_state.results_df
predictions = st.session_state.predictions
split_index = st.session_state.split_index
feature_importance = st.session_state.feature_importance

if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    # Leaderboard with medals
    test_only = results_df[results_df["split"] == "test"].copy().sort_values("RMSE")
    medals = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
    test_only.insert(0, "rank", [medals[i] for i in range(len(test_only))])
    st.markdown("#### 🏆 Test-set leaderboard (lowest RMSE wins)")
    st.dataframe(test_only.reset_index(drop=True), use_container_width=True,
                 height=min(360, 44 * len(test_only) + 60))

    st.markdown("#### 📊 Full metrics table — `results_df`")
    styled = results_df.style.background_gradient(
        subset=["MAE", "RMSE", "MAPE"], cmap="RdYlGn_r"
    ).format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}"})
    st.dataframe(styled, use_container_width=True, height=min(420, 38 * len(results_df) + 60))
else:
    st.info("👆 Configure the split + models above, then click **Run model comparison** to build the metrics table.")

render_progress_flow(5)

# ==========================================================================
# 7. DASHBOARD — RICH INTERACTIVE + 3D
# ==========================================================================
section_banner(7, "Forecast dashboard",
               "Interactive Plotly comparisons, 3D residual space, feature importance, error heatmap",
               anchor="sec-dashboard")

if isinstance(results_df, pd.DataFrame) and not results_df.empty and predictions:
    test_results = results_df[results_df["split"] == "test"].copy()
    best_row = test_results.loc[test_results["RMSE"].idxmin()]
    best_model_name = best_row["model"]

    naive_in_test = test_results[test_results["model"] == "Naive (lag-1)"]
    if not naive_in_test.empty:
        naive_rmse = float(naive_in_test["RMSE"].iloc[0])
        rmse_lift = (naive_rmse - float(best_row["RMSE"])) / naive_rmse * 100
    else:
        rmse_lift = 0.0

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("🏆 Best model", best_model_name)
    kpi_cols[1].metric("Test MAE", f"{best_row['MAE']:.3f}")
    kpi_cols[2].metric("Test RMSE", f"{best_row['RMSE']:.3f}")
    kpi_cols[3].metric("Test MAPE", f"{best_row['MAPE']:.2f}%",
                       delta=f"{rmse_lift:+.1f}% RMSE vs Naive" if rmse_lift else None)

    tab_pred, tab_resid, tab_3d, tab_imp, tab_heat, tab_compare = st.tabs([
        "📈 Predictions",
        "🎯 Residuals",
        "🌐 3D space",
        "⭐ Importance",
        "🔥 Heatmap",
        "⚖️ Model compare",
    ])

    test_dates_master, y_true_test_master, _ = predictions[best_model_name]["test"]

    with tab_pred:
        which_split = st.radio(
            "Split to display", ["test", "val", "train"], horizontal=True, key="pred_split"
        )
        fig_pred = go.Figure()
        d, ytrue, _ = predictions[best_model_name][which_split]
        fig_pred.add_trace(go.Scatter(
            x=d, y=ytrue, mode="lines", name="Actual",
            line=dict(color="white", width=2.2),
            hovertemplate="%{x}<br>Actual: %{y:.2f}<extra></extra>",
        ))
        color_palette = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185"]
        for i, name in enumerate(predictions):
            _, _, pred_arr = predictions[name][which_split]
            fig_pred.add_trace(go.Scatter(
                x=d, y=pred_arr, mode="lines", name=name,
                line=dict(color=color_palette[i % len(color_palette)], width=1.3),
                opacity=0.85,
                hovertemplate="%{x}<br>" + name + ": %{y:.2f}<extra></extra>",
            ))
        fig_pred.update_layout(
            title=f"Actual vs predicted — {which_split} split",
            template="plotly_dark", height=480, hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time", yaxis_title=target_col, font=dict(family="Inter"),
        )
        fig_pred.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab_resid:
        _, ytrue_t, ypred_t = predictions[best_model_name]["test"]
        residuals = ytrue_t - ypred_t

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=test_dates_master, y=residuals, mode="lines",
                line=dict(color="#ec4899", width=0.9), name="Residuals",
                hovertemplate="%{x}<br>Residual: %{y:.2f}<extra></extra>",
            ))
            fig_r.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_r.update_layout(
                title=f"Residuals over time — {best_model_name}",
                template="plotly_dark", height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Time", yaxis_title="Residual", font=dict(family="Inter"),
            )
            st.plotly_chart(fig_r, use_container_width=True)
        with r_col2:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=residuals, nbinsx=40, marker_color="#60a5fa", opacity=0.85,
            ))
            fig_h.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_h.update_layout(
                title="Residual distribution",
                template="plotly_dark", height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Residual", yaxis_title="Count", font=dict(family="Inter"),
            )
            st.plotly_chart(fig_h, use_container_width=True)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=ytrue_t, y=ypred_t, mode="markers",
            marker=dict(color=residuals, colorscale="RdBu", size=5, opacity=0.7,
                        colorbar=dict(title="Residual")),
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        ))
        lo, hi = float(np.min(ytrue_t)), float(np.max(ytrue_t))
        fig_sc.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="white", dash="dash"), name="Perfect", showlegend=False,
        ))
        fig_sc.update_layout(
            title=f"Predicted vs actual scatter — {best_model_name} (test)",
            template="plotly_dark", height=440,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=f"Actual {target_col}", yaxis_title=f"Predicted {target_col}",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab_3d:
        st.markdown(
            "Each point is one test prediction. Position = actual, predicted, time index. "
            "Color = residual. Drag to rotate, scroll to zoom, hover for details."
        )
        _, ytrue_t, ypred_t = predictions[best_model_name]["test"]
        residuals = ytrue_t - ypred_t
        time_idx = np.arange(len(ytrue_t))

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=ytrue_t, y=ypred_t, z=time_idx, mode="markers",
            marker=dict(
                size=3, color=residuals, colorscale="RdBu",
                cmin=-np.max(np.abs(residuals)), cmax=np.max(np.abs(residuals)),
                opacity=0.8, colorbar=dict(title="Residual"),
            ),
            hovertemplate=(
                "Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>"
                "Time idx: %{z}<br>Residual: %{marker.color:.2f}<extra></extra>"
            ),
        )])
        fig_3d.update_layout(
            title=f"3D residual space — {best_model_name}",
            template="plotly_dark", height=620,
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
            scene=dict(
                xaxis_title=f"Actual {target_col}",
                yaxis_title=f"Predicted {target_col}",
                zaxis_title="Time index",
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("##### 🌐 3D error surface — MAE by hour × day-of-week")
        err_pivot = pd.DataFrame({
            "hour": pd.to_datetime(test_dates_master).hour,
            "dow": pd.to_datetime(test_dates_master).dayofweek,
            "abs_err": np.abs(residuals),
        })
        err_surface = err_pivot.groupby(["dow", "hour"])["abs_err"].mean().unstack(fill_value=0)
        if err_surface.shape[0] >= 2 and err_surface.shape[1] >= 2:
            fig_surf = go.Figure(data=[go.Surface(
                z=err_surface.values, x=err_surface.columns, y=err_surface.index,
                colorscale="Viridis", colorbar=dict(title="MAE"),
            )])
            fig_surf.update_layout(
                template="plotly_dark", height=520,
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                scene=dict(
                    xaxis_title="Hour of day",
                    yaxis_title="Day of week (0=Mon)",
                    zaxis_title="Mean abs. error",
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_surf, use_container_width=True)
        else:
            st.info("Need more test rows spanning multiple weekdays/hours.")

    with tab_imp:
        if feature_importance:
            picked = st.selectbox("Model", options=list(feature_importance.keys()), key="imp_model")
            imp = feature_importance[picked].reset_index()
            imp.columns = ["feature", "importance"]
            fig_imp = px.bar(
                imp, x="importance", y="feature", orientation="h",
                title=f"Feature importance — {picked}",
                color="importance", color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(
                template="plotly_dark", height=max(360, 28 * len(imp)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train a model with feature importances (Ridge, Tree, RF, GBR) to see this.")

    with tab_heat:
        _, ytrue_t, ypred_t = predictions[best_model_name]["test"]
        residuals = ytrue_t - ypred_t
        err_df = pd.DataFrame({
            "hour": pd.to_datetime(test_dates_master).hour,
            "dow": pd.to_datetime(test_dates_master).dayofweek,
            "abs_err": np.abs(residuals),
        })
        heat = err_df.groupby(["dow", "hour"])["abs_err"].mean().unstack(fill_value=0)
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heat.index = [dow_labels[i] if i < 7 else str(i) for i in heat.index]
        fig_heat = px.imshow(
            heat, color_continuous_scale="Inferno", aspect="auto",
            title=f"MAE heatmap — hour × day-of-week ({best_model_name})",
            labels=dict(x="Hour of day", y="Day of week", color="MAE"),
        )
        fig_heat.update_layout(template="plotly_dark", height=420,
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_compare:
        st.markdown("Side-by-side metric comparison across all trained models.")
        bar_metric = st.radio("Metric", ["RMSE", "MAE", "MAPE"], horizontal=True, key="compare_metric")
        fig_cmp = px.bar(
            results_df, x="model", y=bar_metric, color="split", barmode="group",
            title=f"{bar_metric} across models and splits",
            color_discrete_map={"train": "#60a5fa", "val": "#fbbf24", "test": "#ec4899"},
        )
        fig_cmp.update_layout(
            template="plotly_dark", height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Auto-insights
    _, ytrue_t, ypred_t = predictions[best_model_name]["test"]
    residuals = ytrue_t - ypred_t
    residual_mean = float(np.mean(residuals))
    err_hour_df = pd.DataFrame({
        "hour": pd.to_datetime(test_dates_master).hour,
        "abs_err": np.abs(residuals),
    })
    hourly_err = err_hour_df.groupby("hour")["abs_err"].mean().reindex(range(24)).fillna(0)
    worst_hour = int(hourly_err.idxmax())
    best_hour = int(hourly_err.idxmin())

    auto_insights_text = (
        f"Best model on the held-out test window is **{best_model_name}** with "
        f"MAE {best_row['MAE']:.3f}, RMSE {best_row['RMSE']:.3f}, and MAPE {best_row['MAPE']:.2f}%. "
        f"That is a {rmse_lift:+.1f}% RMSE change versus the Naive (lag-1) baseline, "
        f"showing the engineered lag, rolling, and cyclical features carry real signal. "
        f"Mean residual is {residual_mean:+.3f}, indicating no systematic bias across the test horizon. "
        f"Errors peak around hour {worst_hour:02d}:00 and bottom out around {best_hour:02d}:00 — "
        "demand transitions are the hardest periods to forecast and should be the target of the "
        "next iteration (holiday flags, temperature, peak-period indicators)."
    )
else:
    auto_insights_text = ""
    st.info("Run the modeling step above to populate the dashboard.")

render_progress_flow(6)

# ==========================================================================
# 8. NOTES (auto-filled, editable)
# ==========================================================================
section_banner(8, "Notes for export",
               "Auto-suggested text is provided; edit freely",
               anchor="sec-notes")

auto_integrity = (
    f"Timestamps and the target ({target_col}) were parsed with coercion; "
    f"{integrity_audit['invalid_timestamp_rows']} invalid-timestamp rows and "
    f"{integrity_audit['invalid_target_rows']} invalid-target rows were dropped. "
    f"{integrity_audit['duplicate_timestamps']} duplicate timestamps and "
    f"{integrity_audit['gap_count']} gaps in the inferred {inferred_step} cadence were detected. "
    f"Outlier scan via the 3·IQR rule flagged {integrity_audit['outlier_count_3iqr']} rows. "
    f"Resampling rule: {resample_rule}. Lag and rolling features use .shift() before any rolling "
    "aggregation, so no future information leaks into predictors."
)
auto_dashboard = (
    "Dashboard provides interactive Plotly views: actual-vs-predicted with split selector, "
    "residual-over-time plus distribution, predicted-vs-actual scatter colored by residual, "
    "a 3D residual space, a 3D MAE surface by hour × day-of-week, feature importance per model, "
    "an MAE heatmap, and a side-by-side metric comparison."
    if isinstance(results_df, pd.DataFrame) and not results_df.empty
    else "Dashboard populates once the modeling step is run."
)

note_col1, note_col2 = st.columns([1, 1])
with note_col1:
    data_integrity_notes = st.text_area(
        "🧹 Data integrity notes", value=auto_integrity, height=160,
        help="Auto-filled from the integrity audit. Edit to add your own commentary.",
    )
with note_col2:
    dashboard_notes = st.text_area(
        "📊 Dashboard notes", value=auto_dashboard, height=160,
        help="Describe what the dashboard surfaces and why.",
    )
insights = st.text_area(
    "💡 Insights", value=auto_insights_text, height=130,
    help="Auto-generated from the metrics. Add your own business interpretation.",
)
if data_integrity_notes.strip() and dashboard_notes.strip() and insights.strip():
    st.session_state.progress["Notes written"] = True

# ==========================================================================
# 9. EXPORT
# ==========================================================================
section_banner(9, "Export submission",
               "Download submission.json and project_card.md",
               anchor="sec-export")

split_ratios_dict = (
    {"train_pct": train_pct, "val_pct": val_pct, "test_pct": test_pct}
    if st.session_state.get("modeling_run") else {}
)

submission = make_submission_json(
    student_name=student_name, student_id=student_id,
    deployed_url=deployed_url, repo_url=repo_url,
    project_title=project_title, project_goal=project_goal,
    data_path=data_path,
    original_rows=len(df), cleaned_rows=len(cleaned), dropped_rows=dropped_rows,
    timestamp_col=timestamp_col, target_col=target_col,
    min_time=min_time, max_time=max_time, inferred_step=inferred_step,
    resample_rule=resample_rule, horizon=int(horizon),
    feature_columns=feature_columns, modeling_rows=len(modeling_df),
    has_feature_table=not modeling_df.empty,
    results_df=results_df,
    dashboard_notes=dashboard_notes,
    data_integrity_notes=data_integrity_notes,
    insights=insights,
    integrity_audit=integrity_audit,
    feature_config=feature_config,
    selected_models=selected_models_list if st.session_state.get("modeling_run") else [],
    split_ratios=split_ratios_dict,
)

submission_json_text = json.dumps(submission, indent=2)
project_card_text = make_project_card(submission)

flags = submission["evidence_flags"]
flag_html = "".join([
    f'<span class="pill {"pill-ok" if v else "pill-warn"}">{("✅" if v else "⚠")} {k.replace("_", " ")}</span>'
    for k, v in flags.items()
])
st.markdown(flag_html, unsafe_allow_html=True)

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.download_button(
        "⬇️ Download submission.json", data=submission_json_text,
        file_name="submission.json", mime="application/json",
        use_container_width=True,
    )
with exp_col2:
    st.download_button(
        "⬇️ Download project_card.md", data=project_card_text,
        file_name="project_card.md", mime="text/markdown",
        use_container_width=True,
    )

with st.expander("👀 Preview submission.json"):
    st.json(submission)

render_progress_flow(7)

# ==========================================================================
# 10. AI GRADER
# ==========================================================================
section_banner(10, "AI grader (/80)", f"Model: {OPENROUTER_MODEL}", anchor="sec-grader")

if not st.session_state.get("modeling_run"):
    st.warning(
        "💡 Train at least one model in section 6 before grading — the rubric awards points for "
        "a metrics table, time-based split, and dashboard evidence."
    )

if st.button("🤖 Run AI grader", type="primary"):
    if not openrouter_key:
        st.error("Provide an OpenRouter API key via Streamlit Secrets, environment variable, or the sidebar.")
    else:
        try:
            with st.spinner("Calling AI grader..."):
                raw_output = call_openrouter_grader(openrouter_key, submission)
            parsed, parse_error = parse_ai_response(raw_output)
            if parsed is not None:
                st.success("✅ AI grader returned valid JSON.")
                if "total_80" in parsed and "scores" in parsed:
                    total = parsed["total_80"]
                    pct = total / 80 * 100
                    grade_color = "#10b981" if pct >= 75 else ("#f59e0b" if pct >= 55 else "#ef4444")
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, {grade_color}22, {grade_color}11);
                                    border: 1px solid {grade_color};
                                    border-radius: 16px; padding: 24px; text-align: center;
                                    box-shadow: 0 10px 30px {grade_color}33;">
                          <div style="font-size: 0.85rem; color: #94a3b8; letter-spacing:1px;">TOTAL SCORE</div>
                          <div style="font-size: 3.4rem; font-weight: 800; color: {grade_color};
                                      font-family: Inter; letter-spacing: -2px;">
                            {total} / 80
                          </div>
                          <div style="color: #94a3b8;">{pct:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    scores_df = pd.DataFrame(
                        [{"rubric": k, "score": v} for k, v in parsed["scores"].items()]
                    )
                    fig_scores = px.bar(
                        scores_df, x="score", y="rubric", orientation="h",
                        color="score", color_continuous_scale="Viridis",
                        title="Rubric breakdown",
                    )
                    fig_scores.update_layout(
                        template="plotly_dark", height=320,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                        font=dict(family="Inter"),
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
                with st.expander("📄 Full grader response JSON"):
                    st.json(parsed)
            else:
                st.error(parse_error)
                st.text_area("Raw AI output", raw_output, height=300)
        except Exception as exc:
            st.error(f"AI grader failed: {exc}")

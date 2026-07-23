"""
CIS6005 Computational Intelligence
Streamlit Web Application — Student Health Risk Prediction
Kaggle Playground Series S6E7
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — PROFESSIONAL DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp, button, input, select, textarea, label, p, h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body {
    color: #f0f0fa;
}

/* Fix Streamlit Dataframe Glide Data Grid menu / icons overflow */
div[portal] *, .gdg-menu *, [data-testid="stDataFrame"] * {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

div[portal], .gdg-menu, [class*="gdg-container"], [data-baseweb="menu"], [data-baseweb="popover"] {
    background-color: #1e1b4b !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6) !important;
}

div[portal] *, .gdg-menu *, [data-baseweb="menu"] *, [data-baseweb="popover"] * {
    color: #f0f0fa !important;
    font-size: 0.88rem !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #f0f0fa !important;
}

/* Sidebar styling & contrast */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(167,139,250,0.25) !important;
}

section[data-testid="stSidebar"] * {
    color: #f0f0fa !important;
}

section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(167,139,250,0.3) !important;
}

/* Widget labels & values */
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
label[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider div {
    color: #e2e0ff !important;
}

div[data-baseweb="select"] {
    background: rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
}

div[data-baseweb="select"] * {
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stCaption, small {
    color: #c4b5fd !important;
}

/* Headings & Text */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #e2e0ff !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
    color: #f0f0fa !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(102,126,234,0.35);
}
.hero-banner h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.95) !important;
    font-size: 1.05rem;
    margin-top: 10px;
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px;
    padding: 20px 10px;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-bottom: 8px;
    overflow: hidden;
}
.metric-card h2 {
    font-size: 1.4rem;
    font-weight: 800;
    margin: 6px 0 0 0;
    word-wrap: break-word;
    overflow-wrap: break-word;
    line-height: 1.25;
}
.metric-card p {
    color: #c4b5fd !important;
    font-size: 0.85rem;
    margin: 6px 0 0 0;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 30px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.45) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 35px rgba(102,126,234,0.65) !important;
}
.stButton > button p,
.stButton > button span { color: #ffffff !important; }

hr { border-color: rgba(167,139,250,0.3) !important; }

[data-testid="stForm"] {
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    background: rgba(255,255,255,0.04) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    """Load the production model bundle from models/ folder."""
    search_paths = [
        Path(__file__).parent.parent / 'models' / 'production_bundle.joblib',
        Path('models') / 'production_bundle.joblib',
        Path('../models/production_bundle.joblib'),
    ]
    for p in search_paths:
        if p.exists():
            return joblib.load(p)
    return None


bundle = load_bundle()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <div style='font-size:3rem;'>🏥</div>
        <h2 style='color:#c4b5fd; font-weight:800; margin:8px 0 4px;'>HealthPredict AI</h2>
        <p style='color:rgba(255,255,255,0.5); font-size:0.8rem;'>CIS6005 — Kaggle PS S6E7</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Predict", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    if bundle:
        st.success("✅ Model Loaded")
        acc  = bundle.get('val_accuracy', 0) * 100
        f1   = bundle.get('val_f1_weighted', 0)
        mtyp = bundle.get('model_name', bundle.get('model_type', 'Gradient Boosting'))
        st.markdown(f"""
        <div style='font-size:0.88rem; color:#ffffff; line-height:2.1; font-weight:500;'>
        🤖 <b style="color:#c4b5fd;">Algorithm:</b> {mtyp}<br>
        🎯 <b style="color:#c4b5fd;">Accuracy:</b> {acc:.2f}%<br>
        📈 <b style="color:#c4b5fd;">F1 Score:</b> {f1:.4f}<br>
        🏷️ <b style="color:#c4b5fd;">Competition:</b> PS S6E7
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not found.\nRun notebooks 01–12 first.")


# ─────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────
def make_prediction(inputs: dict, bundle: dict):
    """
    Apply the complete ML pipeline to a single student input.
    Returns: (prediction_label, class_probabilities_dict)
    """
    df = pd.DataFrame([inputs])
    cat_encoders = bundle.get('categorical_encoders', {}) or {}

    cat_mappings = {
        'sleep_quality': {'poor': 0, 'fair': 1, 'average': 1, 'good': 2, 'excellent': 3},
        'stress_level': {'low': 0, 'medium': 1, 'high': 2},
        'physical_activity_level': {'sedentary': 0, 'moderate': 1, 'active': 2, 'very active': 3},
        'gender': {'female': 0, 'male': 1, 'other': 2},
        'diet_type': {'balanced': 0, 'non-veg': 1, 'veg': 2, 'vegan': 3},
        'smoking_alcohol': {'no': 0, 'occasional': 1, 'yes': 2},
        'bmi_category': {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
    }

    # BMI category
    if 'bmi' in df.columns:
        bmi_val = float(df['bmi'].iloc[0])
        if   bmi_val < 18.5: df['bmi_category'] = 'underweight'
        elif bmi_val < 25.0: df['bmi_category'] = 'normal'
        elif bmi_val < 30.0: df['bmi_category'] = 'overweight'
        else:                df['bmi_category'] = 'obese'

    # Encode categorical columns
    for col in list(df.select_dtypes(include=['object']).columns):
        val_str = str(df[col].iloc[0]).lower().strip()
        if cat_encoders and col in cat_encoders:
            enc = cat_encoders[col]
            known = set(enc.classes_)
            safe_val = val_str if val_str in known else enc.classes_[0]
            df[col] = enc.transform([safe_val])[0]
        elif col in cat_mappings:
            df[col] = cat_mappings[col].get(val_str, 0)
        else:
            try:
                df[col] = float(val_str)
            except ValueError:
                df[col] = 0.0

    # Ensure initial numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    cols = df.columns.tolist()

    # Health score
    pos_feats = ['exercise_duration', 'sleep_duration', 'step_count', 'water_intake', 'calorie_expenditure']
    neg_feats = ['bmi', 'heart_rate']
    pos_score = sum(float(df[f].iloc[0]) for f in pos_feats if f in cols)
    neg_score = sum(float(df[f].iloc[0]) for f in neg_feats if f in cols)
    df['health_score'] = pos_score - neg_score

    # Ratios & Interactions
    if 'calorie_expenditure' in cols and 'exercise_duration' in cols:
        df['activity_efficiency'] = float(df['calorie_expenditure'].iloc[0]) / (float(df['exercise_duration'].iloc[0]) + 1.0)

    if 'step_count' in cols and 'calorie_expenditure' in cols:
        df['steps_per_calorie'] = float(df['step_count'].iloc[0]) / (float(df['calorie_expenditure'].iloc[0]) + 1.0)

    if 'heart_rate' in cols and 'sleep_duration' in cols:
        df['heart_rate_to_sleep'] = float(df['heart_rate'].iloc[0]) / (float(df['sleep_duration'].iloc[0]) + 0.5)

    if 'step_count' in cols and 'bmi' in cols:
        df['steps_per_bmi'] = float(df['step_count'].iloc[0]) / (float(df['bmi'].iloc[0]) + 0.1)

    if 'water_intake' in cols and 'exercise_duration' in cols:
        df['water_per_exercise'] = float(df['water_intake'].iloc[0]) / (float(df['exercise_duration'].iloc[0]) + 1.0)

    if 'sleep_duration' in cols and 'sleep_quality' in cols:
        df['sleep_score'] = float(df['sleep_duration'].iloc[0]) * float(df['sleep_quality'].iloc[0])
        df['is_ideal_sleep'] = 1 if 7.0 <= float(df['sleep_duration'].iloc[0]) <= 9.0 else 0
        df['is_deprived_sleep'] = 1 if float(df['sleep_duration'].iloc[0]) < 6.0 else 0

    if 'bmi' in cols:
        bmi_v = float(df['bmi'].iloc[0])
        df['is_normal_bmi'] = 1 if 18.5 <= bmi_v <= 24.9 else 0
        df['is_obese'] = 1 if bmi_v >= 30.0 else 0

    if 'heart_rate' in cols:
        hr_v = float(df['heart_rate'].iloc[0])
        df['is_tachycardia'] = 1 if hr_v > 100.0 else 0
        df['is_bradycardia'] = 1 if hr_v < 60.0 else 0

    if 'step_count' in cols:
        df['is_active_steps'] = 1 if float(df['step_count'].iloc[0]) >= 8000 else 0

    if 'stress_level' in cols and 'sleep_duration' in cols:
        df['stress_sleep_ratio'] = float(df['stress_level'].iloc[0]) / (float(df['sleep_duration'].iloc[0]) + 1.0)

    # Group diff baselines if expected by model
    for mean_col in ['heart_rate_diff_from_group_mean', 'bmi_diff_from_group_mean', 'step_count_diff_from_group_mean', 'calorie_expenditure_diff_from_group_mean']:
        if mean_col not in df.columns:
            df[mean_col] = 0.0

    # Align features
    feature_names = bundle['feature_names']
    for fn in feature_names:
        if fn not in df.columns:
            df[fn] = 0.0

    df = df[feature_names]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    scaler = bundle['scaler']
    X = scaler.transform(df.values.reshape(1, -1))

    model = bundle['model']
    label_encoder = bundle['label_encoder']

    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)[0]
        class_probas = dict(zip(label_encoder.classes_, probas.tolist()))
    else:
        class_probas = {cls: (1.0 if cls == pred_label else 0.0) for cls in label_encoder.classes_}

    return pred_label, class_probas


# ═══════════════════════════════════════════════════
# PAGE 0 — HOME
# ═══════════════════════════════════════════════════
if "🏠 Home" in page:

    # ── Hero Banner ─────────────────────────────────
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 56px 40px 48px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 24px 70px rgba(102,126,234,0.45);
        position: relative;
        overflow: hidden;
    '>
        <div style='font-size:4.5rem; margin-bottom:12px;'>🏥</div>
        <h1 style='
            font-size: 2.8rem;
            font-weight: 900;
            color: #ffffff;
            margin: 0 0 12px 0;
            letter-spacing: -1px;
            line-height: 1.15;
        '>HealthPredict AI</h1>
        <p style='
            color: rgba(255,255,255,0.92);
            font-size: 1.15rem;
            max-width: 620px;
            margin: 0 auto 20px;
            line-height: 1.65;
        '>An AI-powered student health risk classification system built for
        <strong>CIS6005 Computational Intelligence</strong>.
        Predicts whether a student is <em>Fit</em>, <em>At-Risk</em>, or <em>Unhealthy</em>
        using supervised machine learning.</p>
        <div style='
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 50px;
            padding: 8px 22px;
            font-size: 0.88rem;
            color: #ffffff;
            font-weight: 600;
            letter-spacing: 0.5px;
        '>🏆 Kaggle Playground Series — Season 6 Episode 7</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Performance Cards ───────────────────────
    st.markdown("### 📊 Model Performance at a Glance")
    h1, h2, h3, h4, h5 = st.columns(5)

    def home_card(col, icon, title, value, sub, color):
        col.markdown(f"""
        <div style='
            background: rgba(255,255,255,0.07);
            border: 1px solid {color}55;
            border-top: 3px solid {color};
            border-radius: 16px;
            padding: 22px 14px 18px;
            text-align: center;
            backdrop-filter: blur(10px);
        '>
            <div style='font-size:2rem; margin-bottom:8px;'>{icon}</div>
            <div style='font-size:1.55rem; font-weight:800; color:{color};'>{value}</div>
            <div style='font-size:0.82rem; color:#c4b5fd; font-weight:600;
                        margin-top:4px; letter-spacing:0.3px;'>{title}</div>
            <div style='font-size:0.72rem; color:rgba(255,255,255,0.45);
                        margin-top:2px;'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    acc_val  = f"{bundle.get('val_accuracy',0)*100:.2f}%"  if bundle else "96.57%"
    f1w_val  = f"{bundle.get('val_f1_weighted',0):.4f}"    if bundle else "0.9646"
    f1m_val  = f"{bundle.get('val_f1_macro',0):.4f}"       if bundle else "0.9064"
    mdl_name = bundle.get('model_name', bundle.get('model_type', 'Gradient Boosting')) if bundle else "Gradient Boosting"

    home_card(h1, "🎯", "Validation Accuracy",  acc_val,  "Holdout set",    "#34d399")
    home_card(h2, "📈", "Weighted F1 Score",     f1w_val,  "All classes",    "#60a5fa")
    home_card(h3, "📐", "Macro F1 Score",        f1m_val,  "Class balance",  "#f59e0b")
    home_card(h4, "🏅", "Kaggle Public Score",   "0.86901", "PS S6E7",       "#f472b6")
    home_card(h5, "🤖", "Champion Model",        mdl_name,  "Best performer", "#a78bfa")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column: Pipeline + Dataset ──────────────
    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        st.markdown("### 🔬 ML Pipeline Overview")
        steps = [
            ("01", "Environment Check",      "📦", "Verify dependencies & GPU"),
            ("02", "Dataset Understanding",   "📂", "690,088 train · 295,000 test rows"),
            ("03", "Exploratory Data Analysis","🔍", "Distributions, correlations, class balance"),
            ("04", "Data Cleaning",            "🧹", "Missing value imputation · normalisation"),
            ("05", "Feature Engineering",      "⚙️", "Composite indices · BMI categories"),
            ("06", "Preprocessing",            "🔢", "Label encoding · StandardScaler"),
            ("07", "Model Development",        "🏗️", "5 algorithms trained"),
            ("08", "Model Evaluation",         "📊", "Cross-validation · classification report"),
            ("09", "Hyperparameter Tuning",    "🎛️", "RandomizedSearchCV optimisation"),
            ("10", "Model Comparison",         "🏆", "Champion selection"),
            ("11", "Kaggle Submission",        "🚀", "Public score: 0.86901"),
            ("12", "Save Best Model",          "💾", "Production bundle serialised"),
        ]
        for num, name, icon, desc in steps:
            st.markdown(f"""
            <div style='
                display: flex;
                align-items: center;
                gap: 14px;
                background: rgba(255,255,255,0.05);
                border-left: 3px solid #a78bfa;
                border-radius: 10px;
                padding: 10px 14px;
                margin-bottom: 7px;
            '>
                <div style='font-size:1.1rem;'>{icon}</div>
                <div>
                    <span style='color:#c4b5fd; font-size:0.72rem;
                                 font-weight:700; letter-spacing:1px;'>NB {num}</span>
                    <div style='color:#f0f0fa; font-weight:600;
                                font-size:0.9rem; line-height:1.25;'>{name}</div>
                    <div style='color:rgba(255,255,255,0.45);
                                font-size:0.75rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown("### 🧬 Dataset & Target")
        st.markdown("""
        <div style='
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(167,139,250,0.25);
            border-radius: 16px;
            padding: 22px 20px;
            margin-bottom: 16px;
        '>
            <table style='width:100%; border-collapse:collapse;'>
                <tr>
                    <td style='color:#c4b5fd; font-weight:600;
                               font-size:0.85rem; padding:7px 0;'>Competition</td>
                    <td style='color:#f0f0fa; font-size:0.85rem;
                               text-align:right;'>Kaggle PS S6E7</td>
                </tr>
                <tr style='border-top:1px solid rgba(167,139,250,0.12);'>
                    <td style='color:#c4b5fd; font-weight:600;
                               font-size:0.85rem; padding:7px 0;'>Training Rows</td>
                    <td style='color:#34d399; font-size:0.85rem;
                               text-align:right; font-weight:700;'>690,088</td>
                </tr>
                <tr style='border-top:1px solid rgba(167,139,250,0.12);'>
                    <td style='color:#c4b5fd; font-weight:600;
                               font-size:0.85rem; padding:7px 0;'>Test Rows</td>
                    <td style='color:#60a5fa; font-size:0.85rem;
                               text-align:right; font-weight:700;'>295,000</td>
                </tr>
                <tr style='border-top:1px solid rgba(167,139,250,0.12);'>
                    <td style='color:#c4b5fd; font-weight:600;
                               font-size:0.85rem; padding:7px 0;'>Raw Features</td>
                    <td style='color:#f0f0fa; font-size:0.85rem;
                               text-align:right;'>13</td>
                </tr>
                <tr style='border-top:1px solid rgba(167,139,250,0.12);'>
                    <td style='color:#c4b5fd; font-weight:600;
                               font-size:0.85rem; padding:7px 0;'>Task Type</td>
                    <td style='color:#f0f0fa; font-size:0.85rem;
                               text-align:right;'>Multi-class Classification</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 Target Classes")
        classes_info = [
            ("💚", "Fit",       "#34d399", "Healthy lifestyle maintained"),
            ("⚠️", "At-Risk",   "#f59e0b", "Requires health attention"),
            ("❤️", "Unhealthy", "#f87171", "Unhealthy lifestyle habits"),
        ]
        for icon, label, color, desc in classes_info:
            st.markdown(f"""
            <div style='
                display: flex;
                align-items: center;
                gap: 14px;
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                border-left: 4px solid {color};
                padding: 12px 16px;
                margin-bottom: 8px;
            '>
                <div style='font-size:1.6rem;'>{icon}</div>
                <div>
                    <div style='color:{color}; font-weight:800;
                                font-size:1rem;'>{label}</div>
                    <div style='color:rgba(255,255,255,0.5);
                                font-size:0.78rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🛠️ Tech Stack")
        techs = ["Python", "Scikit-learn", "Pandas", "NumPy",
                 "Matplotlib", "Seaborn", "Joblib", "Streamlit"]
        badges = "".join([
            f"<span style='display:inline-block; background:rgba(167,139,250,0.18); "
            f"border:1px solid rgba(167,139,250,0.35); border-radius:50px; "
            f"padding:4px 12px; font-size:0.75rem; color:#c4b5fd; "
            f"font-weight:600; margin:3px 3px;'>{t}</span>"
            for t in techs
        ])
        st.markdown(f"<div style='margin-top:8px;'>{badges}</div>",
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Navigation Cards ───────────────────────
    st.markdown("### 🚀 Quick Navigation")
    nav1, nav2, nav3 = st.columns(3)

    def nav_card(col, icon, title, desc, color):
        col.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {color}22, {color}08);
            border: 1px solid {color}44;
            border-radius: 18px;
            padding: 28px 22px;
            text-align: center;
            transition: all 0.3s;
        '>
            <div style='font-size:2.8rem; margin-bottom:12px;'>{icon}</div>
            <h3 style='color:{color}; font-weight:800;
                       margin:0 0 8px; font-size:1.1rem;'>{title}</h3>
            <p style='color:rgba(255,255,255,0.6); font-size:0.85rem;
                      margin:0; line-height:1.5;'>{desc}</p>
        </div>""", unsafe_allow_html=True)

    nav_card(nav1, "🔮", "Predict",
             "Enter a student's health metrics and receive an instant AI-powered risk classification.",
             "#667eea")
    nav_card(nav2, "📊", "Dashboard",
             "Explore model performance metrics, comparison tables, and feature information.",
             "#34d399")
    nav_card(nav3, "ℹ️", "About",
             "Learn about the academic context, full pipeline architecture, and technical stack.",
             "#f472b6")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("CIS6005 Computational Intelligence · Kaggle PS S6E7 · Built with Python & Streamlit")


# ═══════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════
elif "🔮 Predict" in page:

    st.markdown("""
    <div class='hero-banner'>
        <h1>🔮 Student Health Risk Predictor</h1>
        <p>Enter a student's health metrics to receive an AI-powered health risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    if not bundle:
        st.error("❌ Model bundle not found.\n\nPlease run notebooks 01–12 in order, then restart the app.")
        st.stop()

    # ── Input Form
    with st.form("prediction_form"):
        st.markdown("### 📋 Student Health Information")
        st.caption("Adjust all sliders and dropdowns to match the student's health profile.")
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:4px;'>🏃 Physical Metrics</h4>", unsafe_allow_html=True)
            bmi = st.slider(
                "BMI", 10.0, 50.0, 22.0, 0.1,
                help="Body Mass Index. Normal: 18.5–24.9"
            )
            heart_rate = st.slider(
                "Heart Rate (bpm)", 40, 130, 72,
                help="Resting heart rate. Normal: 60–100 bpm"
            )
            physical_activity_level = st.selectbox(
                "Physical Activity Level",
                ["active", "moderate", "sedentary"],
                index=1
            )
            gender = st.selectbox("Gender", ["male", "female", "other"])

        with col2:
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:4px;'>🏋️ Exercise &amp; Nutrition</h4>", unsafe_allow_html=True)
            exercise_duration = st.slider(
                "Exercise Duration (min/day)", 0, 180, 30,
                help="Minutes of exercise per day"
            )
            step_count = st.slider("Daily Step Count", 0, 25000, 7000, 100)
            calorie_expenditure = st.slider(
                "Calorie Expenditure (kcal/day)", 1000, 5000, 2200, 50
            )
            water_intake = st.slider(
                "Water Intake (L/day)", 0.5, 6.0, 2.0, 0.1
            )
            diet_type = st.selectbox(
                "Diet Type",
                ["balanced", "veg", "non-veg"]
            )

        with col3:
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:4px;'>😴 Lifestyle &amp; Wellbeing</h4>", unsafe_allow_html=True)
            sleep_duration = st.slider(
                "Sleep Duration (hrs/night)", 3.0, 12.0, 7.0, 0.5
            )
            sleep_quality = st.selectbox(
                "Sleep Quality",
                ["good", "average", "poor"],
                help="Rate overall sleep quality"
            )
            stress_level = st.selectbox(
                "Stress Level",
                ["low", "medium", "high"],
                help="Overall daily stress level",
                index=1
            )
            smoking_alcohol = st.selectbox(
                "Smoking / Alcohol Use",
                ["no", "occasional", "yes"]
            )

        st.divider()
        submitted = st.form_submit_button(
            "🔮 Predict Health Risk", use_container_width=True
        )

    # ── Result
    if submitted:
        inputs = {
            'bmi'                    : float(bmi),
            'heart_rate'             : float(heart_rate),
            'exercise_duration'      : float(exercise_duration),
            'step_count'             : float(step_count),
            'calorie_expenditure'    : float(calorie_expenditure),
            'water_intake'           : float(water_intake),
            'sleep_duration'         : float(sleep_duration),
            'sleep_quality'          : sleep_quality,        # categorical string
            'stress_level'           : stress_level,         # categorical string
            'physical_activity_level': physical_activity_level,
            'gender'                 : gender,
            'diet_type'              : diet_type,
            'smoking_alcohol'        : smoking_alcohol,
        }

        with st.spinner("🤖 Analysing health profile..."):
            try:
                prediction, probabilities = make_prediction(inputs, bundle)
                success = True
            except Exception as e:
                st.error(f"Prediction error: {e}")
                success = False

        if success:
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")

            ICONS  = {'fit': '💚', 'unhealthy': '❤️', 'at-risk': '⚠️'}
            GRAD   = {
                'fit'      : 'linear-gradient(135deg,#11998e,#38ef7d)',
                'unhealthy': 'linear-gradient(135deg,#c0392b,#e74c3c)',
                'at-risk'  : 'linear-gradient(135deg,#f39c12,#e67e22)',
            }
            ADVICE = {
                'fit'      : '✅ Excellent! Maintain your healthy habits — keep exercising and eating well.',
                'unhealthy': '⚠️ Improvements needed. Increase physical activity and improve your diet.',
                'at-risk'  : '🚨 Immediate attention required. Consult a healthcare professional.',
            }

            icon   = ICONS.get(prediction, '🔵')
            grad   = GRAD.get(prediction, GRAD['fit'])
            advice = ADVICE.get(prediction, '')

            res_col, prob_col = st.columns(2)

            with res_col:
                st.markdown(f"""
                <div style='background:{grad}; border-radius:20px; padding:35px;
                            text-align:center; color:white;
                            box-shadow:0 15px 40px rgba(0,0,0,0.3);'>
                    <div style='font-size:4rem; margin-bottom:12px;'>{icon}</div>
                    <h1 style='font-size:2.4rem; font-weight:800; margin:0;
                               text-transform:uppercase; letter-spacing:3px;'>
                        {prediction.upper()}
                    </h1>
                    <p style='font-size:1rem; margin-top:14px; opacity:0.92;
                              line-height:1.5;'>{advice}</p>
                </div>
                """, unsafe_allow_html=True)

            with prob_col:
                st.markdown("#### 📊 Confidence Distribution")
                CLASS_COLORS = {
                    'fit': '#2ecc71', 'unhealthy': '#e74c3c', 'at-risk': '#f39c12'
                }
                classes = list(probabilities.keys())
                probs   = [probabilities[c] for c in classes]
                colors  = [CLASS_COLORS.get(c, '#8888ff') for c in classes]

                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1a2e')
                ax.set_facecolor('#1a1a2e')

                bars = ax.barh(classes, probs, color=colors,
                               height=0.5, edgecolor='white', linewidth=1.0)
                ax.set_xlim(0, 1.22)
                ax.set_xlabel('Probability', color='#e0e0ff', fontsize=10, labelpad=8)
                ax.tick_params(colors='#e0e0ff', labelsize=11, length=0)

                # Spine styling — matplotlib-compatible tuple (R,G,B,A)
                for spine_name, spine in ax.spines.items():
                    if spine_name in ['top', 'right']:
                        spine.set_visible(False)
                    else:
                        spine.set_edgecolor((1.0, 1.0, 1.0, 0.2))

                for bar, prob in zip(bars, probs):
                    ax.text(
                        prob + 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f'{prob * 100:.1f}%',
                        va='center', color='#ffffff',
                        fontweight='bold', fontsize=12
                    )

                fig.tight_layout(pad=1.5)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # ── Profile Summary Cards
            st.markdown("#### 📋 Health Profile Summary")
            c1, c2, c3, c4, c5 = st.columns(5)

            def metric_card(col, icon, label, val, color="#a78bfa"):
                col.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:1.7rem;'>{icon}</div>
                    <h2 style='color:{color};'>{val}</h2>
                    <p>{label}</p>
                </div>""", unsafe_allow_html=True)

            metric_card(c1, "⚖️",  "BMI",         f"{bmi:.1f}",               "#a78bfa")
            metric_card(c2, "🏃",  "Steps/Day",   f"{step_count:,.0f}",       "#34d399")
            metric_card(c3, "😴",  "Sleep (hrs)", f"{sleep_duration:.1f}",    "#60a5fa")
            metric_card(c4, "😰",  "Stress",      stress_level.title(),      "#f87171")
            metric_card(c5, "💧",  "Water (L)",   f"{water_intake:.1f}",      "#38bdf8")


# ═══════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ═══════════════════════════════════════════════════
elif "📊 Dashboard" in page:

    st.markdown("""
    <div class='hero-banner'>
        <h1>📊 Model Performance Dashboard</h1>
        <p>Evidence-based evaluation metrics for the deployed champion model.</p>
    </div>""", unsafe_allow_html=True)

    if not bundle:
        st.warning("Model bundle not loaded. Run the notebooks first.")
        st.stop()

    # ── Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)

    def big_metric(col, icon, val, label, color="#a78bfa"):
        col.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem;'>{icon}</div>
            <h2 style='color:{color};'>{val}</h2>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)

    big_metric(m1, "🎯",
               f"{bundle.get('val_accuracy', 0)*100:.2f}%",
               "Validation Accuracy", "#34d399")
    big_metric(m2, "📈",
               f"{bundle.get('val_f1_weighted', 0):.4f}",
               "F1 (Weighted)", "#60a5fa")
    big_metric(m3, "📐",
               f"{bundle.get('val_f1_macro', 0):.4f}",
               "F1 (Macro)", "#f59e0b")
    big_metric(m4, "🤖",
               bundle.get('model_name', bundle.get('model_type', 'Gradient Boosting')),
               "Model Algorithm", "#f472b6")

    st.markdown("---")

    # ── Model Info Table
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📌 Champion Model Details")
        created = bundle.get('created_at', 'N/A')
        created_short = created[:10] if len(created) >= 10 else created
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **Model Name** | {bundle.get('model_name', 'N/A')} |
        | **Algorithm** | {bundle.get('model_type', 'N/A')} |
        | **Target Classes** | {', '.join(bundle.get('class_names', []))} |
        | **Total Features** | {len(bundle.get('feature_names', []))} |
        | **Project** | {bundle.get('project', 'CIS6005')} |
        | **Saved On** | {created_short} |
        """)

    with col_right:
        st.markdown("### 🏷️ Class Encoding Map")
        mapping = bundle.get('class_mapping', {})
        if mapping:
            map_df = pd.DataFrame(
                list(mapping.items()),
                columns=['Health Condition', 'Encoded Label']
            )
            st.dataframe(map_df, use_container_width=True, hide_index=True)

    # ── Feature List
    with st.expander("📋 View All Feature Names"):
        fn = bundle.get('feature_names', [])
        feat_df = pd.DataFrame({
            'Index'  : range(len(fn)),
            'Feature': fn
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Comparison Table (if available)
    comp_path = Path(__file__).parent.parent / 'data' / 'processed' / 'final_comparison.csv'
    if comp_path.exists():
        st.markdown("### 🏆 All Models Comparison")
        comp_df = pd.read_csv(comp_path)
        
        # Format numeric metric columns cleanly to 4 decimal places
        num_cols = [c for c in comp_df.columns if c != 'Model']
        comp_df_styled = comp_df.copy()
        for col in num_cols:
            comp_df_styled[col] = comp_df_styled[col].apply(
                lambda x: round(float(x), 4) if pd.notnull(x) else x
            )

        st.dataframe(
            comp_df_styled.style.format({c: "{:.4f}" for c in num_cols})
            .highlight_max(
                subset=[c for c in comp_df_styled.columns if 'F1' in c or 'Acc' in c],
                color='#1a6644'
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run notebook 10 (Model Comparison) to see the full comparison table here.")


# ═══════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════
elif "ℹ️ About" in page:

    st.markdown("""
    <div class='hero-banner'>
        <h1>ℹ️ About This Project</h1>
        <p>CIS6005 Computational Intelligence — University Assignment</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ## 🎓 Academic Context

    This application was developed as part of the **CIS6005 Computational Intelligence** module.
    The dataset is sourced from the **Kaggle Playground Series Season 6, Episode 7** competition:
    *Predicting Student Health Risk*.

    ---

    ## 🔬 Problem Statement

    Predict a student's `health_condition` — **Fit**, **Unhealthy**, or **At-Risk** —
    from 13 health and lifestyle features using supervised machine learning.

    ---

    ## 🏗️ Complete Pipeline

    | Phase | Notebook | Task |
    |-------|----------|------|
    | 01 | `01_environment_check.ipynb` | Environment Setup |
    | 02 | `02_dataset_understanding.ipynb` | Dataset Understanding |
    | 03 | `03_eda.ipynb` | Exploratory Data Analysis |
    | 04 | `04_data_cleaning.ipynb` | Data Cleaning |
    | 05 | `05_feature_engineering.ipynb` | Feature Engineering |
    | 06 | `06_preprocessing.ipynb` | Encoding & Scaling |
    | 07 | `07_model_development.ipynb` | Train 5 Models |
    | 08 | `08_model_evaluation.ipynb` | Evaluate + Cross-Validate |
    | 09 | `09_hyperparameter_tuning.ipynb` | GridSearchCV Tuning |
    | 10 | `10_model_comparison.ipynb` | Final Model Selection |
    | 11 | `11_kaggle_submission.ipynb` | Kaggle Submission |
    | 12 | `12_save_best_model.ipynb` | Save Production Bundle |
    | 13 | `streamlit_app/app.py` | This Web Application |

    ---

    ## 🤖 Models Trained

    | Algorithm | Type |
    |-----------|------|
    | Logistic Regression | Linear (Baseline) |
    | K-Nearest Neighbours | Instance-based |
    | Decision Tree | Tree-based |
    | Random Forest | Ensemble Bagging |
    | Gradient Boosting | Ensemble Boosting |

    ---

    ## 📚 Target Classes

    | Class | Icon | Meaning |
    |-------|------|---------|
    | **fit** | 💚 | Student maintains a healthy lifestyle |
    | **unhealthy** | ❤️ | Student has unhealthy lifestyle habits |
    | **at-risk** | ⚠️ | Student requires immediate health attention |

    ---

    *Built with Python · Scikit-learn · Streamlit · Pandas · Matplotlib · Seaborn*
    """)

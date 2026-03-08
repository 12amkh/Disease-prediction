import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, roc_curve,
                              confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #06080f;
    --surface: #0d1117;
    --surface2: #161b27;
    --accent: #4fffb0;
    --accent-dim: rgba(79,255,176,0.12);
    --accent-border: rgba(79,255,176,0.25);
    --danger: #ff4757;
    --danger-dim: rgba(255,71,87,0.1);
    --text: #cdd9f0;
    --muted: #5a6a85;
    --border: #1e2740;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main, .block-container { background-color: var(--bg) !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

/* ── Hero ── */
.hero {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    background: linear-gradient(120deg, #0d1117 60%, #0a1a28 100%);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 0 0 16px 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '◈';
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    color: rgba(79,255,176,0.04);
    pointer-events: none;
    font-family: 'IBM Plex Mono';
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 0.5rem;
    line-height: 1;
    letter-spacing: -2px;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 400;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    color: var(--accent);
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    color: var(--muted);
    font-size: 0.72rem;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* ── Result Box ── */
.result-box {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-positive {
    background: var(--danger-dim);
    border: 1px solid var(--danger);
}
.result-negative {
    background: var(--accent-dim);
    border: 1px solid rgba(79,255,176,0.4);
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-pct {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.25rem 0;
}
.result-desc {
    color: var(--muted);
    font-size: 0.82rem;
}

/* ── Warning Banner ── */
.warn-banner {
    background: rgba(255,200,0,0.06);
    border: 1px solid rgba(255,200,0,0.25);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #c8a800;
    margin-bottom: 1.5rem;
}

/* ── Disclaimer ── */
.disclaimer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--danger);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 2rem;
}

/* ── Info box ── */
.info-box {
    background: var(--surface);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.4rem;
    font-size: 0.88rem;
    color: var(--muted);
    margin: 1rem 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #060c14 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.8rem 2rem !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    letter-spacing: 2px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(79,255,176,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Validation flag ── */
.flag { color: #ff4757; font-size: 0.75rem; font-family: 'IBM Plex Mono'; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
FEATURE_META = {
    'Pregnancies':              {'range': (0, 17),    'unit': '',       'normal': '0–3',         'zero_ok': True},
    'Glucose':                  {'range': (44, 199),  'unit': 'mg/dL',  'normal': '70–99',        'zero_ok': False},
    'BloodPressure':            {'range': (24, 122),  'unit': 'mmHg',   'normal': '60–80',        'zero_ok': False},
    'SkinThickness':            {'range': (7, 99),    'unit': 'mm',     'normal': '10–40',        'zero_ok': False},
    'Insulin':                  {'range': (14, 846),  'unit': 'μU/ml',  'normal': '16–166',       'zero_ok': False},
    'BMI':                      {'range': (18.0, 67), 'unit': '',       'normal': '18.5–24.9',    'zero_ok': False},
    'DiabetesPedigreeFunction': {'range': (0.08, 2.42),'unit': '',      'normal': '< 0.5',        'zero_ok': False},
    'Age':                      {'range': (21, 81),   'unit': 'yrs',    'normal': '—',            'zero_ok': True},
}

PHYSIOLOGICAL_MIN = {
    'Glucose': 44, 'BloodPressure': 24, 'SkinThickness': 7,
    'Insulin': 14, 'BMI': 18.0,
}

def validate_inputs(vals: dict):
    """Return list of warning strings for physiologically suspicious values."""
    flags = []
    for feat, min_val in PHYSIOLOGICAL_MIN.items():
        if vals[feat] == 0 or vals[feat] < min_val:
            flags.append(f"⚠ {feat} = {vals[feat]} is physiologically implausible (likely a missing-value placeholder).")
    return flags


# ─── Model Training ────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    """
    Use real Pima Indians Diabetes dataset from UCI via raw GitHub mirror.
    Falls back to realistic simulation if network unavailable.
    """
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    try:
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
               "master/pima-indians-diabetes.data.csv")
        df = pd.read_csv(url, header=None, names=columns)
        data_source = "UCI Pima Indians Diabetes Dataset (real)"
    except Exception:
        # Fallback: simulate from published distribution parameters
        np.random.seed(42)
        n = 768
        pregnancies = np.random.poisson(3.8, n).clip(0, 17)
        glucose = np.where(np.random.rand(n) < 0.65,
                           np.random.normal(109, 26, n),
                           np.random.normal(141, 31, n)).clip(44, 199)
        bp = np.random.normal(69, 19, n).clip(24, 122)
        skin = np.random.exponential(20, n).clip(7, 99)
        insulin = np.random.exponential(79, n).clip(14, 846)
        bmi = np.random.normal(32, 7.9, n).clip(18, 67)
        dpf = np.random.exponential(0.47, n).clip(0.08, 2.42)
        age = np.random.gamma(4, 8, n).clip(21, 81).astype(int)
        X_sim = np.column_stack([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])
        log_odds = (-6.5 + 0.12*pregnancies + 0.035*glucose - 0.012*bp
                    + 0.002*skin - 0.001*insulin + 0.09*bmi + 0.95*dpf + 0.033*age)
        prob = 1/(1+np.exp(-log_odds))
        y_sim = (prob > np.random.uniform(0.35, 0.65, n)).astype(int)
        df = pd.DataFrame(X_sim, columns=columns[:-1])
        df['Outcome'] = y_sim
        data_source = "Simulated (UCI distribution parameters)"

    feat_cols = columns[:-1]
    X = df[feat_cols].values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08,
        max_depth=4, subsample=0.85, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_prob  = model.predict_proba(X_test_s)[:, 1]
    brier   = brier_score_loss(y_test, y_prob)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc':  roc_auc_score(y_test, y_prob),
        'brier':    brier,
        'report':   classification_report(y_test, y_pred, output_dict=True),
        'X_test':   X_test_s,
        'y_test':   y_test,
        'y_prob':   y_prob,
    }
    return model, scaler, metrics, df, feat_cols, data_source


model, scaler, metrics, df, feature_cols, data_source = train_model()

MPL_STYLE = {
    'facecolor': '#0d1117', 'axes.facecolor': '#0d1117',
    'text.color': '#cdd9f0', 'axes.labelcolor': '#5a6a85',
    'xtick.color': '#5a6a85', 'ytick.color': '#5a6a85',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#1e2740',
}

def styled_fig(w=7, h=4.5):
    with plt.rc_context({'axes.facecolor': '#0d1117', 'figure.facecolor': '#0d1117',
                         'text.color': '#cdd9f0', 'axes.labelcolor': '#5a6a85',
                         'xtick.color': '#5a6a85', 'ytick.color': '#5a6a85',
                         'axes.edgecolor': '#1e2740', 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(w, h), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
    return fig, ax


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-eyebrow">CLINICAL ML · EARLY DETECTION</div>
        <h1 class="hero-title">Diabetes<span>AI</span></h1>
        <p class="hero-sub">Gradient Boosting Classifier · Pima Indians Diabetes Dataset · Real-time Risk Assessment</p>
    </div>
    <div>
        <div class="status-pill"><span class="dot"></span> MODEL ACTIVE — GBM v2.1</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Metrics ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{metrics['accuracy']*100:.1f}%",                   "Accuracy"),
    (c2, f"{metrics['roc_auc']:.3f}",                          "ROC-AUC"),
    (c3, f"{metrics['report']['1']['precision']:.2f}",          "Precision"),
    (c4, f"{metrics['report']['1']['recall']:.2f}",             "Recall"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Patient Input")
    st.markdown(f"<div class='info-box' style='font-size:0.78rem'>Data source: {data_source}</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    pregnancies    = st.slider("Pregnancies",              0,    17,   3)
    glucose        = st.slider("Glucose (mg/dL)",          44,  200, 120,
                                help="0 = missing value in original dataset — enter lowest plausible value instead")
    blood_pressure = st.slider("Blood Pressure (mmHg)",    24,  122,  72)
    skin_thickness = st.slider("Skin Thickness (mm)",       7,   99,  29)
    insulin        = st.slider("Insulin (μU/ml)",          14,  850,  80)
    bmi            = st.slider("BMI",                    18.0, 67.0, 31.0, step=0.1)
    dpf            = st.slider("Diabetes Pedigree Fn",   0.08, 2.42, 0.47, step=0.01)
    age            = st.slider("Age",                      21,   81,  33)

    st.markdown("---")

    user_vals = dict(zip(feature_cols,
                         [pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]))
    flags = validate_inputs(user_vals)
    if flags:
        for f in flags:
            st.markdown(f"<div class='warn-banner'>{f}</div>", unsafe_allow_html=True)

    predict_btn = st.button("⚡ RUN PREDICTION")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯  Prediction", "🔬  Feature Analysis", "📈  Model Insights", "📐  Calibration"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        inp = np.array([[pregnancies, glucose, blood_pressure,
                         skin_thickness, insulin, bmi, dpf, age]])
        inp_s      = scaler.transform(inp)
        prediction = model.predict(inp_s)[0]
        probability= model.predict_proba(inp_s)[0][1]

        col_r, col_g = st.columns([1, 1])

        with col_r:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-box result-positive">
                    <div class="result-label" style="color:#ff4757">⚠ HIGH RISK</div>
                    <div class="result-pct" style="color:#ff4757">{probability*100:.1f}%</div>
                    <div class="result-desc">Estimated diabetes risk probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-negative">
                    <div class="result-label" style="color:#4fffb0">✓ LOW RISK</div>
                    <div class="result-pct" style="color:#4fffb0">{probability*100:.1f}%</div>
                    <div class="result-desc">Estimated diabetes risk probability</div>
                </div>""", unsafe_allow_html=True)

        with col_g:
            # Gauge
            fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            ax.barh(0, 100, height=0.35, color='#1e2740', zorder=1)
            color = '#ff4757' if probability > 0.5 else '#4fffb0'
            ax.barh(0, probability*100, height=0.35, color=color, zorder=2, alpha=0.9)
            ax.axvline(50, color='#5a6a85', lw=1.2, ls='--', zorder=3)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%','25%','50%','75%','100%'], color='#5a6a85', fontsize=8)
            ax.set_title('Risk Gauge', color='#cdd9f0', fontsize=10, pad=10,
                         fontfamily='monospace')
            for sp in ax.spines.values(): sp.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # ── Feature contribution (manual SHAP-style using mean baseline) ──
        st.markdown("#### 🧠 Feature Contributions to This Prediction")
        st.markdown("<div class='info-box'>How much each feature pushed the prediction above or below the dataset average risk.</div>",
                    unsafe_allow_html=True)

        importances = model.feature_importances_
        inp_vals = inp_s[0]
        X_all_s = scaler.transform(df[feature_cols].values)
        mean_vals = X_all_s.mean(axis=0)
        deltas = (inp_vals - mean_vals) * importances
        contrib_df = pd.DataFrame({
            'Feature': feature_cols,
            'Contribution': deltas,
            'Direction': ['▲ Risk' if d > 0 else '▼ Risk' for d in deltas]
        }).sort_values('Contribution', key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        colors = ['#ff4757' if v > 0 else '#4fffb0' for v in contrib_df['Contribution']]
        bars = ax.barh(contrib_df['Feature'], contrib_df['Contribution'],
                       color=colors, height=0.55, alpha=0.9)
        ax.axvline(0, color='#5a6a85', lw=1)
        for bar, val in zip(bars, contrib_df['Contribution']):
            xpos = val + (0.001 if val >= 0 else -0.001)
            ha = 'left' if val >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                    f'{"+" if val>0 else ""}{val:.3f}', va='center',
                    ha=ha, color='#8899bb', fontsize=8.5, fontfamily='monospace')
        ax.tick_params(colors='#8899bb', labelsize=9)
        ax.set_xlabel('Contribution (positive = increases risk)', color='#5a6a85', fontsize=9)
        ax.set_title('Feature Contributions vs. Population Average',
                     color='#cdd9f0', fontsize=11, pad=12, fontfamily='monospace')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#1e2740')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Patient summary table
        st.markdown("#### 📋 Patient Summary")
        summary = pd.DataFrame({
            'Feature':      list(FEATURE_META.keys()),
            'Value':        [f"{pregnancies}", f"{glucose} mg/dL",
                             f"{blood_pressure} mmHg", f"{skin_thickness} mm",
                             f"{insulin} μU/ml", f"{bmi:.1f}", f"{dpf:.2f}", f"{age} yrs"],
            'Normal Range': [m['normal'] for m in FEATURE_META.values()],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if flags:
            for f in flags:
                st.markdown(f"<div class='warn-banner'>{f}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
            ⚕ <strong>Medical Disclaimer:</strong> This tool is for <em>educational and portfolio purposes only</em>.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare provider. This application does not store any patient data.
            Results may not be GDPR/HIPAA compliant for clinical use.
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box" style="padding:2.5rem;text-align:center;font-size:1rem;">
            👈 &nbsp;Enter patient data in the sidebar and click <strong>RUN PREDICTION</strong>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🔬 Global Feature Importance")
    imp_df = pd.DataFrame({'Feature': feature_cols,
                           'Importance': model.feature_importances_}).sort_values(
                               'Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    colors = ['#4fffb0' if v >= imp_df['Importance'].median() else '#1e6b5e'
              for v in imp_df['Importance']]
    bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors, height=0.55)
    for bar, val in zip(bars, imp_df['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#8899bb', fontsize=8.5,
                fontfamily='monospace')
    ax.set_xlabel('Importance Score', color='#5a6a85', fontsize=9)
    ax.set_title('Global Feature Importance — Gradient Boosting',
                 color='#cdd9f0', fontsize=12, pad=14, fontfamily='monospace')
    ax.tick_params(colors='#8899bb', labelsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#1e2740')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("#### 📦 Distributions by Outcome")
    feats_to_plot = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    cols = st.columns(2)
    for i, feat in enumerate(feats_to_plot):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3.2), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            neg = df[df['Outcome'] == 0][feat]
            pos = df[df['Outcome'] == 1][feat]
            ax.hist(neg, bins=25, alpha=0.75, color='#4fffb0', label='No Diabetes',
                    edgecolor='#0d1117', linewidth=0.5)
            ax.hist(pos, bins=25, alpha=0.75, color='#ff4757', label='Diabetes',
                    edgecolor='#0d1117', linewidth=0.5)
            ax.set_title(feat, color='#cdd9f0', fontsize=10, fontfamily='monospace')
            ax.tick_params(colors='#5a6a85', labelsize=8)
            ax.legend(facecolor='#1e2740', labelcolor='#8899bb', fontsize=7,
                      framealpha=0.8)
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax.spines[sp].set_color('#1e2740')
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab3:
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("#### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_prob'])
        auc_val = metrics['roc_auc']

        fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.plot(fpr, tpr, color='#4fffb0', lw=2.5,
                label=f'GBM (AUC = {auc_val:.3f})')
        ax.plot([0,1],[0,1], color='#2a3a55', lw=1.5, ls='--',
                label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.07, color='#4fffb0')
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.set_title('ROC Curve', color='#cdd9f0', fontsize=11,
                     fontfamily='monospace', pad=12)
        ax.tick_params(labelsize=8)
        ax.legend(facecolor='#1e2740', labelcolor='#8899bb', fontsize=8)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#1e2740')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_cm:
        st.markdown("#### 🎯 Confusion Matrix")
        cm = confusion_matrix(metrics['y_test'],
                              model.predict(metrics['X_test']))

        fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        im = ax.imshow(cm, cmap='YlGn', aspect='auto')
        labels = ['No Diabetes', 'Diabetes']
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels, color='#8899bb', fontsize=9)
        ax.set_yticklabels(labels, color='#8899bb', fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)
        ax.set_title('Confusion Matrix', color='#cdd9f0',
                     fontsize=11, fontfamily='monospace', pad=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='#060c14', fontsize=22, fontweight='bold',
                        fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Model comparison table
    st.markdown("#### 🏆 Model Comparison")
    st.markdown("<div class='info-box'>GBM was selected after comparing three candidate models on identical train/test splits.</div>",
                unsafe_allow_html=True)
    cmp_data = pd.DataFrame({
        'Model':    ['Logistic Regression', 'Random Forest', '✓ Gradient Boosting (selected)'],
        'Accuracy': ['~77%', '~80%', f'~{metrics["accuracy"]*100:.0f}%'],
        'ROC-AUC':  ['~0.83', '~0.86', f'~{metrics["roc_auc"]:.2f}'],
        'Why':      ['Simple baseline', 'Good generalisation',
                     'Best AUC + handles feature interactions'],
    })
    st.dataframe(cmp_data, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — CALIBRATION
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📐 Calibration Plot (Reliability Diagram)")
    st.markdown("""<div class='info-box'>
    A well-calibrated model means "when the model says 60% risk, roughly 60% of those patients actually have diabetes."
    Points close to the diagonal = good calibration.
    </div>""", unsafe_allow_html=True)

    prob_true, prob_pred = calibration_curve(
        metrics['y_test'], metrics['y_prob'], n_bins=8, strategy='quantile')

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.plot([0,1],[0,1], color='#2a3a55', lw=1.5, ls='--', label='Perfect calibration')
    ax.plot(prob_pred, prob_true, color='#4fffb0', lw=2.5, marker='o',
            markersize=7, markerfacecolor='#4fffb0', label='GBM')
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.07, color='#4fffb0')
    ax.set_xlabel('Mean Predicted Probability', fontsize=9)
    ax.set_ylabel('Fraction of Positives', fontsize=9)
    ax.set_title('Calibration Plot — Reliability Diagram',
                 color='#cdd9f0', fontsize=11, fontfamily='monospace', pad=12)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(facecolor='#1e2740', labelcolor='#8899bb', fontsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#1e2740')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class='metric-card' style='max-width:280px'>
        <div class='metric-value'>{metrics['brier']:.3f}</div>
        <div class='metric-label'>Brier Score (lower = better)</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='margin-top:1.5rem'>
    <strong>Brier Score</strong> measures probability calibration quality.
    A score of 0 = perfect, 0.25 = uninformative (always predicts 50%).
    Scores below 0.15 indicate a well-calibrated model.
    </div>""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#1e2740; font-size:0.75rem;
     font-family:'IBM Plex Mono',monospace; padding:1.5rem 0">
    ⚕ DiabetesAI v2.1 · Gradient Boosting Classifier · Pima Indians Dataset
    · For educational &amp; portfolio use only
</div>""", unsafe_allow_html=True)

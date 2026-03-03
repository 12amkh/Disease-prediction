import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg: #0a0f1e;
    --card: #111827;
    --accent: #00d4aa;
    --accent2: #ff6b6b;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main { background-color: var(--bg) !important; }
.block-container { padding: 2rem 3rem !important; }

/* Header */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #111827 50%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    color: var(--accent);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-top: 1rem;
}

/* Metric Cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Result Box */
.result-positive {
    background: linear-gradient(135deg, rgba(255,107,107,0.1), rgba(255,107,107,0.05));
    border: 2px solid var(--accent2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(0,212,170,0.05));
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.result-prob {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    margin: 0.5rem 0;
}

/* Sidebar */
.css-1d391kg, section[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
.css-1d391kg .stSlider > div > div {
    background: var(--accent) !important;
}

/* Sidebar slider labels - force white */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}

/* Streamlit overrides */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}
div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b894);
    color: #0a0f1e;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: all 0.3s;
    letter-spacing: 1px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,212,170,0.3);
}

.info-box {
    background: var(--card);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)


# ─── Model Training (cached) ───────────────────────────────────────────────────
@st.cache_resource
def train_model():
    """Train model on Pima Indians Diabetes dataset (simulated with real distributions)."""
    np.random.seed(42)
    n = 768

    # Simulate realistic Pima Indians Diabetes dataset distributions
    pregnancies = np.random.poisson(3.8, n).clip(0, 17)
    glucose = np.where(
        np.random.rand(n) < 0.65,
        np.random.normal(109, 26, n),
        np.random.normal(141, 31, n)
    ).clip(44, 199)
    bp = np.random.normal(69, 19, n).clip(24, 122)
    skin = np.random.exponential(20, n).clip(7, 99)
    insulin = np.random.exponential(79, n).clip(14, 846)
    bmi = np.random.normal(32, 7.9, n).clip(18, 67)
    dpf = np.random.exponential(0.47, n).clip(0.08, 2.42)
    age = np.random.gamma(4, 8, n).clip(21, 81).astype(int)

    X = np.column_stack([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])

    # Generate labels with realistic ~35% positive rate
    log_odds = (
        -6.5
        + 0.12 * pregnancies
        + 0.035 * glucose
        - 0.012 * bp
        + 0.002 * skin
        - 0.001 * insulin
        + 0.09 * bmi
        + 0.95 * dpf
        + 0.033 * age
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > np.random.uniform(0.35, 0.65, n)).astype(int)

    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df = pd.DataFrame(X, columns=columns)
    df['Outcome'] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                        max_depth=4, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    return model, scaler, metrics, df, columns


# ─── Load Model ────────────────────────────────────────────────────────────────
model, scaler, metrics, df, feature_cols = train_model()


# ─── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🩺 DiabetesAI</p>
    <p class="hero-sub">Machine Learning · Early Detection System · Gradient Boosting Classifier</p>
    <span class="badge">▸ MODEL ACTIVE — GBM v2.0</span>
</div>
""", unsafe_allow_html=True)


# ─── Model Performance Metrics ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
        <div class="metric-label">Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{metrics['roc_auc']:.3f}</div>
        <div class="metric-label">ROC-AUC Score</div>
    </div>""", unsafe_allow_html=True)
with col3:
    prec = metrics['report']['1']['precision']
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{prec:.2f}</div>
        <div class="metric-label">Precision</div>
    </div>""", unsafe_allow_html=True)
with col4:
    rec = metrics['report']['1']['recall']
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{rec:.2f}</div>
        <div class="metric-label">Recall</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Patient Data Input")
    st.markdown("<div class='info-box'>Enter patient measurements to generate a diabetes risk prediction.</div>", unsafe_allow_html=True)
    st.markdown("---")

    pregnancies = st.slider("Pregnancies", 0, 17, 3,
                             help="Number of times pregnant")
    glucose = st.slider("Glucose (mg/dL)", 44, 200, 120,
                         help="Plasma glucose concentration (2hr oral glucose tolerance test)")
    blood_pressure = st.slider("Blood Pressure (mmHg)", 24, 122, 72,
                                help="Diastolic blood pressure")
    skin_thickness = st.slider("Skin Thickness (mm)", 7, 99, 29,
                                help="Triceps skin fold thickness")
    insulin = st.slider("Insulin (μU/ml)", 14, 850, 80,
                         help="2-Hour serum insulin")
    bmi = st.slider("BMI", 18.0, 67.0, 31.0, step=0.1,
                     help="Body mass index")
    dpf = st.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.47, step=0.01,
                     help="Family history score")
    age = st.slider("Age", 21, 81, 33,
                     help="Patient age in years")

    st.markdown("---")
    predict_btn = st.button("⚡ RUN PREDICTION")


# ─── Main Content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Feature Analysis", "📈 Model Insights"])


with tab1:
    if predict_btn:
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                  skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        col_r, col_g = st.columns([1, 1])

        with col_r:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <p class="result-title" style="color:#ff6b6b">⚠ HIGH RISK</p>
                    <p class="result-prob" style="color:#ff6b6b">{probability*100:.1f}%</p>
                    <p style="color:#94a3b8;font-size:0.9rem">Diabetes Risk Probability</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <p class="result-title" style="color:#00d4aa">✓ LOW RISK</p>
                    <p class="result-prob" style="color:#00d4aa">{probability*100:.1f}%</p>
                    <p style="color:#94a3b8;font-size:0.9rem">Diabetes Risk Probability</p>
                </div>""", unsafe_allow_html=True)

        with col_g:
            # Risk gauge chart
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#111827')
            ax.set_facecolor('#111827')

            # Background bar
            ax.barh(0, 100, height=0.4, color='#1e293b', zorder=1)
            # Risk fill
            color = '#ff6b6b' if probability > 0.5 else '#00d4aa'
            ax.barh(0, probability * 100, height=0.4, color=color, zorder=2)
            # Threshold line
            ax.axvline(50, color='#64748b', linewidth=1.5, linestyle='--', zorder=3)

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%', '25%', '50%\nThreshold', '75%', '100%'],
                                color='#64748b', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#1e293b')
            ax.set_title('Risk Probability Gauge', color='#e2e8f0',
                         fontfamily='monospace', fontsize=11, pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Patient summary table
        st.markdown("#### 📋 Patient Summary")
        summary_data = {
            'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                        'Insulin', 'BMI', 'Pedigree Function', 'Age'],
            'Value': [pregnancies, f'{glucose} mg/dL', f'{blood_pressure} mmHg',
                      f'{skin_thickness} mm', f'{insulin} μU/ml', f'{bmi:.1f}',
                      f'{dpf:.2f}', f'{age} yrs'],
            'Normal Range': ['0–3', '70–99 mg/dL', '60–80 mmHg', '10–40 mm',
                             '16–166 μU/ml', '18.5–24.9', '< 0.5', '–']
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="info-box" style="font-size:1rem; padding:2rem; text-align:center;">
            👈 &nbsp; Enter patient data in the sidebar and click <strong>RUN PREDICTION</strong>
        </div>""", unsafe_allow_html=True)


with tab2:
    st.markdown("#### 🔬 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#111827')
    ax.set_facecolor('#111827')
    colors = ['#00d4aa' if v > 0.15 else '#1e6b5e' for v in feat_df['Importance']]
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors, height=0.6)

    for bar, val in zip(bars, feat_df['Importance']):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='#94a3b8', fontsize=9)

    ax.set_xlabel('Importance Score', color='#64748b')
    ax.tick_params(colors='#94a3b8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.set_title('Feature Importance — Gradient Boosting', color='#e2e8f0',
                 fontfamily='monospace', fontsize=12, pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 📦 Dataset Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#111827')
        ax.set_facecolor('#111827')
        neg = df[df['Outcome'] == 0]['Glucose']
        pos = df[df['Outcome'] == 1]['Glucose']
        ax.hist(neg, bins=25, alpha=0.7, color='#00d4aa', label='No Diabetes')
        ax.hist(pos, bins=25, alpha=0.7, color='#ff6b6b', label='Diabetes')
        ax.set_title('Glucose Distribution', color='#e2e8f0', fontsize=11,
                     fontfamily='monospace')
        ax.tick_params(colors='#64748b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        ax.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#111827')
        ax.set_facecolor('#111827')
        neg = df[df['Outcome'] == 0]['BMI']
        pos = df[df['Outcome'] == 1]['BMI']
        ax.hist(neg, bins=25, alpha=0.7, color='#00d4aa', label='No Diabetes')
        ax.hist(pos, bins=25, alpha=0.7, color='#ff6b6b', label='Diabetes')
        ax.set_title('BMI Distribution', color='#e2e8f0', fontsize=11,
                     fontfamily='monospace')
        ax.tick_params(colors='#64748b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        ax.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


with tab3:
    st.markdown("#### 📈 ROC Curve")
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_curve

    X_all = df[feature_cols].values
    y_all = df['Outcome'].values
    X_all_s = scaler.transform(X_all)
    y_scores = model.predict_proba(X_all_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_all, y_scores)
    auc = roc_auc_score(y_all, y_scores)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#111827')
    ax.set_facecolor('#111827')
    ax.plot(fpr, tpr, color='#00d4aa', linewidth=2.5, label=f'GBM (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#334155', linewidth=1.5, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#00d4aa')
    ax.set_xlabel('False Positive Rate', color='#64748b')
    ax.set_ylabel('True Positive Rate', color='#64748b')
    ax.set_title('ROC Curve — Receiver Operating Characteristic', color='#e2e8f0',
                 fontfamily='monospace', fontsize=12, pad=15)
    ax.tick_params(colors='#64748b')
    ax.legend(facecolor='#1e293b', labelcolor='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 🎯 Confusion Matrix")
    from sklearn.metrics import confusion_matrix

    X_train_s2 = scaler.transform(X_all)
    y_pred_all = model.predict(X_train_s2)
    cm = confusion_matrix(y_all, y_pred_all)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#111827')
    ax.set_facecolor('#111827')
    im = ax.imshow(cm, cmap='YlGn', aspect='auto')
    labels = ['No Diabetes', 'Diabetes']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color='#94a3b8')
    ax.set_yticklabels(labels, color='#94a3b8')
    ax.set_xlabel('Predicted', color='#64748b')
    ax.set_ylabel('Actual', color='#64748b')
    ax.set_title('Confusion Matrix', color='#e2e8f0', fontfamily='monospace', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='#0a0f1e', fontsize=18, fontweight='bold',
                    fontfamily='monospace')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8rem; font-family:'Space Mono',monospace; padding:1rem 0">
    ⚕ DiabetesAI · Gradient Boosting Classifier · Pima Indians Dataset · Built for Portfolio
    <br><span style="color:#1e293b">─────────────────────────────────────────────────</span>
</div>
""", unsafe_allow_html=True)

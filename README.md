# 🩺 DiabetesAI — Disease Prediction System

> An end-to-end machine learning web application for early diabetes risk detection, built as a portfolio project demonstrating a complete ML pipeline: data ingestion, preprocessing, model training, evaluation, calibration, and interactive deployment.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-4fffb0?style=flat-square&logo=streamlit)](https://disease-prediction-nlmyueqbgc6mvdttcyljyk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Overview

DiabetesAI trains a **Gradient Boosting Classifier** on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (UCI) and serves real-time predictions through a dark-themed Streamlit interface. The app loads the real dataset on startup (falling back to a distribution-matched simulation if the network is unavailable), so every metric you see reflects genuine data.

---

## 🎯 Features

| Feature | Detail |
|---|---|
| **Real-time prediction** | Instant risk assessment from 8 clinical inputs |
| **Calibrated probability** | Outputs a risk percentage backed by a reliability diagram |
| **Feature contributions** | Per-patient bar chart showing how each feature shifts risk vs. the population average |
| **Global feature importance** | Ranked importance scores from the trained GBM |
| **Model comparison table** | Logistic Regression → Random Forest → GBM with selection rationale |
| **Calibration plot** | Reliability diagram + Brier Score verifying probability quality |
| **Input validation** | Flags physiologically implausible values (e.g. Glucose = 0) |
| **Dark UI** | Production-grade Streamlit interface with Syne + IBM Plex Mono typography |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~82% |
| ROC-AUC | ~0.88 |
| Precision | ~0.79 |
| Recall | ~0.73 |
| Brier Score | ~0.14 |

> Evaluated on a stratified 80/20 train/test split. The ROC curve and confusion matrix are available in the **Model Insights** tab; the calibration plot is in the **Calibration** tab.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/12amkh/diabetes-prediction-ai.git
cd diabetes-prediction-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run disease_prediction_app.py
```

The app will attempt to download the real UCI dataset on first run. An internet connection is recommended but not required — it falls back to a simulation automatically.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | GradientBoostingClassifier, StandardScaler, calibration utilities |
| `numpy` / `pandas` | Data processing |
| `matplotlib` | All visualisations |
| `streamlit` | Web app framework & deployment |

---

## 📁 Project Structure

```
diabetes-prediction-ai/
│
├── disease_prediction_app.py   # Main app — model training, UI, all tabs
├── requirements.txt            # Pinned dependencies
├── README.md                   # This file
└── screenshots/                # App screenshots for README / portfolio
```

---

## 📦 Requirements

```
streamlit>=1.28.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## 🧠 How It Works

**1. Data**
The app first tries to load the real Pima Indians Diabetes Dataset (768 samples, 8 features) via a public URL. If unavailable, it generates a statistically equivalent dataset from published distribution parameters. The sidebar displays which source was used.

**2. Preprocessing**
Zero-values in clinical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) are flagged as likely missing-value placeholders — a known quirk of the original dataset. `StandardScaler` normalises all features before training.

**3. Model**
`GradientBoostingClassifier` with 200 estimators, learning rate 0.08, max depth 4, and subsample 0.85. Chosen after comparing Logistic Regression and Random Forest on identical splits (see the Model Insights tab).

**4. Evaluation**
Train/test split (80/20, stratified). Metrics: Accuracy, ROC-AUC, Precision, Recall, Brier Score. Calibration is verified with a reliability diagram.

**5. Prediction & Explanation**
For each new patient, the app computes a risk probability and a feature contribution chart — showing how far each input deviates from the population mean, weighted by feature importance. This provides intuitive, per-patient explanability without external dependencies.

**6. Deployment**
Streamlit Community Cloud. The model is retrained in-memory on each cold start (~2 seconds) — no serialised model files required.

---

## 🌐 Deploy Your Own

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `disease_prediction_app.py` → Deploy

No configuration needed.

---

## 🗺️ Roadmap

- [ ] Replace simulated fallback with a bundled `diabetes.csv` for fully offline use
- [ ] Add SHAP waterfall plots for richer per-prediction explanations
- [ ] Hyperparameter tuning with `GridSearchCV` and cross-validated AUC
- [ ] Unit tests for preprocessing and prediction logic
- [ ] Dockerfile for containerised deployment

---

## ⚠️ Disclaimer

This tool is for **educational and portfolio purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Do not use this application to make real clinical decisions. The model has not been validated for clinical use and is not HIPAA or GDPR compliant.

---

## 👨‍💻 Author

Built by **12amkh** — Intelligent Systems Engineering Student

If you found this useful, consider starring the repo ⭐

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

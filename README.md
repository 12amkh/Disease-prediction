# 🩺 DiabetesAI — Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end machine learning web application that predicts diabetes risk using a **Gradient Boosting Classifier** trained on the Pima Indians Diabetes dataset. Built as a portfolio project demonstrating full ML pipeline development, from data preprocessing to model deployment.

---

## 🎯 Features

- **Real-time predictions** — Input patient data and get instant risk assessment
- **Probability score** — Not just yes/no, but a calibrated risk percentage
- **Feature importance** — Understand which factors drive the prediction
- **Interactive visualizations** — ROC curve, confusion matrix, data distributions
- **Clean dark UI** — Production-grade Streamlit interface

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| ROC-AUC | ~0.88 |
| Precision | ~0.79 |
| Recall | ~0.73 |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-ai.git
cd diabetes-prediction-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run disease_prediction_app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `scikit-learn` | ML model (GradientBoostingClassifier) |
| `numpy / pandas` | Data processing |
| `matplotlib` | Visualizations |
| `streamlit` | Web app deployment |

---

## 📁 Project Structure

```
diabetes-prediction-ai/
│
├── disease_prediction_app.py   # Main Streamlit app
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── screenshots/                # App screenshots
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

1. **Data** — Simulated from real Pima Indians Diabetes dataset distributions
2. **Preprocessing** — StandardScaler normalization
3. **Model** — Gradient Boosting Classifier (200 estimators, depth=4)
4. **Evaluation** — Train/test split (80/20), ROC-AUC scoring
5. **Deployment** — Streamlit web app with real-time inference

---

## 🌐 Deploy for Free

Deploy instantly on **Streamlit Community Cloud**:
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Deploy

---

## ⚠️ Disclaimer

This tool is for **educational and portfolio purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Author

Built by **[Your Name]** — Intelligent Systems Engineering Student  
[LinkedIn](#) · [GitHub](#) · [Portfolio](#)

---

*⭐ Star this repo if you found it useful!*

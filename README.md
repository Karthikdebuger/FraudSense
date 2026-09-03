# FraudSense 🛡️
**AI Fraud-Spike Detector — Razorpay AI Buildathon 2026**

FraudSense is an enterprise-grade, real-time fraud detection architecture designed to process high-velocity payment streams. Unlike standard AI wrappers that rely on slow and expensive LLMs for decision-making, FraudSense uses a **Cost-Optimized Hybrid Architecture** that combines the microsecond speed of tree ensembles with the explainability of Generative AI.

## 🚀 The Architecture

1. **The Brain (Fast ML)**: A highly-optimized ensemble of LightGBM and Random Forest makes split-second blocking decisions locally, easily handling thousands of transactions per second.
2. **The Zero-Day Catcher (Unsupervised AI)**: An Isolation Forest anomaly detector identifies brand new, never-before-seen fraud patterns (like sudden velocity spikes or amount anomalies) purely through statistical deviation.
3. **The Explainer (GenAI)**: Google Gemini is used *asynchronously* to translate mathematical SHAP values (feature importance) into plain English for Risk Ops analysts.

### 💰 Business-Aware Optimization
Most AI models are tuned for generic "accuracy". FraudSense was tuned using **Optuna** to minimize actual INR loss. The model mathematically balances the cost of a false positive (e.g., ₹50 for support time) against the cost of a false negative (e.g., ₹5,000 for a chargeback), finding the exact blocking threshold that saves Razorpay the most money.

## 🛠️ Project Structure
```text
RAZORAI/
├── data/                  # Simulated transaction datasets
├── models/                # Serialized ML models and feature engine states (.joblib)
├── scripts/               
│   ├── generate_data.py   # Simulates log-normal transactions & 5 fraud attack vectors
│   ├── train_model.py     # Optuna tuning, feature engineering, and model training
│   └── run_demo.py        # Boots the FastAPI server
└── src/
    ├── api/               # FastAPI endpoints & Server-Sent Events (SSE) feed
    ├── dashboard/         # Glassmorphism UI (Vanilla JS, HTML, CSS)
    ├── features/          # 22 engineered features without lookahead bias
    ├── llm/               # Gemini API explainer with rate-limit fallbacks
    └── models/            # LightGBM/Random Forest classifier logic
```

## ⚙️ Quickstart Guide

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the root directory and add your Gemini API key:
```env
GEMINI_API_KEY="your_api_key_here"
```

### 3. Generate Data & Train Model
```bash
python scripts/generate_data.py
python scripts/train_model.py
```

### 4. Start the Live Dashboard
```bash
python scripts/run_demo.py
```
Open your browser and navigate to **http://localhost:8000** to view the live streaming dashboard and dynamic SHAP feature explanations!

---
*Built for the Razorpay AI Buildathon.*

# FraudSense — Final Architecture

> **Track 2: AI Risk Manager** | Razorpay AI Buildathon 2026  
> **Deadline**: Sep 5 | **Build window**: Sep 2–4 (3 days)  
> **Cost**: ₹0 | **Solo build**

---

## What You're Building

A **fraud-spike detector** that:
1. Takes a merchant's transaction stream (synthetic)
2. Engineers 22 signal-based features (your denoising intuition)
3. Classifies each transaction as fraud/legit (LightGBM + Random Forest)
4. Explains each flag in plain English (Gemini 2.0 Flash, free)
5. Reports honest precision/recall/FP-cost metrics
6. Shows everything in a live dashboard

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   ① DATA GENERATOR          ② FEATURE ENGINE                  │
│   ┌────────────────┐        ┌─────────────────────┐           │
│   │ 10K synthetic  │───────▶│ 22 engineered       │           │
│   │ transactions   │        │ features per txn     │           │
│   │                │        │                     │           │
│   │ 5 fraud types: │        │ Velocity (4)        │           │
│   │ • Velocity     │        │ Amount (4)          │           │
│   │ • Amount       │        │ Temporal (4)        │           │
│   │ • Geo mismatch │        │ Behavioral (4)      │           │
│   │ • Card testing │        │ Geo (3)             │           │
│   │ • Return fraud │        │ Anomaly scores (3)  │           │
│   └────────────────┘        └──────────┬──────────┘           │
│                                        │                      │
│                                        ▼                      │
│   ③ ML CLASSIFIER            ④ LLM EXPLAINER                 │
│   ┌────────────────┐        ┌─────────────────────┐           │
│   │ LightGBM (0.6) │        │ Gemini 2.0 Flash    │           │
│   │ + RF (0.4)     │───────▶│ (free tier)         │           │
│   │ ensemble       │        │                     │           │
│   │                │        │ Input: top 3        │           │
│   │ Cost-sensitive │        │   features + score  │           │
│   │ threshold      │        │ Output: 1-line      │           │
│   │ (not 0.5)      │        │   explanation       │           │
│   │                │        │                     │           │
│   │ Output:        │        │ Fallback: template  │           │
│   │ • fraud_score  │        │   if API is down    │           │
│   │ • is_flagged   │        └─────────┬───────────┘           │
│   │ • top_features │                  │                       │
│   └────────┬───────┘                  │                       │
│            │                          │                       │
│            ▼                          ▼                       │
│   ⑤ FASTAPI BACKEND                                          │
│   ┌───────────────────────────────────────────────────────┐   │
│   │ POST /api/predict   → score 1 transaction             │   │
│   │ POST /api/batch     → score CSV, return metrics       │   │
│   │ GET  /api/metrics   → precision/recall/FP cost        │   │
│   │ GET  /api/explain/  → explanation for a flagged txn   │   │
│   │ GET  /api/stream    → SSE live transaction feed       │   │
│   │ GET  /api/health    → health check                    │   │
│   └───────────────────────────┬───────────────────────────┘   │
│                               │                               │
│                               ▼                               │
│   ⑥ DASHBOARD (HTML/CSS/JS + Chart.js)                       │
│   ┌───────────────────────────────────────────────────────┐   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐ │   │
│   │  │Precision│ │ Recall  │ │   F1    │ │ FP Cost ₹  │ │   │
│   │  │  0.87   │ │  0.92   │ │  0.89   │ │ ₹4,200/1K  │ │   │
│   │  └─────────┘ └─────────┘ └─────────┘ └────────────┘ │   │
│   │                                                       │   │
│   │  Live Transaction Feed (SSE-powered, auto-scrolling)  │   │
│   │  ┌────┬─────────┬────────┬─────┬───────┬────────────┐│   │
│   │  │ ID │ Amount  │ Method │Score│ Flag  │ Explanation ││   │
│   │  ├────┼─────────┼────────┼─────┼───────┼────────────┤│   │
│   │  │4821│ ₹340    │ UPI    │0.02 │  ✅   │ —          ││   │
│   │  │4822│ ₹1,200  │ Card   │0.11 │  ✅   │ —          ││   │
│   │  │4823│ ₹18,400 │ Card   │0.94 │  🔴   │ "7 txns.."││   │
│   │  └────┴─────────┴────────┴─────┴───────┴────────────┘│   │
│   │                                                       │   │
│   │  PR Curve | Feature Importance | Confusion Matrix     │   │
│   └───────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### ① Data Generator — `src/data/data_generator.py`

Generates **10,000 labeled transactions** with ~5% fraud rate.

**Normal transaction fields:**
```
transaction_id, timestamp, amount, currency, payment_method,
card_id, ip_address, device_fingerprint, merchant_id,
customer_id, customer_city, customer_state, is_fraud (label)
```

**5 fraud injection patterns:**

| Pattern | What It Looks Like | Count |
|---|---|---|
| Velocity spike | 5-15 txns from same card/IP in <30s | ~100 |
| Amount anomaly | Amount 20-100x above merchant median | ~100 |
| Geo mismatch | IP location ≠ cardholder's registered city | ~100 |
| Card testing | Rapid ₹1-10 transactions (stolen card validation) | ~100 |
| Return fraud | Buy → immediate return, slight amount mismatch | ~100 |

**Split**: 80% train / 20% test (stratified — fraud ratio preserved in both)

---

### ② Feature Engine — `src/features/feature_engine.py`

**22 features across 6 categories:**

**Velocity (4)** — *How fast are transactions coming?*
| # | Feature | Description |
|---|---|---|
| 1 | `txn_count_30s` | Transactions from same card/IP in last 30 sec |
| 2 | `txn_count_5m` | Transactions from same card/IP in last 5 min |
| 3 | `txn_count_1h` | Transactions from same card/IP in last 1 hour |
| 4 | `velocity_ratio` | `txn_count_30s / txn_count_1h` — spike indicator |

**Amount (4)** — *How unusual is the amount?*
| # | Feature | Description |
|---|---|---|
| 5 | `amount_zscore` | Z-score vs merchant's historical distribution |
| 6 | `amount_ratio_median` | `amount / merchant_median` |
| 7 | `amount_ratio_p95` | `amount / merchant_95th_percentile` |
| 8 | `rolling_amount_std_5` | Std dev of last 5 transactions |

**Temporal (4)** — *When is it happening?*
| # | Feature | Description |
|---|---|---|
| 9 | `hour_sin` | `sin(2π × hour / 24)` — cyclical encoding |
| 10 | `hour_cos` | `cos(2π × hour / 24)` — cyclical encoding |
| 11 | `is_business_hours` | 1 if 9 AM–6 PM, else 0 |
| 12 | `time_since_last_txn` | Seconds since previous txn from same entity |

**Behavioral (4)** — *How unusual is the pattern?*
| # | Feature | Description |
|---|---|---|
| 13 | `unique_cards_per_ip_1h` | Distinct cards from same IP in 1 hour |
| 14 | `unique_ips_per_card_1h` | Distinct IPs for same card in 1 hour |
| 15 | `payment_method_entropy` | Shannon entropy of payment methods used |
| 16 | `device_changed` | 1 if device ≠ customer's usual device |

**Geo (3)** — *Where is it coming from?*
| # | Feature | Description |
|---|---|---|
| 17 | `geo_distance_km` | Distance from cardholder's usual location |
| 18 | `is_geo_mismatch` | 1 if txn city ≠ registered city |
| 19 | `is_cross_border` | 1 if international |

**Anomaly / Derived (3)** — *Your denoising insight*
| # | Feature | Description |
|---|---|---|
| 20 | `isolation_forest_score` | Unsupervised anomaly score |
| 21 | `local_outlier_factor` | LOF score in merchant's txn neighborhood |
| 22 | `residual_score` | Predicted "clean" pattern minus actual (signal vs noise) |

> Feature 22 (`residual_score`) is your **pitch video differentiator** — it maps your image-restoration experience directly to fraud detection.

---

### ③ ML Classifier — `src/models/classifier.py`

```
Training data (8K txns, 22 features)
         │
         ├──▶ LightGBM (weight: 0.6)
         │         │
         │         ▼
         ├──▶ Random Forest (weight: 0.4)
         │         │
         │         ▼
         └──▶ Weighted Soft Vote ──▶ fraud_score (0.0 to 1.0)
                                          │
                                          ▼
                                   Cost-Optimized Threshold
                                   FP cost: ₹50 (support agent time)
                                   FN cost: ₹5,000 (chargeback)
                                   Threshold ≠ 0.5
                                          │
                                          ▼
                                   is_flagged (True/False)
```

**Evaluation outputs** (`src/models/evaluation.py`):
- Precision, Recall, F1 at optimized threshold
- Precision-Recall curve with AUC
- Confusion matrix with raw counts
- FP cost per 1,000 transactions (in ₹)
- Per-fraud-type recall breakdown
- Top 10 feature importance

---

### ④ LLM Explainer — `src/llm/explainer.py`

```python
# Input (from classifier)
{
  "fraud_score": 0.94,
  "top_features": ["txn_count_30s=7", "amount_ratio_median=47.2", "velocity_ratio=0.88"]
}

# Prompt to Gemini
"A transaction was flagged with score 0.94. Top contributing factors:
 txn_count_30s=7, amount_ratio_median=47.2, velocity_ratio=0.88.
 Write one sentence explaining why this is suspicious. Use specific numbers."

# Output
"7 transactions from the same IP in 12 seconds with an average amount
 47x higher than the merchant baseline — consistent with a card-testing burst."
```

**Fallback** (if API is down):
```python
f"Flagged: {top_features[0]} is abnormally high ({value}), "
f"combined with {top_features[1]} suggesting {fraud_type} pattern."
```

**Config**: `GEMINI_API_KEY` in `.env` | Model: `gemini-2.0-flash` | Free: 15 RPM

---

### ⑤ API — `src/api/main.py`

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/api/predict` | POST | Single transaction JSON | `{ fraud_score, is_flagged, explanation, top_features }` |
| `/api/batch` | POST | CSV file upload | `{ results[], precision, recall, f1, fp_cost }` |
| `/api/metrics` | GET | — | Full metrics object |
| `/api/explain/{id}` | GET | Transaction ID | `{ explanation }` |
| `/api/stream` | GET | — | SSE stream of scored transactions |
| `/api/health` | GET | — | `{ status: "ok" }` |

---

### ⑥ Dashboard — `src/dashboard/`

**Colors**: Background `#0a0a0f` | Cards `#1a1a2e` | Razorpay blue `#267DFF` | Safe `#00C853` | Danger `#FF4444`

**Sections**:
1. Header with project name
2. 4 metric cards (Precision, Recall, F1, FP Cost)
3. Live transaction feed (SSE, color-coded rows)
4. Flagged transactions table with explanations
5. PR Curve + Feature Importance + Confusion Matrix (Chart.js)

---

## 3-Day Build Schedule

### Day 1 (Sep 2) — Data + ML Pipeline
| Block | Hours | What You Build | Done When |
|---|---|---|---|
| Morning | 3h | Setup, venv, Gemini key, data generator | `generate_data.py` produces CSVs |
| Afternoon | 4h | Feature engine (22 features) + tests | Features computed, tests pass |
| Evening | 3h | LightGBM + RF + Optuna + threshold | Precision ≥ 0.85, Recall ≥ 0.90 |

### Day 2 (Sep 3) — API + Explainer + Dashboard
| Block | Hours | What You Build | Done When |
|---|---|---|---|
| Morning | 3h | FastAPI (all 6 endpoints) | Endpoints return correct data |
| Afternoon | 2h | Gemini explainer + fallback | Explanations working |
| Afternoon | 3h | Dashboard HTML/CSS/JS + live feed | Dashboard loads with data |
| Evening | 2h | Charts (PR, features, confusion matrix) | `run_demo.py` → full demo |

### Day 3 (Sep 4) — Polish + Submit
| Block | Hours | What You Build | Done When |
|---|---|---|---|
| Morning | 2h | Bug fixes, edge cases | Demo stable |
| Midday | 2h | README.md + screenshots | README complete |
| Afternoon | 2h | Record 5-min pitch video | Video ready |
| Evening | 1h | Push GitHub, clone-test, submit | ✅ Submitted |

### Sep 5 — Buffer only.

---

## Tech Stack (All Free)

```
pandas numpy scikit-learn lightgbm optuna     ← ML pipeline
fastapi uvicorn pydantic python-dotenv        ← API
google-genai                                   ← LLM (free tier)
Chart.js (CDN)                                 ← Dashboard charts
pytest                                         ← Tests
```

---

## First Command Tomorrow Morning

```bash
cd c:\Users\karth\OneDrive\Documents\RAZORAI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then start coding `src/data/data_generator.py`.

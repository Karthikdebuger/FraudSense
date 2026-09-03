"""
FraudSense FastAPI Backend

Endpoints:
    POST /api/predict    — Score a single transaction
    POST /api/batch      — Score a CSV batch and return metrics
    GET  /api/metrics    — Get evaluation metrics
    GET  /api/explain/{id} — Get explanation for a flagged transaction
    GET  /api/stream     — SSE live transaction feed
    GET  /api/health     — Health check
    GET  /                — Serve the dashboard
"""

import sys
import json
import asyncio
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import io

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import FraudClassifier
from src.features.feature_engine import compute_features, get_feature_columns, load_state
from src.models.evaluation import evaluate_model
from src.api.middleware import RequestLoggingMiddleware

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudsense")

# ─── App Setup ───

app = FastAPI(
    title="FraudSense API",
    description="AI Fraud-Spike Detector — Razorpay AI Buildathon 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ─── Global State ───

classifier: Optional[FraudClassifier] = None
cached_metrics: Optional[dict] = None
scored_transactions: list[dict] = []  # In-memory store for demo
explainer_available: bool = False
_explainer = None


def _load_model():
    """Load the trained classifier on startup."""
    global classifier, cached_metrics, explainer_available, _explainer

    model_dir = PROJECT_ROOT / "models"
    metrics_file = model_dir / "evaluation_metrics.json"

    if (model_dir / "lgb_model.joblib").exists():
        classifier = FraudClassifier()
        classifier.load(str(model_dir))
        load_state(str(model_dir))
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️  No trained model found. Run train_model.py first.")

    if metrics_file.exists():
        with open(metrics_file) as f:
            cached_metrics = json.load(f)
        logger.info("✅ Cached metrics loaded")

    # Try to load explainer
    try:
        from src.llm.explainer import check_api_available, explain_transaction
        _explainer = explain_transaction
        explainer_available = check_api_available()
        logger.info(f"✅ LLM explainer loaded (API available: {explainer_available})")
    except Exception as e:
        logger.warning(f"⚠️  LLM explainer not available: {e}")
        explainer_available = False


@app.on_event("startup")
async def startup():
    _load_model()


# ─── Request/Response Models ───

class TransactionInput(BaseModel):
    transaction_id: str = "TXN-LIVE-001"
    timestamp: str = "2026-09-02T18:00:00"
    amount: float = 15000.0
    currency: str = "INR"
    payment_method: str = "card"
    card_id: str = "CARD-001"
    ip_address: str = "10.0.1.1"
    device_fingerprint: str = "DEV-001"
    merchant_id: str = "MERCHANT-001"
    customer_id: str = "CUST-001"
    customer_city: str = "Mumbai"
    customer_state: str = "Maharashtra"


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_flagged: bool
    explanation: Optional[str] = None
    top_features: list[str] = []


# ─── Dashboard ───

DASHBOARD_DIR = PROJECT_ROOT / "src" / "dashboard"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML."""
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not built yet. Run on Day 2.</h1>")


@app.get("/style.css")
async def serve_css():
    css_path = DASHBOARD_DIR / "style.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(404, "CSS not found")


@app.get("/app.js")
async def serve_js():
    js_path = DASHBOARD_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(404, "JS not found")


# ─── API Endpoints ───

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single(txn: TransactionInput):
    """Score a single transaction for fraud."""
    if classifier is None:
        raise HTTPException(503, "Model not loaded. Run train_model.py first.")

    # Convert to DataFrame for feature computation
    txn_dict = txn.model_dump()
    df = pd.DataFrame([txn_dict])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_fraud"] = 0  # Placeholder
    df["fraud_type"] = None

    # Compute features
    df_features = compute_features(df, fit_unsupervised=False)
    feature_cols = get_feature_columns()
    X = df_features[feature_cols].values

    # Predict
    result = classifier.predict(X)
    fraud_score = float(result["fraud_scores"][0])
    is_flagged = bool(result["is_flagged"][0])
    top_features = result["top_features_per_sample"][0]

    # Get explanation
    explanation = None
    if is_flagged and _explainer:
        try:
            explanation = _explainer(
                fraud_score=fraud_score,
                top_features=top_features,
            )
        except Exception as e:
            logger.warning(f"Explanation failed: {e}")
            explanation = f"Flagged with score {fraud_score:.2f}. Key factors: {', '.join(top_features)}"

    response = PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_score=round(fraud_score, 4),
        is_flagged=is_flagged,
        explanation=explanation,
        top_features=top_features,
    )

    # Store for stream/dashboard
    scored_transactions.append(response.model_dump())
    if len(scored_transactions) > 1000:
        scored_transactions.pop(0)

    return response


@app.post("/api/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Score a CSV batch and return metrics."""
    if classifier is None:
        raise HTTPException(503, "Model not loaded.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Ensure required columns
    required = ["timestamp", "amount", "card_id", "merchant_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    if "fraud_type" not in df.columns:
        df["fraud_type"] = None

    # Compute features and predict
    df_features = compute_features(df, fit_unsupervised=False)
    feature_cols = get_feature_columns()
    X = df_features[feature_cols].values
    y_true = df_features["is_fraud"].values

    result = classifier.predict(X)

    # Evaluate
    metrics = evaluate_model(
        y_true=y_true,
        y_scores=result["fraud_scores"],
        y_pred=result["is_flagged"],
        fraud_types=df_features.get("fraud_type"),
    )

    # Build results
    results = []
    for i in range(len(df)):
        results.append({
            "transaction_id": df.iloc[i].get("transaction_id", f"TXN-{i}"),
            "fraud_score": round(float(result["fraud_scores"][i]), 4),
            "is_flagged": bool(result["is_flagged"][i]),
            "top_features": result["top_features_per_sample"][i],
        })

    return {
        "total_transactions": len(df),
        "flagged_count": int(result["is_flagged"].sum()),
        "metrics": metrics,
        "results": results[:100],  # Cap at 100 for response size
    }


@app.get("/api/metrics")
async def get_metrics():
    """Return cached evaluation metrics."""
    if cached_metrics is None:
        raise HTTPException(404, "No metrics available. Run train_model.py first.")
    return cached_metrics


@app.get("/api/explain/{transaction_id}")
async def get_explanation(transaction_id: str):
    """Get explanation for a specific flagged transaction."""
    for txn in scored_transactions:
        if txn["transaction_id"] == transaction_id:
            return {"transaction_id": transaction_id, "explanation": txn.get("explanation")}
    raise HTTPException(404, f"Transaction {transaction_id} not found")


@app.get("/api/stream")
async def stream_transactions():
    """SSE stream of live scored transactions for the dashboard."""

    async def event_generator():
        # First, send any existing scored transactions
        for txn in scored_transactions[-50:]:
            yield {"event": "transaction", "data": json.dumps(txn)}

        # Then simulate live feed from test data with full history preserved
        test_path = PROJECT_ROOT / "data" / "test.csv"
        train_path = PROJECT_ROOT / "data" / "train.csv"
        
        if test_path.exists() and train_path.exists() and classifier is not None:
            # For the demo, we compute features on train+test together so that 
            # customer history (like most common city) is preserved for the stream.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Take a chunk of test_df to simulate stream (200 rows)
            stream_raw = test_df.iloc[:200].copy()
            
            # Concatenate train + stream chunk to compute features with history
            df_full = pd.concat([train_df, stream_raw]).sort_values("timestamp").reset_index(drop=True)
            df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
            
            # Compute features globally to preserve history dictionaries
            df_features = compute_features(df_full, fit_unsupervised=False)
            
            # Extract just the stream portion
            stream_features = df_features.iloc[-len(stream_raw):].copy()
            stream_raw = df_full.iloc[-len(stream_raw):].copy()

            # Process in small batches for the SSE delay
            batch_size = 5
            for start in range(0, len(stream_raw), batch_size):
                batch_raw = stream_raw.iloc[start:start + batch_size]
                batch_feat = stream_features.iloc[start:start + batch_size]
                
                feature_cols = get_feature_columns()
                X = batch_feat[feature_cols].values
                result = classifier.predict(X)

                for j in range(len(batch_raw)):
                    txn_data = {
                        "transaction_id": batch_raw.iloc[j].get("transaction_id", f"TXN-{start+j}"),
                        "amount": float(batch_raw.iloc[j]["amount"]),
                        "payment_method": str(batch_raw.iloc[j].get("payment_method", "card")),
                        "fraud_score": round(float(result["fraud_scores"][j]), 4),
                        "is_flagged": bool(result["is_flagged"][j]),
                        "top_features": result["top_features_per_sample"][j],
                        "customer_city": str(batch_raw.iloc[j].get("customer_city", "")),
                    }

                    # Get explanation for flagged transactions
                    if txn_data["is_flagged"] and _explainer:
                        try:
                            txn_data["explanation"] = _explainer(
                                fraud_score=txn_data["fraud_score"],
                                top_features=txn_data["top_features"],
                            )
                        except Exception:
                            txn_data["explanation"] = f"Flagged: {', '.join(txn_data['top_features'])}"
                    else:
                        txn_data["explanation"] = None

                    yield {"event": "transaction", "data": json.dumps(txn_data)}
                    scored_transactions.append(txn_data)

                await asyncio.sleep(0.5)  # Simulate real-time delay

        # Send completion event
        yield {"event": "complete", "data": json.dumps({"message": "Stream complete"})}

    return EventSourceResponse(event_generator())


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "metrics_available": cached_metrics is not None,
        "explainer_available": explainer_available,
        "scored_transactions": len(scored_transactions),
    }

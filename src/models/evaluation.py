"""
FraudSense Evaluation Suite

Generates comprehensive metrics:
- Precision, Recall, F1 at optimized threshold
- Precision-Recall curve with AUC
- Confusion matrix
- FP cost per 1,000 transactions (in INR)
- Per-fraud-type recall breakdown
- Top 10 feature importance
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    classification_report,
)

# Cost parameters
FP_COST = 50
FN_COST = 5000


def evaluate_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    fraud_types: Optional[pd.Series] = None,
    feature_importance: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run full evaluation suite.

    Args:
        y_true: True binary labels
        y_scores: Predicted probability scores (0-1)
        y_pred: Predicted binary labels (after threshold)
        fraud_types: Series of fraud type labels for per-type breakdown
        feature_importance: DataFrame with 'feature' and 'importance' columns

    Returns:
        dict with all metrics
    """
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # FP cost per 1000 transactions
    n_total = len(y_true)
    fp_per_1000 = (fp / n_total) * 1000
    fp_cost_per_1000 = fp_per_1000 * FP_COST
    fn_per_1000 = (fn / n_total) * 1000
    fn_cost_per_1000 = fn_per_1000 * FN_COST
    total_cost_per_1000 = fp_cost_per_1000 + fn_cost_per_1000

    # PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recalls, precisions)

    results = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "cost_analysis": {
            "fp_cost_per_1000_txns": round(float(fp_cost_per_1000), 2),
            "fn_cost_per_1000_txns": round(float(fn_cost_per_1000), 2),
            "total_cost_per_1000_txns": round(float(total_cost_per_1000), 2),
            "fp_unit_cost": FP_COST,
            "fn_unit_cost": FN_COST,
        },
        "pr_curve": {
            "precisions": [round(float(p), 4) for p in precisions[::max(1, len(precisions)//100)]],
            "recalls": [round(float(r), 4) for r in recalls[::max(1, len(recalls)//100)]],
            "auc": round(float(pr_auc), 4),
        },
        "total_samples": int(n_total),
        "fraud_count": int(y_true.sum()),
        "fraud_rate": round(float(y_true.mean()), 4),
    }

    # Per-fraud-type recall
    if fraud_types is not None:
        type_recall = {}
        for ftype in fraud_types.dropna().unique():
            if ftype and ftype != "None":
                mask = fraud_types == ftype
                if mask.sum() > 0:
                    type_recall[ftype] = {
                        "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                        "count": int(mask.sum()),
                        "detected": int(y_pred[mask].sum()),
                    }
        results["per_fraud_type_recall"] = type_recall

    # Feature importance
    if feature_importance is not None:
        results["feature_importance"] = feature_importance.head(10).to_dict("records")

    return results


def print_evaluation_report(metrics: dict) -> None:
    """Print a formatted evaluation report to console."""
    print(f"\n{'='*60}")
    print(f"  FraudSense Evaluation Report")
    print(f"{'='*60}")

    print(f"\n📊 Classification Metrics:")
    print(f"   Precision:  {metrics['precision']:.4f}")
    print(f"   Recall:     {metrics['recall']:.4f}")
    print(f"   F1 Score:   {metrics['f1_score']:.4f}")
    print(f"   PR AUC:     {metrics['pr_curve']['auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\n📋 Confusion Matrix:")
    print(f"                 Predicted Legit  Predicted Fraud")
    print(f"   Actual Legit:    {cm['true_negatives']:>6}          {cm['false_positives']:>6}")
    print(f"   Actual Fraud:    {cm['false_negatives']:>6}          {cm['true_positives']:>6}")

    cost = metrics["cost_analysis"]
    print(f"\n💰 Cost Analysis (per 1,000 transactions):")
    print(f"   FP cost (₹{FP_COST}/FP):  ₹{cost['fp_cost_per_1000_txns']:>10,.2f}")
    print(f"   FN cost (₹{FN_COST}/FN):  ₹{cost['fn_cost_per_1000_txns']:>10,.2f}")
    print(f"   Total cost:        ₹{cost['total_cost_per_1000_txns']:>10,.2f}")

    if "per_fraud_type_recall" in metrics:
        print(f"\n🎯 Per-Fraud-Type Recall:")
        for ftype, data in metrics["per_fraud_type_recall"].items():
            bar = "█" * int(data['recall'] * 20)
            print(f"   {ftype:20s} {data['recall']:.3f} {bar} ({data['detected']}/{data['count']})")

    if "feature_importance" in metrics:
        print(f"\n🏆 Top 10 Features:")
        for i, feat in enumerate(metrics["feature_importance"]):
            bar = "█" * int(feat['importance'] / max(f['importance'] for f in metrics['feature_importance']) * 20)
            print(f"   {i+1:2d}. {feat['feature']:25s} {feat['importance']:>6.0f} {bar}")

    print(f"\n{'='*60}")


def save_metrics(metrics: dict, output_dir: str = "models") -> None:
    """Save evaluation metrics to JSON."""
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    with open(path / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {path}/evaluation_metrics.json")

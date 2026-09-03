"""
FraudSense Training Script

Usage:
    python scripts/train_model.py

Loads generated data, computes features, trains the ensemble classifier
with Optuna tuning, evaluates on held-out test set, and saves everything.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engine import compute_features, get_feature_columns, save_state
from src.models.classifier import FraudClassifier
from src.models.evaluation import evaluate_model, print_evaluation_report, save_metrics


def main():
    total_start = time.time()

    print("\n" + "=" * 60)
    print("  FraudSense — Training Pipeline")
    print("=" * 60)

    # Load Data
    data_dir = PROJECT_ROOT / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print("❌ Data not found! Run this first:")
        print("   python scripts/generate_data.py")
        sys.exit(1)

    print("\n📂 Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"   Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # Compute Features
    print("\n🔧 Computing features on training data...")
    t0 = time.time()
    train_df = compute_features(train_df, fit_unsupervised=True)
    print(f"   Done in {time.time() - t0:.1f}s")

    print("\n🔧 Computing features on test data...")
    t0 = time.time()
    test_df = compute_features(test_df, fit_unsupervised=False)
    print(f"   Done in {time.time() - t0:.1f}s")

    # Prepare Train/Test Matrices
    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols].values
    y_train = train_df["is_fraud"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["is_fraud"].values

    print(f"\n📊 Feature matrix: {X_train.shape[1]} features")
    print(f"   Train: {len(y_train)} samples ({y_train.sum()} fraud, {y_train.mean()*100:.1f}%)")
    print(f"   Test:  {len(y_test)} samples ({y_test.sum()} fraud, {y_test.mean()*100:.1f}%)")

    # Train Classifier
    classifier = FraudClassifier()
    train_summary = classifier.train(
        X=X_train,
        y=y_train,
        feature_columns=feature_cols,
        n_optuna_trials=50,
    )

    # Evaluate on Test Set
    print("\n📈 Evaluating on test set...")
    predictions = classifier.predict(X_test)

    fraud_types = test_df["fraud_type"] if "fraud_type" in test_df.columns else None
    feature_importance = classifier.get_feature_importance()

    metrics = evaluate_model(
        y_true=y_test,
        y_scores=predictions["fraud_scores"],
        y_pred=predictions["is_flagged"],
        fraud_types=fraud_types,
        feature_importance=feature_importance,
    )

    print_evaluation_report(metrics)

    # Serialize Models and Metrics
    model_dir = str(PROJECT_ROOT / "models")
    classifier.save(model_dir)
    save_state(model_dir)
    save_metrics(metrics, model_dir)

    elapsed = time.time() - total_start
    print(f"\n⏱️  Total training time: {elapsed:.1f}s")
    print(f"✅ Training complete! Model and metrics saved to models/\n")


if __name__ == "__main__":
    main()

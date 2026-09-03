"""
FraudSense ML Classifier

Ensemble of LightGBM (0.6 weight) + Random Forest (0.4 weight) with
cost-sensitive threshold optimization.

FP cost: ₹50 (support agent time per false alarm)
FN cost: ₹5,000 (chargeback loss per missed fraud)
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Cost parameters (in INR)
FP_COST = 50      # Cost of investigating a false positive
FN_COST = 5000    # Cost of missing a real fraud (chargeback)


class FraudClassifier:
    """
    Ensemble fraud classifier with cost-sensitive threshold optimization.
    
    Architecture:
        LightGBM (weight=0.6) + RandomForest (weight=0.4) → soft vote → threshold
    """

    def __init__(self):
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.threshold: float = 0.5
        self.lgb_weight: float = 0.6
        self.rf_weight: float = 0.4
        self.feature_columns: list[str] = []
        self.best_params: dict = {}

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective for LightGBM hyperparameter tuning."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        }

        model = lgb.LGBMClassifier(
            **params,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        # 3-fold cross-validation predictions
        y_proba = cross_val_predict(model, X, y, cv=3, method="predict_proba")[:, 1]

        # Optimize for cost-weighted metric
        best_cost = float("inf")
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (y_proba >= t).astype(int)
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            total_cost = fp * FP_COST + fn * FN_COST
            best_cost = min(best_cost, total_cost)

        return best_cost

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: list[str],
        n_optuna_trials: int = 50,
    ) -> dict:
        """
        Train the ensemble classifier.

        Args:
            X: Feature matrix (n_samples, 22)
            y: Binary labels (0=legit, 1=fraud)
            feature_columns: List of feature names
            n_optuna_trials: Number of Optuna tuning trials

        Returns:
            dict with training summary
        """
        self.feature_columns = feature_columns
        print(f"\n{'='*60}")
        print(f"Training FraudSense Classifier")
        print(f"{'='*60}")
        print(f"Samples: {len(y)} | Fraud: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"Features: {len(feature_columns)}")

        # Step 1: Optuna tuning for LightGBM
        print(f"\n🔍 Running Optuna tuning ({n_optuna_trials} trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=n_optuna_trials)

        self.best_params = study.best_params
        print(f"   Best cost: ₹{study.best_value:,.0f}")
        print(f"   Best params: {json.dumps(self.best_params, indent=2)}")

        # Step 2: Train LightGBM with best params
        print("\n📊 Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            **self.best_params,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        self.lgb_model.fit(X, y)

        # Step 3: Train Random Forest
        print("🌲 Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=self.best_params.get("max_depth", 7),
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X, y)

        # Step 4: Cost-sensitive threshold optimization
        print("⚖️  Optimizing threshold...")
        self.threshold = self._optimize_threshold(X, y)
        print(f"   Optimal threshold: {self.threshold:.3f}")

        return {
            "best_optuna_cost": study.best_value,
            "threshold": self.threshold,
            "n_features": len(feature_columns),
            "n_samples": len(y),
            "fraud_rate": float(y.mean()),
        }

    def _optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find threshold that minimizes total cost (FP*₹50 + FN*₹5000)."""
        probas = self._ensemble_predict_proba(X)

        best_threshold = 0.5
        best_cost = float("inf")
        cost_curve = []

        for t in np.arange(0.05, 0.95, 0.01):
            preds = (probas >= t).astype(int)
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            total_cost = fp * FP_COST + fn * FN_COST
            cost_curve.append({"threshold": round(t, 2), "cost": total_cost, "fp": int(fp), "fn": int(fn)})

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = t

        return round(best_threshold, 3)

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble probability scores."""
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        return self.lgb_weight * lgb_proba + self.rf_weight * rf_proba

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict fraud scores and flags for input features.

        Returns:
            dict with 'fraud_scores', 'is_flagged', 'top_features_per_sample'
        """
        probas = self._ensemble_predict_proba(X)
        flags = (probas >= self.threshold).astype(int)

        # Get top contributing features per sample using LightGBM feature importance
        top_features = self._get_top_features_per_sample(X)

        return {
            "fraud_scores": probas,
            "is_flagged": flags,
            "top_features_per_sample": top_features,
        }

    def _get_top_features_per_sample(self, X: np.ndarray, top_k: int = 3) -> list[list[str]]:
        """Get top-k contributing features for each sample using LightGBM SHAP values."""
        # Calculate per-sample feature contributions (SHAP values)
        contributions = self.lgb_model.predict(X, pred_contrib=True)
        feature_contribs = contributions[:, :-1]
        
        result = []
        for i in range(len(X)):
            # Get indices of the top_k features with the highest POSITIVE contribution to fraud
            top_indices = np.argsort(feature_contribs[i])[-top_k:][::-1]
            
            features = []
            for idx in top_indices:
                name = self.feature_columns[idx]
                raw_value = X[i, idx]
                features.append(f"{name}={raw_value:.2f}")
            result.append(features)
        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from LightGBM model."""
        importances = self.lgb_model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, model_dir: str = "models") -> None:
        """Save trained models and config to disk."""
        path = Path(model_dir)
        path.mkdir(exist_ok=True)

        joblib.dump(self.lgb_model, path / "lgb_model.joblib")
        joblib.dump(self.rf_model, path / "rf_model.joblib")

        config = {
            "threshold": self.threshold,
            "lgb_weight": self.lgb_weight,
            "rf_weight": self.rf_weight,
            "feature_columns": self.feature_columns,
            "best_params": self.best_params,
        }
        with open(path / "classifier_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Models saved to {path}/")

    def load(self, model_dir: str = "models") -> None:
        """Load trained models and config from disk."""
        path = Path(model_dir)

        self.lgb_model = joblib.load(path / "lgb_model.joblib")
        self.rf_model = joblib.load(path / "rf_model.joblib")

        with open(path / "classifier_config.json", "r") as f:
            config = json.load(f)

        self.threshold = config["threshold"]
        self.lgb_weight = config["lgb_weight"]
        self.rf_weight = config["rf_weight"]
        self.feature_columns = config["feature_columns"]
        self.best_params = config["best_params"]

        print(f"✅ Models loaded from {path}/")

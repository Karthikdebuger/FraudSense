"""
FraudSense Feature Engineering Pipeline

Computes 22 signal-based features from raw transaction data across 6 categories:
- Velocity (4): Transaction frequency / spike detection
- Amount (4): Statistical anomaly in transaction amounts
- Temporal (4): Time-of-day and inter-transaction timing
- Behavioral (4): Usage pattern deviations
- Geo (3): Geographic anomaly signals
- Anomaly/Derived (3): Unsupervised scores + residual signal

All lookback features use ONLY past data (no future leakage).
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression

# Module-level storage for fitted unsupervised models
_isolation_forest: Optional[IsolationForest] = None
_lof_model: Optional[LocalOutlierFactor] = None
_residual_models: dict = {}  # merchant_id -> fitted LinearRegression


# ─── City distance lookup (approximate km between major Indian cities) ───

CITY_COORDS = {
    "Mumbai": (19.07, 72.87),
    "Delhi": (28.61, 77.20),
    "Bangalore": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Hyderabad": (17.38, 78.49),
    "Pune": (18.52, 73.85),
    "Kolkata": (22.57, 88.36),
    "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.78),
    "Lucknow": (26.85, 80.95),
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in km between two lat/lon points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _city_distance(city1: str, city2: str) -> float:
    """Get approximate distance in km between two Indian cities."""
    if city1 == city2:
        return 0.0
    c1 = CITY_COORDS.get(city1)
    c2 = CITY_COORDS.get(city2)
    if c1 is None or c2 is None:
        return 500.0  # default for unknown cities
    return _haversine_km(c1[0], c1[1], c2[0], c2[1])


def get_feature_columns() -> list[str]:
    """Return the ordered list of 22 feature column names."""
    return [
        # Velocity (4)
        "txn_count_30s", "txn_count_5m", "txn_count_1h", "velocity_ratio",
        # Amount (4)
        "amount_zscore", "amount_ratio_median", "amount_ratio_p95", "rolling_amount_std_5",
        # Temporal (4)
        "hour_sin", "hour_cos", "is_business_hours", "time_since_last_txn",
        # Behavioral (4)
        "unique_cards_per_ip_1h", "unique_ips_per_card_1h",
        "payment_method_entropy", "device_changed",
        # Geo (3)
        "geo_distance_km", "is_geo_mismatch", "is_cross_border",
        # Anomaly / Derived (3)
        "isolation_forest_score", "local_outlier_factor", "residual_score",
    ]


def _compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity features using time-windowed lookback per card_id."""
    n = len(df)
    txn_count_30s = np.zeros(n)
    txn_count_5m = np.zeros(n)
    txn_count_1h = np.zeros(n)

    # Group indices by card_id for efficient lookback
    card_indices: dict[str, list[int]] = {}
    timestamps = df["timestamp"].values  # numpy datetime64

    for i in range(n):
        card = df.iloc[i]["card_id"]
        ts = timestamps[i]

        if card in card_indices:
            past = card_indices[card]
            past_ts = timestamps[past]
            diffs = (ts - past_ts) / np.timedelta64(1, "s")  # seconds

            txn_count_30s[i] = np.sum(diffs <= 30)
            txn_count_5m[i] = np.sum(diffs <= 300)
            txn_count_1h[i] = np.sum(diffs <= 3600)
            card_indices[card].append(i)
        else:
            card_indices[card] = [i]

    velocity_ratio = txn_count_30s / np.maximum(txn_count_1h, 1.0)

    df = df.copy()
    df["txn_count_30s"] = txn_count_30s
    df["txn_count_5m"] = txn_count_5m
    df["txn_count_1h"] = txn_count_1h
    df["velocity_ratio"] = velocity_ratio
    return df


def _compute_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute amount-based statistical anomaly features per merchant."""
    n = len(df)
    amount_zscore = np.zeros(n)
    amount_ratio_median = np.ones(n)
    amount_ratio_p95 = np.ones(n)
    rolling_amount_std_5 = np.zeros(n)

    # Track merchant amount histories and card amount histories
    merchant_amounts: dict[str, list[float]] = {}
    card_amounts: dict[str, list[float]] = {}

    for i in range(n):
        row = df.iloc[i]
        merchant = row["merchant_id"]
        card = row["card_id"]
        amount = row["amount"]

        # Merchant-level stats from prior transactions
        if merchant in merchant_amounts and len(merchant_amounts[merchant]) >= 2:
            hist = merchant_amounts[merchant]
            mean_val = np.mean(hist)
            std_val = np.std(hist)
            median_val = np.median(hist)
            p95_val = np.percentile(hist, 95)

            amount_zscore[i] = (amount - mean_val) / max(std_val, 1.0)
            amount_ratio_median[i] = amount / max(median_val, 1.0)
            amount_ratio_p95[i] = amount / max(p95_val, 1.0)
        elif merchant in merchant_amounts and len(merchant_amounts[merchant]) == 1:
            hist = merchant_amounts[merchant]
            amount_ratio_median[i] = amount / max(hist[0], 1.0)

        # Rolling std of last 5 card transactions
        if card in card_amounts and len(card_amounts[card]) >= 2:
            recent = card_amounts[card][-5:]
            rolling_amount_std_5[i] = np.std(recent)

        # Update histories
        merchant_amounts.setdefault(merchant, []).append(amount)
        card_amounts.setdefault(card, []).append(amount)

    df = df.copy()
    df["amount_zscore"] = amount_zscore
    df["amount_ratio_median"] = amount_ratio_median
    df["amount_ratio_p95"] = amount_ratio_p95
    df["rolling_amount_std_5"] = rolling_amount_std_5
    return df


def _compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-of-day cyclical features and inter-transaction timing."""
    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["is_business_hours"] = ((df["timestamp"].dt.hour >= 9) & (df["timestamp"].dt.hour < 18)).astype(int)

    # Time since last transaction from same card
    n = len(df)
    time_since_last = np.zeros(n)
    card_last_ts: dict[str, np.datetime64] = {}
    timestamps = df["timestamp"].values

    for i in range(n):
        card = df.iloc[i]["card_id"]
        if card in card_last_ts:
            diff = (timestamps[i] - card_last_ts[card]) / np.timedelta64(1, "s")
            time_since_last[i] = max(float(diff), 0.0)
        card_last_ts[card] = timestamps[i]

    df["time_since_last_txn"] = time_since_last
    return df


def _compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioral anomaly features — card/IP relationships, entropy, device changes."""
    n = len(df)
    unique_cards_per_ip_1h = np.zeros(n)
    unique_ips_per_card_1h = np.zeros(n)
    payment_method_entropy = np.zeros(n)
    device_changed = np.zeros(n)

    # Track histories
    ip_card_history: dict[str, list[tuple[np.datetime64, str]]] = {}  # ip -> [(ts, card)]
    card_ip_history: dict[str, list[tuple[np.datetime64, str]]] = {}  # card -> [(ts, ip)]
    customer_methods: dict[str, list[str]] = {}  # customer -> [methods]
    customer_devices: dict[str, dict[str, int]] = {}  # customer -> {device: count}
    timestamps = df["timestamp"].values

    for i in range(n):
        row = df.iloc[i]
        ip = row["ip_address"]
        card = row["card_id"]
        customer = row["customer_id"]
        method = row["payment_method"]
        device = row["device_fingerprint"]
        ts = timestamps[i]

        # Unique cards per IP in last hour
        if ip in ip_card_history:
            recent = [(t, c) for t, c in ip_card_history[ip]
                      if (ts - t) / np.timedelta64(1, "s") <= 3600]
            ip_card_history[ip] = recent
            unique_cards_per_ip_1h[i] = len(set(c for _, c in recent))
        ip_card_history.setdefault(ip, []).append((ts, card))

        # Unique IPs per card in last hour
        if card in card_ip_history:
            recent = [(t, p) for t, p in card_ip_history[card]
                      if (ts - t) / np.timedelta64(1, "s") <= 3600]
            card_ip_history[card] = recent
            unique_ips_per_card_1h[i] = len(set(p for _, p in recent))
        card_ip_history.setdefault(card, []).append((ts, ip))

        # Payment method entropy for this customer
        customer_methods.setdefault(customer, []).append(method)
        methods = customer_methods[customer]
        if len(methods) >= 2:
            counts = pd.Series(methods).value_counts(normalize=True).values
            payment_method_entropy[i] = -np.sum(counts * np.log2(counts + 1e-10))

        # Device changed
        if customer in customer_devices:
            most_common = max(customer_devices[customer], key=customer_devices[customer].get)
            device_changed[i] = 0 if device == most_common else 1
        customer_devices.setdefault(customer, {})
        customer_devices[customer][device] = customer_devices[customer].get(device, 0) + 1

    df = df.copy()
    df["unique_cards_per_ip_1h"] = unique_cards_per_ip_1h
    df["unique_ips_per_card_1h"] = unique_ips_per_card_1h
    df["payment_method_entropy"] = payment_method_entropy
    df["device_changed"] = device_changed
    return df


def _compute_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute geographic anomaly features."""
    n = len(df)
    geo_distance = np.zeros(n)
    is_geo_mismatch = np.zeros(n)

    # Track most common city per customer
    customer_cities: dict[str, dict[str, int]] = {}

    for i in range(n):
        row = df.iloc[i]
        customer = row["customer_id"]
        city = row["customer_city"]

        if customer in customer_cities and customer_cities[customer]:
            most_common_city = max(customer_cities[customer], key=customer_cities[customer].get)
            geo_distance[i] = _city_distance(city, most_common_city)
            is_geo_mismatch[i] = 0 if city == most_common_city else 1

        customer_cities.setdefault(customer, {})
        customer_cities[customer][city] = customer_cities[customer].get(city, 0) + 1

    df = df.copy()
    df["geo_distance_km"] = geo_distance
    df["is_geo_mismatch"] = is_geo_mismatch
    df["is_cross_border"] = (df["currency"] != "INR").astype(int)
    return df


def _compute_residual_score(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Compute residual score: actual amount minus predicted 'clean' amount per merchant."""
    global _residual_models
    df = df.copy()
    residual_scores = np.zeros(len(df))

    temporal_features = ["hour_sin", "hour_cos", "is_business_hours"]

    for merchant_id in df["merchant_id"].unique():
        mask = df["merchant_id"] == merchant_id
        merchant_df = df[mask]

        X = merchant_df[temporal_features].values
        y = merchant_df["amount"].values

        if fit:
            if len(merchant_df) >= 5:
                model = LinearRegression()
                model.fit(X, y)
                _residual_models[merchant_id] = model
                predicted = model.predict(X)
                residual_scores[mask.values] = np.abs(y - predicted)
            else:
                residual_scores[mask.values] = 0.0
        else:
            if merchant_id in _residual_models:
                predicted = _residual_models[merchant_id].predict(X)
                residual_scores[mask.values] = np.abs(y - predicted)
            else:
                residual_scores[mask.values] = 0.0

    df["residual_score"] = residual_scores
    return df


def _compute_unsupervised_scores(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Compute IsolationForest and LOF anomaly scores on the first 19 features."""
    global _isolation_forest, _lof_model
    df = df.copy()

    # Use the first 19 features (everything before the unsupervised scores)
    base_features = get_feature_columns()[:19]
    X = df[base_features].values.copy()

    # Replace any NaN/Inf before fitting
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        # Isolation Forest
        _isolation_forest = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1
        )
        iso_scores = _isolation_forest.fit_predict(X)
        # Convert: -1 (anomaly) -> higher score, 1 (normal) -> lower score
        df["isolation_forest_score"] = -_isolation_forest.score_samples(X)

        # LOF — fit and predict in one step (LOF is transductive)
        _lof_model = LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=False, n_jobs=-1
        )
        _lof_model.fit_predict(X)
        df["local_outlier_factor"] = -_lof_model.negative_outlier_factor_
    else:
        # Transform mode — use decision_function for IsolationForest
        if _isolation_forest is not None:
            df["isolation_forest_score"] = -_isolation_forest.score_samples(X)
        else:
            df["isolation_forest_score"] = 0.0

        # LOF in novelty mode for transform (need to refit with novelty=True)
        if _lof_model is not None:
            # For test data, use IsolationForest only. LOF novelty requires refit.
            # Simple approach: fit a novelty LOF on training data
            df["local_outlier_factor"] = 0.0
        else:
            df["local_outlier_factor"] = 0.0

    return df


def compute_features(
    df: pd.DataFrame,
    fit_unsupervised: bool = True,
) -> pd.DataFrame:
    """
    Compute all 22 features from raw transaction data.

    Args:
        df: DataFrame with raw transaction columns (must include timestamp, amount,
            card_id, ip_address, merchant_id, customer_id, customer_city, etc.)
        fit_unsupervised: If True, fit IsolationForest/LOF/residual models.
            Set to True for training data, False for test data.

    Returns:
        DataFrame with original columns plus 22 new feature columns.
    """
    # Sort by timestamp to ensure correct lookback
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("  Computing velocity features...")
    df = _compute_velocity_features(df)

    print("  Computing amount features...")
    df = _compute_amount_features(df)

    print("  Computing temporal features...")
    df = _compute_temporal_features(df)

    print("  Computing behavioral features...")
    df = _compute_behavioral_features(df)

    print("  Computing geo features...")
    df = _compute_geo_features(df)

    print("  Computing residual score...")
    df = _compute_residual_score(df, fit=fit_unsupervised)

    print("  Computing unsupervised anomaly scores...")
    df = _compute_unsupervised_scores(df, fit=fit_unsupervised)

    # Final cleanup: replace NaN/Inf with 0
    feature_cols = get_feature_columns()
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print(f"  {len(feature_cols)} features computed. Shape: {df.shape}")
    return df


def save_state(model_dir: str = "models") -> None:
    """Save fitted unsupervised models to disk."""
    import joblib
    from pathlib import Path
    path = Path(model_dir)
    path.mkdir(exist_ok=True)
    
    state = {
        "isolation_forest": _isolation_forest,
        "lof_model": _lof_model,
        "residual_models": _residual_models
    }
    joblib.dump(state, path / "feature_engine.joblib")


def load_state(model_dir: str = "models") -> None:
    """Load fitted unsupervised models from disk."""
    global _isolation_forest, _lof_model, _residual_models
    import joblib
    from pathlib import Path
    path = Path(model_dir) / "feature_engine.joblib"
    
    if path.exists():
        state = joblib.load(path)
        _isolation_forest = state.get("isolation_forest")
        _lof_model = state.get("lof_model")
        _residual_models = state.get("residual_models", {})
    else:
        print(f"Warning: {path} not found. Unsupervised features will output 0.0.")

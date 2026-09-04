"""
Gemini API-based explanation layer for FraudSense.
"""
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Try importing the GenAI SDK
try:
    import google.genai as genai
except ImportError:
    genai = None

from src.llm.llm_config import MODEL_NAME, MAX_TOKENS, TEMPERATURE, RATE_LIMIT_DELAY_SECONDS

# Load environment variables from .env
load_dotenv()

# Simple in-memory cache to avoid duplicate API calls for identical feature patterns
_explanation_cache: Dict[str, str] = {}


def check_api_available() -> bool:
    """
    Checks if the Gemini API is available (SDK installed and API key set).
    
    Returns:
        bool: True if available, False otherwise.
    """
    if genai is None:
        return False
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    return True


def _fallback_explanation(top_features: List[str], fraud_score: float) -> str:
    """
    Generates a template-based fallback explanation.
    
    Args:
        top_features: List of top contributing features (can be strings with values).
        fraud_score: The fraud score.
        
    Returns:
        str: Fallback explanation.
    """
    explanations = []
    
    for feature in top_features[:2]:  # Use top 2 features for fallback
        feat_lower = feature.lower()
        # Parse value from 'feature_name=value' format
        if "=" in feature:
            feat_name, value = feature.split("=", 1)
        elif ":" in feature:
            feat_name, value = feature.split(":", 1)
        else:
            feat_name, value = feature, "abnormal"
        value = value.strip()
        feat_name = feat_name.strip()
        
        if "txn_count_30s" in feat_lower:
            explanations.append(f"Unusually high transaction velocity detected in a 30-second window (SHAP impact: {value}).")
        elif "amount_ratio_median" in feat_lower or "amount_zscore" in feat_lower:
            explanations.append(f"Transaction amount is highly anomalous compared to the merchant median (SHAP impact: {value}).")
        elif "geo" in feat_lower:
            explanations.append(f"Transaction originated from a different city than the cardholder's registered location (SHAP impact: {value}).")
        elif "velocity_ratio" in feat_lower:
            explanations.append(f"Velocity spike detected, indicating a burst of rapid transactions (SHAP impact: {value}).")
        elif "isolation_forest" in feat_lower or "local_outlier" in feat_lower:
            explanations.append(f"Statistical behavioral anomaly detected on unsupervised analysis (SHAP impact: {value}).")
        elif "residual_score" in feat_lower:
            explanations.append(f"Amount deviates significantly from merchant's expected pattern (SHAP impact: {value}).")
        else:
            explanations.append(f"Suspicious anomaly detected in {feat_name} (SHAP impact: {value}).")
            
    if not explanations:
        return f"Transaction flagged as suspicious with score {fraud_score:.2f} due to anomalous feature patterns."
        
    return " ".join(explanations)


def explain_transaction(fraud_score: float, top_features: List[str], fraud_type: Optional[str] = None) -> str:
    """
    Explains why a transaction is flagged using the Gemini API or a fallback mechanism.
    
    Args:
        fraud_score: The fraud probability score.
        top_features: List of the top contributing features and their values.
        fraud_type: Optional known fraud type.
        
    Returns:
        str: A concise explanation for a risk analyst.
    """
    # Check cache first
    cache_key = f"{fraud_score:.4f}_{','.join(sorted(top_features))}_{fraud_type}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]
        
    features_str = ", ".join(top_features)
    
    if not check_api_available():
        explanation = _fallback_explanation(top_features, fraud_score)
        _explanation_cache[cache_key] = explanation
        return explanation
        
    prompt = (
        f"A payment transaction was flagged with fraud score {fraud_score:.4f}. "
        f"Top contributing factors: {features_str}. "
        f"{f'Known fraud type: {fraud_type}. ' if fraud_type else ''}"
        "Write one sentence explaining why this is suspicious to a risk analyst. "
        "Use the specific numbers provided. Be concise."
    )
    
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
        )
        explanation = response.text.strip()
        _explanation_cache[cache_key] = explanation
        return explanation
    except Exception as e:
        print(f"Warning: Gemini API call failed: {e}. Using fallback.")
        explanation = _fallback_explanation(top_features, fraud_score)
        _explanation_cache[cache_key] = explanation
        return explanation


def explain_batch(transactions: List[dict]) -> List[str]:
    """
    Processes multiple transactions to generate explanations with rate limiting.
    
    Args:
        transactions: List of dictionaries, each containing 'fraud_score', 
                      'top_features', and optionally 'fraud_type'.
                      
    Returns:
        List[str]: List of explanations corresponding to the input transactions.
    """
    explanations = []
    
    for i, txn in enumerate(transactions):
        explanation = explain_transaction(
            fraud_score=txn.get("fraud_score", 0.0),
            top_features=txn.get("top_features", []),
            fraud_type=txn.get("fraud_type")
        )
        explanations.append(explanation)
        
        # Apply rate limiting if making multiple calls (and API is available)
        if i < len(transactions) - 1 and check_api_available():
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
            
    return explanations

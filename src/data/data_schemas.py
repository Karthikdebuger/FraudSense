from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    transaction_id: str
    timestamp: datetime
    amount: float
    currency: str = 'INR'
    payment_method: Literal['card', 'upi', 'netbanking', 'wallet']
    card_id: str
    ip_address: str
    device_fingerprint: str
    merchant_id: str
    customer_id: str
    customer_city: str
    customer_state: str
    is_fraud: int = Field(ge=0, le=1)
    fraud_type: Optional[Literal['velocity_spike', 'amount_anomaly', 'geo_mismatch', 'card_testing', 'return_fraud']] = None

class ScoredTransaction(Transaction):
    fraud_score: float
    is_flagged: bool
    explanation: Optional[str] = None
    top_features: Optional[list[str]] = None

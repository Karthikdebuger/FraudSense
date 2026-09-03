import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

CITIES_STATES = {
    'Mumbai': 'Maharashtra',
    'Delhi': 'Delhi',
    'Bangalore': 'Karnataka',
    'Chennai': 'Tamil Nadu',
    'Hyderabad': 'Telangana',
    'Pune': 'Maharashtra',
    'Kolkata': 'West Bengal',
    'Ahmedabad': 'Gujarat',
    'Jaipur': 'Rajasthan',
    'Lucknow': 'Uttar Pradesh'
}

def generate_dataset(n_transactions: int = 10000, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic transaction data with injected fraud patterns.
    """
    np.random.seed(seed)
    
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud
    
    # Setup base entities
    merchants = [f'M{i:03d}' for i in range(1, 11)]
    merchant_medians = {m: np.random.choice([300, 800, 2000, 5000, 15000]) for m in merchants}
    
    customers = [f'C{i:04d}' for i in range(1, 501)]
    cities = list(CITIES_STATES.keys())
    customer_cities = {c: np.random.choice(cities) for c in customers}
    
    cards = [f'CARD_{i:04d}' for i in range(1, 201)]
    # Assign cards to customers, some customers may share cards or have multiple if we randomly assign
    # For simplicity, assign randomly from the pool
    
    payment_methods = ['card', 'upi', 'netbanking', 'wallet']
    payment_weights = [0.4, 0.4, 0.1, 0.1]
    
    start_date = datetime.now() - timedelta(days=30)
    
    transactions = []
    
    # 1. Generate Legit Transactions
    for _ in range(n_legit):
        cust = np.random.choice(customers)
        merch = np.random.choice(merchants)
        median_amt = merchant_medians[merch]
        amt = np.random.lognormal(mean=np.log(median_amt), sigma=0.5)
        
        city = customer_cities[cust]
        state = CITIES_STATES[city]
        card = np.random.choice(cards)
        ip = f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"
        
        timestamp = start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
        
        transactions.append({
            'transaction_id': f"TXN_{len(transactions):06d}",
            'timestamp': timestamp,
            'amount': round(amt, 2),
            'currency': 'INR',
            'payment_method': np.random.choice(payment_methods, p=payment_weights),
            'card_id': card,
            'ip_address': ip,
            'device_fingerprint': f"DEV_{np.random.randint(1000, 9999)}",
            'merchant_id': merch,
            'customer_id': cust,
            'customer_city': city,
            'customer_state': state,
            'is_fraud': 0,
            'fraud_type': None
        })
        
    # 2. Inject Fraud Patterns
    fraud_patterns = ['velocity_spike', 'amount_anomaly', 'geo_mismatch', 'card_testing', 'return_fraud']
    fraud_counts = {p: n_fraud // len(fraud_patterns) for p in fraud_patterns}
    
    # Distribute any remainder
    for i in range(n_fraud % len(fraud_patterns)):
        fraud_counts[fraud_patterns[i]] += 1
        
    # Velocity Spike (5-15 txns within 30s)
    # Generating in batches to hit the target count
    current_vs = 0
    while current_vs < fraud_counts['velocity_spike']:
        cust = np.random.choice(customers)
        merch = np.random.choice(merchants)
        card = np.random.choice(cards)
        ip = f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"
        base_time = start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
        
        batch_size = min(np.random.randint(5, 16), fraud_counts['velocity_spike'] - current_vs)
        if batch_size == 0: break
        
        for _ in range(batch_size):
            amt = np.random.lognormal(mean=np.log(merchant_medians[merch]), sigma=0.5)
            transactions.append({
                'transaction_id': f"TXN_{len(transactions):06d}",
                'timestamp': base_time + timedelta(seconds=np.random.randint(0, 30)),
                'amount': round(amt, 2),
                'currency': 'INR',
                'payment_method': 'card',
                'card_id': card,
                'ip_address': ip,
                'device_fingerprint': f"DEV_{np.random.randint(1000, 9999)}",
                'merchant_id': merch,
                'customer_id': cust,
                'customer_city': customer_cities[cust],
                'customer_state': CITIES_STATES[customer_cities[cust]],
                'is_fraud': 1,
                'fraud_type': 'velocity_spike'
            })
        current_vs += batch_size
            
    # Amount Anomaly (20-100x above median)
    for _ in range(fraud_counts['amount_anomaly']):
        cust = np.random.choice(customers)
        merch = np.random.choice(merchants)
        amt = merchant_medians[merch] * np.random.uniform(20, 100)
        
        transactions.append({
            'transaction_id': f"TXN_{len(transactions):06d}",
            'timestamp': start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600)),
            'amount': round(amt, 2),
            'currency': 'INR',
            'payment_method': 'card',
            'card_id': np.random.choice(cards),
            'ip_address': f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}",
            'device_fingerprint': f"DEV_{np.random.randint(1000, 9999)}",
            'merchant_id': merch,
            'customer_id': cust,
            'customer_city': customer_cities[cust],
            'customer_state': CITIES_STATES[customer_cities[cust]],
            'is_fraud': 1,
            'fraud_type': 'amount_anomaly'
        })
        
    # Geo Mismatch (wrong city)
    for _ in range(fraud_counts['geo_mismatch']):
        cust = np.random.choice(customers)
        merch = np.random.choice(merchants)
        amt = np.random.lognormal(mean=np.log(merchant_medians[merch]), sigma=0.5)
        
        wrong_cities = [c for c in CITIES_STATES.keys() if c != customer_cities[cust]]
        wrong_city = np.random.choice(wrong_cities)
        
        transactions.append({
            'transaction_id': f"TXN_{len(transactions):06d}",
            'timestamp': start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600)),
            'amount': round(amt, 2),
            'currency': 'INR',
            'payment_method': 'card',
            'card_id': np.random.choice(cards),
            'ip_address': f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}",
            'device_fingerprint': f"DEV_{np.random.randint(1000, 9999)}",
            'merchant_id': merch,
            'customer_id': cust,
            'customer_city': wrong_city,
            'customer_state': CITIES_STATES[wrong_city],
            'is_fraud': 1,
            'fraud_type': 'geo_mismatch'
        })
        
    # Card Testing (rapid 1-10 INR txns)
    current_ct = 0
    while current_ct < fraud_counts['card_testing']:
        cust = np.random.choice(customers)
        card = np.random.choice(cards)
        base_time = start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
        ip = f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"
        
        batch_size = min(np.random.randint(3, 9), fraud_counts['card_testing'] - current_ct)
        if batch_size == 0: break
        
        for _ in range(batch_size):
            transactions.append({
                'transaction_id': f"TXN_{len(transactions):06d}",
                'timestamp': base_time + timedelta(seconds=np.random.randint(0, 60)),
                'amount': round(np.random.uniform(1, 10), 2),
                'currency': 'INR',
                'payment_method': 'card',
                'card_id': card,
                'ip_address': ip,
                'device_fingerprint': f"DEV_{np.random.randint(1000, 9999)}",
                'merchant_id': np.random.choice(merchants),
                'customer_id': cust,
                'customer_city': customer_cities[cust],
                'customer_state': CITIES_STATES[customer_cities[cust]],
                'is_fraud': 1,
                'fraud_type': 'card_testing'
            })
        current_ct += batch_size
            
    # Return Fraud (buy and return slightly different amount)
    current_rf = 0
    while current_rf < fraud_counts['return_fraud']:
        cust = np.random.choice(customers)
        merch = np.random.choice(merchants)
        amt = np.random.lognormal(mean=np.log(merchant_medians[merch]), sigma=0.5)
        base_time = start_date + timedelta(seconds=np.random.randint(0, 30 * 24 * 3600))
        card = np.random.choice(cards)
        ip = f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"
        dev = f"DEV_{np.random.randint(1000, 9999)}"
        
        # Original purchase (added as fraud for labeling purposes, or we just count both as fraud)
        # Adding pairs until we hit target count. If only 1 spot left, just add the purchase.
        
        transactions.append({
            'transaction_id': f"TXN_{len(transactions):06d}",
            'timestamp': base_time,
            'amount': round(amt, 2),
            'currency': 'INR',
            'payment_method': 'card',
            'card_id': card,
            'ip_address': ip,
            'device_fingerprint': dev,
            'merchant_id': merch,
            'customer_id': cust,
            'customer_city': customer_cities[cust],
            'customer_state': CITIES_STATES[customer_cities[cust]],
            'is_fraud': 1,
            'fraud_type': 'return_fraud'
        })
        current_rf += 1
        
        if current_rf < fraud_counts['return_fraud']:
            transactions.append({
                'transaction_id': f"TXN_{len(transactions):06d}",
                'timestamp': base_time + timedelta(minutes=np.random.randint(1, 6)),
                'amount': round(amt * np.random.uniform(0.98, 1.02), 2),
                'currency': 'INR',
                'payment_method': 'card',
                'card_id': card,
                'ip_address': ip,
                'device_fingerprint': dev,
                'merchant_id': merch,
                'customer_id': cust,
                'customer_city': customer_cities[cust],
                'customer_state': CITIES_STATES[customer_cities[cust]],
                'is_fraud': 1,
                'fraud_type': 'return_fraud'
            })
            current_rf += 1

    df = pd.DataFrame(transactions)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    if len(df) > n_transactions:
        df = df.head(n_transactions)
        
    return df

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and test sets with stratified sampling on 'is_fraud'.
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=seed, 
        stratify=df['is_fraud']
    )
    
    return train_df, test_df

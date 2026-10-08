import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "fuel", "restaurant", "electronics", "travel", "online_services", "pharmacy"]
RISK = {"electronics": 0.9, "travel": 0.6, "online_services": 0.7}


def generate_transactions(n=200_000, seed=42):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 90 * 24 * 3600, n))
    hour = (ts // 3600) % 24
    category = rng.choice(CATEGORIES, n, p=[0.28, 0.14, 0.18, 0.08, 0.06, 0.14, 0.12])
    online = rng.random(n) < np.where(np.isin(category, ["online_services", "electronics"]), 0.8, 0.2)
    avg_amount_30d = rng.lognormal(3.6, 0.5, n)
    amount = avg_amount_30d * rng.lognormal(0, 0.7, n)
    distance_km = np.where(online, 0, rng.exponential(8, n))
    account_age_days = rng.integers(1, 3000, n)
    tx_last_24h = rng.poisson(2.2, n)
    new_device = rng.random(n) < 0.06

    score = (
        -5.0
        + 1.3 * np.log(amount / avg_amount_30d).clip(-2, 4)
        + 1.2 * ((hour < 5) | (hour > 23)) * online
        + 2.0 * new_device
        + 0.25 * np.maximum(tx_last_24h - 4, 0)
        + 0.012 * np.minimum(distance_km, 150)
        - 0.0004 * account_age_days
        + 1.1 * np.array([RISK.get(c, 0.0) for c in category])
        + 0.7 * (new_device & online)
    )
    fraud = rng.random(n) < 1 / (1 + np.exp(-score))

    return pd.DataFrame({
        "timestamp": ts,
        "amount": amount.round(2),
        "hour": hour,
        "category": category,
        "online": online.astype(int),
        "distance_km": distance_km.round(1),
        "avg_amount_30d": avg_amount_30d.round(2),
        "account_age_days": account_age_days,
        "tx_last_24h": tx_last_24h,
        "new_device": new_device.astype(int),
        "is_fraud": fraud.astype(int),
    })

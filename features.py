import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["amount", "hour", "online", "distance_km", "avg_amount_30d", "account_age_days",
           "tx_last_24h", "new_device", "amount_ratio", "is_night", "log_amount"]
CATEGORICAL = ["category"]


def add_features(df):
    out = df.copy()
    out["amount_ratio"] = out["amount"] / out["avg_amount_30d"]
    out["is_night"] = ((out["hour"] < 5) | (out["hour"] > 23)).astype(int)
    out["log_amount"] = np.log1p(out["amount"])
    return out


def build_preprocessor(scale=True):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])

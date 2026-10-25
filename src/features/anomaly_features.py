"""Features for anomaly detection on a user's transactions.

Expects the columns date, description, amount and optionally user_id and category.
The returned table has one row per input row, in the same order (index 0..n-1).
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount_z", "log_amount", "day_of_month", "weekday",
    "merchant_count", "days_since_merchant", "same_amount_recent", "is_income",
]
MIN_GROUP_SIZE = 5     # z-scores need some history per user and category
MAX_GAP_DAYS = 400     # value for a merchant's first charge


def merchant_key(descriptions: pd.Series) -> pd.Series:
    """Lower-case text, letters only, so '03/2026' or ids do not split one merchant."""
    cleaned = descriptions.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.replace(r"[^a-zäöüß]+", " ", regex=True)
    return cleaned.str.split().str.join(" ")


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    tx = df.copy().reset_index(drop=True)
    tx["date"] = pd.to_datetime(tx["date"])
    tx["amount"] = pd.to_numeric(tx["amount"])
    if "user_id" not in tx.columns:
        tx["user_id"] = "user"
    if "category" not in tx.columns:
        tx["category"] = "Uncategorized"
    tx["category"] = tx["category"].fillna("Uncategorized")

    tx["merchant"] = merchant_key(tx["description"])
    tx["log_amount"] = np.log1p(tx["amount"].abs())

    # How unusual is the amount for this user and category?
    grouped = tx.groupby(["user_id", "category"])["log_amount"]
    mean, std, size = grouped.transform("mean"), grouped.transform("std"), grouped.transform("size")
    z = (tx["log_amount"] - mean) / std
    tx["amount_z"] = z.where((size >= MIN_GROUP_SIZE) & (std > 0), 0.0).fillna(0.0)

    tx["day_of_month"] = tx["date"].dt.day
    tx["weekday"] = tx["date"].dt.weekday
    tx["is_income"] = (tx["amount"] > 0).astype(int)

    # How common is this merchant for this user?
    tx["merchant_count"] = np.log1p(tx.groupby(["user_id", "merchant"])["amount"].transform("size"))

    # Time since the previous charge of the same merchant, and exact repeats
    ordered = tx.sort_values(["user_id", "merchant", "date"], kind="stable")
    prev = ordered.groupby(["user_id", "merchant"])
    gap = (ordered["date"] - prev["date"].shift(1)).dt.days
    prev_amount = prev["amount"].shift(1)
    ordered["days_since_merchant"] = gap.fillna(MAX_GAP_DAYS).clip(upper=MAX_GAP_DAYS)
    ordered["same_amount_recent"] = ((ordered["amount"] == prev_amount) & (gap <= 3)).astype(int)
    tx = tx.join(ordered[["days_since_merchant", "same_amount_recent"]])

    return tx[FEATURE_COLUMNS].astype(float)

SPIKE_Z = 4.0


def rule_flags(features: pd.DataFrame, spike_z: float = SPIKE_Z) -> pd.DataFrame:
    """Simple, explainable rules that run next to the Isolation Forest.

    duplicate: same merchant and amount within 3 days of the previous charge
    spike:     amount far outside the user's usual range for the category
    Needs some history per category (about 20 rows); with less, nothing is flagged.
    """
    return pd.DataFrame(
        {
            "duplicate": features["same_amount_recent"] == 1,
            "spike": features["amount_z"].abs() > spike_z,
        },
        index=features.index,
    )
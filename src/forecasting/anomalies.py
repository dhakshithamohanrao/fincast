"""Live anomaly detection for one user's transactions (Phase 8.6).

Rule flags (duplicates, amount spikes) are explainable and fairly precise. The
Isolation Forest only adds a limited number of "unusual" suggestions, ranked by
score, because on its own it raises many false alarms.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.features.anomaly_features import MAX_GAP_DAYS, build_anomaly_features, rule_flags
from src.utils.helpers import get_project_root

MIN_TRANSACTIONS = 30    # with less history the per-category statistics mean little
COLUMNS = ["id", "date", "description", "amount", "category",
           "source", "kind", "reason", "score"]


@lru_cache(maxsize=1)
def load_anomaly_model() -> dict:
    return joblib.load(get_project_root() / "models" / "anomaly.joblib")


def detect_anomalies(transactions: pd.DataFrame, max_model_flags: int = 15) -> pd.DataFrame:
    """Flagged transactions, rule flags first, then model suggestions by score.

    transactions needs: id, date, description, amount, category (from crud.get_transactions).
    """
    if transactions is None or len(transactions) < MIN_TRANSACTIONS:
        return pd.DataFrame(columns=COLUMNS)

    tx = transactions.reset_index(drop=True)
    bundle = load_anomaly_model()
    feats = build_anomaly_features(tx)
    rules = rule_flags(feats, bundle["spike_z"])
    X = feats[bundle["features"]]
    scores = bundle["model"].score_samples(X)           # lower = more unusual
    forest = bundle["model"].predict(X) == -1

    dup = rules["duplicate"].to_numpy(dtype=bool)
    spike = rules["spike"].to_numpy(dtype=bool)
    by_rule = dup | spike

    keep = by_rule.copy()
    forest_only = np.flatnonzero(forest & ~by_rule)
    if len(forest_only) and max_model_flags > 0:
        keep[forest_only[np.argsort(scores[forest_only])[:max_model_flags]]] = True

    rows = []
    for i in np.flatnonzero(keep):
        category = tx.loc[i, "category"]
        category = category if pd.notna(category) else "Uncategorized"
        kinds, reasons = [], []
        if dup[i]:
            kinds.append("duplicate")
            reasons.append("Same amount at the same merchant within 3 days of the previous charge.")
        if spike[i]:
            kinds.append("spike")
            size = "larger" if feats.loc[i, "amount_z"] > 0 else "smaller"
            reasons.append(f"The amount is much {size} than your usual for {category}.")
        if not kinds:
            kinds.append("unusual")
            if feats.loc[i, "days_since_merchant"] >= MAX_GAP_DAYS:
                reasons.append("First charge from this merchant, and unusual compared "
                               "with your history.")
            else:
                reasons.append("Unusual mix of amount, timing and merchant compared "
                               "with your history.")
        rows.append({
            "id": int(tx.loc[i, "id"]), "date": tx.loc[i, "date"],
            "description": tx.loc[i, "description"], "amount": float(tx.loc[i, "amount"]),
            "category": category, "source": "rule" if by_rule[i] else "model",
            "kind": " + ".join(kinds), "reason": " ".join(reasons),
            "score": float(-scores[i]),
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["_rank"] = (out["source"] != "rule").astype(int)
    return (out.sort_values(["_rank", "score"], ascending=[True, False])
               .drop(columns="_rank").reset_index(drop=True))
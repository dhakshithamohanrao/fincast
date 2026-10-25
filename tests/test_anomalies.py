import pandas as pd

from src.forecasting.anomalies import detect_anomalies
from src.ingestion.synthetic_generator import generate_user


def _user() -> pd.DataFrame:
    df = generate_user("salaried_single", 30, seed=5)[["date", "description", "amount", "category"]]
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def test_injected_duplicate_and_spike_are_flagged():
    df = _user()
    groceries = df[df["category"] == "Groceries"]
    dup = groceries.iloc[10].copy()
    dup["id"] = 900001                       # the same charge a second time
    spike = groceries.iloc[20].copy()
    spike["id"] = 900002
    spike["amount"] = spike["amount"] * 8    # eight times the usual amount
    df = pd.concat([df, dup.to_frame().T, spike.to_frame().T], ignore_index=True)
    df["id"] = df["id"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])

    flags = detect_anomalies(df)
    assert 900001 in set(flags["id"])
    assert 900002 in set(flags["id"])
    assert "spike" in flags.loc[flags["id"] == 900002, "kind"].iloc[0]


def test_clean_history_gives_a_short_list():
    assert len(detect_anomalies(_user())) <= 40


def test_small_history_returns_nothing():
    assert detect_anomalies(_user().head(10)).empty
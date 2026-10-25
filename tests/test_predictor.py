import pandas as pd

from src.forecasting.predictor import forecast_user
from src.ingestion.synthetic_generator import generate_user

TODAY = pd.Timestamp("2026-10-25")


def _history(months: int) -> pd.DataFrame:
    df = generate_user("salaried_single", 36, seed=3)
    cutoff = pd.Timestamp("2026-09-01") - pd.DateOffset(months=months - 1)
    return df[df["date"] >= cutoff][["date", "description", "amount", "category"]]


def test_full_history_gives_an_ordered_forecast():
    fc = forecast_user(_history(30), today=TODAY)
    assert fc.status == "ok"
    assert fc.target_month == pd.Timestamp("2026-10-01")
    t = fc.table
    assert (t["lower"] <= t["forecast"] + 1e-6).all()
    assert (t["forecast"] <= t["upper"] + 1e-6).all()
    assert 0.0 <= fc.p_negative <= 1.0
    assert fc.income["forecast"] > 0


def test_short_history_is_limited_and_tiny_history_is_refused():
    assert forecast_user(_history(4), today=TODAY).status == "limited"
    assert forecast_user(_history(2), today=TODAY).status == "insufficient"


def test_empty_history_is_handled():
    assert forecast_user(pd.DataFrame(columns=["date", "amount", "category"]),
                         today=TODAY).status == "insufficient"
import pandas as pd
import pytest

from src.forecasting.predictor import Forecast
from src.forecasting.simulator import simulate


def _fc() -> Forecast:
    table = pd.DataFrame({
        "series": ["Dining", "Income", "Rent", "Total expenses"],
        "forecast": [200.0, 3000.0, 800.0, 1000.0],
        "lower": [150.0, 2800.0, 800.0, 900.0],
        "upper": [260.0, 3200.0, 800.0, 1150.0],
        "last_month": [210.0, 3000.0, 800.0, 1010.0],
    })
    table["change"] = table["forecast"] - table["last_month"]
    return Forecast(
        status="ok", table=table, target_month=pd.Timestamp("2026-10-01"),
        income={"forecast": 3000.0, "lower": 2800.0, "upper": 3200.0},
        expenses={"forecast": 1000.0, "lower": 900.0, "upper": 1150.0},
        start_balance=500.0, end_balance=2500.0, p_negative=0.0)


def test_no_changes_gives_the_same_forecast():
    s = simulate(_fc())
    assert s.end_balance == pytest.approx(2500.0)
    assert s.expenses["forecast"] == pytest.approx(1000.0)


def test_cutting_dining_by_20_percent():
    s = simulate(_fc(), {"Dining": -20})
    assert s.expenses["forecast"] == pytest.approx(960.0)
    assert s.end_balance == pytest.approx(2540.0)


def test_income_drop():
    s = simulate(_fc(), income_change=-15)
    assert s.income["forecast"] == pytest.approx(2550.0)
    assert s.end_balance == pytest.approx(2050.0)
    assert s.p_negative >= 0.0


def test_cut_to_zero_and_unknown_category():
    s = simulate(_fc(), {"Dining": -100})
    assert s.table.loc[s.table["series"] == "Dining", "forecast"].iloc[0] == 0.0
    with pytest.raises(ValueError):
        simulate(_fc(), {"Holidays": -10})
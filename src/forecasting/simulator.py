"""What-if engine: change categories or income and recompute the forecast (Phase 8.5).

Works on the Forecast object from predictor.py. A change of -20 means 20% less
than forecast; the whole range (lower and upper) is scaled the same way.
"""
import math
from dataclasses import dataclass

import pandas as pd

from src.features.build_features import TOTAL_EXPENSES
from src.forecasting.predictor import Forecast, Z80

NOT_CATEGORIES = ("Income", TOTAL_EXPENSES)


@dataclass
class Scenario:
    table: pd.DataFrame      # forecast table with an extra column 'original'
    income: dict
    expenses: dict
    end_balance: float
    p_negative: float
    changes: dict            # what was applied


def balance_risk(end: float, income: dict, expenses: dict) -> float:
    """Rough chance the end balance is below zero (normal approximation)."""
    sd = math.hypot((income["upper"] - income["lower"]) / (2 * Z80),
                    (expenses["upper"] - expenses["lower"]) / (2 * Z80))
    if sd <= 0:
        return float(end < 0)
    return 0.5 * (1 + math.erf((-end / sd) / math.sqrt(2)))


def simulate(fc: Forecast, category_changes: dict | None = None,
             income_change: float = 0.0) -> Scenario:
    """Apply percentage changes to categories and/or income.

    category_changes: {"Dining": -20, "Transport": 10}  (percent)
    income_change: percent, e.g. -15 for an income drop of 15%.
    """
    if fc.table is None:
        raise ValueError("There is no forecast to simulate.")
    changes = {k: float(v) for k, v in (category_changes or {}).items() if v}
    table = fc.table.copy()
    table["original"] = table["forecast"]

    factor = pd.Series(1.0, index=table.index)
    for series, pct in changes.items():
        mask = table["series"] == series
        if series in NOT_CATEGORIES or not mask.any():
            raise ValueError(f"Unknown category: {series}")
        factor[mask] = max(0.0, 1 + pct / 100)
    factor[table["series"] == "Income"] = max(0.0, 1 + float(income_change) / 100)
    for col in ("forecast", "lower", "upper"):
        table[col] = table[col] * factor

    # Total spending is the sum of the categories; its range keeps its original shape
    is_cat = ~table["series"].isin(NOT_CATEGORIES)
    old_total = float(fc.expenses["forecast"])
    new_total = float(table.loc[is_cat, "forecast"].sum())
    ratio = new_total / old_total if old_total > 0 else 1.0
    table.loc[table["series"] == TOTAL_EXPENSES, ["forecast", "lower", "upper"]] = [
        new_total, fc.expenses["lower"] * ratio, fc.expenses["upper"] * ratio]
    table["change"] = table["forecast"] - table["last_month"]

    inc = table[table["series"] == "Income"].iloc[0]
    income = {"forecast": float(inc["forecast"]), "lower": float(inc["lower"]),
              "upper": float(inc["upper"])}
    expenses = {"forecast": new_total, "lower": float(fc.expenses["lower"] * ratio),
                "upper": float(fc.expenses["upper"] * ratio)}
    end = float(fc.start_balance) + income["forecast"] - new_total
    return Scenario(table, income, expenses, end, balance_risk(end, income, expenses),
                    {**changes, **({"Income": float(income_change)} if income_change else {})})
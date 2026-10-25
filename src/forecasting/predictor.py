"""Live forecasting for one user (Phase 8.1).

forecast_user(transactions) turns a user's transaction history into next month's
forecast: a median with a lower/upper (80%) range per series, the expected end
balance and a rough risk of going below zero.

It reuses aggregate_monthly_by_category() and build_features() from the training
pipeline, so features are computed exactly as in training. The page passes in the
transactions (from crud.get_transactions), so this module never touches the database.
"""
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import TOTAL_EXPENSES, build_features
from src.preprocessing.aggregate_monthly import aggregate_monthly_by_category
from src.utils.helpers import get_project_root

MIN_MONTHS = 3      # fewer months: no forecast, only a friendly message
FULL_MONTHS = 6     # fewer months: forecast with a visible note
YEAR_MONTHS = 12    # fewer months: no same-month-last-year information
Z80 = 1.2816        # z-score belonging to an 80% interval
USER = 0            # the predictor works on one user at a time
MODEL_FILES = {
    "median": "forecaster_median.joblib",
    "q10": "forecaster_q10.joblib",
    "q90": "forecaster_q90.joblib",
}


@dataclass
class Forecast:
    status: str                         # "ok", "limited" or "insufficient"
    message: str = ""
    notes: list = field(default_factory=list)
    target_month: pd.Timestamp | None = None
    n_months: int = 0
    table: pd.DataFrame | None = None   # series, forecast, lower, upper, last_month, change
    income: dict | None = None          # forecast, lower, upper
    expenses: dict | None = None
    start_balance: float | None = None
    end_balance: float | None = None
    p_negative: float | None = None
    model_version: str = ""


@lru_cache(maxsize=1)
def load_champion():
    """Load the three LightGBM models and their metadata (cached)."""
    models_dir = get_project_root() / "models"
    meta = json.loads((models_dir / "model_meta.json").read_text(encoding="utf-8"))
    if meta.get("champion") != "lgbm":
        raise RuntimeError("The baseline is the champion, so there are no model files to load.")
    models = {name: joblib.load(models_dir / file) for name, file in MODEL_FILES.items()}
    return models, meta


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _prepare_history(transactions: pd.DataFrame, real_categories: list) -> pd.DataFrame:
    """Clean types and map categories the model has never seen (e.g. 'Savings') to 'Other'."""
    df = transactions[["date", "amount", "category"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df["category"] = df["category"].where(df["category"].isin(real_categories), "Other")
    df["user_id"] = USER
    return df


def _complete_categories(monthly: pd.DataFrame, real_categories: list) -> pd.DataFrame:
    """The model expects every category; add zero rows for the ones the user never used."""
    missing = [c for c in real_categories if c not in set(monthly["category"])]
    if not missing:
        return monthly
    base = monthly.drop_duplicates("month")[
        ["user_id", "month", "total_income", "total_expenses", "savings", "balance_end"]
    ]
    extra = [base.assign(category=c, income=0.0, expenses=0.0) for c in missing]
    return pd.concat([monthly] + extra, ignore_index=True)


def _fill_short_history(rows: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Fill features that need more history than the user has.

    Training data never had missing values, and LightGBM would read a missing value
    as 0, which would bias forecasts low. These are rough but sensible stand-ins.
    """
    out = rows.copy()
    out["roll_mean_6"] = out["roll_mean_6"].fillna(out["roll_mean_3"])
    out["roll_std_6"] = out["roll_std_6"].fillna(out["roll_std_3"])
    out["lag_12"] = out["lag_12"].fillna(out["roll_mean_6"])
    incomes = monthly.drop_duplicates("month")["total_income"]
    vol = float(incomes.std()) if len(incomes) > 1 else 0.0
    out["income_vol_6"] = out["income_vol_6"].fillna(0.0 if math.isnan(vol) else vol)
    out["fixed_share_3"] = out["fixed_share_3"].fillna(0.0)
    return out


def forecast_user(transactions: pd.DataFrame, today: pd.Timestamp | None = None,
                  current_balance: float | None = None,
                  macro: pd.DataFrame | None = None) -> Forecast:
    """Forecast the month after the user's last COMPLETE month.

    transactions: columns date, amount, category (expenses negative, income positive).
    current_balance: the user's real balance if known. If None, the cumulative
        savings in the data is used (the opening balance is unknown).
    """
    models, meta = load_champion()
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    real_categories = [c for c in meta["series_categories"] if c != TOTAL_EXPENSES]

    if transactions is None or transactions.empty:
        return Forecast("insufficient", "No transactions yet. Import a bank CSV to get a forecast.")
    df = _prepare_history(transactions, real_categories)
    if df.empty:
        return Forecast("insufficient", "No valid transactions found.")

    notes = []
    this_month = today.to_period("M").to_timestamp()
    if df["date"].max().to_period("M").to_timestamp() == this_month:
        df = df[df["date"] < this_month]          # the current month is not finished yet
        notes.append("The current month is still in progress, so the forecast uses "
                     "completed months only.")
    if df.empty:
        return Forecast("insufficient", "Not enough completed months yet. "
                        "Import at least 3 months of transactions.")

    monthly = aggregate_monthly_by_category(df)
    n_months = int(monthly["month"].nunique())
    if n_months < MIN_MONTHS:
        return Forecast("insufficient", n_months=n_months, notes=notes,
                        message=f"You have {n_months} month(s) of data. FinCast needs at least "
                                f"{MIN_MONTHS} to give a forecast. Import more history.")

    monthly = _complete_categories(monthly, real_categories)
    last = monthly.drop_duplicates("month").sort_values("month").iloc[-1]
    target = pd.Timestamp(last["month"]) + pd.DateOffset(months=1)

    # Placeholder rows for the target month. Every feature only looks at EARLIER
    # months, so these zeros cannot leak into the target month's own features.
    placeholder = pd.DataFrame({
        "user_id": USER, "month": target, "category": real_categories,
        "income": 0.0, "expenses": 0.0, "total_income": 0.0, "total_expenses": 0.0,
        "savings": 0.0, "balance_end": last["balance_end"],
    })
    feats = build_features(pd.concat([monthly, placeholder], ignore_index=True),
                           macro=macro, drop_incomplete=False)
    rows = feats[feats["month"] == target].copy()
    rows = _fill_short_history(rows, monthly)
    rows["series"] = pd.Categorical(rows["category"], categories=meta["series_categories"])
    X = rows[meta["features"]]

    point = np.clip(models["median"].predict(X), 0, None)
    lower = np.clip(models["q10"].predict(X), 0, None)
    upper = np.clip(models["q90"].predict(X), 0, None)
    lower, upper = np.minimum(lower, point), np.maximum(upper, point)

    fixed = rows["category"].isin(meta["copy_last_month"]).to_numpy()   # Rent, Subscriptions
    last_month_values = rows["lag_1"].to_numpy()
    point = np.where(fixed, last_month_values, point)
    lower = np.where(fixed, last_month_values, lower)
    upper = np.where(fixed, last_month_values, upper)

    table = pd.DataFrame({
        "series": rows["category"].to_numpy(), "forecast": point, "lower": lower,
        "upper": upper, "last_month": last_month_values,
    })

    # Total expenses: the sum of the category forecasts (more accurate in the backtest);
    # the range comes from the total-expenses model, kept around the new point.
    parts = table[~table["series"].isin(["Income", TOTAL_EXPENSES])]
    exp_point = float(parts["forecast"].sum())
    is_total = table["series"] == TOTAL_EXPENSES
    exp_lo = min(float(table.loc[is_total, "lower"].iloc[0]), exp_point)
    exp_hi = max(float(table.loc[is_total, "upper"].iloc[0]), exp_point)
    table.loc[is_total, ["forecast", "lower", "upper"]] = [exp_point, exp_lo, exp_hi]
    table["change"] = table["forecast"] - table["last_month"]

    inc = table[table["series"] == "Income"].iloc[0]
    income = {"forecast": float(inc["forecast"]), "lower": float(inc["lower"]),
              "upper": float(inc["upper"])}
    expenses = {"forecast": exp_point, "lower": exp_lo, "upper": exp_hi}

    # Balance and a rough risk of going below zero. The spread comes from the 80%
    # ranges (sd = width / (2 * 1.2816)); income and spending are assumed independent.
    start = float(last["balance_end"]) if current_balance is None else float(current_balance)
    end = start + income["forecast"] - exp_point
    sd = math.hypot((income["upper"] - income["lower"]) / (2 * Z80),
                    (exp_hi - exp_lo) / (2 * Z80))
    p_negative = _norm_cdf(-end / sd) if sd > 0 else float(end < 0)

    if n_months < FULL_MONTHS:
        notes.append(f"You have only {n_months} months of data, so this forecast mostly "
                     "reflects the general model rather than your own habits.")
    if n_months < YEAR_MONTHS:
        notes.append("With less than a year of history, yearly patterns (holidays, summer, "
                     "annual bills) cannot be seen. Treat the forecast as a rough guide.")

    return Forecast(
        status="limited" if n_months < FULL_MONTHS else "ok", notes=notes, target_month=target,
        n_months=n_months, table=table.sort_values("series").reset_index(drop=True),
        income=income, expenses=expenses, start_balance=start, end_balance=end,
        p_negative=float(p_negative), model_version=meta.get("model_version", ""),
    )
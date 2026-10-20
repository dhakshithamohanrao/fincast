"""Forecasting features (Phase 6).

One row per user, series and TARGET month T. A "series" is a category's monthly
expenses, plus 'Income' (monthly income) and 'Total expenses'.

Every data-derived feature only looks at months BEFORE T (lag_1 = month T-1,
lag_12 = same month last year), so nothing from the future leaks in.
Calendar features of T (month of year, holidays) are fine: they are known in advance.
Input months must have no gaps (aggregate_monthly fills them).
"""
import numpy as np
import pandas as pd

from src.ingestion.macro_api import load_macro
from src.utils.helpers import get_project_root, holiday_table

LAGS = (1, 2, 3, 12)
WINDOWS = (3, 6)
FIXED_CATEGORIES = ("Rent", "Utilities", "Subscriptions")
TOTAL_EXPENSES = "Total expenses"
ID_COLUMNS = ["user_id", "category", "month", "target"]


def feature_columns(df: pd.DataFrame) -> list:
    """Every column that is a model input (not an id and not the target)."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def _series_table(monthly: pd.DataFrame) -> pd.DataFrame:
    """Long table: user_id, month, category, target (the value to predict)."""
    m = monthly.copy()
    m["target"] = np.where(m["category"] == "Income", m["income"], m["expenses"])
    base = m[["user_id", "month", "category", "target"]]
    total = (
        monthly.drop_duplicates(["user_id", "month"])[["user_id", "month", "total_expenses"]]
        .rename(columns={"total_expenses": "target"})
    )
    total["category"] = TOTAL_EXPENSES
    return pd.concat([base, total], ignore_index=True)


def _user_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Income volatility and fixed-cost share per user and month (past months only)."""
    users = monthly.drop_duplicates(["user_id", "month"])[
        ["user_id", "month", "total_income", "total_expenses"]
    ]
    fixed = (
        monthly[monthly["category"].isin(FIXED_CATEGORIES)]
        .groupby(["user_id", "month"])["expenses"].sum()
        .rename("fixed_expenses").reset_index()
    )
    users = users.merge(fixed, on=["user_id", "month"], how="left")
    users["fixed_expenses"] = users["fixed_expenses"].fillna(0.0)
    users = users.sort_values(["user_id", "month"])

    g = users.groupby("user_id")
    users["income_vol_6"] = g["total_income"].transform(lambda s: s.shift(1).rolling(6).std())
    fixed_3 = g["fixed_expenses"].transform(lambda s: s.shift(1).rolling(3).sum())
    total_3 = g["total_expenses"].transform(lambda s: s.shift(1).rolling(3).sum())
    users["fixed_share_3"] = fixed_3 / total_3.replace(0, np.nan)
    return users[["user_id", "month", "income_vol_6", "fixed_share_3"]]


def _macro_features(macro: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Macro data lagged one month, forward-filled where it has not been published yet."""
    macro = macro.set_index("month")[["hicp_yoy", "ecb_rate"]].sort_index()
    months = pd.date_range(min(macro.index.min(), first), last, freq="MS")
    macro = macro.reindex(months).ffill().shift(1)   # row month T sees month T-1
    macro.columns = ["hicp_lag1", "ecb_rate_lag1"]
    return macro.rename_axis("month").reset_index()


def build_features(monthly: pd.DataFrame, macro: pd.DataFrame | None = None,
                   country: str = "DE", subdiv: str | None = "BE",
                   drop_incomplete: bool = True) -> pd.DataFrame:
    """Feature table from the output of aggregate_monthly_by_category()."""
    out = _series_table(monthly).sort_values(["user_id", "category", "month"]).reset_index(drop=True)
    g = out.groupby(["user_id", "category"])["target"]

    for k in LAGS:
        out[f"lag_{k}"] = g.shift(k)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        out[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).std())

    out = out.merge(_user_features(monthly), on=["user_id", "month"], how="left")

    out["month_sin"] = np.sin(2 * np.pi * out["month"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].dt.month / 12)
    years = out["month"].dt.year
    cal = holiday_table(int(years.min()), int(years.max()), country, subdiv)
    out = out.merge(cal, on="month", how="left")

    if macro is None:
        macro = load_macro()
    out = out.merge(_macro_features(macro, out["month"].min(), out["month"].max()),
                    on="month", how="left")

    out = out.sort_values(["user_id", "category", "month"]).reset_index(drop=True)
    if drop_incomplete:
        out = out.dropna(subset=["lag_12"]).reset_index(drop=True)
    return out


def build_features_parquet() -> pd.DataFrame:
    root = get_project_root()
    monthly = pd.read_parquet(root / "data" / "processed" / "monthly.parquet")
    feats = build_features(monthly)
    path = root / "data" / "processed" / "features.parquet"
    feats.to_parquet(path, index=False)
    print(f"{len(feats):,} rows, {feats['user_id'].nunique()} users, "
          f"{feats['category'].nunique()} series, "
          f"{feats['month'].min():%Y-%m} to {feats['month'].max():%Y-%m}")
    nan_share = feats[feature_columns(feats)].isna().mean()
    print("Features with missing values:")
    print(nan_share[nan_share > 0].round(3).to_string() or "none")
    print(f"Saved {path}")
    return feats


if __name__ == "__main__":
    build_features_parquet()
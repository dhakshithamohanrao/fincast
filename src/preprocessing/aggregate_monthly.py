"""Turn transactions into monthly series.

aggregate_monthly():              one user, one row per month (used by the Overview page).
aggregate_monthly_by_category():  many users, one row per user, month and category
                                  (used to train the forecasting models).
"""
import pandas as pd

from src.utils.helpers import get_project_root

MONTHLY_COLUMNS = ["month", "income", "expenses", "savings", "balance_end"]
MONEY_COLUMNS = ["income", "expenses", "savings", "balance_end"]


def aggregate_monthly(df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """One row per month: income, expenses (positive numbers), savings and balance_end.

    balance_end is the running total of savings since the first month in the data.
    It is not the real bank balance, because the opening balance is unknown.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    tx = df[["date", "amount"]].copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["amount"] = pd.to_numeric(tx["amount"])
    tx["month"] = tx["date"].dt.to_period("M").dt.to_timestamp()
    tx["income"] = tx["amount"].clip(lower=0)
    tx["expenses"] = (-tx["amount"]).clip(lower=0)

    monthly = tx.groupby("month")[["income", "expenses"]].sum()

    if fill_missing:
        all_months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
        monthly = monthly.reindex(all_months, fill_value=0.0)
    monthly.index.name = "month"

    monthly["savings"] = monthly["income"] - monthly["expenses"]
    monthly["balance_end"] = monthly["savings"].cumsum()
    monthly[MONEY_COLUMNS] = monthly[MONEY_COLUMNS].round(2)
    return monthly.reset_index()


def _build_grid(tx: pd.DataFrame, cats: list) -> pd.MultiIndex:
    """Every user x every month between their first and last x every category."""
    spans = tx.groupby("user_id")["month"].agg(["min", "max"])
    keys = []
    for user, row in spans.iterrows():
        for month in pd.date_range(row["min"], row["max"], freq="MS"):
            for cat in cats:
                keys.append((user, month, cat))
    return pd.MultiIndex.from_tuples(keys, names=["user_id", "month", "category"])


def aggregate_monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user, month and category, with missing months filled with 0.

    Needs columns: user_id, date, amount, category.
    Columns: user_id, month, category, income, expenses (this category only),
    then total_income, total_expenses, savings, balance_end (whole user-month,
    repeated on every category row of that month).
    """
    tx = df[["user_id", "date", "amount", "category"]].copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["amount"] = pd.to_numeric(tx["amount"])
    tx["month"] = tx["date"].dt.to_period("M").dt.to_timestamp()
    tx["income"] = tx["amount"].clip(lower=0)
    tx["expenses"] = (-tx["amount"]).clip(lower=0)

    cats = sorted(tx["category"].unique())
    grouped = tx.groupby(["user_id", "month", "category"])[["income", "expenses"]].sum()
    out = grouped.reindex(_build_grid(tx, cats), fill_value=0.0)

    totals = out.groupby(level=["user_id", "month"])[["income", "expenses"]].sum()
    totals.columns = ["total_income", "total_expenses"]
    totals["savings"] = totals["total_income"] - totals["total_expenses"]
    totals["balance_end"] = totals.groupby(level="user_id")["savings"].cumsum()

    out = out.reset_index().merge(totals.reset_index(), on=["user_id", "month"], how="left")
    money = ["income", "expenses", "total_income", "total_expenses", "savings", "balance_end"]
    out[money] = out[money].round(2)
    return out


def reconcile(raw: pd.DataFrame, monthly: pd.DataFrame) -> float:
    """Largest difference between raw and monthly totals, per user (should be ~0)."""
    amount = pd.to_numeric(raw["amount"])
    raw_inc = amount.clip(lower=0).groupby(raw["user_id"]).sum()
    raw_exp = (-amount).clip(lower=0).groupby(raw["user_id"]).sum()
    mon_inc = monthly.groupby("user_id")["income"].sum()
    mon_exp = monthly.groupby("user_id")["expenses"].sum()
    return float(max((raw_inc - mon_inc).abs().max(), (raw_exp - mon_exp).abs().max()))


def build_monthly_parquet() -> pd.DataFrame:
    """Read the synthetic transactions, aggregate, check totals, save monthly.parquet."""
    root = get_project_root()
    raw = pd.read_csv(root / "data" / "raw" / "synthetic_transactions.csv",
                      parse_dates=["date"])
    monthly = aggregate_monthly_by_category(raw)

    diff = reconcile(raw, monthly)
    if diff > 1.0:  # rounding to 2 decimals can add up to a few cents
        raise ValueError(f"Monthly totals do not match raw data (max diff {diff:.2f})")

    out = root / "data" / "processed" / "monthly.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(out, index=False)
    print(f"{len(monthly):,} rows, {monthly['user_id'].nunique()} users, "
          f"{monthly['category'].nunique()} categories")
    print(f"Max difference vs raw totals: {diff:.4f}")
    print(f"Saved {out}")
    return monthly


if __name__ == "__main__":
    build_monthly_parquet()
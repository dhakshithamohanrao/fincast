"""Turn transactions into monthly series (first version, without categories)."""
import pandas as pd

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
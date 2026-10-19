"""Data quality functions: turn raw transactions into clean ones."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "description", "amount"]


def clean_transactions(df: pd.DataFrame, expenses_positive: bool = False) -> pd.DataFrame:
    """Return a cleaned copy of a transactions table.

    Expects the columns date, description, amount. Expenses end up negative and
    income positive. Set expenses_positive=True if the file shows expenses as
    positive numbers, and the signs will be flipped.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    start = len(df)
    out = df[REQUIRED_COLUMNS].copy()

    out["description"] = (
        out["description"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    invalid = out["date"].isna() | out["amount"].isna() | (out["description"] == "")
    out = out.loc[~invalid].copy()
    n_invalid = int(invalid.sum())

    zero = out["amount"] == 0
    out = out.loc[~zero].copy()
    n_zero = int(zero.sum())

    before = len(out)
    out = out.drop_duplicates(subset=REQUIRED_COLUMNS)
    n_duplicates = before - len(out)

    if expenses_positive:
        out["amount"] = -out["amount"]

    out = out.sort_values("date", kind="stable").reset_index(drop=True)

    logger.info(
        "Cleaning: %d rows in, %d out (invalid: %d, zero amount: %d, duplicates: %d)",
        start, len(out), n_invalid, n_zero, n_duplicates,
    )
    return out

MAX_ABS_AMOUNT = 20_000  # a single transaction above this is almost certainly a data error


def validate_transactions(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems (empty list = data looks fine).

    Expects the standard schema: user_id, date, description, amount.
    """
    problems = []
    required = {"user_id", "date", "description", "amount"}
    missing = required - set(df.columns)
    if missing:
        return [f"Missing columns: {sorted(missing)}"]

    dates = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = int(dates.isna().sum())
    if n_bad_dates:
        problems.append(f"{n_bad_dates} rows have an unreadable date")

    n_future = int((dates > pd.Timestamp.today().normalize()).sum())
    if n_future:
        problems.append(f"{n_future} rows are dated in the future")

    desc = df["description"].astype(str).str.strip()
    n_empty = int((desc == "").sum() + df["description"].isna().sum())
    if n_empty:
        problems.append(f"{n_empty} rows have an empty description")

    amounts = pd.to_numeric(df["amount"], errors="coerce")
    n_bad_amounts = int(amounts.isna().sum())
    if n_bad_amounts:
        problems.append(f"{n_bad_amounts} rows have a non-numeric amount")

    n_huge = int((amounts.abs() > MAX_ABS_AMOUNT).sum())
    if n_huge:
        problems.append(f"{n_huge} rows have an amount above {MAX_ABS_AMOUNT:,}")

    n_zero = int((amounts == 0).sum())
    if n_zero:
        problems.append(f"{n_zero} rows have an amount of exactly 0")

    return problems
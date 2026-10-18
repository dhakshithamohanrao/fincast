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
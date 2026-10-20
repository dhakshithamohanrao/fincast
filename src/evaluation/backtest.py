"""Rolling-origin backtest: train on the past, predict the next month, move forward.

Never use random train/test splits for time series.
"""
import numpy as np
import pandas as pd

# Variable spending is where forecasting adds value (rent and subscriptions are fixed).
VARIABLE_SERIES = ("Dining", "Groceries", "Health", "Leisure", "Other", "Shopping", "Transport")


def backtest_folds(df: pd.DataFrame, min_train_months: int = 12,
                   train_users=None, test_users=None):
    """Yield (train_positions, test_positions, test_month), expanding window.

    Fold k trains on all rows with month < test_month and tests on month == test_month.
    df needs a default RangeIndex and columns user_id, month.
    train_users / test_users: optionally restrict the users (held-out-user test).
    """
    months = np.sort(df["month"].unique())
    if len(months) <= min_train_months:
        raise ValueError(f"Need more than {min_train_months} months, found {len(months)}.")
    for test_month in months[min_train_months:]:
        train = df["month"] < test_month
        test = df["month"] == test_month
        if train_users is not None:
            train &= df["user_id"].isin(train_users)
        if test_users is not None:
            test &= df["user_id"].isin(test_users)
        yield np.flatnonzero(train), np.flatnonzero(test), pd.Timestamp(test_month)


def split_users(df: pd.DataFrame, test_share: float = 0.2, seed: int = 42):
    """Split users into (train_users, holdout_users) for the cold-start test."""
    users = np.sort(df["user_id"].unique())
    np.random.default_rng(seed).shuffle(users)
    n_test = max(1, int(round(len(users) * test_share)))
    return list(users[n_test:]), list(users[:n_test])
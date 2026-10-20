import numpy as np
import pandas as pd

from src.features.build_features import build_features, feature_columns
from src.ingestion.synthetic_generator import generate_user
from src.preprocessing.aggregate_monthly import aggregate_monthly_by_category


def _tiny_monthly() -> pd.DataFrame:
    raw = pd.concat([
        generate_user("freelancer", 30, seed=1, user_id="a"),
        generate_user("salaried_single", 30, seed=2, user_id="b"),
    ])
    return aggregate_monthly_by_category(raw)


def _fake_macro() -> pd.DataFrame:
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    return pd.DataFrame({"month": months, "hicp_yoy": np.linspace(1, 3, len(months)),
                         "ecb_rate": 2.0})


def test_features_do_not_use_the_future():
    monthly, macro = _tiny_monthly(), _fake_macro()
    cutoff = pd.Timestamp(sorted(monthly["month"].unique())[20])

    base = build_features(monthly, macro)

    changed = monthly.copy()
    future = changed["month"] >= cutoff
    for col in ["income", "expenses", "total_income", "total_expenses", "savings", "balance_end"]:
        changed.loc[future, col] = changed.loc[future, col] * 3 + 100
    perturbed = build_features(changed, macro)

    cols = ["user_id", "category", "month"] + feature_columns(base)
    a = base.loc[base["month"] <= cutoff, cols].reset_index(drop=True)
    b = perturbed.loc[perturbed["month"] <= cutoff, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)

    # Sanity check: the change did alter the targets, so the test is not vacuous
    t1 = base.loc[base["month"] == cutoff, "target"].to_numpy()
    t2 = perturbed.loc[perturbed["month"] == cutoff, "target"].to_numpy()
    assert not np.allclose(t1, t2)
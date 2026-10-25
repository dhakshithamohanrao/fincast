"""Phase 7b: route fixed series to lag_1, and a cold-start test on held-out users."""
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.evaluation.backtest import VARIABLE_SERIES, backtest_folds, split_users
from src.evaluation.metrics import coverage, mae
from src.features.build_features import TOTAL_EXPENSES, feature_columns
from src.training.train_lgbm import PARAMS, REQUIRED_IMPROVEMENT, _fit_predict
from src.utils.helpers import get_project_root

COPY_LAST = ("Rent", "Subscriptions")   # fixed amounts: copy last month


def _run_folds(df, feats, train_users=None, test_users=None):
    parts = []
    for tr, te, month in backtest_folds(df, train_users=train_users, test_users=test_users):
        X_tr, y_tr, X_te = df.loc[tr, feats], df.loc[tr, "target"], df.loc[te, feats]
        point = np.clip(_fit_predict("regression_l1", X_tr, y_tr, X_te), 0, None)
        lo = np.clip(_fit_predict("quantile", X_tr, y_tr, X_te, alpha=0.1), 0, None)
        hi = np.clip(_fit_predict("quantile", X_tr, y_tr, X_te, alpha=0.9), 0, None)
        part = df.loc[te, ["user_id", "category", "month", "target", "lag_1", "lag_12"]].copy()
        part["pred"], part["lo"], part["hi"] = point, np.minimum(lo, point), np.maximum(hi, point)
        fixed = part["category"].isin(COPY_LAST)
        for col in ("pred", "lo", "hi"):
            part.loc[fixed, col] = part.loc[fixed, "lag_1"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _report(res, label):
    var = res[res["category"].isin(VARIABLE_SERIES)]
    model_mae = mae(var["target"], var["pred"])
    base_mae = mae(var["target"], var["lag_12"])
    threshold = base_mae * (1 - REQUIRED_IMPROVEMENT)
    cov = coverage(var["target"], var["lo"], var["hi"])

    tot = res[res["category"] == TOTAL_EXPENSES].set_index(["user_id", "month"])
    cats = res[~res["category"].isin(["Income", TOTAL_EXPENSES])]
    summed = cats.groupby(["user_id", "month"])["pred"].sum().reindex(tot.index)
    total_sum = mae(tot["target"], summed)

    print(f"\n[{label}] variable MAE {model_mae:.2f} | baseline {base_mae:.2f} | "
          f"need < {threshold:.2f} | champion {model_mae < threshold}")
    print(f"[{label}] coverage {cov:.3f} | total MAE (sum of categories) {total_sum:.2f}")
    for c in ("Rent", "Subscriptions", "Utilities"):
        g = res[res["category"] == c]
        print(f"[{label}] {c} MAE {mae(g['target'], g['pred']):.2f}")
    return {"variable_mae": model_mae, "baseline_mae": base_mae, "threshold": threshold,
            "coverage_80": cov, "total_mae_sum": total_sum}


def run() -> None:
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet").reset_index(drop=True)
    df["series"] = df["category"].astype(pd.CategoricalDtype(sorted(df["category"].unique())))
    feats = list(dict.fromkeys(feature_columns(df) + ["series"]))
     

    train_users, holdout = split_users(df)
    print(f"Cold start: {len(train_users)} train users, {len(holdout)} held-out users")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("forecaster")

    print("Seen-user backtest (fixed series copied)...")
    seen = _run_folds(df, feats)
    m1 = _report(seen, "seen users")
    with mlflow.start_run(run_name="lgbm_global_fixed_routing"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(m1)

    print("Cold-start backtest...")
    cold = _run_folds(df, feats, train_users=train_users, test_users=holdout)
    m2 = _report(cold, "cold start")
    with mlflow.start_run(run_name="lgbm_cold_start"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(m2)

    seen.to_parquet(root / "data" / "processed" / "backtest_predictions_v2.parquet")


if __name__ == "__main__":
    run()

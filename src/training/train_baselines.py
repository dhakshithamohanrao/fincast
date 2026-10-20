"""The bar to beat: three naive predictors, scored with the rolling backtest.

Baselines use columns that already exist in features.parquet:
  last_month -> lag_1, same_month_last_year -> lag_12, avg_3_months -> roll_mean_3
Results go to MLflow (experiment 'baselines') and to reports/*.csv.
"""
import mlflow
import numpy as np
import pandas as pd

from src.evaluation.backtest import VARIABLE_SERIES, backtest_folds
from src.evaluation.metrics import mae, rmse, smape
from src.utils.helpers import get_project_root

BASELINES = {
    "last_month": "lag_1",
    "same_month_last_year": "lag_12",
    "avg_3_months": "roll_mean_3",
}


def load_features_with_types() -> pd.DataFrame:
    root = get_project_root()
    feats = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    types = pd.read_csv(root / "data" / "raw" / "synthetic_transactions.csv",
                        usecols=["user_id", "archetype"]).drop_duplicates("user_id")
    return feats.merge(types, on="user_id", how="left")


def run() -> None:
    root = get_project_root()
    df = load_features_with_types()
    folds = list(backtest_folds(df))
    test = df.iloc[np.concatenate([t for _, t, _ in folds])].copy()
    var = test[test["category"].isin(VARIABLE_SERIES)]

    for name, col in BASELINES.items():
        test[name] = (test["target"] - test[col]).abs()
    names = list(BASELINES)

    overall = test[test["category"].isin(VARIABLE_SERIES)][names].mean().round(2)
    by_type = test[test["category"].isin(VARIABLE_SERIES)].groupby("archetype")[names].mean().round(2)
    by_series = test.groupby("category")[names].mean().round(2)

    print(f"Backtest: {len(folds)} folds, test months "
          f"{test['month'].min():%Y-%m} to {test['month'].max():%Y-%m}, {len(test):,} test rows\n")
    print("MAE on VARIABLE spending (all users):")
    print(overall.to_string(), "\n")
    print("MAE on variable spending by user type:")
    print(by_type.to_string(), "\n")
    print("MAE by series (all users):")
    print(by_series.to_string(), "\n")
    best = overall.idxmin()
    print(f"Best baseline: {best}  (variable-spending MAE {overall[best]:.2f}). "
          f"A model must beat this by at least 10%: below {overall[best] * 0.9:.2f}.")

    out = root / "reports"
    out.mkdir(exist_ok=True)
    by_series.to_csv(out / "baseline_mae_by_series.csv")
    by_type.to_csv(out / "baseline_mae_by_type.csv")

    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("baselines")
    for name, col in BASELINES.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("feature", col)
            mlflow.log_param("n_folds", len(folds))
            mlflow.log_metric("mae_variable", mae(var["target"], var[col]))
            mlflow.log_metric("rmse_variable", rmse(var["target"], var[col]))
            mlflow.log_metric("smape_variable", smape(var["target"], var[col]))
            for arch, g in var.groupby("archetype"):
                mlflow.log_metric(f"mae_variable_{arch}", mae(g["target"], g[col]))
            for cat, g in test.groupby("category"):
                mlflow.log_metric("mae_" + cat.lower().replace(" ", "_"), mae(g["target"], g[col]))
            for i, (_, t, month) in enumerate(folds):
                sub = df.iloc[t]
                sub = sub[sub["category"].isin(VARIABLE_SERIES)]
                with mlflow.start_run(run_name=f"fold_{i}_{month:%Y-%m}", nested=True):
                    mlflow.log_metric("mae_variable", mae(sub["target"], sub[col]))


if __name__ == "__main__":
    run()
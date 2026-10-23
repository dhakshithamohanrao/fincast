"""Phase 7: global LightGBM (point + 10%/90% quantiles) on the 12-fold backtest."""
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.evaluation.backtest import VARIABLE_SERIES, backtest_folds
from src.evaluation.metrics import coverage, mae, pinball_loss
from src.features.build_features import TOTAL_EXPENSES, feature_columns
from src.utils.helpers import get_project_root

REQUIRED_IMPROVEMENT = 0.10
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)


def _fit_predict(objective, X_tr, y_tr, X_te, alpha=None):
    kwargs = {"alpha": alpha} if alpha is not None else {}
    model = lgb.LGBMRegressor(objective=objective, **kwargs, **PARAMS)
    return model.fit(X_tr, y_tr).predict(X_te)


def run() -> None:
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet").reset_index(drop=True)
    df["series"] = df["category"].astype(pd.CategoricalDtype(sorted(df["category"].unique())))
    feats = list(dict.fromkeys(feature_columns(df) + ["series"]))
    print(f"{len(feats)} features, categorical: series")

    parts = []
    for tr, te, month in backtest_folds(df):
        X_tr, y_tr, X_te = df.loc[tr, feats], df.loc[tr, "target"], df.loc[te, feats]
        point = np.clip(_fit_predict("regression_l1", X_tr, y_tr, X_te), 0, None)
        lo = np.clip(_fit_predict("quantile", X_tr, y_tr, X_te, alpha=0.1), 0, None)
        hi = np.clip(_fit_predict("quantile", X_tr, y_tr, X_te, alpha=0.9), 0, None)
        part = df.loc[te, ["user_id", "category", "month", "target", "lag_12"]].copy()
        part["pred"] = point
        part["lo"] = np.minimum(lo, point)   # keep the interval around the point forecast
        part["hi"] = np.maximum(hi, point)
        parts.append(part)
        print(f"  fold {month:%Y-%m} done")
    res = pd.concat(parts, ignore_index=True)

    # Champion rule on variable spending
    var = res[res["category"].isin(VARIABLE_SERIES)]
    model_mae = mae(var["target"], var["pred"])
    base_mae = mae(var["target"], var["lag_12"])     # same_month_last_year
    threshold = base_mae * (1 - REQUIRED_IMPROVEMENT)
    champion = model_mae < threshold

    cov = coverage(var["target"], var["lo"], var["hi"])
    pin10 = pinball_loss(var["target"], var["lo"], 0.1)
    pin90 = pinball_loss(var["target"], var["hi"], 0.9)

    by_series = {c: mae(g["target"], g["pred"]) for c, g in res.groupby("category")}

    # Total expenses: direct model vs sum of category forecasts
    tot = res[res["category"] == TOTAL_EXPENSES].set_index(["user_id", "month"])
    cats = res[~res["category"].isin(["Income", TOTAL_EXPENSES])]
    summed = cats.groupby(["user_id", "month"])["pred"].sum().reindex(tot.index)
    mae_total_direct = mae(tot["target"], tot["pred"])
    mae_total_sum = mae(tot["target"], summed)

    print(f"\nVariable MAE: model {model_mae:.2f} | baseline {base_mae:.2f} | need < {threshold:.2f}")
    print(f"Champion: {champion}")
    print(f"80% interval coverage: {cov:.3f} (target about 0.80)")
    print(f"Pinball 0.1: {pin10:.2f}, 0.9: {pin90:.2f}")
    print(f"Total expenses MAE: direct {mae_total_direct:.2f} | sum of categories {mae_total_sum:.2f}")
    print("\nMAE by series:")
    print(pd.Series(by_series).round(2).to_string())

    res.to_parquet(root / "data" / "processed" / "backtest_predictions.parquet")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("forecaster")
    with mlflow.start_run(run_name="lgbm_global"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({
            "variable_mae": model_mae, "baseline_mae": base_mae, "threshold": threshold,
            "coverage_80": cov, "pinball_10": pin10, "pinball_90": pin90,
            "total_mae_direct": mae_total_direct, "total_mae_sum": mae_total_sum,
        })
        mlflow.set_tag("champion", str(champion))


if __name__ == "__main__":
    run()
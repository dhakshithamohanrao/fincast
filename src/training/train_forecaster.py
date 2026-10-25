"""Phase 7.9: train the final forecasting models on all data and export the champion.

The champion decision uses the backtest saved by train_lgbm_v2.py. If the model
passes the rule, three models (median, 10% and 90% quantile) are trained on all
rows and saved to models/ together with model_meta.json.
"""
import json
from datetime import date

import joblib
import lightgbm as lgb
import pandas as pd

from src.evaluation.metrics import mae
from src.features.build_features import feature_columns
from src.training.train_lgbm import PARAMS
from src.training.train_lgbm_v2 import COPY_LAST, _report
from src.utils.helpers import get_project_root

MODEL_FILES = {
    "median": "forecaster_median.joblib",
    "q10": "forecaster_q10.joblib",
    "q90": "forecaster_q90.joblib",
}


def load_features():
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "features.parquet").reset_index(drop=True)
    categories = [str(c) for c in sorted(df["category"].unique())]
    df["series"] = df["category"].astype(pd.CategoricalDtype(categories))
    feats = list(dict.fromkeys(feature_columns(df) + ["series"]))
    return df, feats, categories


def load_backtest():
    path = get_project_root() / "data" / "processed" / "backtest_predictions_v2.parquet"
    if not path.exists():
        raise FileNotFoundError("Run  python -m src.training.train_lgbm_v2  first.")
    return pd.read_parquet(path)


def series_table(res: pd.DataFrame) -> pd.DataFrame:
    """MAE of the model and both baselines for every series."""
    rows = {}
    for category, g in res.groupby("category"):
        rows[category] = {
            "model": mae(g["target"], g["pred"]),
            "last_month": mae(g["target"], g["lag_1"]),
            "same_month_last_year": mae(g["target"], g["lag_12"]),
        }
    return pd.DataFrame(rows).T.round(2)


def fit_final(df: pd.DataFrame, feats: list) -> dict:
    X, y = df[feats], df["target"]
    models = {
        "median": lgb.LGBMRegressor(objective="regression_l1", **PARAMS),
        "q10": lgb.LGBMRegressor(objective="quantile", alpha=0.1, **PARAMS),
        "q90": lgb.LGBMRegressor(objective="quantile", alpha=0.9, **PARAMS),
    }
    for name, model in models.items():
        print(f"  fitting {name}...")
        model.fit(X, y)
    return models


def run() -> None:
    root = get_project_root()
    out_dir = root / "models"
    out_dir.mkdir(exist_ok=True)

    df, feats, categories = load_features()
    res = load_backtest()

    metrics = _report(res, "backtest")
    is_champion = metrics["variable_mae"] < metrics["threshold"]

    print("\nMAE by series (model vs baselines):")
    print(series_table(res).to_string())

    df = df[df["target"].notna()]
    meta = {
        "model_version": f"lgbm-{date.today().isoformat()}",
        "created": date.today().isoformat(),
        "champion": "lgbm" if is_champion else "baseline_same_month_last_year",
        "baseline": "same_month_last_year (lag_12)",
        "copy_last_month": list(COPY_LAST),
        "features": feats,
        "categorical_feature": "series",
        "series_categories": categories,
        "trained_months": [f"{df['month'].min():%Y-%m}", f"{df['month'].max():%Y-%m}"],
        "n_rows": int(len(df)),
        "params": PARAMS,
        "backtest_metrics": {k: float(v) for k, v in metrics.items()},
    }

    if is_champion:
        print(f"\nChampion: lgbm. Training final models on {len(df):,} rows...")
        models = fit_final(df, feats)
        for name, model in models.items():
            joblib.dump(model, out_dir / MODEL_FILES[name])
    else:
        print("\nThe model did not beat the baseline by the required margin.")
        print("The baseline stays champion; no model files were written.")

    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'model_meta.json'}")


if __name__ == "__main__":
    run()
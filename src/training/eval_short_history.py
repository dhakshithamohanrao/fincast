"""Does the short-history fill (lag_12 := roll_mean_6) still beat a simple baseline?"""
import pandas as pd

from src.evaluation.backtest import VARIABLE_SERIES, backtest_folds
from src.evaluation.metrics import mae
from src.features.build_features import feature_columns
from src.training.train_lgbm import PARAMS, _fit_predict
import numpy as np

df = pd.read_parquet("data/processed/features.parquet").reset_index(drop=True)
df["series"] = df["category"].astype(pd.CategoricalDtype(sorted(df["category"].unique())))
feats = list(dict.fromkeys(feature_columns(df) + ["series"]))

parts = []
for tr, te, month in backtest_folds(df):
    X_tr, y_tr = df.loc[tr, feats], df.loc[tr, "target"]
    X_te = df.loc[te, feats].copy()
    full = np.clip(_fit_predict("regression_l1", X_tr, y_tr, X_te), 0, None)
    X_te["lag_12"] = X_te["roll_mean_6"]          # the short-history fill
    filled = np.clip(_fit_predict("regression_l1", X_tr, y_tr, X_te), 0, None)
    part = df.loc[te, ["category", "target", "roll_mean_3", "lag_12"]].copy()
    part["full"], part["filled"] = full, filled
    parts.append(part)
    print(f"fold {month:%Y-%m} done")

res = pd.concat(parts)
var = res[res["category"].isin(VARIABLE_SERIES)]
print(f"\nVariable MAE, model with real lag_12 : {mae(var['target'], var['full']):.2f}")
print(f"Variable MAE, model with filled lag_12: {mae(var['target'], var['filled']):.2f}")
print(f"Variable MAE, 3-month average         : {mae(var['target'], var['roll_mean_3']):.2f}")
print(f"Variable MAE, same month last year    : {mae(var['target'], var['lag_12']):.2f}")
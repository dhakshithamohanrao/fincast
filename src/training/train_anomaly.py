"""Phase 7.10: train and export the anomaly model (Isolation Forest + simple rules).

Evaluation: 20% of the users are held out and get fake anomalies injected
(duplicate charges, amount spikes, unknown merchants). The model is trained
only on the other users' real rows.
"""
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features.anomaly_features import (
    FEATURE_COLUMNS,
    SPIKE_Z,
    build_anomaly_features,
    rule_flags,
)
from src.utils.helpers import get_project_root, load_config

REQUIRED_COLUMNS = ["user_id", "date", "description", "amount"]
CONTAMINATION_GRID = [0.001, 0.002, 0.005, 0.01, 0.02]
INJECT_RATE = 0.002          # share of rows copied into EACH kind of fake anomaly
KINDS = ("duplicate", "spike", "new_merchant")
FOREST = dict(n_estimators=200, n_jobs=-1)
SHOW = ["setting", "flagged", "false_alarms", "precision", "recall", "f1",
        "recall_duplicate", "recall_spike", "recall_new_merchant"]


def load_transactions() -> pd.DataFrame:
    path = get_project_root() / "data" / "raw" / "synthetic_transactions.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing}; the file has {list(df.columns)}")
    return df


def inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.assign(kind="normal")
    expenses = df[df["amount"] < 0]
    n = max(1, int(len(df) * INJECT_RATE))
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))

    def sample():
        return expenses.sample(n, random_state=int(rng.integers(1_000_000))).copy()

    dup = sample()
    dup["kind"] = "duplicate"                      # the same charge a second time

    spike = sample()
    spike["amount"] = (spike["amount"] * rng.uniform(5, 10, size=len(spike))).round(2)
    spike["kind"] = "spike"                        # 5-10 times the usual amount

    new = sample()
    new["description"] = ["".join(rng.choice(letters, 10)) + " shop" for _ in range(len(new))]
    new["amount"] = (new["amount"] * rng.uniform(2, 4, size=len(new))).round(2)
    new["kind"] = "new_merchant"                   # unknown shop, 2-4 times the usual amount

    return pd.concat([df, dup, spike, new], ignore_index=True)


def _scores(flagged: np.ndarray, kinds: pd.Series) -> dict:
    injected = (kinds != "normal").to_numpy()
    tp = int((flagged & injected).sum())
    n_flagged = int(flagged.sum())
    precision = tp / max(n_flagged, 1)
    recall = tp / int(injected.sum())
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    row = {"flagged": n_flagged, "false_alarms": int((flagged & ~injected).sum()),
           "precision": precision, "recall": recall, "f1": f1}
    for kind in KINDS:
        mask = (kinds == kind).to_numpy()
        row[f"recall_{kind}"] = float((flagged & mask).sum() / mask.sum())
    return row


def evaluate(scores_train, scores_eval, kinds: pd.Series, rules: pd.DataFrame) -> pd.DataFrame:
    """Rules alone, the forest alone, and both together, for each contamination."""
    by_rule = (rules["duplicate"] | rules["spike"]).to_numpy()
    rows = [{"setting": "rules only", **_scores(by_rule, kinds)}]
    for c in CONTAMINATION_GRID:
        forest = scores_eval < np.quantile(scores_train, c)
        rows.append({"setting": f"forest {c}", "contamination": c, **_scores(forest, kinds)})
        rows.append({"setting": f"forest {c} + rules", "contamination": c,
                     **_scores(forest | by_rule, kinds)})
    return pd.DataFrame(rows)


def run() -> None:
    seed = load_config()["seed"]
    rng = np.random.default_rng(seed)
    df = load_transactions()

    users = np.array(sorted(df["user_id"].unique()))
    eval_users = set(rng.permutation(users)[: len(users) // 5])
    is_eval = df["user_id"].isin(eval_users)
    train_df = df[~is_eval].reset_index(drop=True)
    eval_df = inject_anomalies(df[is_eval].reset_index(drop=True), rng)
    print(f"Train: {len(train_df):,} rows | evaluation: {len(eval_df):,} rows "
          f"({int((eval_df['kind'] != 'normal').sum()):,} fake anomalies)")

    X_train = build_anomaly_features(train_df)
    X_eval = build_anomaly_features(eval_df)
    model = IsolationForest(contamination="auto", random_state=seed, **FOREST).fit(X_train)
    table = evaluate(model.score_samples(X_train), model.score_samples(X_eval),
                     eval_df["kind"], rule_flags(X_eval))

    print("\nDetection of injected anomalies on held-out users:")
    print(table[SHOW].round(3).to_string(index=False))

    combined = table[table["setting"].str.endswith("+ rules")]
    best = combined.loc[combined["f1"].idxmax()]
    best_c = float(best["contamination"])
    print(f"\nBest setting by F1: {best['setting']}")

    final = IsolationForest(contamination=best_c, random_state=seed, **FOREST)
    final.fit(build_anomaly_features(df))
    out_path = get_project_root() / "models" / "anomaly.joblib"
    joblib.dump({"model": final, "features": FEATURE_COLUMNS,
                 "contamination": best_c, "spike_z": SPIKE_Z}, out_path)
    print(f"Saved {out_path}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("anomaly")
    with mlflow.start_run(run_name="isolation_forest_plus_rules"):
        mlflow.log_params({**FOREST, "contamination": best_c, "spike_z": SPIKE_Z,
                           "n_rows": len(df)})
        mlflow.log_metrics({k: float(best[k]) for k in
                            ["precision", "recall", "f1", "false_alarms"]
                            + [f"recall_{x}" for x in KINDS]})


if __name__ == "__main__":
    run()
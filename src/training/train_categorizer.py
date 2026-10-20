"""Train the text categoriser: TF-IDF (character n-grams) + logistic regression.

Run from the project root:  python -m src.training.train_categorizer
"""
import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from src.utils.helpers import get_project_root, load_config

MAX_ROWS = 60_000  # keeps training to about a minute; set to None to use every row


def clean_text(s: pd.Series) -> pd.Series:
    """Lower-case and strip digits/dates so 'Miete 10/2023' and 'Miete 11/2023' match."""
    return (s.astype(str).str.lower()
            .str.replace(r"[\d/]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def load_training_data() -> pd.DataFrame:
    root = get_project_root()
    raw = root / load_config()["paths"]["raw"]
    syn = pd.read_csv(raw / "synthetic_transactions.csv", usecols=["user_id", "description", "category"])
    pub = pd.read_csv(raw / "public_transactions.csv", usecols=["user_id", "description", "category"])
    pub["user_id"] = "pub_" + pub["user_id"].astype(str)
    df = pd.concat([syn, pub], ignore_index=True)
    df["text"] = clean_text(df["description"])
    return df[["user_id", "text", "category"]]


def make_model(seed: int):
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10, random_state=seed),
    )


def main() -> None:
    seed = load_config()["seed"]
    data = load_training_data()
    if MAX_ROWS and len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=seed).reset_index(drop=True)
    print(f"{len(data):,} rows, {data['user_id'].nunique()} users, "
          f"{data['text'].nunique()} unique descriptions")

    # Split by user: test users never appear in training, but shared merchants do
    cv = GroupKFold(n_splits=5)
    pred = cross_val_predict(make_model(seed), data["text"], data["category"],
                             cv=cv, groups=data["user_id"], n_jobs=-1)
    acc = accuracy_score(data["category"], pred)
    macro_f1 = f1_score(data["category"], pred, average="macro")
    print(f"Cross-validated accuracy on unseen users: {acc:.1%}  (macro F1 {macro_f1:.2f})")

    wrong = data[pred != data["category"]].assign(predicted=pred[pred != data["category"]])
    print("\nMost common mistakes:")
    if len(wrong):
        print(wrong.groupby(["text", "category", "predicted"]).size()
              .sort_values(ascending=False).head(15).to_string())
    else:
        print("none")

    report = classification_report(data["category"], pred, output_dict=True, zero_division=0)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("categorizer")
    with mlflow.start_run(run_name="tfidf_char_logreg_user_split"):
        mlflow.log_params({"ngram_range": "2-5", "C": 10, "n_rows": len(data),
                           "cv_folds": 5, "split": "GroupKFold by user"})
        mlflow.log_metrics({"accuracy": acc, "macro_f1": macro_f1})
        for label, scores in report.items():
            if label in data["category"].unique():
                mlflow.log_metric(f"f1_{label}", scores["f1-score"])

        final = make_model(seed).fit(data["text"], data["category"])
        out = get_project_root() / "models" / "categorizer.joblib"
        out.parent.mkdir(exist_ok=True)
        joblib.dump(final, out)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
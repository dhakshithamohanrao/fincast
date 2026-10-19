"""Train the text categoriser: TF-IDF (character n-grams) + logistic regression.

Run from the project root:  python -m src.training.train_categorizer
"""
import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from src.utils.helpers import get_project_root, load_config


def clean_text(s: pd.Series) -> pd.Series:
    """Lower-case and strip digits/dates so 'Miete 10/2023' and 'Miete 11/2023' match."""
    return (s.astype(str).str.lower()
            .str.replace(r"[\d/]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def load_training_data() -> pd.DataFrame:
    root = get_project_root()
    raw = root / load_config()["paths"]["raw"]
    frames = [pd.read_csv(raw / n, usecols=["description", "category"])
              for n in ["synthetic_transactions.csv", "public_transactions.csv"]]
    df = pd.concat(frames, ignore_index=True)
    df["text"] = clean_text(df["description"])
    return df[["text", "category"]].drop_duplicates().reset_index(drop=True)


def make_model(seed: int):
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10, random_state=seed),
    )


def main() -> None:
    seed = load_config()["seed"]
    data = load_training_data()
    print(f"{len(data)} unique descriptions, {data['category'].nunique()} categories")

    # Every description appears once, so each fold tests on descriptions it never saw
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(make_model(seed), data["text"], data["category"], cv=cv)
    acc = accuracy_score(data["category"], pred)
    macro_f1 = f1_score(data["category"], pred, average="macro")
    print(f"Cross-validated accuracy on unseen descriptions: {acc:.1%}  (macro F1 {macro_f1:.2f})")

    wrong = data[pred != data["category"]].assign(predicted=pred[pred != data["category"]])
    print("\nMisclassified:")
    print(wrong.to_string(index=False) if len(wrong) else "none")

    report = classification_report(data["category"], pred, output_dict=True, zero_division=0)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("categorizer")
    with mlflow.start_run(run_name="tfidf_char_logreg"):
        mlflow.log_params({"ngram_range": "2-5", "C": 10, "n_descriptions": len(data), "cv_folds": 5})
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
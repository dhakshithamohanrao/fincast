"""Assign a category to each transaction.

Step 1 (this file, so far): keyword rules from config/categories.yaml.
Step 2 (later): a TF-IDF + logistic regression model for lines the rules miss.
"""
import re
import joblib
import pandas as pd
import yaml

from src.utils.helpers import get_project_root


def load_rules() -> dict[str, re.Pattern]:
    """One compiled regex per category, in the order of categories.yaml."""
    path = get_project_root() / "config" / "categories.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # \b = keyword must start at a word boundary, so "rent" does not match "current"
    return {
        category: re.compile("|".join(r"\b" + re.escape(k.lower()) for k in keywords))
        for category, keywords in raw.items()
    }


def rule_categorize(descriptions: pd.Series, rules: dict | None = None) -> pd.Series:
    """Category per description, or NaN where no keyword matches. First match wins."""
    rules = rules or load_rules()
    lowered = descriptions.astype(str).str.lower().str.strip()
    result = pd.Series(pd.NA, index=descriptions.index, dtype="object")
    for category, pattern in rules.items():
        unmatched = result.isna()
        hit = lowered[unmatched].str.contains(pattern, regex=True)
        result.loc[hit[hit].index] = category
    return result

CONFIDENCE_THRESHOLD = 0.5  # below this the model's guess is replaced by "Other"


def _clean_text(s: pd.Series) -> pd.Series:
    """Must stay identical to clean_text() in src/training/train_categorizer.py.
    Duplicated on purpose so the deployed app does not need mlflow installed."""
    return (s.astype(str).str.lower()
            .str.replace(r"[\d/]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def categorize(df: pd.DataFrame, threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Add 'category' and 'category_source' (rule / model / fallback) to df.

    Rules first, the text model for the rest, and "Other" when the model is unsure.
    Expects a 'description' column.
    """
    out = df.copy()
    out["category"] = rule_categorize(out["description"])
    out["category_source"] = out["category"].notna().map({True: "rule", False: None})

    missing = out["category"].isna()
    if missing.any():
        model_path = get_project_root() / "models" / "categorizer.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            text = _clean_text(out.loc[missing, "description"])
            proba = model.predict_proba(text)
            best = proba.max(axis=1)
            labels = model.classes_[proba.argmax(axis=1)]
            confident = best >= threshold
            out.loc[missing, "category"] = [l if c else "Other" for l, c in zip(labels, confident)]
            out.loc[missing, "category_source"] = ["model" if c else "fallback" for c in confident]
        else:
            out.loc[missing, "category"] = "Other"
            out.loc[missing, "category_source"] = "fallback"
    return out


if __name__ == "__main__":
    from src.utils.helpers import load_config

    raw_dir = get_project_root() / load_config()["paths"]["raw"]
    for name in ["synthetic_transactions.csv", "public_transactions.csv"]:
        df = pd.read_csv(raw_dir / name)
        pred = rule_categorize(df["description"])
        covered = pred.notna()
        print(f"\n== {name}: {len(df):,} rows ==")
        print(f"Coverage (rows the rules label): {covered.mean():.1%}")
        print(f"Accuracy on covered rows:        {(pred[covered] == df.loc[covered, 'category']).mean():.1%}")
        print("Most common descriptions the rules MISS:")
        print(df.loc[~covered, "description"].value_counts().head(8).to_string())
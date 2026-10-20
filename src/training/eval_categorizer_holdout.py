"""Honest test for new merchants: hold out whole item names, not rows.

Run from the project root:  python -m src.training.eval_categorizer_holdout
"""
import numpy as np
import pandas as pd

from src.training.train_categorizer import load_training_data, make_model
from src.utils.helpers import load_config


def main() -> None:
    seed = load_config()["seed"]
    data = load_training_data()
    is_public = data["user_id"].str.startswith("pub_")

    pub_items = np.sort(data.loc[is_public, "text"].unique())
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(pub_items, size=int(0.3 * len(pub_items)), replace=False))

    test_mask = is_public & data["text"].isin(held_out)
    train, test = data[~test_mask], data[test_mask]
    print(f"Train: {len(train):,} rows, {train['text'].nunique()} descriptions")
    print(f"Test:  {len(test):,} rows, {len(held_out)} unseen descriptions")

    model = make_model(seed).fit(train["text"], train["category"])
    pred = model.predict(test["text"])
    print(f"\nAccuracy on unseen descriptions: {(pred == test['category']).mean():.1%}")

    res = test.assign(predicted=pred).drop_duplicates("text")
    print("\nPer description (one row each):")
    print(res[["text", "category", "predicted"]].to_string(index=False))


if __name__ == "__main__":
    main()
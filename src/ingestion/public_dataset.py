"""Loader for the public Kaggle dataset (Spending Habits By Category And Item).

Synthetic, expenses only. Returns the standard schema plus a category column.
"""
import pandas as pd

from src.utils.helpers import get_project_root, load_config

FILENAME = "spending_patterns_detailed.csv"

# Their category names -> our category names. Anything not listed becomes "Other".
CATEGORY_MAP = {
    "Fitness": "Health",
    "Food": "Dining",
    "Friend Activities": "Leisure",
    "Gifts": "Shopping",
    "Groceries": "Groceries",
    "Hobbies": "Leisure",
    "Housing and Utilities": "Utilities",
    "Medical/Dental": "Health",
    "Personal Hygiene": "Health",
    "Shopping": "Shopping",
    "Subscriptions": "Subscriptions",
    "Transportation": "Transport",
    "Travel": "Leisure",
}


def load_public_dataset() -> pd.DataFrame:
    path = get_project_root() / load_config()["paths"]["raw"] / FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Public dataset not found at {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={
        "Customer ID": "user_id",
        "Item": "description",
        "Transaction Date": "date",
        "Category": "raw_category",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Expenses are negative in our schema
    df["amount"] = -df["Total Spent"].abs()
    df["category"] = df["raw_category"].map(CATEGORY_MAP).fillna("Other")

    df = df.dropna(subset=["date", "amount"])
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"] != ""]
        # Drop one-off huge purchases (the dataset's "Car" rows reach six figures)
    df = df[df["amount"].abs() <= 20_000]   
    return df[["user_id", "date", "description", "amount", "category"]]


if __name__ == "__main__":
    out = load_public_dataset()
    print(len(out), "rows,", out["user_id"].nunique(), "users")
    print(out["date"].min().date(), "to", out["date"].max().date())
    print(out["category"].value_counts())
    print(out.head())
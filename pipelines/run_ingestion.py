"""Stage 1 runner: generate synthetic data, load public data, fetch macro data.

Run from the project root:  python pipelines/run_ingestion.py
Writes to data/raw/: synthetic_transactions.csv, public_transactions.csv, macro.csv
"""
import runpy
import sys
from pathlib import Path

# Streamlit-style path fix: make `src` importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.ingestion.macro_api import load_macro  # noqa: E402
from src.ingestion.public_dataset import load_public_dataset  # noqa: E402
from src.utils.helpers import load_config  # noqa: E402


def summarize(name: str, df: pd.DataFrame) -> None:
    print(
        f"  {name}: {len(df):,} rows, {df['user_id'].nunique()} users, "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )


def main() -> None:
    raw = ROOT / load_config()["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)

    print("1/3 Synthetic users")
    runpy.run_module("src.ingestion.synthetic_generator", run_name="__main__")
    synth = pd.read_csv(raw / "synthetic_transactions.csv", parse_dates=["date"])
    summarize("synthetic", synth)

    print("2/3 Public dataset")
    public = load_public_dataset()
    public.to_csv(raw / "public_transactions.csv", index=False)
    summarize("public", public)

    print("3/3 Macro data")
    macro = load_macro(force_refresh=True)
    print(
        f"  macro: {len(macro)} months, "
        f"{macro['month'].min().date()} to {macro['month'].max().date()}"
    )

    synth_months = set(synth["date"].dt.to_period("M").dt.to_timestamp())
    missing = sorted(synth_months - set(macro["month"]))
    print(f"\nSynthetic months with no macro data: {len(missing)}")
    if missing:
        print(f"  {missing[0].date()} to {missing[-1].date()} (handled in Phase 6)")
    print("\nIngestion done. Files are in", raw)


if __name__ == "__main__":
    main()
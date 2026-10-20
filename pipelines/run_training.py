"""Stage 2 runner: aggregate, train, export. Grows as later phases are added."""
from src.preprocessing.aggregate_monthly import build_monthly_parquet


def main():
    print("Building monthly aggregates...")
    build_monthly_parquet()
    # Later: train the categoriser here, then features (Phase 6) and forecasters (Phase 7)


if __name__ == "__main__":
    main()
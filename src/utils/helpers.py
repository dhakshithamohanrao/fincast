from pathlib import Path

import holidays
import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return the project root folder (the one that contains config/ and src/)."""
    return Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    """Load settings from config/config.yaml and return them as a dict."""
    config_path = Path(path) if path else get_project_root() / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def holiday_table(start_year: int, end_year: int, country: str = "DE",
                  subdiv: str | None = None) -> pd.DataFrame:
    """One row per month with the number of public holidays and days in the month.

    Columns: month (first day of month), n_holidays, days_in_month.
    Example: holiday_table(2022, 2026, "DE", "BE") for Berlin.
    """
    hol = holidays.country_holidays(
        country, subdiv=subdiv, years=range(start_year, end_year + 1)
    )
    months = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    counts = pd.Series(1, index=pd.to_datetime(list(hol.keys()))).resample("MS").sum()
    out = counts.reindex(months, fill_value=0).rename_axis("month").reset_index(name="n_holidays")
    out["days_in_month"] = out["month"].dt.days_in_month
    return out
"""Fetch euro-area inflation (HICP) and the ECB key rate, with an offline cache.

Output: data/raw/macro.csv with columns month, hicp_yoy, ecb_rate
Month is the first day of the month. If the APIs fail, the cached file is used.
"""
import io

import pandas as pd
import requests
import holidays

from src.utils.helpers import get_project_root, load_config

CACHE_NAME = "macro.csv"

# Eurostat: HICP annual rate of change, euro area, all items
HICP_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "prc_hicp_manr/M.RCH_A.CP00.EA?format=SDMX-CSV&startPeriod=2022-01"
)
# ECB Data Portal: deposit facility rate, changes only, then forward-filled monthly
ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM/"
    "D.U2.EUR.4F.KR.DFR.LEV?format=csvdata&startPeriod=2022-01-01"
)


def _fetch_hicp() -> pd.Series:
    r = requests.get(HICP_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["month"] = pd.to_datetime(df["TIME_PERIOD"]).dt.to_period("M").dt.to_timestamp()
    return df.set_index("month")["OBS_VALUE"].rename("hicp_yoy").sort_index()


def _fetch_ecb_rate() -> pd.Series:
    r = requests.get(ECB_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["date"] = pd.to_datetime(df["TIME_PERIOD"])
    daily = df.set_index("date")["OBS_VALUE"]
    monthly = daily.resample("MS").last().ffill()
    monthly.index.name = "month"
    return monthly.rename("ecb_rate")


def load_macro(force_refresh: bool = False) -> pd.DataFrame:
    cache = get_project_root() / load_config()["paths"]["raw"] / CACHE_NAME
    try:
        if cache.exists() and not force_refresh:
            return pd.read_csv(cache, parse_dates=["month"])
        macro = pd.concat([_fetch_hicp(), _fetch_ecb_rate()], axis=1).sort_index()
        macro["ecb_rate"] = macro["ecb_rate"].ffill()  # rate only changes on decision dates
        macro = macro.dropna(subset=["hicp_yoy", "ecb_rate"]).reset_index()  # no invented inflation
        macro.to_csv(cache, index=False)
        return macro
    except Exception as e:
        if cache.exists():
            print(f"Macro fetch failed ({e}); using cached file.")
            return pd.read_csv(cache, parse_dates=["month"])
        raise RuntimeError(
            "Could not fetch macro data and no cache exists. "
            "Connect to the internet and run again."
        ) from e


if __name__ == "__main__":
    m = load_macro(force_refresh=True)
    print(len(m), "months")
    print(m["month"].min().date(), "to", m["month"].max().date())
    print(m.tail())

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
import io
from pathlib import Path

import pytest

from src.ingestion.bank_csv_parser import BankCSVError, parse_bank_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "sample_bank_export.csv"


def test_sample_file_parses():
    df = parse_bank_csv(SAMPLE)
    assert list(df.columns) == ["date", "description", "amount"]
    assert len(df) == 60


def test_german_format_semicolon_dots_and_decimal_comma():
    raw = (
        "Buchungstag;Verwendungszweck;Betrag\n"
        "31.01.2026;REWE Markt;-47,32\n"
        "01.02.2026;Gehalt;2.800,00\n"
    ).encode("utf-8")
    df = parse_bank_csv(raw)
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-31", "2026-02-01"]
    assert df["amount"].tolist() == pytest.approx([-47.32, 2800.0])


def test_uploaded_file_object_works():
    raw = io.BytesIO(b"date,description,amount\n2026-01-05,REWE,-10.50\n")
    df = parse_bank_csv(raw)
    assert len(df) == 1
    assert df.loc[0, "amount"] == pytest.approx(-10.5)


def test_wrong_columns_give_friendly_error():
    with pytest.raises(BankCSVError, match="Could not find"):
        parse_bank_csv(b"foo;bar\n1;2\n")


def test_text_in_amount_column_gives_friendly_error():
    with pytest.raises(BankCSVError, match="amount"):
        parse_bank_csv(b"date,description,amount\n2026-01-01,REWE,abc\n")


def test_bad_date_gives_friendly_error():
    with pytest.raises(BankCSVError, match="date"):
        parse_bank_csv(b"date,description,amount\nnot-a-date,REWE,-5\n")


def test_empty_file_gives_friendly_error():
    with pytest.raises(BankCSVError, match="empty"):
        parse_bank_csv(b"")


def test_header_only_file_gives_friendly_error():
    with pytest.raises(BankCSVError, match="no transactions"):
        parse_bank_csv(b"date,description,amount\n")
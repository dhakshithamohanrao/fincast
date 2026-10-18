"""Parse a bank CSV export into the standard schema: date, description, amount."""
import io
import re
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = ["date", "description", "amount"]

COLUMN_ALIASES = {
    "date": [
        "date", "datum", "buchungstag", "buchungsdatum", "booking date",
        "transaction date", "valuta", "wertstellung",
    ],
    "description": [
        "description", "beschreibung", "verwendungszweck", "buchungstext", "text",
        "details", "payee", "empfänger", "empfaenger", "name", "merchant", "reference",
    ],
    "amount": ["amount", "betrag", "value", "umsatz", "summe"],
}

DATE_FORMATS = [
    "%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y",
    "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y",
]


class BankCSVError(ValueError):
    """Raised with a user-friendly message when a file cannot be parsed."""


def _read_bytes(source) -> bytes:
    """Accept a file path, raw bytes, or a file-like object (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        try:
            return Path(source).read_bytes()
        except OSError as exc:
            raise BankCSVError(f"Could not open the file: {source}") from exc
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise BankCSVError("Unsupported file input.")
    return data.encode("utf-8") if isinstance(data, str) else data


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _detect_delimiter(text: str) -> str:
    """Pick the delimiter that appears the same number of times on every line."""
    lines = [line for line in text.splitlines() if line.strip()][:10]
    if not lines:
        raise BankCSVError("The file is empty.")
    best, best_score = ",", -1
    for delimiter in [";", ",", "\t", "|"]:
        counts = [line.count(delimiter) for line in lines]
        if counts[0] == 0:
            continue
        consistent = sum(1 for c in counts if c == counts[0])
        score = consistent * 100 + counts[0]
        if score > best_score:
            best, best_score = delimiter, score
    return best


def read_raw_csv(source) -> pd.DataFrame:
    """Read the file as text columns without interpreting anything (for preview/mapping)."""
    text = _decode(_read_bytes(source))
    delimiter = _detect_delimiter(text)
    try:
        raw = pd.read_csv(
            io.StringIO(text), sep=delimiter, dtype=str,
            keep_default_na=False, skipinitialspace=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise BankCSVError("The file could not be read as a CSV table.") from exc
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.fillna("").apply(lambda col: col.str.strip())
    raw = raw[raw.ne("").any(axis=1)]
    if raw.empty:
        raise BankCSVError("The file contains no transactions.")
    return raw


def detect_columns(columns) -> dict:
    """Guess which file column is the date, description and amount."""
    found = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for column in columns:
            if column.strip().lower() in aliases:
                found[standard] = column
                break
    return found


def _parse_dates(series: pd.Series) -> pd.Series:
    for date_format in DATE_FORMATS:
        parsed = pd.to_datetime(series, format=date_format, errors="coerce")
        if parsed.notna().all():
            return parsed
    for date_format in DATE_FORMATS:
        parsed = pd.to_datetime(series, format=date_format, errors="coerce")
        bad = series[parsed.isna()]
        if len(bad) < len(series):
            raise BankCSVError(
                f"Could not read the date '{bad.iloc[0]}' (row {bad.index[0] + 2}). "
                "Supported formats look like 2026-01-31 or 31.01.2026."
            )
    raise BankCSVError(
        "Could not read the dates. Supported formats look like 2026-01-31 or 31.01.2026."
    )


def _to_number(text: str):
    """Convert text like '-47,32' or '2.800,00' to a float, or None if impossible."""
    s = re.sub(r"[^\d,.\-+()]", "", text.replace("\u2212", "-"))
    if not s:
        return None
    negative = s.startswith("-") or s.endswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("-+()")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def _parse_amounts(series: pd.Series) -> pd.Series:
    values = series.map(_to_number)
    bad = series[values.isna()]
    if len(bad) > 0:
        raise BankCSVError(
            f"Could not read the amount '{bad.iloc[0]}' (row {bad.index[0] + 2}). "
            "Amounts must be numbers like -47.32 or -47,32."
        )
    return values.astype(float)


def parse_bank_csv(source, column_map: dict = None) -> pd.DataFrame:
    """Return a DataFrame with the columns date, description, amount.

    column_map lets the Import page pass a manual mapping, for example
    {"date": "Buchungstag", "description": "Text", "amount": "Betrag"}.
    """
    raw = read_raw_csv(source)
    column_map = column_map or detect_columns(raw.columns)

    missing = [c for c in STANDARD_COLUMNS if not column_map.get(c)]
    if missing:
        raise BankCSVError(
            f"Could not find the column(s): {', '.join(missing)}. "
            f"Columns in your file: {', '.join(raw.columns)}."
        )
    for column in column_map.values():
        if column not in raw.columns:
            raise BankCSVError(f"The column '{column}' does not exist in the file.")

    result = pd.DataFrame(
        {
            "date": _parse_dates(raw[column_map["date"]]),
            "description": raw[column_map["description"]],
            "amount": _parse_amounts(raw[column_map["amount"]]),
        }
    )
    return result.reset_index(drop=True)
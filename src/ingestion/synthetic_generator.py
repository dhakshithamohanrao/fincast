"""Synthetic users for training the forecasting models.

Part 1: archetypes (types of people) and per-user parameters.
Ranges are (low, high). Two integers give a whole-number draw, floats a decimal draw.
"""
import numpy as np

ARCHETYPES = {
    "salaried_single": {
        "income_type": "salary",
        "salary": (2200, 3800),
        "rent": (600, 1100),
        "utilities": (60, 140),
        "groceries_weekly": (40, 70),
        "dining_visits": (3, 8),
        "dining_avg": (18, 40),
        "transport_monthly": (40, 120),
        "n_subscriptions": (2, 4),
        "shopping_monthly": (40, 160),
    },
    "salaried_family": {
        "income_type": "salary",
        "salary": (3200, 5200),
        "rent": (900, 1500),
        "utilities": (120, 260),
        "groceries_weekly": (110, 180),
        "dining_visits": (2, 5),
        "dining_avg": (40, 80),
        "transport_monthly": (100, 250),
        "n_subscriptions": (3, 5),
        "shopping_monthly": (100, 300),
    },
    "freelancer": {
        "income_type": "freelance",
        "invoice_total": (1800, 4500),
        "invoices_per_month": (1, 4),
        "income_gap_prob": (0.05, 0.20),
        "rent": (600, 1100),
        "utilities": (60, 140),
        "groceries_weekly": (40, 80),
        "dining_visits": (4, 10),
        "dining_avg": (20, 45),
        "transport_monthly": (30, 140),
        "n_subscriptions": (2, 5),
        "shopping_monthly": (30, 200),
    },
    "student": {
        "income_type": "salary",
        "salary": (700, 1200),
        "rent": (300, 550),
        "utilities": (30, 70),
        "groceries_weekly": (30, 55),
        "dining_visits": (4, 10),
        "dining_avg": (8, 20),
        "transport_monthly": (0, 60),
        "n_subscriptions": (1, 3),
        "shopping_monthly": (10, 80),
    },
}


def _draw(value, rng):
    """Draw one number from a (low, high) range; pass other values through."""
    if isinstance(value, tuple):
        low, high = value
        if isinstance(low, int) and isinstance(high, int):
            return int(rng.integers(low, high + 1))
        return round(float(rng.uniform(low, high)), 2)
    return value


def draw_user_params(archetype: str, rng: np.random.Generator) -> dict:
    """Concrete numbers for one fake user of the given archetype."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"Unknown archetype '{archetype}'. Choose from {list(ARCHETYPES)}.")
    params = {key: _draw(value, rng) for key, value in ARCHETYPES[archetype].items()}
    params["archetype"] = archetype
    params["rent_day"] = int(rng.integers(1, 4))
    params["payday"] = int(rng.integers(25, 29))
    return params

# ---------------------------------------------------------------------------
# Part 2: building blocks for one month of one user.
# Every function takes (p, month, rng, infl=1.0) and returns a list of rows.
# month = first day of the month, infl = price multiplier (inflation drift).
# ---------------------------------------------------------------------------
import calendar

import pandas as pd

SUBSCRIPTIONS = [
    ("Netflix.com", 12.99),
    ("Spotify AB", 10.99),
    ("FITX Fitnessstudio", 29.90),
    ("Vodafone Mobilfunk", 29.99),
    ("Amazon Prime", 8.99),
    ("Disney Plus", 8.99),
]
GROCERY_SHOPS = ["REWE Markt Berlin", "LIDL Filiale 1042", "ALDI SUED", "EDEKA Center"]
DINING_PLACES = [
    "Restaurant Zur Linde", "Lieferando Bestellung", "Cafe Einstein",
    "Restaurant Il Gabbiano", "Burger Haus", "Baeckerei Schmidt",
]
TRANSPORT_NAMES = ["Deutsche Bahn Ticket", "Shell Tankstelle", "Uber Trip", "BVG Ticket"]
SHOPPING_SHOPS = ["Amazon Marketplace", "Zalando Bestellung", "dm Drogeriemarkt", "MediaMarkt"]


def _day(month, day):
    """The given day of the month, capped at the month's last day."""
    last = calendar.monthrange(month.year, month.month)[1]
    return month + pd.Timedelta(days=min(day, last) - 1)


def _row(date, description, amount, category):
    return {
        "date": date,
        "description": description,
        "amount": round(float(amount), 2),
        "category": category,
    }


def _pick(options, rng):
    return options[int(rng.integers(0, len(options)))]


def _income_base(p):
    return p["salary"] if "salary" in p else p["invoice_total"]


def salary(p, month, rng, infl=1.0):
    """Monthly salary (nearly constant) or freelance invoices (irregular, with gaps)."""
    rows = []
    if p["income_type"] == "salary":
        amount = p["salary"] * float(rng.uniform(0.99, 1.01))
        rows.append(
            _row(_day(month, p["payday"]), f"Gehalt {month:%m/%Y} Arbeitgeber GmbH", amount, "Income")
        )
        if month.month == 12 and p["archetype"] != "student":
            bonus = p["salary"] * float(rng.uniform(0.3, 0.7))
            rows.append(
                _row(_day(month, 15), f"Weihnachtsgeld {month:%Y} Arbeitgeber GmbH", bonus, "Income")
            )
    else:
        if rng.random() >= p["income_gap_prob"]:
            n = max(1, p["invoices_per_month"] + int(rng.integers(-1, 2)))
            for _ in range(n):
                amount = p["invoice_total"] / n * float(rng.uniform(0.6, 1.4))
                client = "ABCDEFGH"[int(rng.integers(0, 8))]
                rows.append(
                    _row(_day(month, int(rng.integers(1, 29))), f"Zahlung Kunde {client} Rechnung", amount, "Income")
                )
    return rows


def rent(p, month, rng, infl=1.0):
    return [_row(_day(month, p["rent_day"]), f"Miete {month:%m/%Y} Hausverwaltung", -p["rent"], "Rent")]


def utilities(p, month, rng, infl=1.0):
    winter = 1.3 if month.month in (11, 12, 1, 2, 3) else 1.0
    amount = p["utilities"] * winter * float(rng.uniform(0.92, 1.08)) * infl
    return [_row(_day(month, 5), "Stadtwerke Strom Gas", -amount, "Utilities")]


def subscriptions(p, month, rng, infl=1.0):
    rows = []
    for i, (name, price) in enumerate(SUBSCRIPTIONS[: p["n_subscriptions"]]):
        rows.append(_row(_day(month, 8 + i), name, -price, "Subscriptions"))
    return rows


def groceries(p, month, rng, infl=1.0):
    """One shop per week, with noisy amounts."""
    rows = []
    for week in range(4):
        day = 2 + week * 7 + int(rng.integers(0, 5))
        amount = p["groceries_weekly"] * float(rng.uniform(0.7, 1.3)) * infl
        rows.append(_row(_day(month, day), _pick(GROCERY_SHOPS, rng), -amount, "Groceries"))
    return rows


def dining(p, month, rng, infl=1.0):
    rows = []
    n = max(0, p["dining_visits"] + int(rng.integers(-1, 2)))
    for _ in range(n):
        amount = p["dining_avg"] * float(rng.uniform(0.6, 1.5)) * infl
        rows.append(
            _row(_day(month, int(rng.integers(1, 29))), _pick(DINING_PLACES, rng), -amount, "Dining")
        )
    return rows


def transport(p, month, rng, infl=1.0):
    if p["transport_monthly"] <= 0:
        return []
    rows = []
    n = int(rng.integers(1, 4))
    for _ in range(n):
        amount = p["transport_monthly"] / n * float(rng.uniform(0.7, 1.3)) * infl
        rows.append(
            _row(_day(month, int(rng.integers(1, 29))), _pick(TRANSPORT_NAMES, rng), -amount, "Transport")
        )
    return rows


def shopping(p, month, rng, infl=1.0):
    rows = []
    n = int(rng.integers(0, 3))
    for _ in range(n):
        amount = p["shopping_monthly"] / n * float(rng.uniform(0.6, 1.4)) * infl
        rows.append(
            _row(_day(month, int(rng.integers(1, 29))), _pick(SHOPPING_SHOPS, rng), -amount, "Shopping")
        )
    return rows


def seasonal(p, month, rng, infl=1.0):
    """December gifts, summer travel and the annual insurance bill."""
    rows = []
    base = _income_base(p)
    if month.month == 12:
        gifts = base * float(rng.uniform(0.04, 0.10)) * infl
        rows.append(_row(_day(month, int(rng.integers(5, 22))), "Weihnachtsgeschenke Amazon", -gifts, "Shopping"))
        party = float(rng.uniform(40, 120)) * infl
        rows.append(_row(_day(month, int(rng.integers(15, 24))), "Restaurant Weihnachtsfeier", -party, "Dining"))
    if month.month in (7, 8) and rng.random() < 0.5:
        trip = base * float(rng.uniform(0.25, 0.60)) * infl
        rows.append(_row(_day(month, int(rng.integers(1, 20))), "Booking.com Urlaub", -trip, "Leisure"))
    if month.month == 4:
        rows.append(
            _row(_day(month, 15), "HUK Versicherung Jahresbeitrag", -float(rng.uniform(150, 450)), "Other")
        )
    return rows


def shocks(p, month, rng, infl=1.0):
    """Rare big costs: car repair, dentist, broken appliance."""
    rows = []
    if rng.random() < 0.06:
        amount = float(rng.uniform(150, 900)) * infl
        rows.append(_row(_day(month, int(rng.integers(1, 29))), "KFZ Werkstatt Reparatur", -amount, "Transport"))
    if rng.random() < 0.05:
        amount = float(rng.uniform(60, 400)) * infl
        rows.append(_row(_day(month, int(rng.integers(1, 29))), "Zahnarzt Dr. Weber", -amount, "Health"))
    if rng.random() < 0.03:
        amount = float(rng.uniform(200, 700)) * infl
        rows.append(_row(_day(month, int(rng.integers(1, 29))), "Elektro Markt Ersatzgeraet", -amount, "Shopping"))
    return rows
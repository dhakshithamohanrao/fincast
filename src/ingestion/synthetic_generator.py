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
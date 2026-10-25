"""Savings goal maths (Phase 8.7)."""
import math
from dataclasses import dataclass

import pandas as pd

MAX_MONTHS = 600     # beyond 50 years we say "not reachable"


@dataclass
class GoalProjection:
    status: str                         # "reached", "on_track", "behind" or "no_savings"
    progress: float                     # 0 to 1
    remaining: float
    monthly_saving: float
    months_needed: int | None
    projected_date: pd.Timestamp | None  # last day of the month the goal is reached
    date_best: pd.Timestamp | None
    date_worst: pd.Timestamp | None
    required_monthly: float | None      # saving per month needed to hit the target date
    months_to_target: int | None


def _reach(remaining: float, saving: float, first_month: pd.Timestamp):
    """(months needed, date the goal is reached) or (None, None) if not reachable."""
    if remaining <= 0:
        return 0, None
    if saving <= 0:
        return None, None
    months = math.ceil(remaining / saving)
    if months > MAX_MONTHS:
        return None, None
    return months, first_month + pd.DateOffset(months=months) - pd.Timedelta(days=1)


def project_goal(target_amount: float, current_saved: float, monthly_saving: float,
                 first_month: pd.Timestamp, target_date: pd.Timestamp | None = None,
                 saving_low: float | None = None,
                 saving_high: float | None = None) -> GoalProjection:
    """Project when a savings goal is reached at a constant monthly saving.

    first_month: first day of the first month in which saving happens (the forecast month).
    saving_low / saving_high: worst and best case monthly saving (optional).
    """
    first_month = pd.Timestamp(first_month).to_period("M").to_timestamp()
    remaining = float(target_amount) - float(current_saved)
    progress = min(max(float(current_saved) / target_amount, 0.0), 1.0) if target_amount > 0 else 0.0

    months_to_target = required = None
    if target_date is not None:
        t = pd.Timestamp(target_date)
        months_to_target = max((t.year - first_month.year) * 12 + t.month - first_month.month + 1, 0)
        if months_to_target > 0 and remaining > 0:
            required = remaining / months_to_target

    if remaining <= 0:
        return GoalProjection("reached", 1.0, remaining, monthly_saving, 0, None, None, None,
                              required, months_to_target)

    months, date = _reach(remaining, monthly_saving, first_month)
    best = _reach(remaining, saving_high, first_month)[1] if saving_high is not None else None
    worst = _reach(remaining, saving_low, first_month)[1] if saving_low is not None else None

    if date is None:
        status = "no_savings"
    elif target_date is None or date <= pd.Timestamp(target_date):
        status = "on_track"
    else:
        status = "behind"
    return GoalProjection(status, progress, remaining, monthly_saving, months, date, best,
                          worst, required, months_to_target)
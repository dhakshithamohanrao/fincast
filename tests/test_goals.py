import pandas as pd
import pytest

from src.forecasting.goals import project_goal

FIRST = pd.Timestamp("2026-10-01")


def test_on_track_and_projected_date():
    p = project_goal(1200, 0, 300, FIRST, target_date=pd.Timestamp("2027-03-01"))
    assert p.status == "on_track"
    assert p.months_needed == 4
    assert p.projected_date == pd.Timestamp("2027-01-31")


def test_behind_shows_required_saving():
    p = project_goal(1200, 0, 300, FIRST, target_date=pd.Timestamp("2026-12-31"))
    assert p.status == "behind"
    assert p.months_to_target == 3
    assert p.required_monthly == pytest.approx(400.0)


def test_no_savings_and_reached():
    assert project_goal(1000, 0, -50, FIRST, pd.Timestamp("2027-06-01")).status == "no_savings"
    assert project_goal(1000, 1500, 100, FIRST, pd.Timestamp("2027-06-01")).status == "reached"


def test_best_and_worst_case_order():
    p = project_goal(3000, 0, 300, FIRST, saving_low=150, saving_high=600)
    assert p.date_best < p.projected_date < p.date_worst
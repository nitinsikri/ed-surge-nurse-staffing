import numpy as np
import pandas as pd

from ed_staffing import (generate_arrivals, to_shifts, TwoStageForecaster, rmse,
                         StaffingPolicy, recommend)


def _data():
    shifts = to_shifts(generate_arrivals(days=200, seed=1))
    cut = shifts["date"].min() + pd.Timedelta(days=150)
    return shifts[shifts.date < cut], shifts[shifts.date >= cut]


def test_three_shifts_per_day():
    shifts = to_shifts(generate_arrivals(days=10, seed=2))
    assert set(shifts["shift"]) == {"07-19", "11-23", "19-07"}


def test_surge_model_not_worse_than_base():
    train, test = _data()
    m = TwoStageForecaster().fit(train)
    ctx = pd.concat([train.tail(90), test])
    base = m.predict_base(test)
    surge = m.predict_surge(ctx)[-len(test):]
    assert rmse(test.arrivals, surge) <= rmse(test.arrivals, base) * 1.05


def test_recommendations_respect_minimum():
    train, test = _data()
    m = TwoStageForecaster().fit(train)
    pol = StaffingPolicy(min_nurses=6)
    plan = recommend(test, m.predict_base(test), m.predict_base(test), pol)
    assert (plan["total_nurses"] >= 6).all()
    assert (plan["surge_callins"] == 0).all()

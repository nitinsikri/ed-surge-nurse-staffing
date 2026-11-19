"""Synthetic emergency-department arrival data.

The generator produces hourly patient arrivals with daily, weekly and yearly
seasonality, a weather effect and occasional surge episodes. It stands in for the
de-identified hospital data used in the paper, which cannot be shared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_arrivals(start: str = "2020-01-01", days: int = 2 * 365 + 140,
                      annual_visits: int = 90_000, seed: int = 7) -> pd.DataFrame:
    """Return an hourly frame with arrivals, census, temperature and precipitation."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    base = annual_visits / (365 * 24)
    hourly = 1 + 0.65 * np.sin((hour - 8) / 24 * 2 * np.pi)          # midday peak
    weekly = np.where(dow == 0, 1.12, np.where(dow >= 5, 0.92, 1.0))  # busy Mondays
    yearly = 1 + 0.08 * np.cos((doy - 15) / 365 * 2 * np.pi)          # winter peak

    temp = 22 + 9 * np.sin((doy - 110) / 365 * 2 * np.pi) + rng.normal(0, 3, len(idx))
    precip = (rng.random(len(idx)) < 0.08) * rng.gamma(2, 3, len(idx))
    weather = 1 + 0.004 * np.clip(temp - 30, 0, None) + 0.01 * (precip > 5)

    surge = np.ones(len(idx))
    n_surges = days // 25
    for s in rng.choice(len(idx) - 72, n_surges, replace=False):
        length = rng.integers(12, 60)
        surge[s:s + length] *= rng.uniform(1.15, 1.45)

    lam = base * hourly * weekly * yearly * weather * surge
    arrivals = rng.poisson(lam)

    # census: patients currently in the ED, with an average stay of ~5 hours
    census = np.zeros(len(idx))
    decay = np.exp(-1 / 5)
    for t in range(1, len(idx)):
        census[t] = census[t - 1] * decay + arrivals[t]

    return pd.DataFrame({"arrivals": arrivals, "census": census.round(1),
                         "temperature": temp.round(1), "precip_mm": precip.round(1)},
                        index=idx)


SHIFTS = {"07-19": (7, 19), "11-23": (11, 23), "19-07": (19, 31)}


def to_shifts(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to the three daily shifts used in the paper."""
    rows = []
    for day, grp in hourly.groupby(hourly.index.normalize()):
        nxt = hourly.loc[day + pd.Timedelta(hours=24): day + pd.Timedelta(hours=31)]
        both = pd.concat([grp, nxt])
        for name, (a, b) in SHIFTS.items():
            sel = both[(both.index >= day + pd.Timedelta(hours=a)) &
                       (both.index < day + pd.Timedelta(hours=b))]
            if len(sel) < b - a:
                continue
            rows.append({"date": day, "shift": name, "arrivals": sel.arrivals.sum(),
                         "mean_census": sel.census.mean(), "peak_census": sel.census.max(),
                         "temperature": sel.temperature.mean(),
                         "precip_mm": sel.precip_mm.sum()})
    return pd.DataFrame(rows)

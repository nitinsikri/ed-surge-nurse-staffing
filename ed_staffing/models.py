"""Two-stage (base and surge) linear forecasting models.

* Base stage (4-6 weeks ahead): calendar features only.
* Surge stage (24 hours ahead): calendar features plus recent census,
  recent arrivals and the weather forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SHIFT_CODES = {"07-19": 0, "11-23": 1, "19-07": 2}


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    X = pd.DataFrame(index=df.index)
    for k in range(7):
        X[f"dow_{k}"] = (d.dt.dayofweek == k).astype(int)
    for name, code in SHIFT_CODES.items():
        X[f"shift_{code}"] = (df["shift"] == name).astype(int)
    doy = d.dt.dayofyear
    X["season_sin"] = np.sin(2 * np.pi * doy / 365)
    X["season_cos"] = np.cos(2 * np.pi * doy / 365)
    return X


def surge_features(df: pd.DataFrame) -> pd.DataFrame:
    X = calendar_features(df)
    # values known 24 hours before the shift starts (same shift, previous day)
    by_shift = df.groupby("shift")
    X["arrivals_lag1"] = by_shift["arrivals"].shift(1)
    X["arrivals_mean7"] = by_shift["arrivals"].transform(lambda s: s.shift(1).rolling(7).mean())
    X["census_lag1"] = by_shift["mean_census"].shift(1)
    X["temperature_fc"] = df["temperature"]          # weather forecast for the shift
    X["heavy_rain_fc"] = (df["precip_mm"] > 5).astype(int)
    return X


class TwoStageForecaster:
    """Fits the base and surge regressions and predicts shift arrivals."""

    def __init__(self):
        self.base = LinearRegression()
        self.surge = LinearRegression()

    def fit(self, train: pd.DataFrame) -> "TwoStageForecaster":
        Xs = surge_features(train).dropna()
        self.base.fit(calendar_features(train).loc[Xs.index], train.loc[Xs.index, "arrivals"])
        self.surge.fit(Xs, train.loc[Xs.index, "arrivals"])
        return self

    def predict_base(self, df: pd.DataFrame) -> np.ndarray:
        return self.base.predict(calendar_features(df))

    def predict_surge(self, df: pd.DataFrame) -> np.ndarray:
        X = surge_features(df)
        pred = self.predict_base(df).astype(float)
        ok = X.notna().all(axis=1).to_numpy()
        pred[ok] = self.surge.predict(X[ok])
        return pred


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)

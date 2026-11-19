"""Turn forecasts into nurse staffing recommendations and evaluate them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StaffingPolicy:
    patients_per_nurse_hour: float = 0.9   # arrivals one nurse can absorb per hour
    min_nurses: int = 6
    base_hourly_rate: float = 55.0          # USD per nurse-hour
    surge_premium: float = 1.5              # incentive multiplier for call-ins
    shift_hours: int = 12

    def nurses_for(self, arrivals: np.ndarray) -> np.ndarray:
        per_hour = np.asarray(arrivals, float) / self.shift_hours
        return np.maximum(self.min_nurses, np.ceil(per_hour / self.patients_per_nurse_hour)).astype(int)


def recommend(df: pd.DataFrame, base_pred: np.ndarray, surge_pred: np.ndarray,
              policy: StaffingPolicy) -> pd.DataFrame:
    """Base roster from the base forecast; surge call-ins when the surge forecast is higher."""
    out = df[["date", "shift", "arrivals", "mean_census"]].copy()
    out["base_forecast"] = base_pred.round(1)
    out["surge_forecast"] = surge_pred.round(1)
    out["base_nurses"] = policy.nurses_for(base_pred)
    needed = policy.nurses_for(surge_pred)
    out["surge_callins"] = np.maximum(0, needed - out["base_nurses"])
    out["total_nurses"] = out["base_nurses"] + out["surge_callins"]
    return out


def evaluate(plan: pd.DataFrame, policy: StaffingPolicy, staff_col: str) -> dict:
    """Workload-staffing alignment and hourly cost for a staffing column."""
    required = policy.nurses_for(plan["arrivals"].to_numpy())
    staff = plan[staff_col].to_numpy()
    alignment = float(np.mean((required - staff) ** 2))
    callins = plan["surge_callins"].to_numpy() if staff_col == "total_nurses" else 0
    base = staff - callins
    cost = (base * policy.base_hourly_rate +
            callins * policy.base_hourly_rate * policy.surge_premium)
    # understaffed nurse-hours are assumed to be covered by overtime at 1.75x
    short = np.maximum(0, required - staff)
    cost = cost + short * policy.base_hourly_rate * 1.75
    return {"alignment_mse": round(alignment, 3),
            "mean_hourly_cost": round(float(np.mean(cost)), 2),
            "understaffed_shift_share": round(float(np.mean(short > 0)), 3)}

"""Train on two years of synthetic data, then compare a base-only roster with
base + surge call-ins over a 69-day evaluation window."""
import pandas as pd

from ed_staffing import (generate_arrivals, to_shifts, TwoStageForecaster, rmse, mape,
                         StaffingPolicy, recommend, evaluate)

hourly = generate_arrivals()
shifts = to_shifts(hourly)
cut = shifts["date"].min() + pd.Timedelta(days=730)
train, test = shifts[shifts.date < cut], shifts[shifts.date >= cut].head(69 * 3)

model = TwoStageForecaster().fit(train)
# give the surge model the history it needs for its lag features
context = pd.concat([train.tail(30 * 3), test])
base_pred = model.predict_base(test)
surge_pred = model.predict_surge(context)[-len(test):]

print(f"Base stage : RMSE {rmse(test.arrivals, base_pred):.2f}  MAPE {mape(test.arrivals, base_pred):.1f}%")
print(f"Surge stage: RMSE {rmse(test.arrivals, surge_pred):.2f}  MAPE {mape(test.arrivals, surge_pred):.1f}%")

policy = StaffingPolicy()
plan = recommend(test, base_pred, surge_pred, policy)
plan["base_only"] = plan["base_nurses"]
print("\nBase roster only     :", evaluate(plan.assign(surge_callins=0), policy, "base_only"))
print("Base + surge call-ins:", evaluate(plan, policy, "total_nurses"))
print("\nSample recommendations:")
print(plan.head(9).to_string(index=False))

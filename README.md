# Two-Stage Predictive Nurse Staffing for Emergency Department Surges

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24380895.svg)](https://doi.org/10.5281/zenodo.24380895)

Reference implementation of the method described in:

**Nitin Sikri.** *Volume-Adaptive Machine Learning Framework for Dynamic Nurse Staffing in Emergency Department Surge Periods.* Proceedings of the 14th International Conference on Frontiers of Intelligent Computing: Theory and Applications (FICTA-2026), London, UK, Springer, 2026. *(In press.)*

> **Reference implementation.** This repository was written in November 2026, after the paper was accepted, to illustrate the method described in the paper and to make it easy to reuse. It is **not** the original experimental code, it runs on **synthetic data**, and its numbers will **differ** from the results reported in the paper. Please cite the paper for the method and its reported results.

## What the paper proposes

A two-tier predictive staffing architecture for emergency departments. A **base-stage** linear model (4–6 weeks ahead, calendar features) sets the posted roster; a **surge-stage** linear model (24 hours ahead) adds recent census, same-day arrivals and the weather forecast, and recommends incentive call-ins when it predicts more demand than the base roster covers. Staffing is evaluated on workload–staffing alignment, hourly labour cost and patient throughput.

## What this repository contains

- `ed_staffing/data.py` – synthetic hourly ED arrivals (daily/weekly/yearly seasonality, weather, multi-hour surges) aggregated to the three shifts used in the paper (07–19, 11–23, 19–07)
- `ed_staffing/models.py` – the base-stage and surge-stage regressions, RMSE and MAPE
- `ed_staffing/staffing.py` – converts forecasts into nurse numbers and call-ins; computes alignment (mean squared gap between required and scheduled nurses) and cost
- `examples/run_pilot.py` – trains on two years of synthetic data and evaluates a 69-day window
- `tests/` – unit tests

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python examples/run_pilot.py
python -m pytest -q
```



## How it maps to the paper

| Paper | Code |
|---|---|
| Base stage (Sect. 3.2) | `TwoStageForecaster.base`, `calendar_features` |
| Surge stage (Sect. 3.2) | `TwoStageForecaster.surge`, `surge_features` |
| Staffing recommendation / call-ins (Sect. 3.3) | `recommend` |
| Alignment and cost metrics (Sect. 3.4) | `evaluate` |

## Limitations

- Synthetic data only; the hospital data used in the paper cannot be shared.
- The paper's decision-support tool (Google Apps Script + daily e-mail) is replaced by a Python pipeline.
- Throughput outcomes (door-to-evaluation, LOS, LWBS) need real operational data and are not simulated.
- On this synthetic data the surge model gives only a small accuracy gain; the paper's reported figures (e.g. RMSE −11.4 %, 34 % alignment improvement, $162/hour saving) come from the real pilot and are **not** reproduced here.

## How to cite

If you use this code, please cite the paper:

```bibtex
@inproceedings{sikri2026volume,
  author    = {Sikri, Nitin},
  title     = {Volume-Adaptive Machine Learning Framework for Dynamic Nurse Staffing in Emergency Department Surge Periods},
  booktitle = {Proceedings of the 14th International Conference on Frontiers of Intelligent Computing: Theory and Applications (FICTA-2026), London, UK},
  publisher = {Springer},
  year      = {2026}
}
```

To cite this software itself, use its Zenodo DOI: [10.5281/zenodo.24380895](https://doi.org/10.5281/zenodo.24380895).

GitHub also shows a **"Cite this repository"** button generated from `CITATION.cff`.

## Licence

MIT – see [LICENSE](LICENSE).

# FrostIQ

ML-based predictive cold-chain temperature monitoring.

## Objective

FrostIQ predicts the **estimated number of minutes remaining before a container's
temperature crosses a configurable threshold** (for example, 8 °C for a 2–8 °C
vaccine fridge). The goal is to warn operators *before* stock is compromised, with
enough lead time to act.

Simpler outputs, such as "will it cross within 30 minutes?", can be derived from
the time-to-threshold estimate. They are not the main objective.

## Labelling principle

Labels come from simulated (and later, real) **future trajectories**:

- If the temperature crosses the threshold at a later point in the observed
  period, the label is the number of minutes from the current reading to that
  crossing.
- If the temperature **never crosses** the threshold within the observed period,
  no remaining-time value is invented. That reading is marked as *censored*: we
  only know that the crossing (if any) happens after the observed period ends.

## Data

Development starts with **documented synthetic data**, which simulates normal
cooling, gradual warming, door-opening disturbances and faster warming scenarios.
All assumptions and limitations are documented in
[docs/SYNTHETIC_DATA.md](docs/SYNTHETIC_DATA.md). Real public datasets may be
investigated later.

> Synthetic results do not establish real-world medical safety or predictive
> accuracy.

Regenerate the data (writes to `data/raw/`, which Git ignores):

```powershell
python -m frostiq.simulate                  # 120 journeys, seed 42, threshold 8.0 degC
python -m frostiq.simulate --threshold 10   # any other threshold
```

## Project structure

```
FrostIQ/
├── data/raw/           # generated or original data (not committed)
├── data/processed/     # cleaned data and features (not committed)
├── notebooks/          # exploration and visual checks
├── src/frostiq/        # core Python package (simulation, features, models)
├── models/             # saved trained models (not committed)
├── app/                # Flask dashboard (later milestone)
├── docs/               # documentation (synthetic data assumptions)
├── tests/              # automated tests
├── requirements.txt
└── README.md
```

## Setup (Windows)

Requires Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Status

- [x] Milestone 1a: environment and project structure
- [x] Milestone 1b: synthetic temperature dataset with time-to-threshold labels
- [ ] Milestone 2: feature engineering
- [ ] Milestone 3: model training and evaluation
- [ ] Milestone 4: Flask dashboard

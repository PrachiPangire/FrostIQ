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
python -m frostiq.simulate --n-journeys 500 # dataset used from Milestone 2 onwards
```

Evaluate the baselines and ML models (prints MAE tables, warning timing, the
crossing classifier's confusion matrix and a journey-level bootstrap interval;
about 20 s):

```powershell
python -m frostiq.evaluate
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for the methodology, metrics and
limitations.

## Website

FrostIQ includes a Flask website with two parts:

- **A 3D story homepage** with eight sections: Home, The Problem, The Journey,
  Temperature Risk, How It Works, AI Predictions, Performance, and Limitations.
  A real-time **three.js** scene sits behind the page: a cold-chain cooler built
  in code (hinged lid, live display, lighting, frost and heat particles), a 3D
  journey route, a temperature column with a threshold ring, an ML "model core",
  and a 3D confusion matrix drawn from the real held-out counts. **GSAP
  ScrollTrigger** links the scenes to scrolling. Scrolling itself is never
  hijacked. If WebGL is unavailable, or the system "reduce motion" setting is on,
  a static SVG cooler is shown instead and every section still works.
- **A monitoring console** (`/dashboard`) that replays the synthetic journeys.

Both pages share the **AI Prediction Centre**, which always shows two cards from
the API for the selected journey and moment:

| Card | Model | Question | Output |
|---|---|---|---|
| **Classification** (risk decision, cyan/mint) | `RandomForestClassifier` | Will the temperature cross 8 °C within the next 120 minutes? | FLAGGED / NOT FLAGGED, plus the uncalibrated score |
| **Regression** (numeric estimate, amber, **Experimental**) | `HistGradientBoostingRegressor` | If a crossing is estimated, how many minutes until it happens? | Estimated minutes, or "Beyond 120 min" when the output sits at the cap |

Classification answers a yes/no risk question. Regression estimates a number,
such as minutes until a possible breach. When there's no valid prediction, both
cards say why instead of showing a value:
- the reading is already above the threshold (an *observed* crossing)
- there's less than 30 minutes of history
- the threshold differs from the model's
- no model is trained

The libraries are **vendored** in `app/static/vendor/` (three.js 0.160.0, MIT;
GSAP 3.12.5 with ScrollTrigger, GreenSock standard "no charge" licence), with
licence files and SHA-256 checksums in `app/static/vendor/README.md`. The 3D
homepage needs no internet connection apart from the optional web fonts.

### Start it (PowerShell, from `D:\FrostIQ`)

```powershell
.venv\Scripts\Activate.ps1
python -m frostiq.simulate --n-journeys 500   # 1. dataset (skip if data/raw/ already has it)
python -m frostiq.model train                 # 2. train and save models/frostiq_model.joblib (~25 s)
python run.py                                 # 3. start the website
```

Open **http://127.0.0.1:5000**. Scroll through the story, or press **Open
Monitoring Console** (also always in the navigation bar). Stop the server with
`Ctrl+C`. After changing templates or code, restart the server so it picks up the
changes.

Step 2 trains the same two models as `frostiq.evaluate`: a gradient-boosting
regressor for the minutes to crossing, and a class-weighted random forest for
"crossing within 120 min?". Both are fitted on the 400 training journeys only,
and the held-out metrics are saved with the model. The model file is ignored by
Git. It's a pickle file, so only load model files you created yourself. Without
a model, the website still works and shows why forecasts are unavailable.

### Using the console

- **Layout:** journey and replay position first; then what was **observed**
  (recorded temperature and recorded status); then what the models **predict**
  (the two-card AI Prediction Centre); then the chart with the playback
  controls. The journey summary, recent readings, status rules and model card
  are in expandable sections below.
- **Journey:** choose any of the synthetic journeys. Filters show held-out
  journeys (never seen in training) or journeys that crossed the threshold. If a
  filter matches nothing, the previous journey's readings are cleared from view
  and an empty state is shown.
- **Replay position:** drag the slider, step reading by reading, or press
  **Play**. The "current" reading is the replay position. Features and forecasts
  only use readings up to that point. This is a replay of recorded data, **not
  live telemetry**.
- **Threshold:** 8 °C is an experimental default. A crossing means a reading
  *strictly above* the threshold. Forecasts exist only for the threshold the
  model was trained on. Other thresholds still update the status, margin, chart
  and readings, and a note says that no estimate is recalculated.
- **AI Prediction Centre:** the Classification card (FLAGGED / NOT FLAGGED,
  with a score bar and the fixed 0.5 decision line) and the experimental
  Regression card (estimated minutes; an output within 5 min of the 120-min cap
  reads as "Beyond 120 min", not as a crossing time). Predictions need at least
  30 minutes of readings. A reading already above the threshold is shown as an
  *observed* crossing, not a prediction. When the two models disagree, the
  Regression card says so.
- **Status rules:**
  - **Above threshold:** the latest reading is strictly above the threshold.
  - **Warning:** not above, and the margin is ≤ 1 °C, or the classifier flags a
    likely crossing within the horizon.
  - **Normal:** neither condition holds.
- **Show hindsight:** reveals the recorded future readings and when the journey
  actually crossed. That's information a real-time system wouldn't have.

The console loads Chart.js 4.4.1 from a CDN, and both pages load web fonts from
Google Fonts. Offline, the chart is replaced by a notice, the latest 12 readings
and all other panels still work, and system fonts are used. The homepage's 3D
libraries are local. Its journey data loads only when you scroll near that
section, and the 3D scene pauses when the tab is hidden.

### Website limitations

- Synthetic data and an academic prototype: **not validated for real medicines
  or vaccines**.
- On held-out journeys, the classifier catches about half of the readings that
  precede a crossing, with many false alarms, and minute estimates are usually
  too late (see [docs/EVALUATION.md](docs/EVALUATION.md)). Door openings and
  sudden failures can't be foreseen from past temperatures.
- Forecasts on *training* journeys are in-sample and look better than they
  really are. The console labels each journey as training or held-out.
- The Flask development server (`python run.py`) is for local use only.

## Running the tests

```powershell
python -m pytest -q
```

## Project structure

```
FrostIQ/
├── data/raw/           # generated or original data (not committed)
├── data/processed/     # cleaned data and features (not committed)
├── notebooks/          # exploration and visual checks
├── src/frostiq/        # core Python package (simulation, features, models)
├── models/             # saved trained models (not committed)
├── app/                # Flask website: routes, services, templates, static files
├── run.py              # starts the website (python run.py)
├── docs/               # synthetic data assumptions, feature and target definitions
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
- [x] Milestone 2: past-only features, 120-minute capped target, journey-level split
  ([docs/FEATURES.md](docs/FEATURES.md))
- [x] Milestone 3A: baselines vs. first ML model ([docs/EVALUATION.md](docs/EVALUATION.md)).
  Result: not yet reliable. On readings that do cross, errors average about 38 min,
  and crossings are predicted too late
- [x] Milestone 3B: warning-timing metrics, crossing classifier, journey-level bootstrap.
  Result: the classifier catches about half of crossing readings (recall 0.51), with many
  false alarms, and regression predictions are usually late. **Not suitable for real
  medicines or vaccines.**
- [x] Milestone 4: Flask website with landing page, journey replay console and
  ML forecast (`python run.py`)

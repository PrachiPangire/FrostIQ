# FrostIQ — An ML-Based Predictive Cold-Chain Monitoring System

FrostIQ is a machine-learning prototype for predictive cold-chain temperature monitoring. It explores how historical temperature readings can help estimate temperature-breach risk and provide early warnings for temperature-sensitive products such as vaccines and insulin.

**Live Demo:** [https://frostiq-qd2b.onrender.com/](https://frostiq-qd2b.onrender.com/)  
**GitHub Repository:** [https://github.com/PrachiPangire/FrostIQ](https://github.com/PrachiPangire/FrostIQ)

> **Important:** FrostIQ is an academic prototype built using synthetic data. Its models have not been validated for real medicines, vaccines, or operational cold-chain safety.

---

## 1. Project Overview

Maintaining the required temperature range during cold-chain transportation and storage is essential for temperature-sensitive products. A temperature excursion may occur because of door openings, cooling interruptions, power failures, or changing environmental conditions.

FrostIQ explores a predictive approach that uses historical temperature readings to estimate the risk of a future temperature breach.

The primary objective is to estimate the **number of minutes remaining before the temperature crosses a configurable threshold**, such as 8 °C for a hypothetical 2–8 °C storage scenario.

The system also provides a classification model to flag readings that indicate a potential breach within the next 120 minutes.

## 2. Key Features

- **Interactive 3D homepage:** Presents the cold-chain problem, monitoring journey, temperature risk, ML predictions, performance, and limitations.
- **3D cold-chain visualization:** Includes a cooler, temperature display, lighting, frost and heat particles, and a journey visualization.
- **Monitoring dashboard:** Replays synthetic temperature journeys and displays readings, status, charts, and prediction results.
- **ML classification:** Estimates whether the temperature will cross 8 °C within the next 120 minutes.
- **ML regression:** Estimates the remaining time until a potential temperature breach.
- **Journey replay controls:** Supports selecting journeys, adjusting the replay position, stepping through readings, and playing recorded sequences.
- **Temperature-risk indicators:** Displays normal, warning, and above-threshold states.
- **Model evaluation:** Includes baseline comparisons, held-out journey evaluation, warning-timing metrics, and a journey-level bootstrap interval.
- **Responsive fallback behavior:** Provides a static SVG cooler when WebGL is unavailable or reduced motion is preferred.
- **Automated tests:** Includes tests for model behavior, features, evaluation, and the Flask web application.

## 3. Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Web framework | Flask |
| Machine learning | scikit-learn |
| Data processing | pandas, NumPy |
| Scientific computing | SciPy |
| Model persistence | joblib |
| Visualization | Matplotlib, Chart.js |
| 3D graphics | three.js |
| Animation and scrolling | GSAP, ScrollTrigger |
| Frontend | HTML, CSS, JavaScript |
| Testing | pytest |
| Production server | Gunicorn |
| Version control | Git, GitHub |
| Deployment | Render |

The three.js and GSAP assets used by the 3D homepage are included in `app/static/vendor/`, with relevant license information documented in that directory. The monitoring console loads Chart.js from a CDN, and the website uses Google Fonts for typography.

## 4. Machine-Learning Approach

FrostIQ uses two complementary prediction tasks.

| Component | Algorithm | Prediction question | Output |
|---|---|---|---|
| Classification | `RandomForestClassifier` | Will the temperature cross 8 °C within the next 120 minutes? | `FLAGGED` or `NOT FLAGGED`, with an uncalibrated score |
| Regression | `HistGradientBoostingRegressor` | How many minutes remain until a potential crossing? | Estimated minutes, capped at 120 minutes |

### Classification

The classifier predicts whether a temperature crossing will occur within the next 120 minutes. It uses a fixed 0.5 decision threshold and a class-weighted random forest.

The displayed score is **not calibrated as a probability**.

### Regression

The regressor estimates the remaining time until a potential temperature crossing. The target is capped at 120 minutes.

When the estimate is near the upper limit, the interface displays `Beyond 120 min` rather than presenting the cap as an exact breach time.

### Prediction rules

The application avoids presenting a forecast when the available information does not support one. For example:

- The latest reading is already above the threshold, indicating an observed crossing.
- Less than 30 minutes of historical readings are available.
- The selected threshold differs from the threshold used to train the model.
- A required trained model is unavailable.

The classification and regression models may disagree. When they do, the prediction interface indicates the disagreement rather than treating both outputs as equivalent.

### Data leakage prevention

Features use readings available only up to the current replay position. Training and evaluation are separated by journey, so readings from a held-out journey are not used to train the models.

The current dataset contains 500 synthetic journeys, with 400 training journeys used for model fitting. Evaluation uses held-out journeys.

## 5. Dataset and Data Generation

FrostIQ currently uses a documented synthetic dataset rather than real cold-chain telemetry.

The simulator models scenarios such as:

- Normal cooling behavior.
- Gradual temperature increases.
- Door-opening disturbances.
- Faster warming scenarios.

Labels are derived from future temperature trajectories. When a crossing occurs during the observed period, the remaining time is measured from the current reading to the crossing. When no crossing is observed, the remaining-time target is treated as censored rather than assigning an invented crossing time.

Dataset assumptions, generation details, and limitations are documented in [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md).

The committed dataset is located at:

`data/raw/synthetic_readings.csv`

**Synthetic data cannot establish real-world predictive accuracy or medical safety.** Real public datasets may be investigated in future work.

## 6. Model Evaluation and Limitations

The project evaluates baseline methods and ML models using held-out journeys. Evaluation includes regression errors, warning timing, classification confusion matrices, and a journey-level bootstrap interval.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the evaluation methodology, metrics, and detailed findings.

Current evaluation indicates that the models are **not yet reliable enough for real-world cold-chain decisions**:

- The classifier catches approximately half of the readings preceding a crossing, with a reported recall of about 0.51 and many false alarms.
- For readings that cross the threshold, time-to-crossing estimates have errors averaging approximately 38 minutes.
- Regression predictions are usually too late.
- Door openings and sudden equipment failures cannot necessarily be predicted from past temperatures alone.
- Results on training journeys are in-sample and may look better than performance on unseen journeys.

These limitations are important when interpreting the dashboard and prediction results. FrostIQ is intended for academic demonstration and further experimentation, not as a substitute for validated monitoring equipment, established safety procedures, or qualified professional judgment.

## 7. Website and Dashboard

### Interactive homepage

The homepage presents eight sections:

1. Home
2. The Problem
3. The Journey
4. Temperature Risk
5. How It Works
6. AI Predictions
7. Performance
8. Limitations

The three.js scene includes a programmatically constructed cold-chain cooler, hinged lid, display, lighting, frost and heat particles, a journey route, a temperature column with a threshold ring, an ML model core, and a confusion matrix based on held-out counts.

GSAP ScrollTrigger connects scene changes to page scrolling without hijacking normal scrolling. When WebGL is unavailable or reduced motion is enabled, the page uses a static SVG cooler fallback.

### Monitoring console

The console is available at `/dashboard` and provides:

- Synthetic journey selection and replay.
- Recorded temperature and status information.
- Classification and regression prediction cards.
- Temperature charts and playback controls.
- Training versus held-out journey indicators.
- Expandable journey summaries, recent readings, status rules, and model information.

The replay controls operate on recorded synthetic data. **This is not a live sensor or IoT telemetry system.**

### Temperature-status rules

- **Above threshold:** The latest reading is strictly greater than the selected threshold.
- **Warning:** The temperature is not above the threshold, but the margin is at most 1 °C, or the classifier flags a likely crossing within the prediction horizon.
- **Normal:** Neither warning condition applies.

The default threshold is 8 °C. Forecasts are available only for the threshold used during model training. Changing the display threshold updates the status and chart, but does not retrain the model or recalculate forecasts for a new threshold.

## 8. Live Deployment

FrostIQ is deployed as a Python web service on Render.

- **Platform:** Render
- **Application framework:** Flask
- **Production server:** Gunicorn
- **GitHub branch:** `main`

**Open the deployed application:** [FrostIQ Live Demo](https://frostiq-qd2b.onrender.com/)

The application loads the committed dataset and trained model from the repository. The Render service uses the following configuration.

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
gunicorn --bind 0.0.0.0:$PORT run:app
```

Render's free instances may spin down after inactivity, so the first request after a period of inactivity may take longer to respond.

The application should be monitored for startup failures, memory limitations, and dependency compatibility as the project evolves.

## 9. Local Setup and Installation

### Prerequisites

- Python 3.14 or a compatible version satisfying the project's Python requirement.
- Git.
- A terminal such as Windows PowerShell.
- Internet access to install dependencies and load the console's external chart library and web fonts.

### Clone the repository

```powershell
git clone https://github.com/PrachiPangire/FrostIQ.git
cd FrostIQ
```

### Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can permit local activation scripts for your Windows user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Only change this setting if needed and if you understand the implications.

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses an editable local package installation through its requirements configuration.

### Run the website

The repository includes the dataset and trained model needed by the deployed application. For the normal local run, start the website with:

```powershell
python run.py
```

Open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

To open the monitoring console directly, visit:

[http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard)

Stop the local server with `Ctrl+C`.

## 10. Regenerating Data and Training Models

**Note:** Regenerating data or retraining models changes local project artifacts. These steps are optional and are not required simply to run the existing application.

Run commands from the repository root with the virtual environment activated.

### Generate synthetic journeys

```powershell
python -m frostiq.simulate
```

The default generation settings are 120 journeys, seed 42, and an 8.0 °C threshold.

Generate the larger dataset used from Milestone 2 onwards:

```powershell
python -m frostiq.simulate --n-journeys 500
```

Generate data with another threshold:

```powershell
python -m frostiq.simulate --threshold 10
```

### Train and save the models

```powershell
python -m frostiq.model train
```

The trained model is saved to `models/frostiq_model.joblib`. Training takes approximately 25 seconds under the documented development setup; actual time depends on the environment.

**Warning:** Retraining replaces the saved model artifact. Keep a backup if you need to preserve the current model.

### Evaluate the models

```powershell
python -m frostiq.evaluate
```

This command evaluates baseline methods and ML models, prints regression metrics and warning-timing results, produces the crossing classifier's confusion matrix, and calculates a journey-level bootstrap interval.

See [`docs/FEATURES.md`](docs/FEATURES.md) for feature and target definitions and [`docs/EVALUATION.md`](docs/EVALUATION.md) for the evaluation methodology.

## 11. Running the Tests

Run the automated test suite from the repository root:

```powershell
python -m pytest -q
```

The tests cover model behavior, feature generation, evaluation, and the Flask web application.

## 12. Project Structure

```text
FrostIQ/
├── app/
│   ├── routes.py
│   ├── services.py
│   ├── templates/
│   └── static/
│       ├── css/
│       ├── js/
│       └── vendor/
├── data/
│   └── raw/
│       └── synthetic_readings.csv
├── docs/
│   ├── EVALUATION.md
│   ├── FEATURES.md
│   └── SYNTHETIC_DATA.md
├── models/
│   └── frostiq_model.joblib
├── src/
│   └── frostiq/
│       ├── simulate.py
│       ├── features.py
│       ├── model.py
│       └── evaluate.py
├── tests/
├── requirements.txt
├── pyproject.toml
├── run.py
└── README.md
```

The repository also includes frontend assets, vendored JavaScript libraries and their license information, HTML templates, and automated tests.

## 13. Development Milestones

- [x] **Milestone 1A:** Environment and project structure.
- [x] **Milestone 1B:** Synthetic temperature dataset with time-to-threshold labels.
- [x] **Milestone 2:** Past-only features, 120-minute capped target, and journey-level data split.
- [x] **Milestone 3A:** Baseline comparison and initial ML model evaluation.
- [x] **Milestone 3B:** Warning-timing metrics, crossing classifier, and journey-level bootstrap evaluation.
- [x] **Milestone 4:** Flask website with 3D homepage, monitoring console, and ML forecasts.
- [x] **Deployment:** GitHub repository and live Render web service.

The evaluation milestones identified substantial model limitations. Improving generalization, reducing false alarms, improving warning lead time, and evaluating on real-world data remain important future research directions.

## 14. Future Improvements

Potential next steps include:

- Investigating suitable public cold-chain and temperature-monitoring datasets.
- Comparing model performance on real-world data when available.
- Improving warning lead time and reducing false alarms.
- Exploring better handling of censored time-to-event targets.
- Testing model calibration and decision thresholds.
- Adding real sensor integration only after suitable validation.
- Expanding deployment monitoring and reproducibility documentation.

These are proposed improvements, not claims that the features have already been implemented.

## 15. License and Third-Party Assets

The repository includes license information for its vendored three.js and GSAP assets under `app/static/vendor/`. Consult those files before redistributing or modifying the third-party libraries.

A separate project-wide license has not been specified in this README.

---

**FrostIQ** is an academic machine-learning prototype demonstrating a complete development workflow: synthetic data generation, feature engineering, model training, evaluation, web application development, testing, version control, and deployment.

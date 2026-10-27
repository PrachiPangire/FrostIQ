"""Train, save, load and apply the FrostIQ models used by the website.

Reuses the leakage-safe pipeline and the exact model settings from
``evaluate.py``:

* regressor  -> HistGradientBoostingRegressor predicting minutes to threshold,
                capped at the horizon (``target_min``)
* classifier -> class-weighted RandomForestClassifier predicting
                ``crosses_within_horizon``

Both are fitted on the *training journeys* of the usual journey-level split, so
the held-out test journeys stay unseen. The artifact records which journeys were
used for training, so the dashboard can say whether a journey was seen in
training. Held-out metrics are computed with the functions in ``evaluate.py``.

Usage::

    python -m frostiq.model train          # writes models/frostiq_model.joblib

The artifact is a joblib (pickle) file: only load files you created yourself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from frostiq.config import DEFAULT_SEED, DEFAULT_THRESHOLD_C, MODELS_DIR, RAW_DATA_DIR
from frostiq.evaluate import (
    CLASS_LABEL,
    DECISION_THRESHOLD,
    TARGET,
    TEST_SIZE,
    classification_metrics,
    crossing_probability,
    feature_matrix,
    fit_classifier,
    fit_models,
    masked_mae,
    warning_timing,
)
from frostiq.features import (
    DEFAULT_HORIZON_MIN,
    FEATURE_COLUMNS,
    build_features,
    make_capped_target,
    split_by_journey,
)
from frostiq.simulate import READINGS_FILE

MODEL_FILE = "frostiq_model.joblib"
DEFAULT_MODEL_PATH = MODELS_DIR / MODEL_FILE
ARTIFACT_VERSION = 1

# A prediction needs this much history, so the 30-minute trend features exist.
MIN_HISTORY_MIN = 30

STATUS_PREDICTED = "predicted"
STATUS_ABOVE = "observed_above"
STATUS_INSUFFICIENT = "insufficient_history"
STATUS_THRESHOLD_MISMATCH = "threshold_mismatch"


class ModelUnavailable(Exception):
    """The model artifact is missing, unreadable or incompatible."""


def train_model(readings: pd.DataFrame, horizon=DEFAULT_HORIZON_MIN, threshold_c=DEFAULT_THRESHOLD_C,
                seed=DEFAULT_SEED, test_size=TEST_SIZE) -> dict:
    """Fit both models on training journeys and measure them on held-out journeys."""
    table = make_capped_target(build_features(readings, threshold_c=threshold_c), horizon=horizon)
    train, test = split_by_journey(table, test_size=test_size, seed=seed)
    regressor = fit_models(train, seed=seed)["hist_gb"]
    classifier = fit_classifier(train, seed=seed)

    minutes = np.clip(regressor.predict(feature_matrix(test)), 0.0, horizon)
    flagged = crossing_probability(classifier, test) >= DECISION_THRESHOLD
    crossing = test[CLASS_LABEL].to_numpy(bool)
    timing = warning_timing(test[TARGET], minutes, crossing)
    held_out = {
        "journeys": int(test["journey_id"].nunique()),
        "rows": int(len(test)),
        "mae_all": masked_mae(test[TARGET], minutes)[0],
        "mae_crossing": masked_mae(test[TARGET][crossing], minutes[crossing])[0],
        "late_pct": timing["late_pct"],
        "mean_lateness": timing["mean_lateness"],
        **{k: v for k, v in classification_metrics(crossing, flagged).items()
           if k in ("positives", "negatives", "caught", "missed", "recall", "precision",
                    "false_positive_rate", "tp", "fp", "fn", "tn")},
    }
    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": version("scikit-learn"),
        "feature_columns": list(FEATURE_COLUMNS),
        "horizon_min": float(horizon),
        "threshold_c": float(threshold_c),
        "decision_threshold": DECISION_THRESHOLD,
        "seed": seed,
        "train_journeys": sorted(train["journey_id"].unique()),
        "test_journeys": sorted(test["journey_id"].unique()),
        "held_out_metrics": held_out,
        "regressor": regressor,
        "classifier": classifier,
    }


def save_model(artifact: dict, path=DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_model(path=DEFAULT_MODEL_PATH) -> dict:
    """Load and check an artifact. Raises ``ModelUnavailable`` with a readable reason."""
    path = Path(path)
    if not path.exists():
        raise ModelUnavailable(f"No trained model found at {path}. Train one with: python -m frostiq.model train")
    try:
        artifact = joblib.load(path)
    except Exception as err:  # corrupted or incompatible pickle
        raise ModelUnavailable(f"Model file {path.name} could not be loaded ({type(err).__name__}). "
                               "Retrain with: python -m frostiq.model train") from err
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ModelUnavailable("Model file has an unexpected format. Retrain with: python -m frostiq.model train")
    if artifact.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ModelUnavailable("Model was trained with different features. Retrain with: python -m frostiq.model train")
    return artifact


def predict_latest(artifact: dict, features: pd.DataFrame, threshold_c: float) -> dict:
    """Predict for the *last* row of one journey's features (output of ``build_features``).

    Only that row's past-only features are used. Returns a dict whose ``status`` is:

    * ``predicted``             minutes and crossing score are available
    * ``observed_above``        the latest reading is already above the threshold
    * ``insufficient_history``  less than ``MIN_HISTORY_MIN`` minutes of readings
    * ``threshold_mismatch``    the requested threshold differs from the model's
    """
    if features.empty:
        raise ValueError("no readings to predict from")
    horizon = artifact["horizon_min"]
    last = features.iloc[[-1]]
    temperature = float(last["temperature_c"].iloc[0])
    history = (features["timestamp"].iloc[-1] - features["timestamp"].iloc[0]).total_seconds() / 60
    base = {"horizon_min": horizon, "model_threshold_c": artifact["threshold_c"], "history_min": history,
            "decision_threshold": artifact["decision_threshold"],
            "minutes": None, "capped": None, "crossing_score": None, "flags_crossing": None}

    if temperature > threshold_c:
        return {**base, "status": STATUS_ABOVE,
                "message": "The latest reading is already above the threshold: an observed crossing, not a prediction."}
    if float(threshold_c) != artifact["threshold_c"]:
        return {**base, "status": STATUS_THRESHOLD_MISMATCH,
                "message": (f"The model was trained for a {artifact['threshold_c']:g} °C threshold. "
                            f"Predictions for {threshold_c:g} °C would need a model retrained for that threshold.")}
    if history < MIN_HISTORY_MIN:
        return {**base, "status": STATUS_INSUFFICIENT,
                "message": (f"Needs {MIN_HISTORY_MIN} minutes of readings for the trend features; "
                            f"this journey has {history:g} so far.")}

    X = feature_matrix(last)
    minutes = float(np.clip(artifact["regressor"].predict(X)[0], 0.0, horizon))
    score = float(crossing_probability(artifact["classifier"], last)[0])
    return {**base, "status": STATUS_PREDICTED, "message": None,
            "minutes": minutes,
            # Within one 5-minute reading of the cap: read as "not expected within the horizon".
            "capped": minutes >= horizon - 5,
            "crossing_score": score,
            "flags_crossing": score >= artifact["decision_threshold"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and save the FrostIQ website models.")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--readings", type=Path, default=RAW_DATA_DIR / READINGS_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    if not args.readings.exists():
        raise SystemExit(f"{args.readings} not found. Generate it first:\n"
                         "    python -m frostiq.simulate --n-journeys 500")
    artifact = train_model(pd.read_csv(args.readings))
    path = save_model(artifact, args.out)
    m = artifact["held_out_metrics"]
    print(f"Trained on {len(artifact['train_journeys'])} journeys; held out {m['journeys']} journeys ({m['rows']} rows).")
    print(f"Held-out regressor MAE: {m['mae_all']:.1f} min overall, {m['mae_crossing']:.1f} min on crossing rows; "
          f"late on {m['late_pct']:.1f}% of crossing rows.")
    print(f"Held-out classifier: caught {m['caught']} of {m['positives']} crossing rows "
          f"(recall {m['recall']:.3f}), precision {m['precision']:.3f}.")
    print(f"Saved model -> {path}")
    print("Synthetic data only: these numbers do not establish real-world safety or accuracy.")


if __name__ == "__main__":
    main()

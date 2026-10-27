"""Data and prediction services for the FrostIQ website.

Everything the website displays comes from here: recorded synthetic readings,
features from ``frostiq.features`` and predictions from ``frostiq.model``.
Nothing is invented. Missing data or a missing model produce explicit
"unavailable" states rather than placeholder numbers.

Dashboard status rules (also shown in the dashboard legend):

* ABOVE THRESHOLD  latest reading is strictly above the threshold (observed).
* WARNING          not above, and the margin is at most ``WARNING_MARGIN_C``,
                   or the crossing classifier flags a crossing within the horizon.
* NORMAL           neither of the above.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from frostiq.features import build_features
from frostiq.model import STATUS_PREDICTED, ModelUnavailable, load_model, predict_latest
from frostiq.simulate import CONFIG_FILE, STATUS_WILL_CROSS, add_threshold_labels

WARNING_MARGIN_C = 1.0
RECENT_READINGS = 12
THRESHOLD_LIMITS_C = (-30.0, 50.0)

LEVEL_NORMAL = "normal"
LEVEL_WARNING = "warning"
LEVEL_ABOVE = "above"


class DataUnavailable(Exception):
    """The synthetic readings file is missing or unreadable."""


class InvalidRequest(ValueError):
    """A request parameter is invalid (shown to the user as a 400 error)."""


def _num(value, digits=2):
    """JSON-safe rounded float (NaN/inf -> None)."""
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else round(value, digits)


class JourneyStore:
    """Loads the readings CSV once and serves individual journeys."""

    def __init__(self, readings_path):
        self.readings_path = Path(readings_path)
        self._readings = None
        self._groups = None

    def _load(self):
        if self._readings is not None:
            return
        if not self.readings_path.exists():
            raise DataUnavailable(
                f"Synthetic dataset not found at {self.readings_path}. "
                "Generate it with: python -m frostiq.simulate --n-journeys 500")
        try:
            readings = pd.read_csv(self.readings_path)
            readings["timestamp"] = pd.to_datetime(readings["timestamp"], format="ISO8601", utc=True)
        except Exception as err:
            raise DataUnavailable(f"Synthetic dataset could not be read ({type(err).__name__}).") from err
        missing = {"journey_id", "timestamp", "temperature_c"} - set(readings.columns)
        if missing:
            raise DataUnavailable(f"Synthetic dataset is missing columns: {sorted(missing)}")
        readings = readings.sort_values(["journey_id", "timestamp"], kind="stable").reset_index(drop=True)
        self._readings = readings
        self._groups = readings.groupby("journey_id", sort=True).indices

    def dataset_info(self) -> dict:
        """Journey and reading counts, plus the generator config if present."""
        self._load()
        info = {"journeys": len(self._groups), "readings": int(len(self._readings)),
                "n_journeys_config": None, "seed": None}
        config = self.readings_path.with_name(CONFIG_FILE)
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
            info["n_journeys_config"], info["seed"] = data.get("n_journeys"), data.get("seed")
        except (OSError, ValueError, AttributeError):
            pass
        return info

    def journey_ids(self) -> list:
        self._load()
        return list(self._groups)

    def journey(self, journey_id: str) -> pd.DataFrame:
        self._load()
        if journey_id not in self._groups:
            raise KeyError(journey_id)
        return self._readings.iloc[self._groups[journey_id]].reset_index(drop=True)

    def journey_list(self, threshold_c: float) -> list:
        """One summary per journey, for the selector."""
        self._load()
        r = self._readings
        grouped = r.groupby("journey_id", sort=True)
        crossed = grouped["temperature_c"].max() > threshold_c
        scenario = grouped["scenario"].first() if "scenario" in r.columns else None
        counts = grouped.size()
        return [
            {"id": jid,
             "scenario": None if scenario is None else str(scenario[jid]),
             "readings": int(counts[jid]),
             "crossed": bool(crossed[jid])}
            for jid in counts.index
        ]


class ModelProvider:
    """Loads the model artifact lazily; remembers why it is unavailable."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._artifact = None
        self._error = None

    def get(self):
        """Return (artifact, None) or (None, reason)."""
        if self._artifact is None and self._error is None:
            try:
                self._artifact = load_model(self.model_path)
            except ModelUnavailable as err:
                self._error = str(err)
        return self._artifact, self._error


def parse_threshold(raw, default: float) -> float:
    if raw is None or raw == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InvalidRequest("threshold must be a number in °C") from None
    low, high = THRESHOLD_LIMITS_C
    if not math.isfinite(value) or not low <= value <= high:
        raise InvalidRequest(f"threshold must be between {low:g} and {high:g} °C")
    return value


def parse_position(raw, n_readings: int) -> int:
    """Replay position = index of the 'current' reading (default: the latest)."""
    if raw is None or raw == "":
        return n_readings - 1
    try:
        position = int(raw)
    except (TypeError, ValueError):
        raise InvalidRequest("at must be a whole number (reading index)") from None
    if not 0 <= position < n_readings:
        raise InvalidRequest(f"at must be between 0 and {n_readings - 1} for this journey")
    return position


def status_level(temperature: float, threshold_c: float, prediction: dict | None) -> dict:
    """Apply the documented dashboard status rules."""
    margin = threshold_c - temperature
    if temperature > threshold_c:
        return {"level": LEVEL_ABOVE, "reasons": [f"Latest reading is above {threshold_c:g} °C (observed)."]}
    reasons = []
    if margin <= WARNING_MARGIN_C:
        reasons.append(f"Margin {margin:.2f} °C is within {WARNING_MARGIN_C:g} °C of the threshold.")
    if prediction and prediction.get("status") == STATUS_PREDICTED and prediction["flags_crossing"]:
        reasons.append(f"Classifier flags a likely crossing within {prediction['horizon_min']:g} min.")
    if reasons:
        return {"level": LEVEL_WARNING, "reasons": reasons}
    return {"level": LEVEL_NORMAL, "reasons": ["Below the threshold with no warning condition met."]}


CLASSIFIER_NAME = "RandomForestClassifier"
REGRESSOR_NAME = "HistGradientBoostingRegressor"


def prediction_outputs(prediction: dict | None, model_error: str | None) -> tuple[dict, dict]:
    """Split the model output into the two website cards: classification and regression.

    Classification answers "will it cross within the horizon?" (FLAGGED / NOT FLAGGED).
    Regression estimates "how many minutes until it crosses?" and is experimental.
    Neither value is ever filled in when the model gave no valid prediction.
    """
    if prediction is None:
        message = model_error or "No trained model is loaded."
        return (
            {"model": CLASSIFIER_NAME, "available": False, "status": "model_unavailable",
             "flagged": None, "score": None, "decision_threshold": None, "horizon_min": None, "message": message},
            {"model": REGRESSOR_NAME, "experimental": True, "available": False, "status": "model_unavailable",
             "minutes": None, "model_output_min": None, "horizon_min": None, "message": message},
        )

    horizon = prediction["horizon_min"]
    if prediction["status"] != STATUS_PREDICTED:
        # Observed crossing, too little history, or a threshold the model wasn't trained for.
        return (
            {"model": CLASSIFIER_NAME, "available": False, "status": prediction["status"], "flagged": None,
             "score": None, "decision_threshold": prediction.get("decision_threshold"), "horizon_min": horizon,
             "message": prediction["message"]},
            {"model": REGRESSOR_NAME, "experimental": True, "available": False, "status": prediction["status"],
             "minutes": None, "model_output_min": None, "horizon_min": horizon, "message": prediction["message"]},
        )

    flagged = bool(prediction["flags_crossing"])
    output = _num(prediction["minutes"], 1)
    classification = {
        "model": CLASSIFIER_NAME, "available": True, "status": "predicted", "flagged": flagged,
        "score": _num(prediction["crossing_score"], 3), "decision_threshold": prediction.get("decision_threshold"),
        "horizon_min": horizon, "message": None,
    }
    if prediction["capped"]:
        # The output sits at the horizon cap: it means "not within the horizon", not a crossing time.
        regression = {
            "model": REGRESSOR_NAME, "experimental": True, "available": True, "status": "beyond_horizon",
            "minutes": None, "model_output_min": output, "horizon_min": horizon,
            "message": (f"No crossing is estimated within {horizon:g} minutes. The model output ({output:g} min) is at "
                        f"the {horizon:g}-minute cap of its training target, not a predicted crossing time."
                        + (" The classifier does flag a likely crossing, so the two models disagree: treat this as uncertain."
                           if flagged else "")),
        }
    else:
        note = None if flagged else ("The classifier did not flag a crossing, so the two models disagree. "
                                     "Treat this estimate with extra caution.")
        regression = {
            "model": REGRESSOR_NAME, "experimental": True, "available": True, "status": "estimated",
            "minutes": output, "model_output_min": output, "horizon_min": horizon, "message": note,
        }
    return classification, regression


def _summary(rows: pd.DataFrame, threshold_c: float) -> dict:
    temps = rows["temperature_c"].to_numpy(float)
    above = temps > threshold_c
    starts = above & ~np.concatenate(([False], above[:-1]))
    minutes = (rows["timestamp"] - rows["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60
    step = float(np.median(np.diff(minutes))) if len(minutes) > 1 else 0.0
    return {
        "readings": int(len(rows)),
        "duration_min": _num(minutes[-1], 1),
        "min_c": _num(temps.min()), "max_c": _num(temps.max()), "mean_c": _num(temps.mean()),
        "excursions": int(starts.sum()),
        "readings_above": int(above.sum()),
        # Approximate: each reading above the threshold represents one sampling interval.
        "minutes_above_approx": _num(above.sum() * step, 1),
    }


def _hindsight(rows: pd.DataFrame, position: int, threshold_c: float) -> dict:
    """What actually happened after the replay position (uses future readings)."""
    labelled = add_threshold_labels(rows[["journey_id", "timestamp", "temperature_c"]], threshold_c)
    row = labelled.iloc[position]
    if row["label_status"] == STATUS_WILL_CROSS:
        return {"outcome": "crossed_later", "minutes_to_crossing": _num(row["minutes_to_threshold"], 1)}
    if row["temperature_c"] > threshold_c:
        return {"outcome": "already_above", "minutes_to_crossing": None}
    return {"outcome": "no_crossing_observed", "minutes_to_crossing": None,
            "observed_for_min": _num(row["minutes_to_window_end"], 1)}


def journey_payload(store: JourneyStore, models: ModelProvider, journey_id: str,
                    position_raw=None, threshold_raw=None, default_threshold=8.0) -> dict:
    """Everything the dashboard needs for one journey at one replay position."""
    threshold_c = parse_threshold(threshold_raw, default_threshold)
    try:
        rows = store.journey(journey_id)
    except KeyError:
        raise
    position = parse_position(position_raw, len(rows))
    visible = rows.iloc[: position + 1]

    features = build_features(visible[["journey_id", "timestamp", "temperature_c"]], threshold_c=threshold_c)
    artifact, model_error = models.get()
    prediction = predict_latest(artifact, features, threshold_c) if artifact else None

    latest = visible.iloc[-1]
    temperature = float(latest["temperature_c"])
    minutes = (visible["timestamp"] - rows["timestamp"].iloc[0]).dt.total_seconds() / 60
    future = rows.iloc[position + 1:]
    future_minutes = (future["timestamp"] - rows["timestamp"].iloc[0]).dt.total_seconds() / 60

    split = None
    if artifact:
        split = ("train" if journey_id in set(artifact["train_journeys"])
                 else "test" if journey_id in set(artifact["test_journeys"]) else "unseen")

    recent = visible.iloc[-RECENT_READINGS:][::-1]
    classification, regression = prediction_outputs(prediction, model_error)
    return {
        "journey_id": journey_id,
        "scenario": str(latest["scenario"]) if "scenario" in rows.columns else None,
        "split": split,
        "threshold_c": threshold_c,
        "position": position,
        "total_readings": int(len(rows)),
        "latest": {
            "timestamp": latest["timestamp"].isoformat(),
            "elapsed_min": _num(minutes.iloc[-1], 1),
            "temperature_c": _num(temperature),
            "margin_c": _num(threshold_c - temperature),
        },
        "status": status_level(temperature, threshold_c, prediction),
        "summary": _summary(visible, threshold_c),
        "chart": {
            "elapsed_min": [_num(m, 1) for m in minutes],
            "temperature_c": [_num(t) for t in visible["temperature_c"]],
            "future_elapsed_min": [_num(m, 1) for m in future_minutes],
            "future_temperature_c": [_num(t) for t in future["temperature_c"]],
        },
        "recent": [
            {"timestamp": r["timestamp"].isoformat(),
             "elapsed_min": _num((r["timestamp"] - rows["timestamp"].iloc[0]).total_seconds() / 60, 1),
             "temperature_c": _num(r["temperature_c"]),
             "above": bool(r["temperature_c"] > threshold_c)}
            for _, r in recent.iterrows()
        ],
        "prediction": None if prediction is None else {
            **prediction,
            "minutes": _num(prediction["minutes"], 1),
            "crossing_score": _num(prediction["crossing_score"], 3),
            "history_min": _num(prediction["history_min"], 1),
        },
        "classification": classification,
        "regression": regression,
        "model_error": model_error,
        "hindsight": _hindsight(rows, position, threshold_c),
    }


def model_card(models: ModelProvider) -> dict | None:
    """Held-out metrics recorded at training time (None if no model)."""
    artifact, _ = models.get()
    if artifact is None:
        return None
    m = dict(artifact["held_out_metrics"])
    if "tp" not in m:  # older model files: derive the confusion counts from stored metrics
        m["tp"], m["fn"] = m["caught"], m["missed"]
        m["fp"] = int(round(m["false_positive_rate"] * m["negatives"]))
        m["tn"] = m["negatives"] - m["fp"]
    return {
        "created_utc": artifact["created_utc"],
        "threshold_c": artifact["threshold_c"],
        "horizon_min": artifact["horizon_min"],
        "decision_threshold": artifact["decision_threshold"],
        "train_journeys": len(artifact["train_journeys"]),
        "test_journeys": len(artifact["test_journeys"]),
        **{k: (_num(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
    }

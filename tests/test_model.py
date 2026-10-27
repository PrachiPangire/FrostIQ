"""Tests for training, saving, loading and applying the website model."""

import numpy as np
import pandas as pd
import pytest

from frostiq.evaluate import crossing_probability, feature_matrix
from frostiq.features import FEATURE_COLUMNS, build_features
from frostiq.model import (
    MIN_HISTORY_MIN,
    STATUS_ABOVE,
    STATUS_INSUFFICIENT,
    STATUS_PREDICTED,
    STATUS_THRESHOLD_MISMATCH,
    ModelUnavailable,
    load_model,
    predict_latest,
    save_model,
    train_model,
)
from frostiq.simulate import SimulationConfig, generate_dataset


@pytest.fixture(scope="module")
def readings():
    return generate_dataset(SimulationConfig(n_journeys=40, seed=3))[0]


@pytest.fixture(scope="module")
def artifact(readings):
    return train_model(readings)


def _journey_features(readings, journey_id, upto=None, threshold_c=8.0):
    rows = readings[readings["journey_id"] == journey_id]
    if upto is not None:
        rows = rows.iloc[: upto + 1]
    return build_features(rows[["journey_id", "timestamp", "temperature_c"]], threshold_c=threshold_c)


def test_artifact_records_features_split_and_metrics(artifact):
    assert artifact["feature_columns"] == list(FEATURE_COLUMNS)
    assert list(artifact["regressor"].feature_names_in_) == list(FEATURE_COLUMNS)
    assert list(artifact["classifier"].feature_names_in_) == list(FEATURE_COLUMNS)
    assert not set(artifact["train_journeys"]) & set(artifact["test_journeys"])
    assert artifact["threshold_c"] == 8.0 and artifact["horizon_min"] == 120.0
    m = artifact["held_out_metrics"]
    assert m["caught"] + m["missed"] == m["positives"]
    assert m["journeys"] == len(artifact["test_journeys"])


def test_training_is_reproducible(readings, artifact):
    again = train_model(readings)
    X = feature_matrix(build_features(readings).dropna(subset=["slope_30"]).head(200))
    np.testing.assert_array_equal(again["regressor"].predict(X), artifact["regressor"].predict(X))
    np.testing.assert_array_equal(again["classifier"].predict_proba(X), artifact["classifier"].predict_proba(X))
    assert again["train_journeys"] == artifact["train_journeys"]


def test_save_and_load_round_trip(artifact, tmp_path):
    path = save_model(artifact, tmp_path / "m.joblib")
    loaded = load_model(path)
    assert loaded["train_journeys"] == artifact["train_journeys"]
    assert loaded["feature_columns"] == artifact["feature_columns"]


def test_load_model_missing_file(tmp_path):
    with pytest.raises(ModelUnavailable, match="python -m frostiq.model train"):
        load_model(tmp_path / "missing.joblib")


def test_load_model_corrupted_file(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"this is not a model")
    with pytest.raises(ModelUnavailable, match="could not be loaded"):
        load_model(path)


def test_load_model_rejects_incompatible_artifacts(artifact, tmp_path):
    save_model({**artifact, "feature_columns": ["temperature_c"]}, tmp_path / "old.joblib")
    with pytest.raises(ModelUnavailable, match="different features"):
        load_model(tmp_path / "old.joblib")
    save_model({"something": "else"}, tmp_path / "other.joblib")
    with pytest.raises(ModelUnavailable, match="unexpected format"):
        load_model(tmp_path / "other.joblib")


def _find_row(readings, condition):
    for journey_id, rows in readings.groupby("journey_id"):
        rows = rows.reset_index(drop=True)
        for i in range(len(rows)):
            if condition(rows, i):
                return journey_id, i
    raise AssertionError("no matching row in test data")


def test_prediction_matches_models_on_last_row(readings, artifact):
    journey_id, i = _find_row(readings, lambda r, i: i >= 20 and r["temperature_c"][i] <= 8.0)
    features = _journey_features(readings, journey_id, upto=i)
    result = predict_latest(artifact, features, 8.0)
    assert result["status"] == STATUS_PREDICTED
    last = features.iloc[[-1]]
    expected = float(np.clip(artifact["regressor"].predict(feature_matrix(last))[0], 0, 120))
    assert result["minutes"] == pytest.approx(expected)
    assert result["crossing_score"] == pytest.approx(float(crossing_probability(artifact["classifier"], last)[0]))
    assert 0 <= result["minutes"] <= 120
    assert result["capped"] == (result["minutes"] >= 115)
    assert result["flags_crossing"] == (result["crossing_score"] >= 0.5)


def test_prediction_ignores_future_readings(readings, artifact):
    journey_id, i = _find_row(readings, lambda r, i: i == 30 and r["temperature_c"][i] <= 8.0)
    truncated = predict_latest(artifact, _journey_features(readings, journey_id, upto=i), 8.0)
    rows = readings[readings["journey_id"] == journey_id].reset_index(drop=True).copy()
    rows.loc[i + 1:, "temperature_c"] = 30.0  # a wildly different future
    altered = build_features(rows[["journey_id", "timestamp", "temperature_c"]]).iloc[: i + 1]
    assert predict_latest(artifact, altered, 8.0) == truncated


def test_already_above_is_observed_not_predicted(readings, artifact):
    journey_id, i = _find_row(readings, lambda r, i: i >= 10 and r["temperature_c"][i] > 8.0)
    result = predict_latest(artifact, _journey_features(readings, journey_id, upto=i), 8.0)
    assert result["status"] == STATUS_ABOVE
    assert result["minutes"] is None and result["crossing_score"] is None


def test_insufficient_history(readings, artifact):
    journey_id, i = _find_row(readings, lambda r, i: i == 2 and r["temperature_c"][i] <= 8.0)
    result = predict_latest(artifact, _journey_features(readings, journey_id, upto=i), 8.0)
    assert result["status"] == STATUS_INSUFFICIENT
    assert result["history_min"] < MIN_HISTORY_MIN and result["minutes"] is None


def test_threshold_mismatch_gives_no_prediction(readings, artifact):
    journey_id, i = _find_row(readings, lambda r, i: i >= 20 and r["temperature_c"][i] <= 8.0)
    result = predict_latest(artifact, _journey_features(readings, journey_id, upto=i, threshold_c=9.0), 9.0)
    assert result["status"] == STATUS_THRESHOLD_MISMATCH and result["minutes"] is None


def test_prediction_rejects_empty_input(artifact):
    with pytest.raises(ValueError):
        predict_latest(artifact, pd.DataFrame(columns=["timestamp", "temperature_c"]), 8.0)

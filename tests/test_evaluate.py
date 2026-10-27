"""Tests for the evaluation pipeline. No test depends on a particular model score."""

import json

import numpy as np
import pandas as pd
import pytest

from frostiq.evaluate import (
    BOOTSTRAP_RESAMPLES,
    DECISION_THRESHOLD,
    MODEL_NAMES,
    TARGET,
    bootstrap_mae_difference,
    classification_metrics,
    crossing_probability,
    feature_matrix,
    fit_classifier,
    fit_models,
    load_dataset_info,
    mae_by_segment,
    main,
    masked_mae,
    predict_all,
    print_report,
    recall_by_scenario,
    run_evaluation,
    warning_timing,
)
from frostiq.features import FEATURE_COLUMNS, NEVER_FEATURES
from frostiq.simulate import SimulationConfig, generate_dataset

CFG = SimulationConfig(n_journeys=60, seed=5)


@pytest.fixture(scope="module")
def readings():
    return generate_dataset(CFG)[0]


@pytest.fixture(scope="module")
def result(readings):
    return run_evaluation(readings)


# --- Metric calculations -----------------------------------------------------


def test_masked_mae_on_hand_made_values():
    assert masked_mae([10, 20, 30], [12, 18, 33]) == (pytest.approx(7 / 3), 3)


def test_masked_mae_ignores_missing_predictions():
    mae, scored = masked_mae([10, 20, 30], [12, np.nan, 40])
    assert scored == 2
    assert mae == pytest.approx((2 + 10) / 2)


def test_masked_mae_with_no_predictions():
    mae, scored = masked_mae([1.0], [np.nan])
    assert np.isnan(mae) and scored == 0


def test_mae_by_segment_on_hand_made_table():
    df = pd.DataFrame({
        TARGET: [10.0, 120.0, 30.0, 120.0],
        "crosses_within_horizon": [True, False, True, False],
        "scenario": ["a", "a", "b", "b"],
    })
    preds = pd.DataFrame({"m": [20.0, 100.0, 30.0, 120.0], "eta": [np.nan, 120.0, 60.0, np.nan]})
    res = mae_by_segment(df, preds).set_index("segment")
    assert res.loc["all", "m_mae"] == pytest.approx((10 + 20 + 0 + 0) / 4)
    assert res.loc["crosses_within_horizon=True", "m_mae"] == pytest.approx(5.0)
    assert res.loc["crosses_within_horizon=False", "m_mae"] == pytest.approx(10.0)
    assert res.loc["scenario=a", "m_mae"] == pytest.approx(15.0)
    assert res.loc["all", "eta_rows"] == 2 and res.loc["all", "rows"] == 4
    assert res.loc["all", "eta_mae"] == pytest.approx((0 + 30) / 2)

    common = mae_by_segment(df, preds, common_rows_only=True).set_index("segment")
    assert common.loc["all", "rows"] == 2  # only rows where every prediction exists
    assert common.loc["all", "m_mae"] == pytest.approx((20 + 0) / 2)
    assert common.loc["all", "m_rows"] == common.loc["all", "eta_rows"] == 2


def test_report_segments_cover_all_test_rows(result):
    res = result["all_rows"].set_index("segment")
    n = len(result["test"])
    assert res.loc["all", "rows"] == n
    assert res.loc["crosses_within_horizon=True", "rows"] + res.loc["crosses_within_horizon=False", "rows"] == n
    assert res.filter(like="scenario=", axis=0)["rows"].sum() == n


def test_common_rows_are_where_linear_eta_exists(result):
    common = result["common_rows"].set_index("segment").loc["all"]
    assert common["rows"] == result["predictions"]["linear_eta"].notna().sum()
    for name in MODEL_NAMES:
        assert common[f"{name}_rows"] == common["rows"]


# --- Predictions -------------------------------------------------------------


def test_predictions_stay_within_target_range(result):
    preds = result["predictions"]
    for name in MODEL_NAMES:
        values = preds[name].dropna()
        assert values.between(0, 120).all(), name


def test_linear_eta_is_capped_and_keeps_missing_values(result):
    test, preds = result["test"], result["predictions"]
    assert preds["linear_eta"].isna().equals(test["linear_eta_min"].isna())
    has = test["linear_eta_min"].notna()
    np.testing.assert_allclose(preds.loc[has, "linear_eta"], np.minimum(test.loc[has, "linear_eta_min"], 120))


def test_dummy_predicts_training_median_only(result):
    train, test = result["train"], result["test"]
    constant = result["models"]["dummy_median"].constant_.ravel()[0]
    assert constant == pytest.approx(train[TARGET].median())
    assert (result["predictions"]["dummy_median"] == min(constant, 120)).all()


# --- Feature selection -------------------------------------------------------


def test_feature_matrix_selects_exactly_feature_columns(result):
    X = feature_matrix(result["test"])
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert not set(X.columns) & set(NEVER_FEATURES)
    for forbidden in ("crosses_within_horizon", "scenario", "journey_id", "timestamp",
                      "target_min", "minutes_to_threshold", "label_status", "elapsed_min"):
        assert forbidden not in X.columns


def test_models_were_fitted_on_feature_columns_only(result):
    assert list(result["models"]["hist_gb"].feature_names_in_) == list(FEATURE_COLUMNS)


REQUIRED_SCRAMBLE = {"crosses_within_horizon", "target_min", "minutes_to_threshold",
                     "minutes_to_window_end", "label_status", "scenario", "journey_id",
                     "timestamp", "elapsed_min"}


def _scramble_forbidden(df):
    """Replace every NEVER_FEATURES column present with different values."""
    scrambled = df.copy()
    columns = [c for c in NEVER_FEATURES if c in df.columns]
    for col in columns:
        values = df[col]
        if pd.api.types.is_bool_dtype(values):
            scrambled[col] = ~values
        elif pd.api.types.is_datetime64_any_dtype(values):
            scrambled[col] = values + pd.Timedelta(days=1000)
        elif pd.api.types.is_numeric_dtype(values):
            scrambled[col] = -values.fillna(0.0) - 999.0
        else:
            scrambled[col] = "scrambled"
        assert not scrambled[col].equals(values), col
    assert REQUIRED_SCRAMBLE <= set(columns)
    return scrambled


def test_predictions_ignore_forbidden_columns(result):
    test = result["test"]
    pd.testing.assert_frame_equal(
        predict_all(result["models"], _scramble_forbidden(test)), predict_all(result["models"], test)
    )


# --- Journey-level separation and training-only fitting ---------------------


def test_train_and_test_share_no_journeys(result):
    assert not set(result["train"]["journey_id"]) & set(result["test"]["journey_id"])


def test_models_depend_only_on_training_rows(result):
    # Refitting on the training rows alone must reproduce the same predictions.
    refit = fit_models(result["train"])
    pd.testing.assert_frame_equal(predict_all(refit, result["test"]), result["predictions"])


# --- Reproducibility ---------------------------------------------------------


def test_evaluation_is_reproducible(readings, result):
    again = run_evaluation(readings)
    pd.testing.assert_frame_equal(again["predictions"], result["predictions"])
    pd.testing.assert_frame_equal(again["all_rows"], result["all_rows"])
    pd.testing.assert_frame_equal(again["common_rows"], result["common_rows"])
    pd.testing.assert_frame_equal(again["timing"], result["timing"])
    np.testing.assert_array_equal(again["crossing_probability"], result["crossing_probability"])
    assert again["classification"] == result["classification"]
    assert again["bootstrap"] == result["bootstrap"]


def test_cli_runs_on_csv_and_prints_report(readings, tmp_path, capsys):
    path = tmp_path / "readings.csv"
    readings.to_csv(path, index=False)
    main(["--readings", str(path)])
    out = capsys.readouterr().out
    assert "A) All test rows" in out and "B) Common rows only" in out
    assert "C) Warning timing" in out and "D) Crossing classifier" in out
    assert "E) Journey-level bootstrap" in out and "not a measure of real-world reliability" in out
    for name in MODEL_NAMES:
        assert name in out


# --- Warning-timing metrics --------------------------------------------------


def test_warning_timing_on_hand_made_values():
    m = warning_timing(y_true=[10, 20, 120, 120], y_pred=[15, 10, 100, np.nan],
                       crosses=[True, True, False, False])
    assert m["rows_scored"] == 3
    assert m["mean_signed_error"] == pytest.approx((5 - 10 - 20) / 3)
    assert m["crossing_rows"] == 2
    assert m["crossing_mean_signed_error"] == pytest.approx(-2.5)
    assert m["late_rows"] == 1
    assert m["late_pct"] == pytest.approx(50.0)
    assert m["mean_lateness"] == pytest.approx(5.0)


def test_warning_timing_when_nothing_is_late():
    m = warning_timing([30, 60], [30, 40], [True, True])  # on time, and early
    assert m["late_rows"] == 0 and m["late_pct"] == 0.0
    assert np.isnan(m["mean_lateness"])


def test_timing_table_uses_common_rows(result):
    timing = result["timing"].set_index("model")
    common = result["predictions"].notna().all(axis=1).sum()
    assert (timing["rows_scored"] == common).all()
    assert (timing["late_rows"] <= timing["crossing_rows"]).all()


# --- Classifier metrics ------------------------------------------------------


def test_classification_metrics_on_hand_made_values():
    m = classification_metrics([True, True, True, False, False, False, False],
                               [True, False, True, True, False, False, False])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
    assert (m["positives"], m["negatives"], m["caught"], m["missed"]) == (3, 4, 2, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)
    assert m["accuracy"] == pytest.approx(5 / 7)


def test_classifier_that_never_flags_catches_nothing():
    m = classification_metrics([True, False, False], [False, False, False])
    assert m["recall"] == 0.0 and m["caught"] == 0 and m["missed"] == 1
    assert np.isnan(m["precision"]) and m["f1"] == 0.0
    assert m["accuracy"] == pytest.approx(2 / 3)  # looks fine, yet misses every crossing


def test_recall_by_scenario_on_hand_made_values():
    df = pd.DataFrame({"crosses_within_horizon": [True, True, True, False],
                       "scenario": ["a", "a", "b", "b"]})
    res = recall_by_scenario(df, [True, False, False, True]).set_index("scenario")
    assert res.loc["a", "positives"] == 2 and res.loc["a", "caught"] == 1 and res.loc["a", "missed"] == 1
    assert res.loc["b", "recall"] == 0.0


def test_reported_classification_counts_match_test_rows(result):
    m = result["classification"]
    test = result["test"]
    assert m["positives"] == test["crosses_within_horizon"].sum()
    assert m["negatives"] == (~test["crosses_within_horizon"]).sum()
    assert m["caught"] + m["missed"] == m["positives"]
    assert result["recall_by_scenario"]["positives"].sum() == m["positives"]


# --- Classifier leakage and training-only fitting ---------------------------


def test_decision_threshold_is_fixed_in_advance(result):
    assert DECISION_THRESHOLD == 0.5
    np.testing.assert_array_equal(result["flagged"], result["crossing_probability"] >= DECISION_THRESHOLD)


def test_classifier_uses_feature_columns_only(result):
    assert list(result["classifier"].feature_names_in_) == list(FEATURE_COLUMNS)
    assert result["classifier"].class_weight == "balanced"


def test_classifier_ignores_forbidden_columns(result):
    np.testing.assert_array_equal(crossing_probability(result["classifier"], _scramble_forbidden(result["test"])),
                                  result["crossing_probability"])


def test_classifier_depends_only_on_training_rows(result):
    refit = fit_classifier(result["train"])
    np.testing.assert_array_equal(crossing_probability(refit, result["test"]), result["crossing_probability"])


def test_crossing_probabilities_are_valid(result):
    p = result["crossing_probability"]
    assert len(p) == len(result["test"])
    assert ((p >= 0) & (p <= 1)).all()


# --- Bootstrap ---------------------------------------------------------------


def _bootstrap_table(errors_a, errors_b, journeys, target=50.0):
    df = pd.DataFrame({"journey_id": journeys, TARGET: target, "crosses_within_horizon": True})
    preds = pd.DataFrame({"hist_gb": target + np.asarray(errors_a, float),
                          "linear_eta": target + np.asarray(errors_b, float)})
    return df, preds


def test_bootstrap_uses_at_least_1000_resamples_by_default(result):
    assert BOOTSTRAP_RESAMPLES >= 1000
    assert result["bootstrap"]["n_resamples"] == BOOTSTRAP_RESAMPLES


def test_bootstrap_estimate_is_the_crossing_mae_difference(result):
    test, preds = result["test"], result["predictions"]
    rows = preds[["hist_gb", "linear_eta"]].notna().all(axis=1) & test["crosses_within_horizon"]
    expected = ((preds.loc[rows, "hist_gb"] - test.loc[rows, TARGET]).abs().mean()
                - (preds.loc[rows, "linear_eta"] - test.loc[rows, TARGET]).abs().mean())
    b = result["bootstrap"]
    assert b["estimate"] == pytest.approx(expected)
    assert b["rows"] == rows.sum()
    assert b["journeys"] == test.loc[rows, "journey_id"].nunique()
    assert b["ci_low"] <= b["ci_high"]


def test_bootstrap_is_reproducible_and_seed_dependent(result):
    test, preds = result["test"], result["predictions"]
    a = bootstrap_mae_difference(test, preds, seed=1)
    assert bootstrap_mae_difference(test, preds, seed=1) == a
    b = bootstrap_mae_difference(test, preds, seed=2)
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_bootstrap_resamples_whole_journeys():
    # One journey only: resampling journeys can only redraw that journey, so the
    # interval collapses to the estimate. Row-level resampling would vary.
    df, preds = _bootstrap_table([1, 9, 30, 2], [0, 0, 0, 0], ["J1"] * 4)
    b = bootstrap_mae_difference(df, preds, n_resamples=1000)
    assert b["journeys"] == 1
    assert b["ci_low"] == pytest.approx(b["estimate"]) and b["ci_high"] == pytest.approx(b["estimate"])


def test_bootstrap_interval_stays_within_journey_level_extremes():
    # Journey J1 always favours linear_eta by 10, J2 is a tie: every resample lies in [0, 10].
    df, preds = _bootstrap_table([10, 0, 0, 0], [0, 0, 0, 0], ["J1", "J2", "J2", "J2"])
    b = bootstrap_mae_difference(df, preds, n_resamples=1000)
    assert 0.0 <= b["ci_low"] <= b["estimate"] <= b["ci_high"] <= 10.0


def test_bootstrap_skips_rows_missing_a_prediction_or_not_crossing():
    df, preds = _bootstrap_table([5, 5, 5], [1, np.nan, 1], ["J1", "J1", "J2"])
    df.loc[2, "crosses_within_horizon"] = False
    b = bootstrap_mae_difference(df, preds, n_resamples=1000)
    assert b["rows"] == 1 and b["journeys"] == 1


def test_bootstrap_rejects_empty_input():
    df, preds = _bootstrap_table([5], [1], ["J1"])
    df["crosses_within_horizon"] = False
    with pytest.raises(ValueError, match="no crossing rows"):
        bootstrap_mae_difference(df, preds)


def test_load_dataset_info_reads_config_next_to_readings(tmp_path):
    (tmp_path / "synthetic_config.json").write_text(
        json.dumps({"n_journeys": 500, "seed": 42, "threshold_c": 8.0, "other": 1}), encoding="utf-8")
    info, note = load_dataset_info(tmp_path / "synthetic_readings.csv")
    assert info == {"n_journeys": 500, "seed": 42, "threshold_c": 8.0} and note is None


def test_load_dataset_info_handles_missing_and_broken_config(tmp_path):
    info, note = load_dataset_info(tmp_path / "synthetic_readings.csv")
    assert info is None and "not found" in note
    (tmp_path / "synthetic_config.json").write_text("{not json", encoding="utf-8")
    info, note = load_dataset_info(tmp_path / "synthetic_readings.csv")
    assert info is None and "could not be read" in note


def test_cli_reports_dataset_config_when_present(readings, tmp_path, capsys):
    readings.to_csv(tmp_path / "synthetic_readings.csv", index=False)
    (tmp_path / "synthetic_config.json").write_text(
        json.dumps({"n_journeys": 60, "seed": 5, "threshold_c": 8.0}), encoding="utf-8")
    main(["--readings", str(tmp_path / "synthetic_readings.csv")])
    assert "Dataset: 60 journeys, seed 5" in capsys.readouterr().out


def test_cli_still_runs_without_dataset_config(readings, tmp_path, capsys):
    readings.to_csv(tmp_path / "readings.csv", index=False)
    main(["--readings", str(tmp_path / "readings.csv")])
    out = capsys.readouterr().out
    assert "journey count and seed unknown" in out and "A) All test rows" in out


def test_report_header_uses_actual_split_size_and_clean_horizon(readings, capsys):
    result = run_evaluation(readings, horizon=90, test_size=0.25, n_resamples=1000)
    print_report(result, horizon=90.0)
    out = capsys.readouterr().out
    assert "Split: journey-level, 75%/25%" in out
    assert "Horizon: 90 min" in out
    assert "capped 90-min target" in out and "crossed within 90 min?" in out
    for messy in ("Horizon: 90.0", "capped 90.0-min", "within 90.0 min"):
        assert messy not in out


def test_cli_explains_missing_dataset(tmp_path):
    with pytest.raises(SystemExit, match="python -m frostiq.simulate"):
        main(["--readings", str(tmp_path / "missing.csv")])

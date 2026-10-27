"""Tests for past-only features, capped targets and journey-level splits."""

import numpy as np
import pandas as pd
import pytest

from frostiq.features import (
    DEFAULT_HORIZON_MIN,
    ETA_CAP_MIN,
    FEATURE_COLUMNS,
    NEVER_FEATURES,
    build_features,
    make_capped_target,
    split_by_journey,
)
from frostiq.simulate import STATUS_ABOVE, SimulationConfig, generate_dataset

H = DEFAULT_HORIZON_MIN
CFG = SimulationConfig(n_journeys=40, seed=11)


@pytest.fixture(scope="module")
def readings():
    return generate_dataset(CFG)[0]


@pytest.fixture(scope="module")
def features(readings):
    return build_features(readings)


@pytest.fixture(scope="module")
def table(features):
    return make_capped_target(features)


def _journey(temps, interval_min=5, journey_id="J0001", start="2026-01-01"):
    t0 = pd.Timestamp(start, tz="UTC")
    return pd.DataFrame(
        {
            "journey_id": journey_id,
            "timestamp": t0 + pd.to_timedelta(np.arange(len(temps)) * interval_min, unit="min"),
            "temperature_c": np.asarray(temps, dtype=float),
        }
    )


def _assert_features_equal(a, b):
    pd.testing.assert_frame_equal(
        a[list(FEATURE_COLUMNS)].reset_index(drop=True),
        b[list(FEATURE_COLUMNS)].reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


# --- Feature definitions -----------------------------------------------------


def test_feature_list_excludes_forbidden_columns():
    assert not set(FEATURE_COLUMNS) & set(NEVER_FEATURES)
    for forbidden in ("scenario", "elapsed_min", "label_status", "minutes_to_threshold",
                      "minutes_to_window_end", "failure_onset_min", "crosses_threshold"):
        assert forbidden not in FEATURE_COLUMNS


def test_features_on_hand_made_journey():
    # Steady warming of 0.1 degC per 5 min = 0.02 degC/min.
    temps = 5.0 + 0.1 * np.arange(20)
    f = build_features(_journey(temps), threshold_c=8.0)
    row = f.iloc[12]  # 60 minutes in
    assert row["margin_c"] == pytest.approx(8.0 - temps[12])
    assert row["delta_15"] == pytest.approx(0.3)
    assert row["delta_30"] == pytest.approx(0.6)
    assert row["delta_60"] == pytest.approx(1.2)
    assert row["slope_30"] == pytest.approx(0.02)
    assert row["slope_60"] == pytest.approx(0.02)
    assert row["accel_30"] == pytest.approx(0.0, abs=1e-12)
    assert row["mean_60"] == pytest.approx(temps[0:13].mean())
    assert row["min_60"] == pytest.approx(temps[0])
    assert row["max_60"] == pytest.approx(temps[12])
    assert row["linear_eta_min"] == pytest.approx((8.0 - temps[12]) / 0.02)


def test_slope_matches_numpy_polyfit(features):
    journey = features[features["journey_id"] == features["journey_id"].iloc[0]].reset_index(drop=True)
    for i in (10, 40, 90):
        now = journey["elapsed_min"][i]
        window = journey[(journey["elapsed_min"] >= now - 30) & (journey["elapsed_min"] <= now)]
        expected = np.polyfit(window["elapsed_min"], window["temperature_c"], 1)[0]
        assert journey["slope_30"][i] == pytest.approx(expected, abs=1e-9)


def test_window_features_are_missing_until_enough_history():
    f = build_features(_journey(np.full(20, 5.0)))
    assert f["delta_15"].iloc[:3].isna().all() and f["delta_15"].iloc[3:].notna().all()
    assert f["slope_30"].iloc[:6].isna().all() and f["slope_30"].iloc[6:].notna().all()
    for name in ("delta_60", "slope_60", "accel_30", "mean_60", "std_60", "min_60", "max_60"):
        assert f[name].iloc[:12].isna().all() and f[name].iloc[12:].notna().all(), name


def test_excursion_features_count_only_past_crossings():
    f = build_features(_journey([5, 9, 9, 6, 5, 9, 6, 5]), threshold_c=8.0)
    assert list(f["n_excursions_so_far"]) == [0, 1, 1, 1, 1, 2, 2, 2]
    assert np.isnan(f["minutes_since_last_above"].iloc[0])
    assert list(f["minutes_since_last_above"].iloc[1:]) == [0, 0, 5, 10, 0, 5, 10]


def test_linear_eta_when_cooling_or_above_threshold():
    cooling = build_features(_journey(7.0 - 0.05 * np.arange(10)))
    assert (cooling["linear_eta_min"].dropna() == ETA_CAP_MIN).all()
    above = build_features(_journey(9.0 + 0.1 * np.arange(10)))
    assert (above["linear_eta_min"].dropna() == 0.0).all()


def test_linear_eta_follows_strict_crossing_rule_at_threshold():
    # Regression test: exactly at the threshold is NOT a crossing.
    # Flat temperature at 8.00 -> "not warming" (cap), not "already above" (0).
    at_threshold_flat = build_features(_journey(np.full(10, 8.0)), threshold_c=8.0)
    assert (at_threshold_flat["linear_eta_min"].dropna() == ETA_CAP_MIN).all()
    # Rising to exactly 8.00 -> margin / slope = 0 minutes (it is right at the limit).
    rising = build_features(_journey([7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0]), threshold_c=8.0)
    assert rising["linear_eta_min"].iloc[-1] == pytest.approx(0.0)
    # Strictly above -> 0.
    above = build_features(_journey(np.full(10, 8.01)), threshold_c=8.0)
    assert (above["linear_eta_min"].dropna() == 0.0).all()


def test_irregular_sampling_uses_real_time():
    df = _journey([5.0, 5.5, 6.0, 6.5])
    df["timestamp"] = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10",
                                      "2026-01-01 00:25", "2026-01-01 00:30"], utc=True)
    f = build_features(df)
    # At 00:30 the reading at or before 00:15 is the 00:10 one (5.5).
    assert f["delta_15"].iloc[3] == pytest.approx(1.0)
    assert f["slope_30"].iloc[3] == pytest.approx(np.polyfit([0, 10, 25, 30], [5.0, 5.5, 6.0, 6.5], 1)[0])


def test_build_features_accepts_csv_text_timestamps(readings, features, tmp_path):
    path = tmp_path / "readings.csv"
    readings.to_csv(path, index=False)
    _assert_features_equal(build_features(pd.read_csv(path)), features)


# --- Future leakage ------------------------------------------------------------


def test_features_do_not_change_when_future_rows_are_removed(features, readings):
    for journey_id in features["journey_id"].unique()[:8]:
        full = features[features["journey_id"] == journey_id]
        raw = readings[readings["journey_id"] == journey_id]
        for cutoff in (len(raw) // 4, len(raw) // 2, len(raw) - 1):
            truncated = build_features(raw.iloc[: cutoff + 1])
            _assert_features_equal(truncated, full.iloc[: cutoff + 1])


def test_features_do_not_change_when_future_temperatures_change(readings):
    journey = readings[readings["journey_id"] == readings["journey_id"].iloc[0]].reset_index(drop=True)
    cutoff = len(journey) // 2
    altered = journey.copy()
    altered.loc[cutoff + 1 :, "temperature_c"] = 30.0  # a wildly different future
    _assert_features_equal(build_features(altered).iloc[: cutoff + 1],
                           build_features(journey).iloc[: cutoff + 1])


def test_row_order_of_input_does_not_matter(readings, features):
    shuffled = readings.sample(frac=1.0, random_state=0)
    _assert_features_equal(build_features(shuffled), features)


# --- Journey boundaries --------------------------------------------------------


def test_features_never_mix_journeys():
    a = _journey(5.0 + 0.1 * np.arange(15), journey_id="J0001")
    b = _journey(np.full(15, 4.0), journey_id="J0002", start="2026-01-01 01:00")
    together = build_features(pd.concat([a, b], ignore_index=True))
    _assert_features_equal(together[together["journey_id"] == "J0002"], build_features(b))
    _assert_features_equal(together[together["journey_id"] == "J0001"], build_features(a))


def test_first_readings_of_every_journey_have_no_history(features):
    first = features.groupby("journey_id").head(1)
    assert first["delta_15"].isna().all()
    assert first["slope_30"].isna().all()


# --- Capped target and censoring -------------------------------------------


def test_capped_target_on_hand_made_journey():
    # 5-minute readings: below for 40 min, then above at minute 40, then below again.
    temps = [5.0] * 8 + [9.0] + [5.0] * 30
    table = make_capped_target(build_features(_journey(temps)), horizon=30)
    by_minute = table.assign(
        minute=(table["timestamp"] - table["timestamp"].iloc[0]).dt.total_seconds() / 60
    ).set_index("minute")
    assert by_minute.loc[0, "target_min"] == 30 and not by_minute.loc[0, "crosses_within_horizon"]
    assert by_minute.loc[10, "target_min"] == 30 and by_minute.loc[10, "crosses_within_horizon"]
    assert by_minute.loc[35, "target_min"] == 5
    assert 40 not in by_minute.index  # already above: no target
    last_minute = (len(temps) - 1) * 5
    assert by_minute.index.max() == last_minute - 30  # later rows lack 30 min of follow-up


def test_targets_are_within_horizon_and_never_above_threshold_rows(table):
    assert table["target_min"].between(0, H, inclusive="right").all()
    assert (table["label_status"] != STATUS_ABOVE).all()
    assert (table["horizon_min"] == H).all()


def test_crossing_targets_equal_the_first_future_crossing(table):
    crossing = table[table["crosses_within_horizon"]]
    assert len(crossing) > 0
    assert (crossing["target_min"] == crossing["minutes_to_threshold"]).all()
    assert (crossing["minutes_to_threshold"] <= H).all()


def test_horizon_value_is_only_used_when_no_crossing_is_proven(table):
    capped = table[~table["crosses_within_horizon"]]
    assert (capped["target_min"] == H).all()
    proven_by_later_crossing = capped["minutes_to_threshold"] > H
    proven_by_follow_up = capped["minutes_to_threshold"].isna() & (capped["minutes_to_window_end"] >= H)
    assert (proven_by_later_crossing | proven_by_follow_up).all()


def test_rows_with_insufficient_follow_up_are_dropped(features, table):
    labelled = make_capped_target(features, horizon=10_000)  # longer than any journey
    assert labelled["crosses_within_horizon"].all()  # only crossings can be proven
    kept = set(zip(table["journey_id"], table["timestamp"]))
    unknown = features[
        features["temperature_c"].le(8.0)
        & (features.groupby("journey_id")["elapsed_min"].transform("max") - features["elapsed_min"] < H)
    ]
    for _, row in unknown.iterrows():
        future = features[(features["journey_id"] == row["journey_id"])
                          & (features["elapsed_min"] > row["elapsed_min"])]
        if not (future["temperature_c"] > 8.0).any():
            assert (row["journey_id"], row["timestamp"]) not in kept


def test_target_keeps_feature_columns(table):
    assert set(FEATURE_COLUMNS) <= set(table.columns)


def test_target_threshold_must_match_feature_threshold(readings):
    f = build_features(readings, threshold_c=8.0)
    with pytest.raises(ValueError, match="threshold"):
        make_capped_target(f, threshold_c=10.0)
    assert (make_capped_target(build_features(readings, threshold_c=10.0))["threshold_c"] == 10.0).all()


def test_invalid_horizon_is_rejected(features):
    with pytest.raises(ValueError, match="horizon"):
        make_capped_target(features, horizon=0)


# --- Reproducibility --------------------------------------------------------


def test_pipeline_is_reproducible(readings, table):
    again = make_capped_target(build_features(readings))
    pd.testing.assert_frame_equal(again, table)
    tr1, te1 = split_by_journey(table, seed=3)
    tr2, te2 = split_by_journey(again, seed=3)
    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(te1, te2)


def test_different_split_seed_gives_different_test_journeys(table):
    _, te1 = split_by_journey(table, seed=1)
    _, te2 = split_by_journey(table, seed=2)
    assert set(te1["journey_id"]) != set(te2["journey_id"])


# --- Journey-level split ----------------------------------------------------


def test_split_has_zero_journey_overlap_and_loses_no_rows(table):
    train, test = split_by_journey(table)
    assert not set(train["journey_id"]) & set(test["journey_id"])
    assert len(train) + len(test) == len(table)
    assert set(train["journey_id"]) | set(test["journey_id"]) == set(table["journey_id"])


def test_split_is_by_journey_count_and_balanced_by_scenario():
    big = make_capped_target(build_features(generate_dataset(SimulationConfig(n_journeys=200))[0]))
    train, test = split_by_journey(big, test_size=0.2)
    assert test["journey_id"].nunique() == 40
    test_mix = test.drop_duplicates("journey_id")["scenario"].value_counts(normalize=True)
    all_mix = big.drop_duplicates("journey_id")["scenario"].value_counts(normalize=True)
    assert (test_mix - all_mix).abs().max() < 0.05


def test_split_works_without_scenario_column(table):
    train, test = split_by_journey(table.drop(columns="scenario"))
    assert not set(train["journey_id"]) & set(test["journey_id"])


def test_split_rejects_journey_with_two_scenarios(table):
    broken = table.copy()
    first = broken["journey_id"] == broken["journey_id"].iloc[0]
    broken.loc[first[first].index[0], "scenario"] = "something_else"
    with pytest.raises(ValueError, match="single"):
        split_by_journey(broken)

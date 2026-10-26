"""Tests for the synthetic journey generator and threshold labels."""

import numpy as np
import pandas as pd
import pytest

from frostiq.simulate import (
    SCENARIOS,
    STATUS_ABOVE,
    STATUS_CENSORED,
    STATUS_WILL_CROSS,
    SimulationConfig,
    add_threshold_labels,
    generate_dataset,
    save_dataset,
)

SMALL = SimulationConfig(n_journeys=24, seed=7)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(SMALL, threshold_c=8.0)


def _hand_made(temps, interval_min=5, journey_id="J0001"):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "journey_id": journey_id,
            "timestamp": start + pd.to_timedelta(np.arange(len(temps)) * interval_min, unit="min"),
            "temperature_c": temps,
        }
    )


# --- Reproducibility -------------------------------------------------------


def test_same_seed_gives_identical_data():
    r1, j1 = generate_dataset(SMALL)
    r2, j2 = generate_dataset(SMALL)
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(j1, j2)


def test_different_seed_gives_different_data():
    r1, _ = generate_dataset(SMALL)
    r2, _ = generate_dataset(SimulationConfig(n_journeys=24, seed=8))
    assert not r1["temperature_c"].equals(r2["temperature_c"])


def test_saved_csv_matches_generated_data(dataset, tmp_path):
    readings, journeys = dataset
    paths = save_dataset(readings, journeys, SMALL, 8.0, tmp_path)
    loaded = pd.read_csv(paths["readings"])
    assert len(loaded) == len(readings)
    np.testing.assert_allclose(loaded["temperature_c"], readings["temperature_c"])
    assert paths["config"].exists()


# --- Valid temperatures and timestamps -------------------------------------


def test_temperatures_are_finite_and_physically_plausible(dataset):
    readings, journeys = dataset
    merged = readings.merge(journeys[["journey_id", "setpoint_c", "ambient_c"]], on="journey_id")
    assert np.isfinite(merged["temperature_c"]).all()
    # Never colder than the cooling setpoint or hotter than outside air, allowing for noise.
    assert (merged["temperature_c"] >= merged["setpoint_c"] - 1.0).all()
    assert (merged["temperature_c"] <= merged["ambient_c"] + 1.0).all()


def test_timestamps_are_regular_and_increasing_within_each_journey(dataset):
    readings, _ = dataset
    assert readings["timestamp"].notna().all()
    for _, group in readings.groupby("journey_id"):
        steps = group["timestamp"].diff().dropna()
        assert (steps == pd.Timedelta(minutes=SMALL.sampling_interval_min)).all()


def test_journey_metadata_is_consistent(dataset):
    readings, journeys = dataset
    assert journeys["journey_id"].is_unique
    assert set(readings["journey_id"]) == set(journeys["journey_id"])
    assert set(journeys["scenario"]) <= set(SCENARIOS)


def test_all_scenarios_appear_in_default_dataset():
    _, journeys = generate_dataset(SimulationConfig(n_journeys=60))
    assert set(journeys["scenario"]) == set(SCENARIOS)


def test_scenarios_behave_as_designed():
    _, journeys = generate_dataset(SimulationConfig(n_journeys=120))
    crossed = journeys.groupby("scenario")["crosses_threshold"].mean()
    assert crossed["normal"] == 0.0
    assert crossed["fast_warming"] > 0.9


# --- Threshold-crossing labels ---------------------------------------------


def test_labels_on_hand_made_journey():
    labelled = add_threshold_labels(_hand_made([5.0, 6.0, 9.0, 7.0, 5.0]), threshold_c=8.0)
    assert list(labelled["label_status"]) == [
        STATUS_WILL_CROSS,
        STATUS_WILL_CROSS,
        STATUS_ABOVE,
        STATUS_CENSORED,
        STATUS_CENSORED,
    ]
    assert labelled["minutes_to_threshold"].iloc[0] == 10.0
    assert labelled["minutes_to_threshold"].iloc[1] == 5.0
    assert labelled["minutes_to_threshold"].iloc[2:].isna().all()


def test_reading_equal_to_threshold_is_not_a_crossing():
    labelled = add_threshold_labels(_hand_made([7.0, 8.0, 8.0]), threshold_c=8.0)
    assert (labelled["label_status"] == STATUS_CENSORED).all()


def test_threshold_is_configurable():
    df = _hand_made([5.0, 9.0, 11.0])
    at_8 = add_threshold_labels(df, threshold_c=8.0)
    at_10 = add_threshold_labels(df, threshold_c=10.0)
    assert at_8["minutes_to_threshold"].iloc[0] == 5.0
    assert at_10["minutes_to_threshold"].iloc[0] == 10.0
    assert (at_10["threshold_c"] == 10.0).all()


def test_labels_never_look_into_another_journey():
    a = _hand_made([5.0, 6.0], journey_id="J0001")
    b = _hand_made([9.0, 9.0], journey_id="J0002")
    labelled = add_threshold_labels(pd.concat([a, b], ignore_index=True), threshold_c=8.0)
    first = labelled[labelled["journey_id"] == "J0001"]
    assert (first["label_status"] == STATUS_CENSORED).all()


def test_will_cross_labels_point_at_first_reading_above_threshold(dataset):
    readings, _ = dataset
    rows = readings[readings["label_status"] == STATUS_WILL_CROSS].sample(200, random_state=0)
    for _, row in rows.iterrows():
        journey = readings[readings["journey_id"] == row["journey_id"]]
        later = journey[journey["elapsed_min"] > row["elapsed_min"]]
        target = row["elapsed_min"] + row["minutes_to_threshold"]
        at_target = later[later["elapsed_min"] == target]
        before_target = later[later["elapsed_min"] < target]
        assert at_target["temperature_c"].iloc[0] > 8.0
        assert (before_target["temperature_c"] <= 8.0).all()


def test_minutes_to_threshold_is_positive_and_within_window(dataset):
    readings, _ = dataset
    crossing = readings[readings["label_status"] == STATUS_WILL_CROSS]
    assert (crossing["minutes_to_threshold"] > 0).all()
    assert (crossing["minutes_to_threshold"] <= crossing["minutes_to_window_end"]).all()


# --- Timestamps loaded from CSV and input validation ------------------------


def _text_journey(timestamps, temps, journey_id="J0001"):
    return pd.DataFrame({"journey_id": journey_id, "timestamp": timestamps, "temperature_c": temps})


LABEL_COLUMNS = ["journey_id", "label_status", "minutes_to_threshold", "minutes_to_window_end"]


def test_csv_loaded_timestamps_give_identical_labels(dataset, tmp_path):
    readings, journeys = dataset
    save_dataset(readings, journeys, SMALL, 8.0, tmp_path)
    loaded = pd.read_csv(tmp_path / "synthetic_readings.csv")
    assert not pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])  # really text

    relabelled = add_threshold_labels(loaded[["journey_id", "timestamp", "temperature_c"]], 8.0)
    pd.testing.assert_frame_equal(relabelled[LABEL_COLUMNS], readings[LABEL_COLUMNS])
    assert str(relabelled["timestamp"].dt.tz) == "UTC"


def test_same_instants_in_different_utc_offsets_give_same_labels():
    utc = add_threshold_labels(
        _text_journey(["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z", "2026-01-01T00:20:00Z"],
                      [5.0, 6.0, 9.0]))
    india = add_threshold_labels(
        _text_journey(["2026-01-01T05:30:00+05:30", "2026-01-01T05:40:00+05:30", "2026-01-01T05:50:00+05:30"],
                      [5.0, 6.0, 9.0]))
    pd.testing.assert_frame_equal(utc[LABEL_COLUMNS], india[LABEL_COLUMNS])
    assert list(utc["minutes_to_threshold"].iloc[:2]) == [20.0, 10.0]


def test_offset_change_within_journey_is_ordered_by_real_time():
    # Clocks jump forward 1 hour (as on a daylight-saving change), but only 5 minutes pass.
    # As text, "02:00+01:00" sorts after "01:30+00:00", though it is the earlier instant.
    labelled = add_threshold_labels(
        _text_journey(["2026-03-29T01:30:00+00:00", "2026-03-29T00:55:00+00:00", "2026-03-29T02:00:00+01:00"],
                      [9.0, 5.0, 6.0]))
    assert list(labelled["temperature_c"]) == [5.0, 6.0, 9.0]
    assert list(labelled["minutes_to_threshold"].iloc[:2]) == [35.0, 30.0]


def test_naive_text_timestamps_are_accepted():
    labelled = add_threshold_labels(
        _text_journey(["2026-01-01 00:00:00", "2026-01-01 00:05:00"], [5.0, 9.0]))
    assert labelled["minutes_to_threshold"].iloc[0] == 5.0
    assert labelled["timestamp"].dt.tz is None


def test_unsorted_input_is_sorted_before_labelling():
    labelled = add_threshold_labels(
        _text_journey(["2026-01-01 00:10:00", "2026-01-01 00:00:00", "2026-01-01 00:05:00"],
                      [9.0, 5.0, 6.0]))
    assert list(labelled["minutes_to_threshold"].iloc[:2]) == [10.0, 5.0]


@pytest.mark.parametrize(
    "timestamps, message",
    [
        (["2026-01-01 00:00:00", "not a date"], "not an ISO 8601"),
        (["2026-02-30 00:00:00", "2026-03-01 00:00:00"], "not an ISO 8601"),
        (["2026-01-01 00:00:00", None], "missing value"),
        (["2026-01-01 00:00:00", ""], "missing value"),
        (["2026-01-01 00:00:00", "2026-01-01 00:05:00+00:00"], "with and without a time zone"),
        (["2026-01-01 00:00:00", "2026-01-01 00:00:00"], "repeat a timestamp"),
    ],
)
def test_invalid_timestamps_are_rejected(timestamps, message):
    with pytest.raises(ValueError, match=message):
        add_threshold_labels(_text_journey(timestamps, [5.0, 6.0]))


def test_missing_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature_c"):
        add_threshold_labels(_text_journey(["2026-01-01 00:00", "2026-01-01 00:05"], [5.0, None]))


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="journey_id"):
        add_threshold_labels(pd.DataFrame({"timestamp": ["2026-01-01"], "temperature_c": [5.0]}))


def test_same_timestamp_in_different_journeys_is_allowed():
    df = pd.concat([_text_journey(["2026-01-01 00:00"], [5.0], "J0001"),
                    _text_journey(["2026-01-01 00:00"], [9.0], "J0002")], ignore_index=True)
    assert len(add_threshold_labels(df)) == 2


# --- Censored observations -------------------------------------------------


def test_censored_rows_have_no_invented_remaining_time(dataset):
    readings, _ = dataset
    censored = readings[readings["label_status"] == STATUS_CENSORED]
    assert len(censored) > 0
    assert censored["minutes_to_threshold"].isna().all()


def test_censored_rows_really_have_no_later_crossing(dataset):
    readings, _ = dataset
    for journey_id, journey in readings.groupby("journey_id"):
        censored = journey[journey["label_status"] == STATUS_CENSORED]
        if censored.empty:
            continue
        after_first_censored = journey[journey["elapsed_min"] >= censored["elapsed_min"].min()]
        assert (after_first_censored["temperature_c"] <= 8.0).all(), journey_id


def test_never_crossing_journey_is_fully_censored(dataset):
    readings, journeys = dataset
    never = journeys.loc[~journeys["crosses_threshold"], "journey_id"]
    assert len(never) > 0
    rows = readings[readings["journey_id"].isin(never)]
    assert (rows["label_status"] == STATUS_CENSORED).all()


def test_censored_window_end_is_a_lower_bound(dataset):
    readings, _ = dataset
    last = readings.groupby("journey_id")["elapsed_min"].transform("max")
    np.testing.assert_allclose(readings["minutes_to_window_end"], last - readings["elapsed_min"])

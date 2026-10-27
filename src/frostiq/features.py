"""Feature engineering, capped time-to-threshold targets and journey-level splits.

Pipeline (order matters)::

    features = build_features(readings)          # ALL readings, so history is complete
    table = make_capped_target(features)         # then keep rows with a known target
    train, test = split_by_journey(table)        # whole journeys go to one side only

Every feature for a reading uses only that reading and earlier readings from the
same journey, so a live system could compute it at that moment. Feature
definitions, target rules and limitations are documented in ``docs/FEATURES.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from frostiq.config import DEFAULT_SEED, DEFAULT_THRESHOLD_C
from frostiq.simulate import (
    STATUS_ABOVE,
    STATUS_CENSORED,
    STATUS_WILL_CROSS,
    add_threshold_labels,
    validate_readings,
)

DEFAULT_HORIZON_MIN = 120
DELTA_WINDOWS_MIN = (15, 30, 60)
SLOPE_WINDOWS_MIN = (30, 60)
ROLLING_WINDOW_MIN = 60
# linear_eta_min value meaning "not warming, no crossing expected at this rate".
ETA_CAP_MIN = 24 * 60

FEATURE_COLUMNS = (
    "temperature_c",
    "margin_c",
    *(f"delta_{n}" for n in DELTA_WINDOWS_MIN),
    *(f"slope_{n}" for n in SLOPE_WINDOWS_MIN),
    "accel_30",
    f"mean_{ROLLING_WINDOW_MIN}",
    f"std_{ROLLING_WINDOW_MIN}",
    f"min_{ROLLING_WINDOW_MIN}",
    f"max_{ROLLING_WINDOW_MIN}",
    "n_excursions_so_far",
    "minutes_since_last_above",
    "linear_eta_min",
)

# Columns that must never be model inputs: identifiers, hidden simulator facts,
# labels, future information, or (for now) excluded by decision.
NEVER_FEATURES = (
    "journey_id",
    "scenario",
    "timestamp",
    "elapsed_min",
    "threshold_c",
    "label_status",
    "minutes_to_threshold",
    "minutes_to_window_end",
    "target_min",
    "crosses_within_horizon",
    "horizon_min",
)


def _value_at_or_before(t, values, query_t):
    """For each query time, the value at the last reading at or before it (NaN if none)."""
    idx = np.searchsorted(t, query_t, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)


def _journey_features(timestamps: pd.Series, temps: np.ndarray, threshold_c: float) -> dict:
    """Features for one journey whose readings are already sorted by time."""
    t = ((timestamps - timestamps.iloc[0]).dt.total_seconds() / 60).to_numpy()
    index = pd.DatetimeIndex(timestamps)
    f = {"temperature_c": temps, "margin_c": threshold_c - temps}

    # A window feature is only defined once the journey has that much history.
    for n in DELTA_WINDOWS_MIN:
        f[f"delta_{n}"] = temps - _value_at_or_before(t, temps, t - n)

    # Least-squares slope over [t - n, t], from rolling sums (works for irregular sampling).
    sums_input = pd.DataFrame({"n": 1.0, "x": t, "y": temps, "xy": t * temps, "xx": t * t}, index=index)
    for n in SLOPE_WINDOWS_MIN:
        s = sums_input.rolling(f"{n}min", closed="both").sum()
        numerator = (s["n"] * s["xy"] - s["x"] * s["y"]).to_numpy()
        denominator = (s["n"] * s["xx"] - s["x"] ** 2).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(denominator > 0, numerator / denominator, np.nan)
        f[f"slope_{n}"] = np.where(t >= n, slope, np.nan)

    f["accel_30"] = f["slope_30"] - _value_at_or_before(t, f["slope_30"], t - 30)

    w = ROLLING_WINDOW_MIN
    window = pd.Series(temps, index=index).rolling(f"{w}min", closed="both")
    has_window = t >= w
    for name, values in (("mean", window.mean()), ("std", window.std()),
                         ("min", window.min()), ("max", window.max())):
        f[f"{name}_{w}"] = np.where(has_window, values.to_numpy(), np.nan)

    above = temps > threshold_c
    starts = above & ~np.concatenate(([False], above[:-1]))
    f["n_excursions_so_far"] = np.cumsum(starts).astype(float)
    last_above_t = pd.Series(np.where(above, t, np.nan)).ffill().to_numpy()
    f["minutes_since_last_above"] = t - last_above_t

    slope_30 = f["slope_30"]
    margin = f["margin_c"]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(slope_30 > 0, np.minimum(margin / slope_30, ETA_CAP_MIN), ETA_CAP_MIN)
    # Only a reading strictly above the threshold is a crossing (margin < 0).
    eta = np.where(margin < 0, 0.0, eta)
    f["linear_eta_min"] = np.where(np.isnan(slope_30), np.nan, eta)
    return f


def build_features(readings: pd.DataFrame, threshold_c: float = DEFAULT_THRESHOLD_C) -> pd.DataFrame:
    """Add past-only features (``FEATURE_COLUMNS``) to every reading.

    Needs ``journey_id``, ``timestamp`` and ``temperature_c`` (validated the same
    way as ``add_threshold_labels``). Rows are sorted by journey and time; other
    columns are kept. Pass *all* readings, including above-threshold and censored
    ones, so every row has its full history. A window feature is NaN until the
    journey has that many minutes of history.
    """
    out = validate_readings(readings)
    out = out.sort_values(["journey_id", "timestamp"], kind="stable").reset_index(drop=True)
    temps = out["temperature_c"].to_numpy(dtype=float)

    columns = {name: np.full(len(out), np.nan) for name in FEATURE_COLUMNS}
    for idx in out.groupby("journey_id", sort=False).indices.values():
        for name, values in _journey_features(out["timestamp"].iloc[idx], temps[idx], threshold_c).items():
            columns[name][idx] = values

    for name in FEATURE_COLUMNS:
        out[name] = columns[name]
    out["threshold_c"] = float(threshold_c)
    return out


def make_capped_target(df: pd.DataFrame, horizon=DEFAULT_HORIZON_MIN, threshold_c=None) -> pd.DataFrame:
    """Keep rows whose capped time-to-threshold is known, and add the target.

    ``target_min`` = minutes until the first later reading above the threshold,
    capped at ``horizon``:

    * crosses within the horizon -> the exact minutes (``crosses_within_horizon`` True)
    * a later crossing beyond the horizon, or censored but observed for at least
      ``horizon`` more minutes -> ``horizon`` (proven: no crossing within it)
    * censored with less than ``horizon`` minutes of follow-up -> dropped (unknown)
    * already above the threshold -> dropped (no time remaining to predict)

    ``threshold_c`` defaults to the threshold ``build_features`` used; passing a
    different one raises, so features and targets always share one threshold.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    used = df["threshold_c"].unique() if "threshold_c" in df.columns else []
    if len(used) > 1:
        raise ValueError(f"rows were built with several thresholds: {sorted(used)}")
    if threshold_c is None:
        threshold_c = float(used[0]) if len(used) else DEFAULT_THRESHOLD_C
    elif len(used) and float(used[0]) != float(threshold_c):
        raise ValueError(f"features were built for threshold {used[0]}, not {threshold_c}")

    labelled = add_threshold_labels(df, threshold_c)
    status = labelled["label_status"]
    to_threshold = labelled["minutes_to_threshold"]
    follow_up = labelled["minutes_to_window_end"]

    crosses = (status == STATUS_WILL_CROSS) & (to_threshold <= horizon)
    proven_beyond = ((status == STATUS_WILL_CROSS) & (to_threshold > horizon)) | (
        (status == STATUS_CENSORED) & (follow_up >= horizon)
    )
    keep = (crosses | proven_beyond) & (status != STATUS_ABOVE)

    out = labelled[keep].copy()
    out["target_min"] = np.where(crosses[keep], to_threshold[keep], float(horizon))
    out["crosses_within_horizon"] = crosses[keep].to_numpy()
    out["horizon_min"] = float(horizon)
    return out.reset_index(drop=True)


def split_by_journey(df: pd.DataFrame, test_size=0.2, seed=DEFAULT_SEED, stratify_col="scenario"):
    """Split rows into (train, test) so that each journey lands on one side only.

    Journeys, not rows, are split. If ``stratify_col`` is present (by default the
    simulator's ``scenario``), each side gets a similar mix of its values. It is
    used only for balancing and is never a model feature (see ``NEVER_FEATURES``).
    """
    if stratify_col is not None and stratify_col in df.columns:
        per_journey = df.groupby("journey_id")[stratify_col]
        if (per_journey.nunique() > 1).any():
            raise ValueError(f"each journey must have a single {stratify_col!r} value")
        journeys = per_journey.first()
        strata = journeys.to_numpy()
    else:
        journeys = pd.Series(index=pd.Index(df["journey_id"].unique()), dtype=object)
        strata = None

    train_ids, test_ids = train_test_split(
        journeys.index.to_numpy(), test_size=test_size, random_state=seed, stratify=strata
    )
    train = df[df["journey_id"].isin(train_ids)].reset_index(drop=True)
    test = df[df["journey_id"].isin(test_ids)].reset_index(drop=True)
    return train, test

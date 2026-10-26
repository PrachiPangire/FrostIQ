"""Synthetic cold-chain journey generator for FrostIQ.

Each *journey* is one insulated, refrigerated container observed for a few hours.
Its temperature is simulated minute by minute with a simple heat-balance model,
then sampled like a data logger would (every ``sampling_interval_min`` minutes).

Heat-balance model (Newton's law of cooling, applied to two "pulls"):

    dT/dt = -k_cool * e(t) * (T - setpoint)          # refrigeration pulls T down
            - (k_leak(t) + k_door(t)) * (T - ambient) # outside air pulls T up

* ``e(t)`` is cooling effectiveness (1 = working, 0 = failed).
* ``k_leak`` is heat leaking through the insulation.
* ``k_door`` is extra heat exchange while a door is open.

Over each one-minute step the rates are constant, so the equation is solved
exactly rather than approximated. Small random noise is added to the true
temperature (process noise) and to each logged reading (sensor noise).

Scenarios
---------
normal          Cooling works all journey. Loaded slightly warm, then pulled down.
door_openings   Cooling works, but random door openings cause short warm spikes.
gradual_warming Cooling effectiveness declines linearly after a random onset
                (e.g. a degrading compressor), down to a small residual level.
fast_warming    Cooling stops completely at a random onset and heat leak rises
                (e.g. power cut plus a damaged seal), so temperature rises fast.

Labels
------
Labels are derived only from *future readings in the same journey* (see
``add_threshold_labels``). A reading whose journey never goes above the threshold
again within the observation window is **censored**: its remaining time is left
missing, never guessed.

All assumptions and limitations are documented in ``docs/SYNTHETIC_DATA.md``.
This data is for software development only. It does not establish real-world
medical safety or predictive accuracy.

Usage::

    python -m frostiq.simulate --n-journeys 120 --seed 42 --threshold 8.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from frostiq.config import DEFAULT_SEED, DEFAULT_THRESHOLD_C, RAW_DATA_DIR

SCENARIOS = ("normal", "door_openings", "gradual_warming", "fast_warming")

STATUS_WILL_CROSS = "will_cross"
STATUS_CENSORED = "censored"
STATUS_ABOVE = "above_threshold"

READINGS_FILE = "synthetic_readings.csv"
JOURNEYS_FILE = "synthetic_journeys.csv"
CONFIG_FILE = "synthetic_config.json"


@dataclass(frozen=True)
class SimulationConfig:
    """All simulation assumptions in one place. Ranges are (low, high), sampled
    uniformly per journey. Rates are per minute."""

    n_journeys: int = 120
    seed: int = DEFAULT_SEED
    sampling_interval_min: int = 5
    journey_hours: tuple[float, float] = (8.0, 16.0)
    scenario_weights: dict[str, float] = field(
        default_factory=lambda: {
            "normal": 0.30,
            "door_openings": 0.30,
            "gradual_warming": 0.20,
            "fast_warming": 0.20,
        }
    )
    start_date: str = "2026-01-01"
    start_spread_days: int = 30

    # Container and environment
    ambient_c: tuple[float, float] = (20.0, 35.0)
    setpoint_c: tuple[float, float] = (3.0, 5.0)
    initial_c: tuple[float, float] = (5.0, 7.5)
    cooling_rate: tuple[float, float] = (0.015, 0.025)
    leak_rate: tuple[float, float] = (0.0006, 0.0010)

    # Noise (standard deviations, °C)
    process_noise_c: float = 0.02
    sensor_noise_c: float = 0.10

    # door_openings
    door_events_per_hour: tuple[float, float] = (0.25, 0.50)
    door_open_minutes: tuple[int, int] = (2, 10)
    door_exchange_rate: tuple[float, float] = (0.005, 0.025)

    # gradual_warming and fast_warming
    failure_onset_hours: tuple[float, float] = (1.0, 6.0)
    degradation_hours: tuple[float, float] = (2.0, 8.0)
    residual_cooling: tuple[float, float] = (0.0, 0.15)
    fast_leak_multiplier: tuple[float, float] = (3.0, 6.0)

    def __post_init__(self):
        if self.n_journeys < 1:
            raise ValueError("n_journeys must be at least 1")
        if self.sampling_interval_min < 1:
            raise ValueError("sampling_interval_min must be at least 1")
        if set(self.scenario_weights) - set(SCENARIOS):
            raise ValueError(f"scenario_weights keys must be in {SCENARIOS}")


def _simulate_journey(rng, cfg, journey_id, scenario, start_time):
    """Simulate one journey. Returns (readings DataFrame, journey metadata dict)."""
    dt = cfg.sampling_interval_min
    n_minutes = int(round(rng.uniform(*cfg.journey_hours) * 60 / dt)) * dt
    minutes = np.arange(n_minutes + 1)

    ambient = rng.uniform(*cfg.ambient_c)
    setpoint = rng.uniform(*cfg.setpoint_c)
    initial = rng.uniform(*cfg.initial_c)
    k_cool = rng.uniform(*cfg.cooling_rate)
    k_leak = rng.uniform(*cfg.leak_rate)

    effectiveness = np.ones(n_minutes + 1)
    leak = np.full(n_minutes + 1, k_leak)
    door = np.zeros(n_minutes + 1)
    failure_onset = np.nan
    n_door_events = 0

    if scenario == "gradual_warming":
        failure_onset = rng.uniform(*cfg.failure_onset_hours) * 60
        duration = rng.uniform(*cfg.degradation_hours) * 60
        residual = rng.uniform(*cfg.residual_cooling)
        progress = np.clip((minutes - failure_onset) / duration, 0.0, 1.0)
        effectiveness = 1.0 - progress * (1.0 - residual)
    elif scenario == "fast_warming":
        failure_onset = rng.uniform(*cfg.failure_onset_hours) * 60
        failed = minutes >= failure_onset
        effectiveness[failed] = 0.0
        leak[failed] *= rng.uniform(*cfg.fast_leak_multiplier)
    elif scenario == "door_openings":
        rate = rng.uniform(*cfg.door_events_per_hour)
        n_door_events = int(rng.poisson(rate * n_minutes / 60))
        lo, hi = cfg.door_open_minutes
        for start in rng.integers(0, n_minutes, size=n_door_events):
            length = int(rng.integers(lo, hi + 1))
            k_door = rng.uniform(*cfg.door_exchange_rate)
            door[start : start + length] = np.maximum(door[start : start + length], k_door)

    # Exact solution of the linear ODE over each 1-minute step.
    k_c = k_cool * effectiveness
    k_warm = leak + door
    k_total = k_c + k_warm
    equilibrium = (k_c * setpoint + k_warm * ambient) / k_total
    decay = np.exp(-k_total)
    process_noise = rng.normal(0.0, cfg.process_noise_c, size=n_minutes)

    true_temp = np.empty(n_minutes + 1)
    true_temp[0] = initial
    for m in range(n_minutes):
        true_temp[m + 1] = (
            equilibrium[m] + (true_temp[m] - equilibrium[m]) * decay[m] + process_noise[m]
        )

    sample_min = minutes[::dt]
    measured = true_temp[::dt] + rng.normal(0.0, cfg.sensor_noise_c, size=len(sample_min))

    readings = pd.DataFrame(
        {
            "journey_id": journey_id,
            "scenario": scenario,
            "timestamp": start_time + pd.to_timedelta(sample_min, unit="min"),
            "elapsed_min": sample_min.astype(float),
            "temperature_c": np.round(measured, 2),
        }
    )
    journey = {
        "journey_id": journey_id,
        "scenario": scenario,
        "start_time": start_time,
        "duration_min": float(n_minutes),
        "ambient_c": round(ambient, 2),
        "setpoint_c": round(setpoint, 2),
        "initial_c": round(initial, 2),
        "cooling_rate_per_min": k_cool,
        "leak_rate_per_min": k_leak,
        "failure_onset_min": failure_onset,
        "n_door_events": n_door_events,
    }
    return readings, journey


def simulate_journeys(cfg: SimulationConfig | None = None):
    """Generate unlabelled readings and per-journey metadata.

    Returns (readings, journeys). The same config (including seed) always gives
    identical output.
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    names = list(cfg.scenario_weights)
    weights = np.array([cfg.scenario_weights[n] for n in names], dtype=float)
    scenarios = rng.choice(names, size=cfg.n_journeys, p=weights / weights.sum())

    base = pd.Timestamp(cfg.start_date, tz="UTC")
    slots = cfg.start_spread_days * 24 * 60 // cfg.sampling_interval_min
    offsets = rng.integers(0, slots, size=cfg.n_journeys) * cfg.sampling_interval_min

    all_readings, all_journeys = [], []
    for i, (scenario, offset) in enumerate(zip(scenarios, offsets), start=1):
        start_time = base + pd.Timedelta(minutes=int(offset))
        readings, journey = _simulate_journey(rng, cfg, f"J{i:04d}", str(scenario), start_time)
        all_readings.append(readings)
        all_journeys.append(journey)

    return pd.concat(all_readings, ignore_index=True), pd.DataFrame(all_journeys)


LABEL_INPUT_COLUMNS = ("journey_id", "timestamp", "temperature_c")

# A time followed by "Z" or a UTC offset such as +05:30 / -0400.
_OFFSET_PATTERN = r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[Zz]|[+-]\d{2}:?\d{2})$"


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Convert a timestamp column to pandas datetimes, rejecting ambiguous input.

    * Already-datetime columns are returned unchanged.
    * Text must be ISO 8601 (e.g. ``2026-01-05 04:55:00+00:00``).
    * If every value carries a time zone ("Z" or an offset), all are converted to
      UTC, so readings with different offsets are still ordered correctly.
    * If no value carries a time zone, they stay time-zone-naive. Only time
      *differences* are used, so this is safe as long as one clock was used.
    * A mix of the two is rejected: a value without a zone could mean any zone.
    * Missing or unparseable values are rejected.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = values
    else:
        text = values.astype("string").str.strip()
        has_zone = text.str.contains(_OFFSET_PATTERN, regex=True).fillna(False)
        if has_zone.any() and not has_zone[text.notna()].all():
            raise ValueError(
                "timestamp column mixes values with and without a time zone; "
                "make them consistent before labelling"
            )
        try:
            parsed = pd.to_datetime(text, format="ISO8601", utc=bool(has_zone.any()))
        except (ValueError, TypeError) as err:
            raise ValueError(f"timestamp column has a value that is not an ISO 8601 date-time: {err}") from err

    missing = parsed.isna()
    if missing.any():
        rows = list(values.index[missing][:5])
        raise ValueError(f"timestamp column has {int(missing.sum())} missing value(s), e.g. at rows {rows}")
    return parsed


def _validate_label_input(readings: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and values; return a copy with parsed timestamps."""
    absent = [c for c in LABEL_INPUT_COLUMNS if c not in readings.columns]
    if absent:
        raise ValueError(f"readings are missing required column(s): {absent}")

    out = readings.copy()
    out["timestamp"] = _parse_timestamps(out["timestamp"])

    if out["journey_id"].isna().any():
        raise ValueError("journey_id column has missing values")
    temps = pd.to_numeric(out["temperature_c"], errors="coerce")
    if temps.isna().any():
        raise ValueError(
            f"temperature_c has {int(temps.isna().sum())} missing or non-numeric value(s); "
            "a gap could hide a threshold crossing, so clean or split the journey first"
        )
    out["temperature_c"] = temps
    duplicated = out.duplicated(["journey_id", "timestamp"])
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} reading(s) repeat a timestamp within the same journey"
        )
    return out


def add_threshold_labels(readings: pd.DataFrame, threshold_c: float = DEFAULT_THRESHOLD_C):
    """Label each reading with the minutes until its journey next goes above
    ``threshold_c``, using only later readings from the same journey.

    Needs only ``journey_id``, ``timestamp`` and ``temperature_c``, so it also
    works on data loaded from CSV (see ``_parse_timestamps`` for timestamp rules).
    Rows are sorted by journey and time; irregular intervals are fine. Raises
    ``ValueError`` for missing columns, missing/invalid/ambiguous timestamps,
    missing temperatures, or repeated timestamps within a journey.

    "Above" means strictly greater than the threshold (8.0 °C is still in range
    for an 8 °C threshold). Adds these columns:

    * ``threshold_c``: the threshold used.
    * ``label_status``:
        - ``will_cross``: a later reading in this journey is above the threshold.
        - ``censored``: no later reading is above the threshold before the
          observation window ends. The true remaining time is unknown.
        - ``above_threshold``: this reading is already above the threshold.
    * ``minutes_to_threshold``: minutes until the first later reading above the
      threshold. Filled only for ``will_cross``, otherwise missing (NaN).
    * ``minutes_to_window_end``: minutes until the journey's last reading. For
      censored rows this is a lower bound: the crossing, if it ever happens,
      happens later than this.

    Precision is limited by the sampling interval: the true crossing happened
    somewhere between the last reading at or below the threshold and the first
    reading above it.
    """
    out = _validate_label_input(readings)
    out = out.sort_values(["journey_id", "timestamp"], kind="stable").reset_index(drop=True)
    temps = out["temperature_c"].to_numpy(dtype=float)
    status = np.empty(len(out), dtype=object)
    to_threshold = np.full(len(out), np.nan)
    to_window_end = np.empty(len(out))

    for idx in out.groupby("journey_id", sort=False).indices.values():
        ts = out["timestamp"].iloc[idx]
        t = ((ts - ts.iloc[0]).dt.total_seconds() / 60).to_numpy()
        to_window_end[idx] = t[-1] - t

        next_above = np.nan
        for j in range(len(idx) - 1, -1, -1):
            row = idx[j]
            if temps[row] > threshold_c:
                status[row] = STATUS_ABOVE
                next_above = t[j]
            elif np.isnan(next_above):
                status[row] = STATUS_CENSORED
            else:
                status[row] = STATUS_WILL_CROSS
                to_threshold[row] = next_above - t[j]

    out["threshold_c"] = float(threshold_c)
    out["label_status"] = status
    out["minutes_to_threshold"] = to_threshold
    out["minutes_to_window_end"] = to_window_end
    return out


def summarize_crossings(readings: pd.DataFrame, journeys: pd.DataFrame, threshold_c: float):
    """Add ``crosses_threshold`` and ``first_crossing_min`` to the journey table."""
    above = readings[readings["temperature_c"] > threshold_c]
    first = above.groupby("journey_id")["elapsed_min"].min()
    out = journeys.copy()
    out["crosses_threshold"] = out["journey_id"].isin(first.index)
    out["first_crossing_min"] = out["journey_id"].map(first)
    return out


def generate_dataset(cfg: SimulationConfig | None = None, threshold_c: float = DEFAULT_THRESHOLD_C):
    """Simulate and label. Returns (labelled readings, journeys)."""
    readings, journeys = simulate_journeys(cfg)
    return (
        add_threshold_labels(readings, threshold_c),
        summarize_crossings(readings, journeys, threshold_c),
    )


def save_dataset(readings, journeys, cfg, threshold_c, out_dir=RAW_DATA_DIR):
    """Write readings, journey metadata and the exact config used. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "readings": out_dir / READINGS_FILE,
        "journeys": out_dir / JOURNEYS_FILE,
        "config": out_dir / CONFIG_FILE,
    }
    readings.to_csv(paths["readings"], index=False)
    journeys.to_csv(paths["journeys"], index=False)
    meta = {"threshold_c": threshold_c, **asdict(cfg)}
    paths["config"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic FrostIQ journeys.")
    parser.add_argument("--n-journeys", type=int, default=SimulationConfig.n_journeys)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_C,
                        help="temperature threshold in degC (experimental default: 8.0)")
    parser.add_argument("--interval", type=int, default=SimulationConfig.sampling_interval_min,
                        help="minutes between logged readings")
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args(argv)

    cfg = SimulationConfig(
        n_journeys=args.n_journeys, seed=args.seed, sampling_interval_min=args.interval
    )
    readings, journeys = generate_dataset(cfg, args.threshold)
    paths = save_dataset(readings, journeys, cfg, args.threshold, args.out_dir)

    print(f"Journeys: {len(journeys)}   Readings: {len(readings)}   "
          f"Threshold: {args.threshold} degC   Seed: {args.seed}")
    print("\nJourneys crossing the threshold, by scenario:")
    print(journeys.groupby("scenario")["crosses_threshold"].agg(["sum", "count"])
          .rename(columns={"sum": "crossed", "count": "journeys"}).to_string())
    print("\nReading label status:")
    print(readings["label_status"].value_counts().to_string())
    print()
    for name, path in paths.items():
        print(f"Saved {name:9} -> {path}")


if __name__ == "__main__":
    main()

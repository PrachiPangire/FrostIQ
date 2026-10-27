# FrostIQ Features, Targets and Data Split

> **Reminder:** everything here is built on synthetic data. Good results show
> only that a model can learn the patterns programmed into the simulator. They
> do **not** establish real-world medical safety or predictive accuracy. See
> [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md).

Code: [`src/frostiq/features.py`](../src/frostiq/features.py)

## Pipeline

```python
import pandas as pd
from frostiq.features import FEATURE_COLUMNS, build_features, make_capped_target, split_by_journey

readings = pd.read_csv("data/raw/synthetic_readings.csv")
features = build_features(readings)        # 1. ALL readings, so every row has its full history
table = make_capped_target(features)       # 2. keep rows whose 120-minute target is known
train, test = split_by_journey(table)      # 3. whole journeys go to train OR test

X_train, y_train = train[list(FEATURE_COLUMNS)], train["target_min"]
```

**The order matters.** If you filter rows *before* building features, some
readings lose part of their history. For example, an above-threshold reading
that was removed would vanish from the next reading's 60-minute window.

Milestone 2 uses 500 journeys (seed 42, all other settings unchanged):

```powershell
python -m frostiq.simulate --n-journeys 500
```

## The golden rule: past-only features

A feature is allowed only if a live monitoring system could calculate it **at
the moment of the reading**. It may use that reading and earlier readings from
the **same journey**, and nothing else. The tests check this in two ways:

- **Delete the future:** removing every later reading must not change any
  feature of earlier readings.
- **Change the future:** replacing later temperatures with 30 °C must not change
  any feature of earlier readings.

I also checked that these tests catch real mistakes. When I deliberately made
`delta_15` read 5 minutes ahead, or made `mean_60` use a centred window, both
tests failed, as they should.

## The threshold-crossing rule

**A reading is above the threshold, and counts as a crossing, only when
`temperature_c > threshold_c`: strictly greater than, not greater than or
equal to.**

With the default 8.0 °C threshold:

| `temperature_c` | Above the threshold? |
|---|---|
| 7.99 | No |
| **8.00** | **No.** Exactly at the threshold is still in range |
| 8.01 | Yes |

The same rule is used everywhere a crossing matters:
- the labels (`add_threshold_labels` in `simulate.py`), and therefore
  `label_status`, `minutes_to_threshold`, `target_min` and
  `crosses_within_horizon`
- the excursion features `n_excursions_so_far` and `minutes_since_last_above`

The threshold itself is configurable (`build_features(..., threshold_c=...)`).
8.0 °C is only an experimental default.

## Feature definitions (`FEATURE_COLUMNS`)

"Window [t − N, t]" means readings from N minutes ago up to and including now.

| Feature | Meaning | Missing (NaN) when |
|---|---|---|
| `temperature_c` | Current reading | never |
| `margin_c` | `threshold_c − temperature_c`: degrees of headroom left. Positive = below; **0 = exactly at the threshold, still in range**; negative = above (a crossing) | never |
| `delta_15`, `delta_30`, `delta_60` | Current reading minus the reading at or just before N minutes ago | less than N minutes of history |
| `slope_30`, `slope_60` | Warming rate in °C per minute: a straight-line (least-squares) fit through window [t − N, t] | less than N minutes of history |
| `accel_30` | `slope_30` now minus `slope_30` 30 minutes ago. Positive = warming is speeding up | less than 60 minutes of history |
| `mean_60`, `std_60`, `min_60`, `max_60` | Average, wobble (standard deviation), lowest and highest reading over window [t − 60, t] | less than 60 minutes of history |
| `n_excursions_so_far` | How many times this journey has gone above the threshold (`temperature_c > threshold_c`) so far, counting each separate excursion once | never |
| `minutes_since_last_above` | Minutes since the most recent reading with `temperature_c > threshold_c` (0 if this reading is one) | the journey hasn't been above yet (most rows) |
| `linear_eta_min` | Simple physics estimate: `margin_c / slope_30` if warming; 1440 (24 h, meaning "not warming") otherwise; **0 when `margin_c < 0`**, i.e. strictly above the threshold. See the note below | less than 30 minutes of history |

**Note on `linear_eta_min` at exactly the threshold.** This feature follows the
strict crossing rule: it's 0 only when `margin_c < 0`. A reading of exactly
8.00 °C that is warming gets `0 / slope = 0` minutes (it's right at the limit),
and one that is flat or cooling gets 1440 ("not warming"), not 0.

An earlier version used `margin_c ≤ 0`, which gave 0 to flat or cooling readings
at exactly the threshold. Fixing it changed 2 of the 72,361 rows in the
500-journey dataset (a regression test now covers it). The retrained models
shifted slightly. Recall and crossing-row MAE were unchanged, and precision
moved from 0.351 to 0.350. [EVALUATION.md](EVALUATION.md) shows the current
numbers.

**Why time-based windows?** Windows are measured in minutes, not "the last 12
rows". Real loggers sometimes skip readings, and a time-based window still
covers the right period. The slope is a proper straight-line fit, so a skipped
reading doesn't distort it.

**Why not a 5-minute change?** Sensor noise (about ±0.1 °C) is as large as a
typical 5-minute change, so the shortest window used is 15 minutes.

**Missing values.** Each window feature stays empty until the journey has that
much history. Tree models such as scikit-learn's `HistGradientBoostingRegressor`
handle empty values directly. Linear models need them filled or the rows dropped,
and that choice belongs to Milestone 3.

### Columns that are never features (`NEVER_FEATURES`)

| Column | Reason |
|---|---|
| `scenario` | A real system never knows the failure type; detecting it is the point. Used **only** to balance the train/test split and to report results per scenario |
| `elapsed_min` | Excluded by decision for now. Failures in the simulator are programmed to start 1–6 h in, so the model could learn the simulator's schedule instead of physics. To be tested later |
| `timestamp` | Start dates are random and outside temperature is constant, so the date and hour carry no real signal. Used only to calculate time differences |
| `journey_id` | An identifier, used for grouping and splitting |
| `threshold_c` | Always the same value. `margin_c` carries the useful information |
| `label_status`, `minutes_to_threshold`, `target_min`, `crosses_within_horizon` | The answers |
| `minutes_to_window_end` | Depends on when the journey *ends*: future information |
| Everything in `synthetic_journeys.csv` | Hidden simulator settings, e.g. `failure_onset_min`, `crosses_threshold`, `ambient_c` |

## Target: minutes to threshold, capped at 120

`make_capped_target(features, horizon=120)` adds:

- `target_min`: minutes until the first later reading with
  `temperature_c > threshold_c` (strictly greater), capped at 120. A value of **120 means "no crossing within the next 2 hours"**, not
  "exactly 2 hours".
- `crosses_within_horizon`: `True` if a crossing happens within 120 minutes.
  This tells apart "crosses at exactly 120" and "no crossing within 120", which
  both have `target_min` = 120.
- `horizon_min`: the horizon that was used.

How each reading is treated:

| Reading | `target_min` | Kept? |
|---|---|---|
| A crossing happens within 120 min | exact minutes (5–120) | ✅ |
| A crossing happens, but later than 120 min | 120 (proven: the readings show no crossing within 120 min) | ✅ |
| No later crossing, and the journey was observed for **at least** 120 more minutes | 120 (proven by observation) | ✅ |
| No later crossing, but the journey ended **less than** 120 min later | unknown: it might have crossed after recording stopped | ❌ dropped, never guessed |
| Already above the threshold (`temperature_c > threshold_c`) | no time remaining to predict | ❌ dropped (handled by a simple "excursion in progress" rule) |

A reading **exactly at** the threshold (`temperature_c == threshold_c`) isn't
above it. It's treated like any other in-range reading and gets a normal target.
Likewise, a later reading exactly at the threshold doesn't count as the crossing
that `target_min` measures to.

The threshold always matches the one used for the features. Asking for targets
at a different threshold raises an error.

### What the 500-journey dataset gives (threshold 8.0 °C, horizon 120 min)

| | Rows |
|---|---|
| All readings | 72,361 |
| Dropped: already above the threshold | 10,718 |
| Dropped: less than 120 min of follow-up, no crossing seen | 8,213 |
| **Kept** | **53,430** |
| ↳ crossing within 120 min (exact target) | 6,368 (12%) |
| ↳ no crossing within 120 min (target = 120) | 47,062 (88%) |

## Journey-level split

`split_by_journey(table, test_size=0.2, seed=42)` divides **journeys**, not
rows. Every reading of a journey goes to the same side. Otherwise readings five
minutes apart, which are nearly identical, would appear on both sides, and the
model could score well just by recognising the journey.

The split is balanced by `scenario`, so both sides get a similar mix. With 500
journeys: 400 for training and 100 for testing, with zero overlap.

| Scenario | Train journeys | Test journeys |
|---|---|---|
| normal | 122 | 31 |
| door_openings | 118 | 29 |
| gradual_warming | 86 | 21 |
| fast_warming | 74 | 19 |

Use the test set **once**, for the final evaluation. To choose features and
settings, use cross-validation grouped by journey within the training set.

## Limitations and known issues

- **Synthetic only.** The features capture patterns of a simplified simulator
  (see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md#limitations-what-this-data-does-not-capture)).
  A real system needs real data and validation.
- **Imbalanced target.** 88% of rows have `target_min` = 120. A model can look
  accurate overall by always predicting 120 while missing real crossings.
  Evaluation must focus on rows where `crosses_within_horizon` is `True`, and
  report results per scenario.
- **Dropped rows come from journey ends.** Readings in the last 2 hours of
  non-crossing journeys are removed. This is a small, known bias.
- **Label resolution is 5 minutes.** A target of 25 means the crossing was first
  *seen* 25 minutes later. The true crossing happened between 20 and 25 minutes.
- **Door spikes are partly unpredictable.** A door opening at a random time
  can't be foreseen from past temperatures.
- **The first hour of each journey has incomplete features.** 60-minute
  features are empty there, about 8% of rows.

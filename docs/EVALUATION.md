# FrostIQ Evaluation: Baselines, ML Models and Safety-Oriented Metrics

> **These results come from synthetic data.** They show how well each method
> recovers patterns that we programmed into a simplified simulator. They **cannot**
> validate real-world cold-chain safety, and nothing here makes FrostIQ suitable
> for monitoring real medicines or vaccines. See
> [Why this cannot validate real-world safety](#why-this-cannot-validate-real-world-safety).

Code: [`src/frostiq/evaluate.py`](../src/frostiq/evaluate.py)

## How to run

```powershell
.venv\Scripts\Activate.ps1
python -m frostiq.simulate --n-journeys 500   # dataset (ignored by Git)
python -m frostiq.evaluate                    # prints sections A–E below (about 20 s)
```

Options:

| Option | Default | Effect |
|---|---|---|
| `--readings PATH` | `data/raw/synthetic_readings.csv` | Readings file to evaluate |
| `--horizon MINUTES` | 120 | Prediction horizon for the capped target |
| `--seed N` | 42 | Controls **all** randomness in the evaluation: which journeys go to train or test, the models' random seeds, and the bootstrap resampling. A different seed changes all three at once. It doesn't regenerate the data; that's `python -m frostiq.simulate --seed` |

The first line of the report names the dataset: journey count, generator seed
and threshold, read from `synthetic_config.json` next to the readings file. If
that file is missing or unreadable, the report says so and the evaluation still
runs. Check this line before comparing results with the numbers below, which
come from 500 journeys and seed 42.

No model files are saved. Two runs give byte-identical output.

## Methodology

1. **Data:** 500 synthetic journeys (seed 42) from `data/raw/synthetic_readings.csv`.
2. **Features:** `build_features()` on *all* readings, using past-only features
   (see [FEATURES.md](FEATURES.md)).
3. **Target:** `make_capped_target(horizon=120)`: minutes to the first crossing,
   capped at 120, plus the yes/no label `crosses_within_horizon`. Rows whose
   answer is unknown are dropped, never guessed.
4. **Split:** `split_by_journey(test_size=0.2, seed=42)`: 400 training journeys
   and 100 test journeys, with **zero** shared journeys, balanced by scenario.
5. **Fitting:** every model sees training rows only. Inputs are exactly
   `FEATURE_COLUMNS`. Never used as inputs: `crosses_within_horizon`, `scenario`,
   IDs, timestamps, `elapsed_min`, or any target or label column. Tests replace
   **every** `NEVER_FEATURES` column in the test set with different values:
   `crosses_within_horizon`, `target_min`, `minutes_to_threshold`,
   `minutes_to_window_end`, `label_status`, `scenario`, `journey_id`,
   `timestamp`, `elapsed_min`, `threshold_c` and `horizon_min`. The tests then
   confirm that neither the regression predictions nor the classifier
   probabilities change.
6. **No tuning against the reported test metrics.** Model settings and the
   classifier decision threshold were fixed, and no model selection or threshold
   tuning was performed against these test results. However, the test results
   were inspected during development, so this holdout should not be treated as a
   completely untouched final evaluation. Future model selection should use
   journey-grouped cross-validation on training data and a fresh holdout.

### The predictors

| Name | Type | What it is |
|---|---|---|
| `dummy_median` | regression | `DummyRegressor(strategy="median")`: always predicts the training median, which is **120** |
| `hist_gb` | regression | `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=42)` |
| `linear_eta` | regression | `min(linear_eta_min, 120)`: "at the current 30-minute warming rate, how long until the threshold?" |
| crossing classifier | classification | `RandomForestClassifier(n_estimators=200, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=1)`. Flags a row when P(crossing) ≥ **0.5** |

Settings explained:
- **`early_stopping=False`:** scikit-learn's automatic early stopping sets aside a
  random sample of *rows*, which would mix readings from the same journeys.
- **`class_weight="balanced"`:** crossing rows are rare (12%), so each one gets
  more weight. In total, the two classes count equally during training.
- **Decision threshold 0.5:** with balanced weights, 0.5 is the neutral point.
  It was fixed in advance, not chosen by looking at test results.
- **`n_jobs=1`:** multi-core prediction adds up tree votes in a varying order,
  which changes probabilities in the 16th decimal place. One core makes results
  exactly reproducible (about 12 s instead of 3 s).
- **Clipping:** regression predictions are clipped to 0–120, the range a capped
  target can take.

## Metrics

### A/B: MAE in minutes

**Mean absolute error** is the average distance between the predicted and true
minutes. It's reported for all rows, for crossing and non-crossing rows, and per
scenario. `linear_eta` needs 30 minutes of history, so it can't score 598 of the
10,695 test rows:
- **Table A** shows all rows, with `(n=...)` where fewer rows were scored.
- **Table B** scores every model on the same 10,097 rows. Compare methods using
  Table B.

### C: Warning timing (regression models)

| Metric | Meaning |
|---|---|
| Signed error | prediction − target, averaged. **Positive = predicted later than the truth** |
| Signed error on crossing rows | the same, on rows that really cross within 120 min |
| Late % | share of crossing rows where the prediction is *after* the actual crossing |
| Mean lateness | among those late predictions, the average minutes late |

MAE treats "10 minutes early" and "10 minutes late" the same. For cold-chain
monitoring, late is far worse: by the time the warning arrives, the excursion
has already happened. These metrics separate the two.

> **Important:** these use the **capped 120-minute target** on synthetic data,
> at 5-minute label resolution. They describe the capped prediction task, not
> validated real-world warning times.

### D: Crossing classifier

This answers a yes/no question: "will this container cross the threshold within
120 minutes?"

- **Confusion matrix:** counts of each outcome. *True positive* = crossing,
  flagged. *False negative* = crossing, **missed**. *False positive* = no
  crossing, flagged anyway (a false alarm). *True negative* = no crossing, not
  flagged.
- **Recall** = caught ÷ all real crossings. The key safety number: what share of
  real crossings got a warning?
- **Precision** = real crossings ÷ all flags. How trustworthy is a warning?
- **F1:** a single number balancing precision and recall.
- **False-positive rate** = false alarms ÷ all non-crossing rows.
- **Accuracy** is printed for reference only. With 88% non-crossing rows, a model
  that never warns would score 88% accuracy while missing every crossing, so it's
  never used to choose a model.

### E: Journey-level bootstrap interval

This answers: "Is `hist_gb`'s advantage over `linear_eta` on crossing rows real,
or could it just be luck in which journeys ended up in the test set?"

1. Take the test journeys that contain crossing rows where both models have a
   prediction.
2. Draw the same number of journeys **with replacement**: some appear twice,
   some not at all. Always take **whole journeys**, because readings within a
   journey are strongly related.
3. Compute MAE(`hist_gb`) − MAE(`linear_eta`) on that resample.
4. Repeat 2,000 times (seed 42). The middle 95% of results is the interval.

A negative value favours `hist_gb`. **This measures uncertainty from sampling
synthetic test journeys only, not real-world reliability.**

## Results (500 journeys, seed 42, horizon 120 min)

Test set: 100 journeys and 10,695 rows. **1,242 rows cross** within the horizon
and **9,453 don't**.

### Table B: common rows (10,097), MAE in minutes

| Segment | Rows | dummy_median | hist_gb | linear_eta |
|---|---|---|---|---|
| all | 10,097 | **7.2** | 9.2 | 7.4 |
| crosses within 120 min | 1,211 | 60.0 | **38.4** | 42.6 |
| no crossing within 120 min | 8,886 | **0.0** | 5.2 | 2.6 |
| door_openings | 3,438 | **9.6** | 15.6 | 15.3 |
| fast_warming | 788 | 32.6 | 20.8 | **18.4** |
| gradual_warming | 2,223 | 6.2 | 5.2 | **3.4** |
| normal | 3,648 | **0.0** | 3.0 | **0.0** |

Table A (all rows) tells the same story. `dummy_median` and `hist_gb` score 6.9
and 8.9 overall, and 59.6 and 38.5 on crossing rows.

### C: Warning timing (common rows; 1,211 crossing rows)

| Model | Signed error, all rows | Signed error, crossing rows | Late rows | Late % | Mean lateness (min) |
|---|---|---|---|---|---|
| dummy_median | +7.2 | +60.0 | 1,167 | 96.4% | 62.3 |
| hist_gb | −0.5 | +33.7 | 1,006 | 83.1% | 43.4 |
| linear_eta | +1.8 | +34.1 | 940 | 77.6% | 49.4 |

### D: Crossing classifier (decision threshold 0.5, all 10,695 test rows)

| | Predicted no | Predicted yes |
|---|---|---|
| **Actually no crossing** (9,453) | 8,274 (true negatives) | 1,179 (false alarms) |
| **Actually crosses** (1,242) | **608 (missed)** | 634 (caught) |

| Recall | Precision | F1 | False-positive rate | Accuracy (reference only) |
|---|---|---|---|---|
| **0.510** | 0.350 | 0.415 | 0.125 | 0.833 |

Crossing rows caught per scenario:

| Scenario | Crossing rows | Caught | Missed | Recall |
|---|---|---|---|---|
| door_openings | 566 | 268 | 298 | 0.473 |
| fast_warming | 448 | 149 | 299 | 0.333 |
| gradual_warming | 228 | 217 | 11 | 0.952 |

**Why is fast-warming recall so low?** A one-off diagnostic, which used the
hidden failure times *for explanation only* and never as model input, found the
cause. In fast warming, the crossing comes 24–65 minutes after a sudden failure.
So many readings within 120 minutes of the crossing were taken **before the
failure started**, while the container still looked normal:
- 280 of the 448 fast-warming crossing rows came from before the failure; only
  8 of those were caught.
- After the failure began, 141 of 168 were caught (recall 0.84).

No method can foresee a sudden power cut from earlier normal temperatures.

### E: Bootstrap (crossing rows, common rows)

MAE(`hist_gb`) − MAE(`linear_eta`) = **−4.2 min**, with a 95% interval of
**[−6.7, −1.6]**. That's 2,000 resamples of 42 journeys and 1,211 rows, seed 42.

## What the results mean

- **Nothing here is safe for real medicines or vaccines.** Even on synthetic
  data, the system misses about half of the crossing rows.
- **Overall MAE and accuracy are misleading.** The dummy has the best overall MAE
  (7.2) and would score 88% accuracy by never warning, yet it misses every
  crossing.
- **Predictions are usually late.** On crossing rows, `hist_gb` predicts later
  than the truth 83% of the time, by 43 minutes on average when late. The physics
  rule is similar (78%, 49 minutes).
- **The classifier catches 51% of crossing rows** (634 of 1,242), with 1,179
  false alarms (12.5% of non-crossing rows). Of every three warnings, roughly one
  is real. It does well on gradual warming (95%), but door spikes and the
  pre-failure part of fast warming are largely unpredictable from past
  temperatures.
- **ML beats the physics rule on crossing rows, modestly.** The bootstrap
  interval for the gap (−6.7 to −1.6 minutes) excludes zero, so the advantage is
  unlikely to be an accident of this test sample, but it's small. Over all rows,
  `hist_gb` is slightly worse.

## Limitations

- **One split, one seed.** The bootstrap covers sampling of test journeys, but
  not variation from a different simulator seed or a different training set.
- **Row-level, not event-level.** Recall counts *readings*, not *excursions*. A
  journey whose excursion was warned about 30 minutes ahead still counts its
  earlier unflagged readings as misses. An event-level metric ("was each
  excursion warned at least X minutes ahead?") would be closer to how an operator
  experiences the system.
- **The 0.5 threshold is not optimised for safety.** A lower threshold would
  catch more crossings and raise more false alarms. Choosing it properly needs
  cross-validation grouped by journey on *training* journeys only, never the test
  set.
- **Capped target.** Timing metrics only see up to 120 minutes, and "120" hides
  how far beyond the horizon a crossing really was.
- **5-minute label resolution,** and **default model settings** (no tuning, by
  design).

## Why this cannot validate real-world safety

- The data comes from a simplified simulator whose parameters are educated
  guesses, not measurements (see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md#limitations-what-this-data-does-not-capture)).
- The models learn the simulator's rules. A good score means "it learned our
  equations", not "it understands real refrigerators, products or failures".
- Real cold chains have effects the simulator lacks: compressor cycling, day/night
  temperatures, sensor drift and gaps, combined failures, and product thermal mass.
- Bootstrap intervals describe uncertainty within this synthetic test sample.
  They say nothing about performance on real equipment.
- Medical and pharmaceutical cold-chain decisions require validation on real data,
  regulatory requirements and review by qualified people. None of that has
  happened here.

## Recommended next steps

1. **Event-level evaluation:** for each excursion, was it flagged in time, and
   how many minutes ahead?
2. **Safety-oriented threshold:** choose the decision threshold to reach a target
   recall, using cross-validation grouped by journey within the training
   journeys. Report the false alarms that costs.
3. **Uncertainty beyond one seed:** repeat with several simulator seeds.

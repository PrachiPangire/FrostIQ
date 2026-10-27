"""Baseline evaluation and model comparison for FrostIQ (Milestones 3A and 3B).

Compares three predictors of ``target_min`` (minutes to threshold, capped at the
horizon) on a journey-level held-out test set:

* ``dummy_median``    always predicts the training-set median (DummyRegressor).
* ``hist_gb``         HistGradientBoostingRegressor on ``FEATURE_COLUMNS`` only.
* ``linear_eta``      the physics rule ``linear_eta_min``, capped at the horizon.
                      Needs 30 minutes of history, so it cannot score every row.

Milestone 3B adds safety-oriented measurements:

* warning timing: signed error, how often and how much predictions are late;
* a class-weighted RandomForestClassifier for ``crosses_within_horizon``,
  judged by crossings caught and missed rather than accuracy;
* a journey-level bootstrap interval for the hist_gb vs linear_eta MAE gap.

Models are fitted on training journeys only, with fixed settings (including the
classifier's decision threshold) chosen before looking at any test result.
Methodology and limitations: ``docs/EVALUATION.md``.

Usage::

    python -m frostiq.simulate --n-journeys 500
    python -m frostiq.evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix

from frostiq.config import DEFAULT_SEED, RAW_DATA_DIR
from frostiq.features import (
    DEFAULT_HORIZON_MIN,
    FEATURE_COLUMNS,
    NEVER_FEATURES,
    build_features,
    make_capped_target,
    split_by_journey,
)
from frostiq.simulate import CONFIG_FILE, READINGS_FILE

TARGET = "target_min"
MODEL_NAMES = ("dummy_median", "hist_gb", "linear_eta")
TEST_SIZE = 0.2

# Fixed in advance, not tuned on test results. Early stopping is off because its
# internal validation set is a random sample of *rows*, which would mix readings
# from the same journeys.
HIST_GB_PARAMS = {"max_iter": 200, "learning_rate": 0.1, "early_stopping": False}

CLASS_LABEL = "crosses_within_horizon"
# Fixed in advance. "balanced" weights each class inversely to its frequency, so
# rare crossing rows count as much in total as the common non-crossing rows.
# n_jobs=1: multi-threaded prediction sums tree votes in varying order, which
# changes probabilities in the last decimal place between runs.
CLASSIFIER_PARAMS = {"n_estimators": 200, "min_samples_leaf": 10, "class_weight": "balanced", "n_jobs": 1}
# Chosen before any test evaluation and never adjusted from test results: with
# balanced class weights, 0.5 is the neutral point between the two classes.
DECISION_THRESHOLD = 0.5

BOOTSTRAP_RESAMPLES = 2000


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the model inputs, in a fixed order."""
    columns = list(FEATURE_COLUMNS)
    leaked = set(columns) & set(NEVER_FEATURES)
    if leaked:
        raise ValueError(f"forbidden columns in FEATURE_COLUMNS: {sorted(leaked)}")
    return df[columns]


def fit_models(train: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict:
    """Fit the learned models on training rows only."""
    X, y = feature_matrix(train), train[TARGET]
    return {
        "dummy_median": DummyRegressor(strategy="median").fit(X, y),
        "hist_gb": HistGradientBoostingRegressor(random_state=seed, **HIST_GB_PARAMS).fit(X, y),
    }


def predict_all(models: dict, df: pd.DataFrame, horizon=DEFAULT_HORIZON_MIN) -> pd.DataFrame:
    """One prediction column per model. ``linear_eta`` is NaN where it has no value.

    Learned-model predictions are clipped to [0, horizon], the range a capped
    target can take.
    """
    X = feature_matrix(df)
    preds = pd.DataFrame(index=df.index)
    for name in ("dummy_median", "hist_gb"):
        preds[name] = np.clip(models[name].predict(X), 0.0, horizon)
    preds["linear_eta"] = np.minimum(df["linear_eta_min"], horizon)  # NaN stays NaN
    return preds


def masked_mae(y_true, y_pred):
    """MAE over rows where the prediction exists. Returns (mae, rows_scored)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = ~np.isnan(y_pred)
    if not ok.any():
        return np.nan, 0
    return float(np.mean(np.abs(y_true[ok] - y_pred[ok]))), int(ok.sum())


def _segments(df: pd.DataFrame):
    yield "all", np.ones(len(df), dtype=bool)
    yield "crosses_within_horizon=True", df["crosses_within_horizon"].to_numpy()
    yield "crosses_within_horizon=False", ~df["crosses_within_horizon"].to_numpy()
    if "scenario" in df.columns:
        for scenario in sorted(df["scenario"].unique()):
            yield f"scenario={scenario}", (df["scenario"] == scenario).to_numpy()


def mae_by_segment(df: pd.DataFrame, preds: pd.DataFrame, common_rows_only=False) -> pd.DataFrame:
    """MAE per model and segment, with row counts.

    With ``common_rows_only`` every model is scored on the same rows: those where
    all predictions exist (i.e. where ``linear_eta`` has a value).
    """
    y = df[TARGET].to_numpy(dtype=float)
    available = preds.notna().all(axis=1).to_numpy() if common_rows_only else np.ones(len(df), bool)
    rows = []
    for segment, mask in _segments(df):
        mask = mask & available
        row = {"segment": segment, "rows": int(mask.sum())}
        for name in preds.columns:
            mae, scored = masked_mae(y[mask], preds[name].to_numpy()[mask])
            row[f"{name}_mae"] = mae
            row[f"{name}_rows"] = scored
        rows.append(row)
    return pd.DataFrame(rows)


def warning_timing(y_true, y_pred, crosses) -> dict:
    """Warning-timing metrics for one model, over rows where the prediction exists.

    * ``mean_signed_error``: mean of (prediction - target) over all scored rows.
      Positive means predictions are later than the target on average.
    * On actual crossing rows (``crosses`` True) only:
      ``late_pct`` = share of rows predicted later than the actual crossing, and
      ``mean_lateness`` = average minutes late among those late rows.

    Targets are capped at the horizon, so these are capped-target quantities, not
    validated real-world warning times.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    crosses = np.asarray(crosses, dtype=bool)
    ok = ~np.isnan(y_pred)
    error = y_pred[ok] - y_true[ok]
    crossing_error = error[crosses[ok]]
    late = crossing_error[crossing_error > 0]
    return {
        "rows_scored": int(ok.sum()),
        "mean_signed_error": float(error.mean()) if error.size else np.nan,
        "crossing_rows": int(crossing_error.size),
        "crossing_mean_signed_error": float(crossing_error.mean()) if crossing_error.size else np.nan,
        "late_rows": int(late.size),
        "late_pct": 100.0 * late.size / crossing_error.size if crossing_error.size else np.nan,
        "mean_lateness": float(late.mean()) if late.size else np.nan,
    }


def timing_table(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """``warning_timing`` per model, on rows where every model has a prediction."""
    common = preds.notna().all(axis=1).to_numpy()
    y, crosses = df[TARGET].to_numpy(float)[common], df[CLASS_LABEL].to_numpy(bool)[common]
    return pd.DataFrame(
        [{"model": name, **warning_timing(y, preds[name].to_numpy()[common], crosses)} for name in preds.columns]
    )


def fit_classifier(train: pd.DataFrame, seed: int = DEFAULT_SEED) -> RandomForestClassifier:
    """Class-weighted random forest for ``crosses_within_horizon``, training rows only."""
    return RandomForestClassifier(random_state=seed, **CLASSIFIER_PARAMS).fit(
        feature_matrix(train), train[CLASS_LABEL].astype(bool)
    )


def crossing_probability(classifier, df: pd.DataFrame) -> np.ndarray:
    """Predicted probability that a crossing happens within the horizon."""
    positive = list(classifier.classes_).index(True)
    return classifier.predict_proba(feature_matrix(df))[:, positive]


def classification_metrics(y_true, y_flag) -> dict:
    """Confusion matrix and positive-class (crossing) metrics.

    Precision: of the rows flagged, how many really cross. Recall: of the rows
    that really cross, how many were flagged (caught). Accuracy is included for
    reference only; with rare crossings it is misleading.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_flag = np.asarray(y_flag, dtype=bool)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_flag, labels=[False, True]).ravel())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if tp + fp + fn else np.nan)
    return {
        "positives": tp + fn,
        "negatives": tn + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "caught": tp,
        "missed": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else np.nan,
    }


def recall_by_scenario(df: pd.DataFrame, y_flag) -> pd.DataFrame:
    """Crossing rows caught and missed per scenario (reporting only)."""
    positives = df[CLASS_LABEL].to_numpy(bool)
    rows = df[positives].assign(caught=np.asarray(y_flag, dtype=bool)[positives])
    out = rows.groupby("scenario")["caught"].agg(positives="size", caught="sum")
    out["missed"] = out["positives"] - out["caught"]
    out["recall"] = out["caught"] / out["positives"]
    return out.reset_index()


def bootstrap_mae_difference(df: pd.DataFrame, preds: pd.DataFrame, model_a="hist_gb", model_b="linear_eta",
                             n_resamples=BOOTSTRAP_RESAMPLES, seed=DEFAULT_SEED, confidence=0.95) -> dict:
    """Journey-level bootstrap interval for MAE(model_a) - MAE(model_b) on crossing rows.

    Uses rows where both models have a prediction and a crossing happens within
    the horizon. Each resample draws *journeys* with replacement (all of a
    journey's rows together), because readings within a journey are strongly
    related. Negative values favour ``model_a``. This describes uncertainty from
    which synthetic test journeys happened to be sampled, nothing more.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rows = preds[[model_a, model_b]].notna().all(axis=1).to_numpy() & df[CLASS_LABEL].to_numpy(bool)
    y = df[TARGET].to_numpy(float)[rows]
    per_row = pd.DataFrame({
        "journey_id": df["journey_id"].to_numpy()[rows],
        "err_a": np.abs(preds[model_a].to_numpy()[rows] - y),
        "err_b": np.abs(preds[model_b].to_numpy()[rows] - y),
        "n": 1,
    })
    if per_row.empty:
        raise ValueError("no crossing rows where both models have predictions")
    journeys = per_row.groupby("journey_id")[["err_a", "err_b", "n"]].sum()
    sum_a, sum_b, counts = (journeys[c].to_numpy(float) for c in ("err_a", "err_b", "n"))

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(journeys), size=(n_resamples, len(journeys)))
    resampled_n = counts[picks].sum(axis=1)
    diffs = (sum_a[picks].sum(axis=1) - sum_b[picks].sum(axis=1)) / resampled_n
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "estimate": float((sum_a.sum() - sum_b.sum()) / counts.sum()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "journeys": len(journeys),
        "rows": int(counts.sum()),
        "seed": seed,
    }


def run_evaluation(readings: pd.DataFrame, horizon=DEFAULT_HORIZON_MIN, test_size=TEST_SIZE,
                   seed=DEFAULT_SEED, n_resamples=BOOTSTRAP_RESAMPLES) -> dict:
    """Full pipeline: features -> capped target -> journey split -> fit -> evaluate."""
    table = make_capped_target(build_features(readings), horizon=horizon)
    train, test = split_by_journey(table, test_size=test_size, seed=seed)
    models = fit_models(train, seed=seed)
    preds = predict_all(models, test, horizon=horizon)

    classifier = fit_classifier(train, seed=seed)
    probability = crossing_probability(classifier, test)
    flagged = probability >= DECISION_THRESHOLD
    return {
        "test_size": test_size,
        "train": train,
        "test": test,
        "models": models,
        "predictions": preds,
        "all_rows": mae_by_segment(test, preds),
        "common_rows": mae_by_segment(test, preds, common_rows_only=True),
        "timing": timing_table(test, preds),
        "classifier": classifier,
        "crossing_probability": probability,
        "flagged": flagged,
        "classification": classification_metrics(test[CLASS_LABEL], flagged),
        "recall_by_scenario": recall_by_scenario(test, flagged),
        "bootstrap": bootstrap_mae_difference(test, preds, n_resamples=n_resamples, seed=seed),
    }


def _format_table(results: pd.DataFrame) -> str:
    shown = pd.DataFrame({"segment": results["segment"], "rows": results["rows"]})
    for name in MODEL_NAMES:
        mae = results[f"{name}_mae"].map(lambda v: "-" if np.isnan(v) else f"{v:6.1f}")
        scored = results[f"{name}_rows"]
        shown[name] = [m if s == r else f"{m} (n={s})" for m, s, r in zip(mae, scored, results["rows"])]
    return shown.to_string(index=False)


def load_dataset_info(readings_path: Path):
    """Read the generator's config saved next to the readings file.

    Returns ``(info, note)``: ``info`` is a dict with ``n_journeys``, ``seed`` and
    ``threshold_c`` (None if unavailable) and ``note`` explains why it is missing.
    A missing or unreadable config never stops the evaluation.
    """
    path = Path(readings_path).with_name(CONFIG_FILE)
    if not path.exists():
        return None, f"{CONFIG_FILE} not found next to the readings file"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        return {key: config.get(key) for key in ("n_journeys", "seed", "threshold_c")}, None
    except (OSError, ValueError, AttributeError) as err:
        return None, f"{CONFIG_FILE} could not be read ({type(err).__name__})"


def _describe_dataset(info, note) -> str:
    if info is None:
        return f"Dataset: journey count and seed unknown ({note})"
    return (f"Dataset: {info['n_journeys']} journeys, seed {info['seed']}, "
            f"threshold {info['threshold_c']} degC (from {CONFIG_FILE})")


def print_report(result: dict, horizon=DEFAULT_HORIZON_MIN, dataset=(None, "not provided")) -> None:
    train, test, preds = result["train"], result["test"], result["predictions"]
    test_size = result.get("test_size", TEST_SIZE)
    print(_describe_dataset(*dataset))
    print(f"Horizon: {horizon:g} min   Split: journey-level, {1 - test_size:.0%}/{test_size:.0%}")
    print(f"Train: {train['journey_id'].nunique()} journeys, {len(train)} rows   "
          f"Test: {test['journey_id'].nunique()} journeys, {len(test)} rows   "
          f"Shared journeys: {len(set(train['journey_id']) & set(test['journey_id']))}")
    print(f"Test rows crossing within horizon: {int(test['crosses_within_horizon'].sum())}   "
          f"not crossing: {int((~test['crosses_within_horizon']).sum())}")
    print(f"Dummy median prediction: {result['models']['dummy_median'].constant_.ravel()[0]:.1f} min")
    print(f"linear_eta has a value for {int(preds['linear_eta'].notna().sum())} of {len(test)} test rows "
          "(needs 30 min of history)")
    print("\nMAE in minutes (lower is better). '(n=...)' = model scored on fewer rows than the segment has.")
    print("\nA) All test rows")
    print(_format_table(result["all_rows"]))
    print("\nB) Common rows only: every model scored on the same rows (where linear_eta exists)")
    print(_format_table(result["common_rows"]))

    timing = result["timing"]
    print(f"\nC) Warning timing on common rows (capped {horizon:g}-min target; not real-world warning times)")
    print("   signed error = prediction - target (positive = later than actual). Late = predicted after the crossing.")
    shown = pd.DataFrame({
        "model": timing["model"],
        "signed_err_all": timing["mean_signed_error"].round(1),
        "signed_err_crossing": timing["crossing_mean_signed_error"].round(1),
        "crossing_rows": timing["crossing_rows"],
        "late_rows": timing["late_rows"],
        "late_%": timing["late_pct"].round(1),
        "mean_lateness_min": timing["mean_lateness"].round(1),
    })
    print(shown.to_string(index=False))

    m = result["classification"]
    print(f"\nD) Crossing classifier: will the threshold be crossed within {horizon:g} min?")
    print(f"   RandomForestClassifier, class_weight='balanced', decision threshold {DECISION_THRESHOLD} "
          "(fixed before testing)")
    print(f"   Test rows: {m['positives']} positive (cross), {m['negatives']} negative")
    print("   Confusion matrix:            predicted no   predicted yes")
    print(f"     actually no crossing      {m['tn']:>12}   {m['fp']:>13}")
    print(f"     actually crosses          {m['fn']:>12}   {m['tp']:>13}")
    print(f"   Caught {m['caught']} of {m['positives']} crossing rows, missed {m['missed']} (false negatives)")
    print(f"   Recall {m['recall']:.3f}   Precision {m['precision']:.3f}   F1 {m['f1']:.3f}   "
          f"False-positive rate {m['false_positive_rate']:.3f}")
    print(f"   (Accuracy {m['accuracy']:.3f}: shown for reference only; misleading with rare crossings)")
    by_scenario = result["recall_by_scenario"].assign(recall=lambda d: d["recall"].round(3))
    print("   Crossing rows caught per scenario:")
    print("   " + by_scenario.to_string(index=False).replace("\n", "\n   "))

    b = result["bootstrap"]
    print(f"\nE) Journey-level bootstrap: MAE({b['model_a']}) - MAE({b['model_b']}) on crossing rows")
    print(f"   {b['estimate']:+.1f} min, {b['confidence']:.0%} interval [{b['ci_low']:+.1f}, {b['ci_high']:+.1f}]  "
          f"({b['n_resamples']} resamples of {b['journeys']} journeys, {b['rows']} rows, seed {b['seed']}).")
    print("   Negative favours the first model. This is uncertainty for this synthetic test sample only,")
    print("   not a measure of real-world reliability.")
    print("\nSynthetic data only: these numbers do not establish real-world safety or accuracy.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate FrostIQ baselines and models.")
    parser.add_argument("--readings", type=Path, default=RAW_DATA_DIR / READINGS_FILE)
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON_MIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="controls the journey split, model randomness and bootstrap resampling")
    args = parser.parse_args(argv)

    if not args.readings.exists():
        raise SystemExit(f"{args.readings} not found. Generate it first:\n"
                         "    python -m frostiq.simulate --n-journeys 500")
    result = run_evaluation(pd.read_csv(args.readings), horizon=args.horizon, seed=args.seed)
    print_report(result, horizon=args.horizon, dataset=load_dataset_info(args.readings))
    return result


if __name__ == "__main__":
    main()

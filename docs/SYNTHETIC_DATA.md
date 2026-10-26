# FrostIQ Synthetic Data: Assumptions and Limitations

> **Important:** This dataset is computer-generated for software development and
> learning. Any model trained or tested on it shows only that the model can learn
> the patterns *we programmed in*. It does **not** establish real-world medical
> safety, regulatory compliance or predictive accuracy for real cold-chain
> equipment, vaccines or medicines. Real-world use would require validation on
> real sensor data and review by qualified people.

Generator: [`src/frostiq/simulate.py`](../src/frostiq/simulate.py)

## Regenerating the data

```powershell
.venv\Scripts\Activate.ps1
python -m frostiq.simulate                          # defaults: 120 journeys, seed 42, threshold 8.0 degC
python -m frostiq.simulate --threshold 10 --seed 1  # example with other settings
```

The generator writes three files to `data/raw/`. Git ignores them because they
can always be regenerated:

| File | Contents |
|---|---|
| `synthetic_readings.csv` | One row per logged reading, with labels |
| `synthetic_journeys.csv` | One row per journey: hidden simulation settings and whether it crossed |
| `synthetic_config.json` | The exact settings and seed used |

The same seed and settings always produce the same files.

## What is a "journey"?

A journey is one refrigerated container (a truck, a reefer box or a vaccine
carrier) observed for 8–16 hours. A data logger records its temperature every
5 minutes. Journeys are independent: nothing carries over between them.

## How temperature is simulated

Two forces act on the temperature inside the box at every minute:

1. **Refrigeration** pulls the temperature *down* towards a setpoint (3–5 °C).
2. **Outside air** pulls it *up* towards the ambient temperature (20–35 °C),
   through the insulation, and much faster while a door is open.

Each pull is stronger the further the temperature is from its target. This is
Newton's law of cooling. When the refrigeration works, the two pulls balance at a
steady temperature somewhere between **3.4 and 6.9 °C**, depending on that
journey's setpoint, cooling strength, insulation and outside temperature. With
seed 42, 90% of `normal` readings fall between 4.0 and 6.2 °C, and the warmest is
7.42 °C. When the refrigeration weakens or stops, outside air wins and the
temperature climbs.

Two kinds of random noise are added:
- **Process noise** (standard deviation 0.02 °C per minute): small real
  fluctuations, such as air currents.
- **Sensor noise** (standard deviation 0.10 °C per reading): logger measurement
  error.

## The four scenarios

Each journey's scenario is picked at random using the **probabilities** below.
Because it's random, the **actual counts** differ a little from those
proportions, just as 10 coin flips don't always give exactly 5 heads.

| Scenario | Probability | What happens |
|---|---|---|
| `normal` | 30% | Loaded slightly warm (5–7.5 °C), cooled down, held steady |
| `door_openings` | 30% | Normal cooling plus random door openings (0.25–0.5 per hour, 2–10 min each) |
| `gradual_warming` | 20% | After 1–6 h, cooling weakens steadily over 2–8 h to 0–15% of normal |
| `fast_warming` | 20% | After 1–6 h, cooling stops completely and heat leak rises 3–6× (about 1.6–11 °C/h of warming at 5 °C) |

What the default run (120 journeys, seed 42, threshold 8.0 °C) actually produced:

| Scenario | Actual journeys | Share of 120 | Crossed 8 °C | Time from failure to first crossing |
|---|---|---|---|---|
| `normal` | 34 | 28% | 0 | (no failure) |
| `door_openings` | 36 | 30% | 17 | (no failure; brief spikes from door openings) |
| `gradual_warming` | 32 | 27% | 16 | 224–602 min (3.7–10.0 h). The other 16 were still at or below 8 °C when observation ended, so they are censored |
| `fast_warming` | 18 | 15% | 18 | 24–65 min |

These outcomes describe this seed only. A different seed or more journeys will
give somewhat different counts and ranges.

## All numeric assumptions

Every value lives in `SimulationConfig` in `simulate.py`. Ranges are sampled
uniformly, once per journey.

| Setting | Value | Meaning |
|---|---|---|
| `sampling_interval_min` | 5 | Minutes between logged readings |
| `journey_hours` | 8–16 | Observation window length |
| `ambient_c` | 20–35 °C | Outside air temperature, constant for the whole journey |
| `setpoint_c` | 3–5 °C | Refrigeration target |
| `initial_c` | 5–7.5 °C | Temperature at loading |
| `cooling_rate` | 0.015–0.025 /min | Strength of the refrigeration pull |
| `leak_rate` | 0.0006–0.0010 /min | Insulation heat leak: about 0.5–1.8 °C/h of warming at 5 °C if cooling stops, slowing as the box approaches outside temperature |
| `door_exchange_rate` | 0.005–0.025 /min | Extra heat exchange while a door is open |
| `fast_leak_multiplier` | 3–6× | Heat-leak increase in `fast_warming` |
| `residual_cooling` | 0–15% | Cooling left at the end of `gradual_warming` |
| `seed` | 42 | Random seed, for reproducibility |

These values were chosen to give plausible-looking curves and a useful mix of
outcomes. They were **not** fitted to measurements from any real device.

## How labels are built

The threshold is **configurable** (`--threshold`). 8.0 °C is only an
experimental default, the upper limit of the common 2–8 °C storage range.
"Crossing" means a reading **strictly above** the threshold, so 8.0 °C counts as
still in range.

For every reading, the labeller looks **forward in the same journey only**:

| `label_status` | Meaning | `minutes_to_threshold` |
|---|---|---|
| `will_cross` | A later reading in this journey is above the threshold | Minutes until the first such reading |
| `censored` | No later reading goes above the threshold before the journey's observation ends | **Left empty.** We don't know when, or whether, it would cross. Inventing a number would teach a model false information |
| `above_threshold` | This reading is already above the threshold | Empty. The excursion is already happening |

`minutes_to_window_end` is filled for every row. For censored rows it is a
**lower bound**: if a crossing ever happens, it happens later than this. This is
the standard way to handle "we stopped watching before the event" in survival
analysis (also called time-to-event analysis). Later milestones can use it
instead of throwing censored rows away.

After a door spike ends, temperature can drop back below the threshold. Readings
after that point are labelled by the *next* crossing in the journey, or censored
if there isn't one.

## Limitations (what this data does NOT capture)

- **Not fitted to real equipment.** The parameters are educated guesses, not
  measurements.
- **Simplified physics.** The box is treated as one uniform temperature. Real
  containers have hot and cold spots, airflow patterns, product thermal mass and
  sensor placement effects.
- **No thermostat cycling.** Real compressors switch on and off, which creates a
  saw-tooth pattern. Here cooling is smooth.
- **Constant ambient temperature.** Real journeys see day/night cycles, sun
  exposure, and moves between warehouses and trucks.
- **One failure type per journey.** Real failures can combine, for example door
  openings during a compressor fault.
- **Door openings are random.** No model can predict a random door opening hours
  ahead, so `will_cross` labels for `door_openings` journeys are partly
  unpredictable by design.
- **Simple sensor noise.** Real loggers can drift, lose readings, record
  irregular timestamps, or round to 0.5 °C.
- **Label precision equals the sampling interval.** The true crossing happened
  somewhere between two readings up to 5 minutes apart.
- **Hidden settings are recorded separately.** `synthetic_journeys.csv` contains
  information a real system would not know, such as the failure onset time. Use
  it for analysis only, never as model input. Doing so would be *data leakage*:
  giving the model the answer in disguise.

## Moving to real data later

Real public datasets can be investigated in a later milestone. The labeller,
`add_threshold_labels`, only needs `journey_id`, `timestamp` and
`temperature_c`. It computes times from the timestamps themselves, so it handles
irregular sampling intervals and rows in any order. It accepts data loaded
straight from CSV, where timestamps arrive as text.

How timestamps are handled:

| Input | What happens |
|---|---|
| ISO 8601 text with a zone, e.g. `2026-01-05 04:55:00+00:00`, `...Z` or `...+05:30` | Converted to UTC. Readings with different offsets, such as a daylight-saving change, are still ordered by real time |
| ISO 8601 text without a zone, e.g. `2026-01-05 04:55:00` | Accepted as-is. Only time *differences* are used, so this is fine as long as one clock was used throughout |
| A mix of values with and without a zone | **Rejected.** A value without a zone could mean any zone, so the order would be a guess |
| Missing or blank timestamps | **Rejected.** The reading can't be placed in time |
| Text that isn't a valid date-time, e.g. `not a date` or `2026-02-30` | **Rejected** |
| The same timestamp twice in one journey | **Rejected.** The order of the two readings would be ambiguous |

Missing temperatures are also rejected. A gap could hide a threshold crossing, so
real data with gaps must be cleaned, or split into separate journeys, before
labelling. Every rejection raises a `ValueError` with a message explaining the
problem. Nothing is silently dropped or guessed.

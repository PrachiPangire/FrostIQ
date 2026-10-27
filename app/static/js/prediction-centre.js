// Renders the two AI Prediction Centre cards from an API journey payload.
// Shared by the homepage and the console. Values come only from the API.
(function () {
  "use strict";

  const UNAVAILABLE_TITLE = {
    observed_above: "Crossing observed",
    insufficient_history: "Not enough readings yet",
    threshold_mismatch: "No prediction",
    model_unavailable: "Model unavailable",
  };

  function part(card, name) { return card.querySelector(`[data-pc-${name}]`); }

  // Plain-English reasons for "no prediction". The API's technical message is
  // kept only for a missing model, because it contains the command that fixes it.
  function unavailableText(block, payload) {
    const limit = payload.prediction && payload.prediction.model_threshold_c;
    switch (block.status) {
      case "observed_above":
        return "The latest reading is already above the limit. That has really happened (it was recorded), so there is nothing left to predict.";
      case "insufficient_history":
        return "The models need at least 30 minutes of readings before they can look ahead. Move further into the journey.";
      case "threshold_mismatch":
        return `The models only know the ${limit != null ? limit : 8}°C limit they were trained on, so they make no prediction for other limits.`;
      default:
        return block.message || "No prediction is available right now.";
    }
  }

  function setUnavailable(card, block, payload) {
    card.dataset.state = block.status === "observed_above" ? "observed" : "unavailable";
    part(card, "value").textContent = UNAVAILABLE_TITLE[block.status] || "Unavailable";
    part(card, "detail").textContent = unavailableText(block, payload);
    const meter = part(card, "meter");
    if (meter) meter.hidden = true;
  }

  function renderClassification(card, c, payload) {
    if (!c.available) { setUnavailable(card, c, payload); return; }
    card.dataset.state = c.flagged ? "flagged" : "clear";
    part(card, "value").textContent = c.flagged ? "FLAGGED" : "NOT FLAGGED";
    const meter = part(card, "meter");
    meter.hidden = false;
    part(card, "fill").style.width = `${Math.round(c.score * 100)}%`;
    part(card, "detail").textContent =
      `Risk score: ${c.score.toFixed(3)} out of 1 (a rough score, not a certainty). It's flagged at ${c.decision_threshold} or more.` +
      (c.flagged
        ? " The temperature is still within the limit; this is a warning about what may happen next."
        : ` No problem is expected in the next ${c.horizon_min} minutes.`);
  }

  function renderRegression(card, r, c, payload) {
    if (!r.available) { setUnavailable(card, r, payload); return; }
    const disagree = " Note: the two cards disagree, so treat this answer with extra caution.";
    if (r.status === "beyond_horizon") {
      card.dataset.state = "beyond";
      part(card, "value").textContent = `Beyond ${r.horizon_min} min`;
      part(card, "detail").textContent =
        `No crossing is expected in the next ${r.horizon_min} minutes. (The model's raw output, ${r.model_output_min} min, ` +
        `is just its upper limit, not a predicted time.)` + (c.flagged ? disagree : "");
      return;
    }
    card.dataset.state = "estimated";
    part(card, "value").textContent = `≈ ${Math.round(r.minutes)} min`;
    part(card, "detail").textContent =
      `About ${Math.round(r.minutes)} minutes until the temperature may first go above the limit ` +
      `(model output: ${r.minutes} min). It only looks up to ${r.horizon_min} minutes ahead.` + (c.available && !c.flagged ? disagree : "");
  }

  function render(root, payload) {
    if (!root) return;
    const cards = {
      classification: root.querySelector('[data-pc-card="classification"]'),
      regression: root.querySelector('[data-pc-card="regression"]'),
    };
    if (!payload || !payload.classification || !payload.regression) { reset(root, "No prediction data was returned."); return; }
    renderClassification(cards.classification, payload.classification, payload);
    renderRegression(cards.regression, payload.regression, payload.classification, payload);
  }

  function reset(root, message) {
    if (!root) return;
    root.querySelectorAll("[data-pc-card]").forEach((card) => {
      card.dataset.state = "idle";
      part(card, "value").textContent = "—";
      part(card, "detail").textContent = message || "Waiting for a journey…";
      const meter = part(card, "meter");
      if (meter) meter.hidden = true;
    });
  }

  window.FrostIQPredictions = { render, reset };
})();

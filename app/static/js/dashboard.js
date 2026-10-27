// FrostIQ telemetry console. Every number shown comes from the JSON API.
(function () {
  "use strict";
  const root = document.getElementById("console");
  if (!root || !document.getElementById("journey-select")) return;

  const $ = (id) => document.getElementById(id);
  const API_LIST = root.dataset.apiJourneys;
  const API_ONE = root.dataset.apiJourney;
  const DEFAULT_THRESHOLD = parseFloat(root.dataset.threshold);
  const card = JSON.parse($("model-card").textContent || "null");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COLORS = { cyan: "#38d6f5", amber: "#f7b955", red: "#ff6b81", grey: "#9db0cf", grid: "rgba(125,211,252,0.08)", text: "#9db0cf" };
  const LEVEL_TEXT = { normal: "Normal", warning: "Warning", above: "Above threshold" };

  const params = new URLSearchParams(window.location.search);
  const state = {
    journeys: [],
    filter: "all",
    id: root.dataset.selected,
    position: params.has("at") ? parseInt(params.get("at"), 10) : null,
    threshold: params.has("threshold") ? parseFloat(params.get("threshold")) : DEFAULT_THRESHOLD,
    hindsight: params.get("hindsight") === "1",
    playing: false,
    data: null,
  };
  if (!Number.isFinite(state.threshold)) state.threshold = DEFAULT_THRESHOLD;
  if (!Number.isInteger(state.position)) state.position = null;

  let controller = null;
  let chart = null;
  let loadingTimer = null;
  let sliderTimer = null;

  // ---------- formatting ----------
  const fmtTemp = (v) => (v === null || v === undefined ? "—" : v.toFixed(2));
  const fmtClock = (m) => {
    const h = Math.floor(m / 60);
    return `${h}:${String(Math.round(m % 60)).padStart(2, "0")}`;
  };
  const fmtDuration = (m) => {
    if (m === null || m === undefined) return "—";
    const h = Math.floor(m / 60);
    const mm = Math.round(m % 60);
    return h ? `${h} h ${String(mm).padStart(2, "0")} min` : `${mm} min`;
  };
  const fmtUtc = (iso) => iso.replace("T", " ").slice(0, 16);

  // ---------- network ----------
  async function getJSON(url, signal) {
    const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
    let body = null;
    try { body = await response.json(); } catch (_) { /* non-JSON error page */ }
    if (!response.ok) throw new Error(body && body.error ? body.error : `Request failed (${response.status})`);
    return body;
  }

  function setLoading(on) {
    clearTimeout(loadingTimer);
    if (on) loadingTimer = setTimeout(() => ($("loading").hidden = false), 180);
    else $("loading").hidden = true;
  }

  function showError(message) {
    const box = $("api-error");
    box.textContent = message;
    box.hidden = !message;
  }

  // ---------- journey list ----------
  async function loadList() {
    const data = await getJSON(`${API_LIST}?threshold=${encodeURIComponent(state.threshold)}`);
    state.journeys = data.journeys;
    const hasSplit = state.journeys.some((j) => j.split);
    document.querySelector('[data-filter="test"]').disabled = !hasSplit;
    updateThresholdNote();
    return renderSelect();
  }

  // Explain that a different threshold does not get a recalculated forecast.
  function updateThresholdNote() {
    const note = $("threshold-note");
    if (card && state.threshold !== card.threshold_c) {
      note.textContent = `The model was trained for ${card.threshold_c} °C, so no ML estimate is calculated at ${state.threshold} °C. Status, margin, chart and readings use the new threshold.`;
      note.hidden = false;
    } else {
      note.hidden = true;
    }
  }

  // Empty filter result: hide (and stop updating) the previous journey's readings.
  function showEmpty(on) {
    $("empty-state").hidden = !on;
    $("journey-view").hidden = on;
    if (on) {
      stopPlay();
      if (controller) controller.abort();
      state.data = null;
      if (window.FrostIQPredictions) window.FrostIQPredictions.reset($("console-pc"), "No journey selected.");
    }
  }

  function filtered() {
    if (state.filter === "test") return state.journeys.filter((j) => j.split === "test");
    if (state.filter === "crossed") return state.journeys.filter((j) => j.crossed);
    return state.journeys;
  }

  function renderSelect() {
    const select = $("journey-select");
    const list = filtered();
    select.replaceChildren(...list.map((j) => {
      const parts = [j.id, j.scenario || "unknown scenario"];
      if (j.split) parts.push(j.split === "test" ? "held-out" : j.split === "train" ? "training" : j.split);
      if (j.crossed) parts.push(`crossed ${state.threshold} °C`);
      const option = new Option(parts.join(" · "), j.id);
      return option;
    }));
    $("journey-count").textContent = `Showing ${list.length} of ${state.journeys.length} journeys`;
    if (!list.length) {
      select.append(new Option("No journeys match this filter", ""));
      select.value = "";
      showEmpty(true);
      return "empty";
    }
    showEmpty(false);
    if (list.some((j) => j.id === state.id)) {
      select.value = state.id;
      return state.data ? "same" : "changed"; // reload if the view was cleared
    }
    state.id = list[0].id;
    state.position = null;
    select.value = state.id;
    return "changed";
  }

  // ---------- one journey ----------
  async function loadJourney() {
    if (!state.id) return;
    if (controller) controller.abort();
    controller = new AbortController();
    const query = new URLSearchParams({ threshold: state.threshold });
    if (state.position !== null) query.set("at", state.position);
    setLoading(true);
    try {
      const data = await getJSON(`${API_ONE.replace("__ID__", encodeURIComponent(state.id))}?${query}`, controller.signal);
      state.data = data;
      state.position = data.position;
      showError("");
      render(data);
      updateUrl();
    } catch (err) {
      if (err.name === "AbortError") return;
      showError(err.message);
      stopPlay();
    } finally {
      setLoading(false);
    }
  }

  function updateUrl() {
    const q = new URLSearchParams({ journey: state.id, at: state.position });
    if (state.threshold !== DEFAULT_THRESHOLD) q.set("threshold", state.threshold);
    if (state.hindsight) q.set("hindsight", "1");
    history.replaceState(null, "", `${window.location.pathname}?${q}`);
  }

  // ---------- rendering ----------
  function render(d) {
    const last = d.total_readings - 1;
    const slider = $("position-slider");
    slider.max = last;
    slider.value = d.position;
    $("position-label").textContent = `Reading ${d.position + 1} of ${d.total_readings} · ${fmtDuration(d.latest.elapsed_min)} in`;
    $("step-back").disabled = d.position === 0;
    $("step-fwd").disabled = d.position === last;
    $("to-latest").disabled = d.position === last;
    if (d.position === last && state.playing) stopPlay();

    $("now-journey").textContent = d.journey_id;
    $("now-scenario").textContent = d.scenario ? `· ${d.scenario.replace(/_/g, " ")} (simulated)` : "";
    $("now-position").textContent = `Reading ${d.position + 1} of ${d.total_readings} · ${fmtDuration(d.latest.elapsed_min)} in`;

    renderStatus(d);
    renderPrediction(d);
    renderJourney(d);
    renderRecent(d);
    renderChart(d);
    renderHindsight(d);
  }

  function renderStatus(d) {
    const level = d.status.level;
    $("status-card").dataset.level = level;
    const el = $("status-level");
    el.dataset.level = level;
    el.textContent = LEVEL_TEXT[level] || level;
    $("status-reasons").replaceChildren(...d.status.reasons.map((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      return li;
    }));

    $("latest-temp").innerHTML = "";
    $("latest-temp").append(fmtTemp(d.latest.temperature_c), small(" °C"));
    $("latest-time").textContent = `${fmtUtc(d.latest.timestamp)} UTC · ${fmtDuration(d.latest.elapsed_min)} into journey`;

    const margin = d.latest.margin_c;
    const marginEl = $("margin");
    marginEl.innerHTML = "";
    marginEl.append(`${margin > 0 ? "+" : ""}${fmtTemp(margin)}`, small(" °C"));
    marginEl.className = `mono ${margin < 0 ? "coral-text" : margin <= 1 ? "amber-text" : "mint-text"}`;
    $("threshold-text").textContent = `Threshold ${d.threshold_c} °C · margin = threshold − latest reading`;
  }

  function small(text) {
    const s = document.createElement("small");
    s.textContent = text;
    return s;
  }

  // The AI Prediction Centre: classification + regression cards (shared renderer).
  function renderPrediction(d) {
    if (window.FrostIQPredictions) window.FrostIQPredictions.render($("console-pc"), d);
  }

  function renderJourney(d) {
    const s = d.summary;
    $("j-id").textContent = d.journey_id;
    $("j-scenario").textContent = d.scenario || "—";
    $("j-readings").textContent = `${s.readings} of ${d.total_readings}`;
    $("j-duration").textContent = fmtDuration(s.duration_min);
    $("j-range").textContent = `${fmtTemp(s.min_c)} / ${fmtTemp(s.mean_c)} / ${fmtTemp(s.max_c)} °C`;
    $("j-excursions").textContent = String(s.excursions);
    $("j-above").textContent = `${s.readings_above} (~${fmtDuration(s.minutes_above_approx)})`;
    const badge = $("split-badge");
    const note = $("split-note");
    if (d.split === "test") {
      badge.hidden = false; badge.className = "badge test"; badge.textContent = "Held-out";
      note.textContent = "Held-out journey: the model never saw it during training.";
    } else if (d.split === "train") {
      badge.hidden = false; badge.className = "badge train"; badge.textContent = "Training";
      note.textContent = "Training journey: the model learned from it, so its forecasts here are optimistic (in-sample).";
    } else {
      badge.hidden = true;
      note.textContent = d.split === "unseen" ? "This journey was not part of the model's dataset split." : "";
    }
  }

  function renderRecent(d) {
    $("recent-body").replaceChildren(...d.recent.map((r) => {
      const tr = document.createElement("tr");
      if (r.above) tr.className = "is-above";
      const cells = [fmtUtc(r.timestamp).slice(11), fmtClock(r.elapsed_min), fmtTemp(r.temperature_c), r.above ? "Above" : "In range"];
      cells.forEach((text, i) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (i === 2) td.className = "num";
        tr.append(td);
      });
      return tr;
    }));
  }

  function renderHindsight(d) {
    const box = $("hindsight-box");
    $("legend-future").hidden = !state.hindsight;
    box.hidden = !state.hindsight;
    if (!state.hindsight) return;
    const h = d.hindsight;
    let text = "Hindsight (uses future readings a real-time system would not have): ";
    if (h.outcome === "crossed_later") {
      text += `the recorded data first went above ${d.threshold_c} °C ${h.minutes_to_crossing} min after this reading.`;
      if (d.prediction && d.prediction.status === "predicted") {
        text += d.prediction.capped
          ? ` The forecast said "not within ${d.prediction.horizon_min} min".`
          : ` The forecast said about ${d.prediction.minutes} min.`;
      }
    } else if (h.outcome === "already_above") {
      text += "this reading is already above the threshold.";
    } else {
      text += `no crossing in the remaining ${fmtDuration(h.observed_for_min)} of recorded data.`;
    }
    box.textContent = text;
  }

  // ---------- chart ----------
  function renderChart(d) {
    if (typeof window.Chart === "undefined") {
      $("chart-fallback").hidden = false;
      $("temp-chart").hidden = true;
      return;
    }
    const thr = d.threshold_c;
    const past = d.chart.elapsed_min.map((x, i) => ({ x, y: d.chart.temperature_c[i] }));
    const future = d.chart.future_elapsed_min.map((x, i) => ({ x, y: d.chart.future_temperature_c[i] }));
    if (future.length && past.length) future.unshift(past[past.length - 1]);
    const xMax = future.length ? future[future.length - 1].x : past[past.length - 1].x;
    const ys = past.concat(future).map((p) => p.y).concat([thr]);
    const yMin = Math.floor(Math.min(...ys) - 0.5);
    const yMax = Math.ceil(Math.max(...ys) + 0.5);
    const lastIndex = past.length - 1;
    const datasets = [
      {
        label: "Recorded temperature",
        data: past,
        borderColor: COLORS.cyan,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: (ctx) => (ctx.dataIndex === lastIndex ? 5 : ctx.raw && ctx.raw.y > thr ? 2.5 : 0),
        pointBackgroundColor: (ctx) => (ctx.raw && ctx.raw.y > thr ? COLORS.red : COLORS.cyan),
        pointBorderColor: (ctx) => (ctx.raw && ctx.raw.y > thr ? COLORS.red : COLORS.cyan),
        segment: { borderColor: (ctx) => (ctx.p0.parsed.y > thr || ctx.p1.parsed.y > thr ? COLORS.red : COLORS.cyan) },
      },
      {
        label: "Threshold",
        data: [{ x: 0, y: thr }, { x: xMax, y: thr }],
        borderColor: COLORS.amber,
        borderDash: [6, 5],
        borderWidth: 1.5,
        pointRadius: 0,
      },
      {
        label: "Future readings (hindsight)",
        data: state.hindsight ? future : [],
        borderColor: COLORS.grey,
        borderDash: [3, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.25,
      },
    ];

    if (!chart) {
      chart = new window.Chart($("temp-chart"), {
        type: "line",
        data: { datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: reduceMotion ? false : { duration: 300 },
          parsing: false,
          normalized: true,
          interaction: { mode: "nearest", intersect: false, axis: "x" },
          scales: {
            x: {
              type: "linear", min: 0,
              title: { display: true, text: "Elapsed time (h:mm)", color: COLORS.text },
              ticks: { color: COLORS.text, callback: (v) => fmtClock(v), maxTicksLimit: 10 },
              grid: { color: COLORS.grid },
            },
            y: {
              title: { display: true, text: "Temperature (°C)", color: COLORS.text },
              ticks: { color: COLORS.text },
              grid: { color: COLORS.grid },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => `+${fmtClock(items[0].parsed.x)}`,
                label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(2)} °C`,
              },
            },
          },
        },
      });
    }
    chart.data.datasets = datasets;
    chart.options.scales.x.max = xMax;
    chart.options.scales.y.min = yMin;
    chart.options.scales.y.max = yMax;
    chart.update(state.playing || reduceMotion ? "none" : undefined);
  }

  // ---------- replay ----------
  function stopPlay() {
    state.playing = false;
    $("play").textContent = "▶ Play";
  }

  async function playLoop() {
    while (state.playing && state.data) {
      if (state.position >= state.data.total_readings - 1) { stopPlay(); break; }
      state.position += 1;
      await loadJourney();
      await new Promise((resolve) => setTimeout(resolve, 450));
    }
  }

  function goTo(position) {
    stopPlay();
    state.position = position;
    loadJourney();
  }

  // ---------- events ----------
  $("journey-select").addEventListener("change", (e) => {
    if (!e.target.value) return;
    stopPlay();
    state.id = e.target.value;
    state.position = null;
    loadJourney();
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
      state.filter = chip.dataset.filter;
      if (renderSelect() === "changed") { stopPlay(); loadJourney(); }
    });
  });
  $("empty-reset").addEventListener("click", () => document.querySelector('[data-filter="all"]').click());

  $("threshold-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const value = parseFloat($("threshold-input").value);
    if (!Number.isFinite(value) || value < -30 || value > 50) {
      showError("Threshold must be a number between -30 and 50 °C.");
      return;
    }
    stopPlay();
    state.threshold = value;
    try {
      if ((await loadList()) !== "empty") await loadJourney();
    } catch (err) {
      showError(err.message);
    }
  });

  $("threshold-reset").addEventListener("click", () => {
    $("threshold-input").value = DEFAULT_THRESHOLD;
    $("threshold-form").requestSubmit();
  });

  $("position-slider").addEventListener("input", (e) => {
    stopPlay();
    const position = parseInt(e.target.value, 10);
    clearTimeout(sliderTimer);
    sliderTimer = setTimeout(() => { state.position = position; loadJourney(); }, 120);
  });

  $("step-back").addEventListener("click", () => state.data && goTo(Math.max(0, state.position - 1)));
  $("step-fwd").addEventListener("click", () => state.data && goTo(Math.min(state.data.total_readings - 1, state.position + 1)));
  $("to-latest").addEventListener("click", () => state.data && goTo(state.data.total_readings - 1));
  $("play").addEventListener("click", () => {
    if (!state.data) return;
    if (state.playing) { stopPlay(); return; }
    if (state.position >= state.data.total_readings - 1) state.position = -1; // restart from the beginning
    state.playing = true;
    $("play").textContent = "❚❚ Pause";
    playLoop();
  });

  $("hindsight-toggle").checked = state.hindsight;
  $("hindsight-toggle").addEventListener("change", (e) => {
    state.hindsight = e.target.checked;
    if (state.data) { renderChart(state.data); renderHindsight(state.data); updateUrl(); }
  });

  // ---------- start ----------
  $("threshold-input").value = state.threshold;
  (async () => {
    try {
      if ((await loadList()) !== "empty") await loadJourney();
    } catch (err) {
      showError(err.message);
    }
  })();
})();

// "Explore a journey": a lightweight replay of real recorded (simulated) journeys
// using the same JSON API as the monitoring console. Nothing here is invented.
(function () {
  "use strict";
  const root = document.getElementById("journey-story");
  if (!root) return;

  const $ = (id) => document.getElementById(id);
  const API_LIST = root.dataset.apiJourneys;
  const API_ONE = root.dataset.apiJourney;
  const CONSOLE = root.dataset.console;
  const THRESHOLD = parseFloat(root.dataset.threshold);
  const cooler = document.getElementById("journey-cooler");
  const STEP = 2; // readings per playback tick (2 × 5 min)

  // Plain-English descriptions of the simulator's scenarios.
  const SCENARIO_TEXT = {
    normal: "Steady cooling: the cooling works all journey. The container is loaded slightly warm and cooled down.",
    door_openings: "Door openings: cooling works, but doors open at random moments and warm air rushes in for a few minutes.",
    gradual_warming: "Gradual warming: partway through, the cooling slowly weakens, like a failing compressor.",
    fast_warming: "Power loss: partway through, cooling stops completely, so the temperature rises quickly.",
  };
  const LEVEL = {
    normal: { text: "Normal", state: "safe", meaning: "Below the threshold with more than 1 °C to spare." },
    warning: { text: "Warning", state: "warning", meaning: "Still below the threshold, but close to it or the model flags a likely crossing." },
    above: { text: "Above threshold", state: "unsafe", meaning: "This recorded reading is above the threshold: an observed crossing, not a prediction." },
  };

  const state = { picks: {}, scenario: null, id: null, position: 0, total: 0, temps: [], minutes: [], playing: false, started: false };
  let controller = null;
  let rangeTimer = null;

  const fmtDuration = (m) => {
    const h = Math.floor(m / 60), mm = Math.round(m % 60);
    return h ? `${h} h ${String(mm).padStart(2, "0")} min` : `${mm} min`;
  };

  async function getJSON(url, signal) {
    const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
    let body = null;
    try { body = await response.json(); } catch (_) { /* not JSON */ }
    if (!response.ok) throw new Error(body && body.error ? body.error : `Request failed (${response.status})`);
    return body;
  }

  function showError(message) {
    $("js-error").textContent = message || "";
    $("js-error").hidden = !message;
  }

  // Choose one held-out journey per scenario: one that crosses for the failure
  // scenarios, one that does not for steady cooling.
  function pickJourneys(journeys) {
    const picks = {};
    for (const scenario of Object.keys(SCENARIO_TEXT)) {
      const ofScenario = journeys.filter((j) => j.scenario === scenario);
      const wantCrossed = scenario !== "normal";
      picks[scenario] = (
        ofScenario.find((j) => j.split === "test" && j.crossed === wantCrossed) ||
        ofScenario.find((j) => j.split === "test") ||
        ofScenario[0] || null
      );
    }
    return picks;
  }

  async function init() {
    if (state.started) return;
    state.started = true;
    try {
      const list = await getJSON(`${API_LIST}?threshold=${THRESHOLD}`);
      state.picks = pickJourneys(list.journeys);
      document.querySelectorAll("[data-scenario]").forEach((chip) => {
        chip.disabled = !state.picks[chip.dataset.scenario];
        chip.addEventListener("click", () => selectScenario(chip.dataset.scenario));
      });
      const first = ["gradual_warming", "fast_warming", "door_openings", "normal"].find((s) => state.picks[s]);
      if (first) await selectScenario(first);
      else showError("No journeys available in the dataset.");
    } catch (err) {
      showError(err.message);
      $("js-journey-meta").textContent = "Recorded journeys could not be loaded.";
    }
  }

  async function selectScenario(scenario) {
    stopPlay();
    state.scenario = scenario;
    state.id = state.picks[scenario].id;
    document.querySelectorAll("[data-scenario]").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.scenario === scenario)));
    $("js-scenario-text").textContent = SCENARIO_TEXT[scenario];
    try {
      // The full recorded series sets the timeline's scale; replay starts at the first reading.
      const full = await getJSON(API_ONE.replace("__ID__", encodeURIComponent(state.id)) + `?threshold=${THRESHOLD}`);
      state.temps = full.chart.temperature_c;
      state.minutes = full.chart.elapsed_min;
      state.total = full.total_readings;
      $("js-range").max = state.total - 1;
      await goTo(0);
    } catch (err) {
      showError(err.message);
    }
  }

  async function goTo(position) {
    state.position = Math.max(0, Math.min(position, state.total - 1));
    if (controller) controller.abort();
    controller = new AbortController();
    try {
      const d = await getJSON(API_ONE.replace("__ID__", encodeURIComponent(state.id)) +
                              `?threshold=${THRESHOLD}&at=${state.position}`, controller.signal);
      showError("");
      render(d);
    } catch (err) {
      if (err.name !== "AbortError") { showError(err.message); stopPlay(); }
    }
  }

  function render(d) {
    const level = LEVEL[d.status.level] || LEVEL.normal;
    const t = d.latest.temperature_c;
    if (cooler) { // SVG fallback cooler (shown when 3D is not running)
      cooler.dataset.state = level.state;
      cooler.querySelector("[data-cooler-value]").textContent = t.toFixed(1);
    }

    const split = d.split === "test" ? "held-out, never used in training"
      : d.split === "train" ? "a training journey" : "simulated";
    $("js-journey-meta").textContent = `Recorded simulated journey ${d.journey_id} (${split}) · reading ${d.position + 1} of ${d.total_readings}`;
    $("js-time").textContent = `${fmtDuration(d.latest.elapsed_min)} into the journey · ${d.latest.timestamp.replace("T", " ").slice(0, 16)} UTC (simulated)`;
    $("js-temp").textContent = `${t.toFixed(2)} °C`;
    $("js-thr").textContent = `${d.threshold_c} °C`;

    $("js-state-box").dataset.level = d.status.level;
    $("js-state").textContent = level.text;
    $("js-state-note").textContent = level.meaning;

    // The two AI cards (classification + regression), straight from the API.
    if (window.FrostIQPredictions) window.FrostIQPredictions.render(document.getElementById("home-pc"), d);
    // Let the 3D cooler show the recorded status and reading.
    document.dispatchEvent(new CustomEvent("frostiq:status", { detail: { level: d.status.level, value: `${t.toFixed(1)} °C` } }));
    drawSpark(d.position);

    $("js-range").value = d.position;
    $("js-back").disabled = d.position === 0;
    $("js-fwd").disabled = d.position === d.total_readings - 1;
    $("js-open-moment").href = `${CONSOLE}?journey=${encodeURIComponent(d.journey_id)}&at=${d.position}`;
  }

  function drawSpark(position) {
    const W = 600, H = 140, PAD = 10;
    const temps = state.temps, mins = state.minutes;
    if (!temps.length) return;
    const lo = Math.min(...temps, THRESHOLD) - 0.5;
    const hi = Math.max(...temps, THRESHOLD) + 0.5;
    const xMax = mins[mins.length - 1] || 1;
    const x = (m) => (m / xMax) * W;
    const y = (v) => H - PAD - ((v - lo) / (hi - lo)) * (H - 2 * PAD);
    const pts = temps.slice(0, position + 1).map((v, i) => `${x(mins[i]).toFixed(1)},${y(v).toFixed(1)}`);
    $("js-spark-line").setAttribute("d", pts.length ? `M${pts.join(" L")}` : "");
    $("js-spark-area").setAttribute("d", pts.length ? `M${x(mins[0])},${H} L${pts.join(" L")} L${x(mins[position])},${H} Z` : "");
    const ty = y(THRESHOLD).toFixed(1);
    const thr = $("js-spark-threshold");
    thr.setAttribute("y1", ty); thr.setAttribute("y2", ty);
    const dot = $("js-spark-dot");
    dot.setAttribute("cx", x(mins[position]).toFixed(1));
    dot.setAttribute("cy", y(temps[position]).toFixed(1));
    const color = temps[position] > THRESHOLD ? "var(--coral)" : "var(--cyan)";
    dot.style.fill = color;
    $("js-spark-line").style.stroke = color;
  }

  // --- playback ---
  function stopPlay() {
    state.playing = false;
    $("js-play").textContent = "▶ Play";
  }
  async function playLoop() {
    while (state.playing) {
      if (state.position >= state.total - 1) { stopPlay(); break; }
      await goTo(Math.min(state.position + STEP, state.total - 1));
      await new Promise((r) => setTimeout(r, 280));
    }
  }
  $("js-play").addEventListener("click", () => {
    if (!state.id) return;
    if (state.playing) { stopPlay(); return; }
    if (state.position >= state.total - 1) state.position = 0;
    state.playing = true;
    $("js-play").textContent = "❚❚ Pause";
    playLoop();
  });
  $("js-back").addEventListener("click", () => { stopPlay(); goTo(state.position - 1); });
  $("js-fwd").addEventListener("click", () => { stopPlay(); goTo(state.position + 1); });
  $("js-range").addEventListener("input", (e) => {
    stopPlay();
    const position = parseInt(e.target.value, 10);
    drawSpark(position);
    clearTimeout(rangeTimer);
    rangeTimer = setTimeout(() => goTo(position), 120);
  });

  // Load data only when the section approaches the viewport (keeps first paint fast).
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(([entry]) => { if (entry.isIntersecting) { io.disconnect(); init(); } },
                                        { rootMargin: "400px 0px" });
    io.observe(root);
  } else {
    init();
  }
})();

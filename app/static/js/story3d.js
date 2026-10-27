// Homepage 3D storytelling: starts the WebGL stage and links it to scrolling.
// Normal page scrolling is never hijacked: sections only *trigger* or *scrub* scene changes.
// If WebGL, the 3D module or the visitor's motion preference rule it out, the
// static SVG fallback (visible by default) simply stays on screen.

const root = document.documentElement;
const canvas = document.getElementById("stage");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function webglAvailable() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (_) {
    return false;
  }
}

function fallback(reason) {
  root.classList.remove("has-3d");
  root.dataset.fallback = reason;
}

const narrow = () => window.innerWidth < 900;

// Camera and object targets for each section. Mobile centres the object.
function sceneValues(name) {
  const m = narrow();
  const lookX = m ? 0 : -1.7;
  const camZ = (z) => (m ? z + 4 : z + 2.2); // pulled back so the scene breathes
  const reset = { warmth: 0, danger: 0, safe: 0, power: 1, frost: 1, heat: 0, lid: 0, route: 0, column: 0, core: 0, matrix: 0, rings: 1, spin: 1, coolerX: 0, coolerY: 0, coolerScale: 1 };
  const scenes = {
    // On phones the camera looks below the cooler so it sits above the headline.
    home: { ...reset, camX: 0, camY: 1.4, camZ: camZ(7.2), lookX, lookY: m ? -1.15 : -0.2, lookZ: 0 },
    problem: { ...reset, safe: 1, camX: 1.4, camY: 1.7, camZ: camZ(6.6), lookX: m ? 0 : -1.5, lookY: 0.25, lookZ: 0 },
    journey: { ...reset, route: 1, frost: 0.3, rings: 0, spin: 0.2, camX: 0, camY: m ? 10 : 7.5, camZ: m ? 11 : 8.5, lookX: m ? 0 : -1.2, lookY: -0.9, lookZ: 0 },
    risk: { ...reset, column: 1, coolerX: -0.7, camX: 0.6, camY: 1.3, camZ: camZ(7.8), lookX: m ? 0.4 : -1.3, lookY: 0.45, lookZ: 0 },
    how: { ...reset, core: 1, coolerX: -1.2, coolerY: -0.15, coolerScale: 0.55, rings: 0.3, camX: 0, camY: 1.0, camZ: camZ(8.2), lookX: m ? 0 : -1.3, lookY: 0.35, lookZ: 0 },
    predict: { ...reset, camX: 0.4, camY: 1.35, camZ: camZ(6.9), lookX: m ? 0 : -1.8, lookY: 0.2, lookZ: 0 },
    performance: { ...reset, coolerScale: 0, matrix: 1, rings: 0, camX: 4.6, camY: 4.2, camZ: m ? 10 : 7.6, lookX: m ? 0 : -1.9, lookY: 0.7, lookZ: 0 },
    limits: { ...reset, spin: 0.6, camX: 0, camY: 1.9, camZ: camZ(10), lookX: 0, lookY: 0.2, lookZ: 0 },
  };
  return scenes[name] || scenes.home;
}

// Overlays for the problem-section steps.
const PROBLEM_STEPS = {
  safe: { safe: 1, lid: 0, power: 1, warmth: 0, heat: 0, frost: 1 },
  door: { safe: 0, lid: 1, power: 1, warmth: 0.45, heat: 0, frost: 0.4 },
  power: { safe: 0, lid: 0, power: 0, warmth: 0.15, heat: 0, frost: 0 },
  gradual: { safe: 0, lid: 0, power: 1, warmth: 1, heat: 0, frost: 0.25 },
  ambient: { safe: 0, lid: 0, power: 1, warmth: 0.7, heat: 1, frost: 0.2 },
};
const STATUS_LOOK = {
  normal: { safe: 1, warmth: 0, danger: 0 },
  warning: { safe: 0, warmth: 1, danger: 0 },
  above: { safe: 0, warmth: 1, danger: 1 },
};

async function boot() {
  let createStage;
  try {
    ({ createStage } = await import("./stage/frostiq-stage.js"));
  } catch (err) {
    fallback("module-failed");
    return;
  }
  const countsEl = document.getElementById("model-counts");
  let counts = null;
  try { counts = countsEl ? JSON.parse(countsEl.textContent) : null; } catch (_) { counts = null; }
  const threshold = canvas.dataset.threshold || "8";
  const stage = createStage(canvas, { mobile: narrow(), counts, label: "Threshold", value: `${threshold} °C` });
  if (!stage) { fallback("renderer-failed"); return; }
  root.classList.add("has-3d");
  stage.onContextLost(() => fallback("context-lost"));

  const gsap = window.gsap;
  const ScrollTrigger = window.ScrollTrigger;
  const useGsap = !!(gsap && ScrollTrigger);
  if (useGsap) gsap.registerPlugin(ScrollTrigger);
  else stage.setDamping(true);

  let current = "home";
  let lastStatus = null;
  let lastProblemStep = "safe";

  function apply(values, duration = 1.4) {
    if (useGsap) gsap.to(stage.state, { ...values, duration, ease: "power2.inOut", overwrite: "auto" });
    else Object.assign(stage.goal, values);
  }
  function goTo(name) {
    current = name;
    canvas.dataset.scene = name; // visible state for debugging and browser checks
    const values = sceneValues(name);
    if (name === "problem") Object.assign(values, PROBLEM_STEPS[lastProblemStep]);
    if (name === "risk") { delete values.warmth; delete values.danger; delete values.safe; } // owned by the scroll scrub
    if (name === "predict" && lastStatus) Object.assign(values, STATUS_LOOK[lastStatus.level] || {});
    apply(values);
    if (name === "predict" && lastStatus) stage.setDisplay("Recorded", lastStatus.value);
    else stage.setDisplay("Threshold", `${threshold} °C`);
    document.body.classList.toggle("stage-dim", name === "limits");
  }

  // Scrubbed progress for the route and the temperature column.
  function setRoute(p) { stage.state.routeT = p; if (!useGsap) stage.goal.routeT = p; }
  function setColumn(p) {
    const values = { columnFill: p, warmth: Math.min(1, Math.max(0, (p - 0.35) / 0.27)), danger: p > 0.66 ? 1 : 0, safe: p < 0.35 ? 1 : 0 };
    if (current !== "risk") return;
    if (useGsap) Object.assign(stage.state, values); else Object.assign(stage.goal, values);
    const label = document.getElementById("risk-state");
    if (label) label.textContent = p > 0.66 ? "Above threshold (observed)" : p > 0.45 ? "Warning: close to the threshold" : "Normal: well below the threshold";
  }

  const sections = Array.from(document.querySelectorAll("[data-scene]"));
  if (useGsap) {
    sections.forEach((section) => {
      ScrollTrigger.create({
        trigger: section, start: "top 55%", end: "bottom 45%",
        onEnter: () => goTo(section.dataset.scene), onEnterBack: () => goTo(section.dataset.scene),
      });
    });
    const route = document.getElementById("journey");
    if (route) ScrollTrigger.create({ trigger: route, start: "top 70%", end: "bottom 50%", scrub: true, onUpdate: (s) => setRoute(s.progress) });
    const risk = document.getElementById("risk");
    if (risk) ScrollTrigger.create({ trigger: risk, start: "top 60%", end: "bottom 60%", scrub: true, onUpdate: (s) => setColumn(s.progress) });
    // Gentle parallax on section copy, scrubbed by scroll (transform only; scrolling is untouched).
    gsap.utils.toArray(".scene-copy").forEach((el) => {
      gsap.fromTo(el, { y: 40 }, { y: -20, ease: "none", scrollTrigger: { trigger: el, start: "top bottom", end: "bottom top", scrub: true } });
    });
  } else {
    // Fallback without GSAP: IntersectionObserver for sections, scroll math for scrubs.
    const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) goTo(e.target.dataset.scene); }),
                                        { rootMargin: "-45% 0px -45% 0px" });
    sections.forEach((s) => io.observe(s));
    const progressOf = (el) => { const r = el.getBoundingClientRect(); return Math.min(1, Math.max(0, (innerHeight * 0.7 - r.top) / (r.height))); };
    window.addEventListener("scroll", () => {
      const route = document.getElementById("journey"); if (route) setRoute(progressOf(route));
      const risk = document.getElementById("risk"); if (risk) setColumn(progressOf(risk));
    }, { passive: true });
  }

  // Problem steps (activated by story.js) and the recorded journey player (journey-story.js).
  document.addEventListener("frostiq:problem", (e) => {
    lastProblemStep = e.detail.scene in PROBLEM_STEPS ? e.detail.scene : "safe";
    canvas.dataset.problem = lastProblemStep;
    if (current === "problem") apply(PROBLEM_STEPS[lastProblemStep], 1.0);
  });
  document.addEventListener("frostiq:status", (e) => {
    lastStatus = e.detail;
    canvas.dataset.status = lastStatus.level;
    if (current === "predict") {
      apply(STATUS_LOOK[lastStatus.level] || {}, 0.6);
      stage.setDisplay("Recorded", lastStatus.value);
    }
  });

  // Subtle pointer parallax on fine pointers only.
  if (window.matchMedia("(pointer: fine)").matches) {
    window.addEventListener("pointermove", (e) => stage.setPointer(e.clientX / innerWidth - 0.5, e.clientY / innerHeight - 0.5), { passive: true });
  }
  window.addEventListener("resize", () => goTo(current));
  goTo("home");
}

if (!canvas) fallback("no-canvas");
else if (reduceMotion) fallback("reduced-motion");
else if (!webglAvailable()) fallback("no-webgl");
else boot();

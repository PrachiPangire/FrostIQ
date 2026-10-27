// Story homepage: scroll reveals and the problem-section steps.
// Steps announce the active scene with a "frostiq:problem" event (used by the 3D stage)
// and also drive the SVG fallback cooler when 3D is not running.
(function () {
  "use strict";
  const hasIO = "IntersectionObserver" in window;

  // --- Reveal elements as they scroll into view ---
  const revealables = document.querySelectorAll(".on-scroll");
  if (hasIO) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-visible"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -10% 0px" });
    revealables.forEach((el) => io.observe(el));
  } else {
    revealables.forEach((el) => el.classList.add("is-visible"));
  }

  // --- Problem steps ---
  const steps = Array.from(document.querySelectorAll(".scene-step"));
  const fallbackCooler = document.getElementById("problem-cooler");
  const caption = document.getElementById("problem-caption");

  function activate(step) {
    if (!step) return;
    steps.forEach((s) => {
      const active = s === step;
      s.classList.toggle("is-active", active);
      s.querySelector(".scene-btn").setAttribute("aria-pressed", String(active));
    });
    const scene = step.dataset.sceneStep;
    if (fallbackCooler) {
      fallbackCooler.dataset.state = scene;
      fallbackCooler.querySelector("[data-cooler-value]").textContent = step.dataset.label;
    }
    if (caption) caption.textContent = `Illustration · ${step.querySelector(".scene-title").textContent} · not recorded data`;
    document.dispatchEvent(new CustomEvent("frostiq:problem", { detail: { scene } }));
  }
  steps.forEach((s) => s.querySelector(".scene-btn").addEventListener("click", () => activate(s)));
  if (hasIO && steps.length) {
    const sceneIO = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) activate(e.target); });
    }, { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach((s) => sceneIO.observe(s));
  }
})();

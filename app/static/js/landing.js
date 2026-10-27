// Hero atmosphere: particle field and gentle pointer parallax on the cooler (decorative only).
(function () {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const small = window.matchMedia("(max-width: 900px)").matches;

  // --- Parallax tilt (desktop pointers only) ---
  const hero = document.getElementById("home");
  const tilt = document.querySelector("#hero-cooler .cooler-tilt");
  if (hero && tilt && !reduceMotion && window.matchMedia("(pointer: fine)").matches) {
    hero.addEventListener("pointermove", (e) => {
      const r = hero.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      tilt.style.setProperty("--ry", `${(x * 10).toFixed(2)}deg`);
      tilt.style.setProperty("--rx", `${(-y * 8).toFixed(2)}deg`);
    });
    hero.addEventListener("pointerleave", () => {
      tilt.style.setProperty("--ry", "0deg");
      tilt.style.setProperty("--rx", "0deg");
    });
  }

  // --- Particle field ---
  const canvas = document.getElementById("particles");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const COUNT = small ? 18 : 46;
  const LINK = small ? 80 : 120;
  let width = 0, height = 0, points = [], frame = null, inView = true;

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  function seed() {
    points = Array.from({ length: COUNT }, () => ({
      x: Math.random() * width, y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.22, vy: (Math.random() - 0.5) * 0.22,
      r: Math.random() * 1.5 + 0.5,
    }));
  }
  function draw() {
    ctx.clearRect(0, 0, width, height);
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      for (let j = i + 1; j < points.length; j++) {
        const b = points[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < LINK) {
          ctx.strokeStyle = `rgba(125, 211, 252, ${0.13 * (1 - d / LINK)})`;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
      ctx.fillStyle = "rgba(165, 236, 255, 0.7)";
      ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2); ctx.fill();
    }
  }
  function step() {
    for (const p of points) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;
    }
    draw();
    frame = requestAnimationFrame(step);
  }
  const start = () => { if (frame === null && !reduceMotion && inView && !document.hidden) frame = requestAnimationFrame(step); };
  const stop = () => { if (frame !== null) cancelAnimationFrame(frame); frame = null; };

  resize(); seed(); draw(); start();
  window.addEventListener("resize", () => { resize(); seed(); draw(); });
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  // Pause the loop while the hero is off screen.
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; inView ? start() : stop(); }).observe(canvas);
  }
})();

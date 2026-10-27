// Navigation: mobile menu (keyboard friendly) and active-section highlighting.
(function () {
  "use strict";
  const toggle = document.querySelector(".nav-toggle");
  const links = document.getElementById("nav-links");

  if (toggle && links) {
    const setOpen = (open, returnFocus) => {
      links.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
      if (open) links.querySelector("a")?.focus();
      else if (returnFocus) toggle.focus();
    };
    toggle.addEventListener("click", () => setOpen(!links.classList.contains("open"), false));
    links.addEventListener("click", (event) => { if (event.target.closest("a")) setOpen(false, false); });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && links.classList.contains("open")) setOpen(false, true);
    });
    window.matchMedia("(min-width: 1141px)").addEventListener("change", (e) => { if (e.matches) setOpen(false, false); });
  }

  // Highlight the homepage section currently in view.
  const navLinks = Array.from(document.querySelectorAll(".nav-links a[data-section]"));
  const sections = navLinks.map((a) => document.getElementById(a.dataset.section)).filter(Boolean);
  if (!sections.length || !("IntersectionObserver" in window)) return;
  const visible = new Map();
  const update = () => {
    let best = null;
    for (const s of sections) {
      if (visible.get(s.id) && (!best || s.offsetTop > best.offsetTop)) best = s;
    }
    navLinks.forEach((a) => {
      if (best && a.dataset.section === best.id) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
  };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => visible.set(e.target.id, e.isIntersecting));
    update();
  }, { rootMargin: "-35% 0px -55% 0px" });
  sections.forEach((s) => observer.observe(s));
})();

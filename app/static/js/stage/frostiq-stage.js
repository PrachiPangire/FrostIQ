// FrostIQ 3D stage: one WebGL scene shared by every homepage section.
// Everything is built from code (no model files). Scroll timelines animate `stage.state`.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/RoundedBoxGeometry.js";

const COLORS = {
  cyan: new THREE.Color(0x38d6f5),
  mint: new THREE.Color(0x4ff5c6),
  amber: new THREE.Color(0xf7b955),
  coral: new THREE.Color(0xff6b81),
  off: new THREE.Color(0x33415c),
  navy: 0x070b16,
};

// Numeric parameters the scroll story animates (also the defaults).
const DEFAULT_STATE = {
  camX: 0, camY: 1.4, camZ: 7.2, lookX: -1.7, lookY: 0.15, lookZ: 0,
  coolerX: 0, coolerY: 0, coolerScale: 1, spin: 1,
  lid: 0, warmth: 0, danger: 0, safe: 0, power: 1, frost: 1, heat: 0, rings: 1,
  route: 0, routeT: 0, column: 0, columnFill: 0, core: 0, matrix: 0,
};

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export function createStage(canvas, options = {}) {
  const mobile = !!options.mobile;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: !mobile, alpha: true, powerPreference: "high-performance" });
  } catch (err) {
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(COLORS.navy, 0.04);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 120);
  const state = { ...DEFAULT_STATE };
  const goal = { ...DEFAULT_STATE };
  let damping = false;
  const pointer = { x: 0, y: 0, sx: 0, sy: 0 };
  const disposables = [];
  const track = (o) => { disposables.push(o); return o; };

  // ---------- lights ----------
  scene.add(new THREE.HemisphereLight(0x7dd3fc, 0x070b16, 0.55));
  const key = new THREE.DirectionalLight(0xe4f7ff, 1.5);
  key.position.set(4, 6, 5);
  scene.add(key);
  const rim = new THREE.PointLight(0x38d6f5, 34, 14, 1.6);
  rim.position.set(-3.2, 2.2, -2.6);
  scene.add(rim);
  const fill = new THREE.PointLight(0x4ff5c6, 10, 10, 1.6);
  fill.position.set(3, -0.4, 3);
  scene.add(fill);
  const warm = new THREE.PointLight(0xf7b955, 0, 10, 1.4);
  warm.position.set(0.4, 2.6, 1.8);
  scene.add(warm);

  // ---------- floor: grid, rings, soft shadow ----------
  const floorY = -0.95;
  const grid = new THREE.GridHelper(40, 80, 0x38d6f5, 0x1b3a55);
  grid.position.y = floorY - 0.01;
  grid.material.transparent = true;
  grid.material.opacity = 0.12;
  grid.material.depthWrite = false;
  scene.add(grid);

  const shadowTex = (() => {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d");
    const grad = g.createRadialGradient(64, 64, 4, 64, 64, 64);
    grad.addColorStop(0, "rgba(0,0,0,0.75)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    return track(new THREE.CanvasTexture(c));
  })();
  const shadow = new THREE.Mesh(track(new THREE.PlaneGeometry(3.6, 2.6)),
    track(new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false })));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = floorY;

  const rings = [1.7, 2.2, 2.75].map((r, i) => {
    const ring = new THREE.Mesh(
      track(new THREE.RingGeometry(r, r + (i === 1 ? 0.012 : 0.022), 96, 1, 0, i === 0 ? Math.PI * 1.6 : Math.PI * 2)),
      track(new THREE.MeshBasicMaterial({ color: i === 2 ? 0x4ff5c6 : 0x38d6f5, transparent: true, opacity: 0.5,
        side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false })));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = floorY + 0.005;
    return ring;
  });

  // ---------- the cooler ----------
  const cooler = new THREE.Group();
  const shell = new THREE.Group(); // bobbing + spinning part
  cooler.add(shell);
  scene.add(cooler);
  shell.add(shadow);
  rings.forEach((r) => shell.add(r));

  const bodyMat = track(new THREE.MeshPhysicalMaterial({
    color: 0x10243b, metalness: 0.55, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.2,
    emissive: 0x38d6f5, emissiveIntensity: 0.04,
  }));
  const lidMat = track(bodyMat.clone());
  lidMat.color.set(0x16304c);
  const W = 2.0, H = 1.25, D = 1.3;
  const body = new THREE.Mesh(track(new RoundedBoxGeometry(W, H, D, 4, 0.1)), bodyMat);
  body.position.y = floorY + 0.12 + H / 2;
  shell.add(body);

  const edgeMat = track(new THREE.LineBasicMaterial({ color: 0x38d6f5, transparent: true, opacity: 0.85 }));
  const bodyEdges = new THREE.LineSegments(track(new THREE.EdgesGeometry(new THREE.BoxGeometry(W + 0.02, H + 0.02, D + 0.02))), edgeMat);
  bodyEdges.position.copy(body.position);
  shell.add(bodyEdges);

  const glowMat = track(new THREE.MeshBasicMaterial({ color: 0x38d6f5, toneMapped: false }));
  const strip = new THREE.Mesh(track(new THREE.BoxGeometry(W + 0.04, 0.045, D + 0.04)), glowMat);
  strip.position.set(0, body.position.y + 0.32, 0);
  shell.add(strip);

  // lid on a hinge along the back top edge
  const hinge = new THREE.Group();
  hinge.position.set(0, body.position.y + H / 2, -D / 2);
  shell.add(hinge);
  const lid = new THREE.Mesh(track(new RoundedBoxGeometry(W + 0.06, 0.22, D + 0.04, 3, 0.08)), lidMat);
  lid.position.set(0, 0.11, D / 2);
  hinge.add(lid);
  const lidEdges = new THREE.LineSegments(track(new THREE.EdgesGeometry(new THREE.BoxGeometry(W + 0.08, 0.24, D + 0.06))), edgeMat);
  lidEdges.position.copy(lid.position);
  hinge.add(lidEdges);
  const handle = new THREE.Mesh(track(new THREE.TorusGeometry(0.34, 0.045, 12, 36, Math.PI)), glowMat);
  handle.position.set(0, 0.22, D / 2);
  hinge.add(handle);

  // interior glow (visible when the lid opens)
  const insideMat = track(new THREE.MeshBasicMaterial({ color: 0xf7b955, transparent: true, opacity: 0, toneMapped: false }));
  const inside = new THREE.Mesh(track(new THREE.PlaneGeometry(W - 0.2, D - 0.2)), insideMat);
  inside.rotation.x = -Math.PI / 2;
  inside.position.set(0, body.position.y + H / 2 - 0.02, 0);
  shell.add(inside);

  // front display (live canvas texture)
  const displayCanvas = document.createElement("canvas");
  displayCanvas.width = 512;
  displayCanvas.height = 224;
  const displayTex = track(new THREE.CanvasTexture(displayCanvas));
  displayTex.colorSpace = THREE.SRGBColorSpace;
  const display = new THREE.Mesh(track(new THREE.PlaneGeometry(0.98, 0.43)),
    track(new THREE.MeshBasicMaterial({ map: displayTex, toneMapped: false, transparent: true })));
  display.position.set(-0.42, body.position.y + 0.02, D / 2 + 0.012);
  shell.add(display);
  let displayText = { label: options.label || "THRESHOLD", value: options.value || "8 °C" };
  let displayColorKey = "";

  function drawDisplay(color, dim) {
    const key = `${displayText.label}|${displayText.value}|${color.getHexString()}|${dim.toFixed(1)}`;
    if (key === displayColorKey) return;
    displayColorKey = key;
    const g = displayCanvas.getContext("2d");
    g.clearRect(0, 0, 512, 224);
    g.fillStyle = "rgba(3, 10, 22, 0.92)";
    g.beginPath();
    g.roundRect(4, 4, 504, 216, 26);
    g.fill();
    g.strokeStyle = `#${color.getHexString()}`;
    g.globalAlpha = 0.7 * dim + 0.2;
    g.lineWidth = 5;
    g.stroke();
    g.globalAlpha = 0.55 + 0.45 * dim;
    g.fillStyle = "#9db0cf";
    g.font = "600 34px 'JetBrains Mono', Consolas, monospace";
    g.fillText(displayText.label.toUpperCase(), 34, 66);
    g.fillStyle = `#${color.getHexString()}`;
    g.shadowColor = `#${color.getHexString()}`;
    g.shadowBlur = 24 * dim;
    g.font = "700 96px 'JetBrains Mono', Consolas, monospace";
    g.fillText(displayText.value, 30, 180);
    g.shadowBlur = 0;
    g.globalAlpha = 1;
    displayTex.needsUpdate = true;
  }

  // medical cross and side vents
  const crossMat = track(new THREE.MeshBasicMaterial({ color: 0x4ff5c6, toneMapped: false }));
  const crossV = new THREE.Mesh(track(new THREE.BoxGeometry(0.11, 0.4, 0.02)), crossMat);
  const crossH = new THREE.Mesh(track(new THREE.BoxGeometry(0.4, 0.11, 0.02)), crossMat);
  [crossV, crossH].forEach((m) => { m.position.set(0.56, body.position.y + 0.02, D / 2 + 0.015); shell.add(m); });
  for (let i = 0; i < 4; i++) {
    const vent = new THREE.Mesh(track(new THREE.BoxGeometry(0.02, 0.04, 0.8)), edgeMat);
    vent.position.set(W / 2 + 0.012, body.position.y - 0.25 + i * 0.12, 0);
    shell.add(vent);
  }
  [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sz]) => {
    const foot = new THREE.Mesh(track(new THREE.CylinderGeometry(0.07, 0.08, 0.12, 16)), bodyMat);
    foot.position.set(sx * (W / 2 - 0.18), floorY + 0.06, sz * (D / 2 - 0.18));
    shell.add(foot);
  });

  // ---------- particles ----------
  function makePoints(count, color, size, spread) {
    const pos = new Float32Array(count * 3);
    const seed = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * spread[0];
      pos[i * 3 + 1] = Math.random() * spread[1];
      pos[i * 3 + 2] = (Math.random() - 0.5) * spread[2];
      seed[i] = Math.random();
    }
    const geo = track(new THREE.BufferGeometry());
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const mat = track(new THREE.PointsMaterial({ color, size, transparent: true, opacity: 0.8,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true }));
    const pts = new THREE.Points(geo, mat);
    pts.userData = { seed, spread };
    return pts;
  }
  const frost = makePoints(mobile ? 40 : 90, 0xa5ecff, 0.045, [2.2, 2.0, 1.4]);
  frost.position.y = body.position.y + H / 2;
  shell.add(frost);
  const heat = makePoints(mobile ? 50 : 120, 0xf7b955, 0.05, [9, 4, 6]);
  heat.position.y = floorY;
  cooler.add(heat);
  const stars = (() => {
    const n = mobile ? 220 : 480;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const r = 16 + Math.random() * 18, th = Math.random() * Math.PI * 2, ph = Math.acos(Math.random() * 2 - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
      pos[i * 3 + 1] = Math.abs(r * Math.cos(ph)) * 0.6 - 2;
      pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
    }
    const geo = track(new THREE.BufferGeometry());
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    return new THREE.Points(geo, track(new THREE.PointsMaterial({ color: 0x9fdcf5, size: 0.06, transparent: true, opacity: 0.55, depthWrite: false })));
  })();
  scene.add(stars);

  // ---------- the journey route ----------
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-7.5, floorY, 2.2), new THREE.Vector3(-4.5, floorY, -1.6), new THREE.Vector3(-1.4, floorY, 1.6),
    new THREE.Vector3(1.8, floorY, -1.3), new THREE.Vector3(4.8, floorY, 1.5), new THREE.Vector3(7.8, floorY, -0.6),
  ]);
  const tubeGeo = track(new THREE.TubeGeometry(curve, 360, 0.04, 8));
  const routeBase = new THREE.Mesh(tubeGeo, track(new THREE.MeshBasicMaterial({ color: 0x38d6f5, transparent: true, opacity: 0, toneMapped: false })));
  const routeDone = new THREE.Mesh(tubeGeo.clone(), track(new THREE.MeshBasicMaterial({ color: 0x4ff5c6, transparent: true, opacity: 0, toneMapped: false })));
  track(routeDone.geometry);
  const routeIndexCount = routeDone.geometry.index.count;
  scene.add(routeBase, routeDone);
  const waypointMat = track(new THREE.MeshBasicMaterial({ color: 0x38d6f5, transparent: true, opacity: 0, toneMapped: false }));
  const waypoints = [0.04, 0.36, 0.66, 0.96].map((t) => {
    const g = new THREE.Group();
    const p = curve.getPointAt(t);
    g.position.copy(p);
    const pillar = new THREE.Mesh(track(new THREE.CylinderGeometry(0.03, 0.03, 0.9, 8)), waypointMat);
    pillar.position.y = 0.45;
    const cap = new THREE.Mesh(track(new THREE.SphereGeometry(0.1, 16, 16)), waypointMat);
    cap.position.y = 0.95;
    const halo = new THREE.Mesh(track(new THREE.RingGeometry(0.22, 0.26, 40)), waypointMat);
    halo.rotation.x = -Math.PI / 2;
    halo.position.y = 0.01;
    g.add(pillar, cap, halo);
    scene.add(g);
    return g;
  });

  // ---------- temperature column (risk) ----------
  const column = new THREE.Group();
  column.position.set(1.9, floorY, 0.2);
  scene.add(column);
  const COL_H = 2.6, THRESH_AT = 0.65; // threshold sits at 65% of the column
  const glass = new THREE.Mesh(track(new THREE.CylinderGeometry(0.17, 0.17, COL_H, 32, 1, true)),
    track(new THREE.MeshPhysicalMaterial({ color: 0x9fdcf5, transparent: true, opacity: 0, roughness: 0.1, side: THREE.DoubleSide })));
  glass.position.y = COL_H / 2;
  const fillMat = track(new THREE.MeshBasicMaterial({ color: 0x4ff5c6, transparent: true, opacity: 0, toneMapped: false }));
  const fillMesh = new THREE.Mesh(track(new THREE.CylinderGeometry(0.12, 0.12, 1, 24)), fillMat);
  const threshMat = track(new THREE.MeshBasicMaterial({ color: 0xf7b955, transparent: true, opacity: 0, side: THREE.DoubleSide, toneMapped: false, depthWrite: false }));
  const threshRing = new THREE.Mesh(track(new THREE.RingGeometry(0.3, 0.36, 48)), threshMat);
  threshRing.rotation.x = -Math.PI / 2;
  threshRing.position.y = COL_H * THRESH_AT;
  const threshPlaneMat = track(new THREE.MeshBasicMaterial({ color: 0xf7b955, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  const threshPlane = new THREE.Mesh(track(new THREE.PlaneGeometry(6, 3)), threshPlaneMat);
  threshPlane.rotation.x = -Math.PI / 2;
  threshPlane.position.set(-1.9, COL_H * THRESH_AT, -0.2);
  column.add(glass, fillMesh, threshRing, threshPlane);

  // ---------- ML model core (how it works) ----------
  const core = new THREE.Group();
  core.position.set(1.0, 0.55, 0);
  scene.add(core);
  const coreWireMat = track(new THREE.LineBasicMaterial({ color: 0x38d6f5, transparent: true, opacity: 0 }));
  const coreWire = new THREE.LineSegments(track(new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(0.95, 1))), coreWireMat);
  const coreInnerMat = track(new THREE.MeshStandardMaterial({ color: 0x0d2a40, emissive: 0x38d6f5, emissiveIntensity: 0.9, transparent: true, opacity: 0, roughness: 0.4 }));
  const coreInner = new THREE.Mesh(track(new THREE.IcosahedronGeometry(0.55, 3)), coreInnerMat);
  const outCls = new THREE.Mesh(track(new THREE.SphereGeometry(0.2, 24, 24)), track(new THREE.MeshBasicMaterial({ color: 0x38d6f5, transparent: true, opacity: 0, toneMapped: false })));
  const outReg = new THREE.Mesh(track(new THREE.SphereGeometry(0.2, 24, 24)), track(new THREE.MeshBasicMaterial({ color: 0xf7b955, transparent: true, opacity: 0, toneMapped: false })));
  outCls.position.set(2.1, 0.95, 0);
  outReg.position.set(2.1, -0.75, 0);
  core.add(coreWire, coreInner, outCls, outReg);
  const STREAM = mobile ? 40 : 80;
  const streamPos = new Float32Array(STREAM * 3);
  const streamCol = new Float32Array(STREAM * 3);
  const streamGeo = track(new THREE.BufferGeometry());
  streamGeo.setAttribute("position", new THREE.BufferAttribute(streamPos, 3));
  streamGeo.setAttribute("color", new THREE.BufferAttribute(streamCol, 3));
  const streamMat = track(new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  const stream = new THREE.Points(streamGeo, streamMat);
  core.add(stream);
  const streamStart = new THREE.Vector3(-2.2, -0.2, 0); // core-local: the small cooler at world x ≈ -1.2

  // ---------- confusion matrix bars (model performance) ----------
  const matrix = new THREE.Group();
  matrix.position.set(0, floorY, 0);
  scene.add(matrix);
  const counts = options.counts || null;
  const bars = [];
  if (counts) {
    const max = Math.max(counts.tp, counts.fp, counts.fn, counts.tn, 1);
    const spec = [
      ["tp", -0.75, -0.75, COLORS.mint], ["fn", 0.75, -0.75, COLORS.coral],
      ["fp", -0.75, 0.75, COLORS.amber], ["tn", 0.75, 0.75, COLORS.cyan],
    ];
    const plate = new THREE.Mesh(track(new THREE.BoxGeometry(3.4, 0.05, 3.4)), track(new THREE.MeshStandardMaterial({ color: 0x0c1a2c, transparent: true, opacity: 0 })));
    plate.position.y = 0.025;
    matrix.add(plate);
    bars.push({ mesh: plate, height: 0, isPlate: true });
    for (const [name, x, z, color] of spec) {
      const h = 0.2 + 2.6 * Math.sqrt(counts[name] / max);
      const mat = track(new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, roughness: 0.35, metalness: 0.2, transparent: true, opacity: 0 }));
      const bar = new THREE.Mesh(track(new THREE.BoxGeometry(1, 1, 1)), mat);
      bar.position.set(x, 0, z);
      matrix.add(bar);
      bars.push({ mesh: bar, height: h });
    }
  }

  // ---------- frame loop ----------
  const clock = new THREE.Clock();
  const edgeColor = new THREE.Color();
  const tmp = new THREE.Vector3();
  const home = new THREE.Vector3();
  let raf = null;
  let running = false;

  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function updateParticles(points, dt, speed, wrapHeight) {
    const p = points.geometry.attributes.position;
    const { seed, spread } = points.userData;
    for (let i = 0; i < p.count; i++) {
      let y = p.getY(i) + dt * speed * (0.4 + seed[i]);
      if (y > wrapHeight) {
        y = 0;
        p.setX(i, (Math.random() - 0.5) * spread[0]);
        p.setZ(i, (Math.random() - 0.5) * spread[2]);
      }
      p.setY(i, y);
    }
    p.needsUpdate = true;
  }

  function frame() {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;
    if (damping) {
      const k = 1 - Math.pow(0.0025, dt);
      for (const name in goal) state[name] += (goal[name] - state[name]) * k;
    }
    pointer.sx += (pointer.x - pointer.sx) * 0.05;
    pointer.sy += (pointer.y - pointer.sy) * 0.05;
    const s = state;

    // camera
    camera.position.set(s.camX + pointer.sx * 0.35, s.camY - pointer.sy * 0.25, s.camZ);
    camera.lookAt(s.lookX, s.lookY, s.lookZ);

    // colour: cyan -> mint (safe) -> amber (warmth) -> coral (danger), dimmed by power loss
    edgeColor.copy(COLORS.cyan).lerp(COLORS.mint, s.safe).lerp(COLORS.amber, s.warmth).lerp(COLORS.coral, s.danger);
    edgeColor.lerp(COLORS.off, 1 - s.power);
    edgeMat.color.copy(edgeColor);
    glowMat.color.copy(edgeColor);
    rim.color.copy(edgeColor);
    rim.intensity = 34 * (0.25 + 0.75 * s.power);
    fill.intensity = 10 * s.power;
    warm.intensity = 26 * Math.max(s.warmth * 0.8, s.heat, s.lid * 0.6);
    bodyMat.emissive.copy(edgeColor);
    lidMat.emissive.copy(edgeColor);
    bodyMat.emissiveIntensity = 0.05 * s.power;
    crossMat.color.copy(COLORS.mint).lerp(COLORS.off, 1 - s.power);
    drawDisplay(edgeColor, s.power);
    rings.forEach((r, i) => {
      r.material.opacity = 0.45 * s.rings * (0.35 + 0.65 * s.power);
      r.material.color.copy(i === 2 ? COLORS.mint : edgeColor);
      r.rotation.z = t * (i % 2 ? -0.12 : 0.18) * s.power;
    });

    // cooler pose: home position, or travelling along the route
    home.set(s.coolerX, s.coolerY, 0);
    if (s.route > 0.001) {
      curve.getPointAt(Math.min(0.999, Math.max(0.001, s.routeT)), tmp);
      tmp.y -= floorY; // cooler origin sits at floor level offset
      cooler.position.lerpVectors(home, tmp, s.route);
    } else {
      cooler.position.copy(home);
    }
    const scale = THREE.MathUtils.lerp(s.coolerScale, 0.34, s.route);
    cooler.scale.setScalar(Math.max(scale, 0.0001));
    cooler.visible = scale > 0.01;
    shell.position.y = Math.sin(t * 1.2) * 0.05 * s.spin;
    shell.rotation.y = -0.55 + Math.sin(t * 0.25) * 0.35 * s.spin + pointer.sx * 0.25;
    hinge.rotation.x = -s.lid * 1.15;
    insideMat.opacity = s.lid * 0.85;
    insideMat.color.copy(COLORS.amber).lerp(COLORS.cyan, s.power * (1 - s.warmth) * 0.4);

    // particles
    frost.material.opacity = 0.85 * s.frost * s.power;
    if (frost.material.opacity > 0.01) updateParticles(frost, dt, 0.35, 2.0);
    heat.material.opacity = 0.75 * s.heat;
    if (heat.material.opacity > 0.01) updateParticles(heat, dt, 0.5, 4.0);
    stars.rotation.y = t * 0.01;

    // route
    routeBase.material.opacity = 0.35 * s.route;
    routeDone.material.opacity = 0.95 * s.route;
    routeDone.geometry.setDrawRange(0, Math.floor((routeIndexCount * s.routeT) / 3) * 3);
    waypointMat.opacity = 0.9 * s.route;
    waypoints.forEach((w, i) => { w.visible = s.route > 0.01; w.children[1].position.y = 0.95 + Math.sin(t * 2 + i) * 0.05; });
    routeBase.visible = routeDone.visible = s.route > 0.01;

    // temperature column
    column.visible = s.column > 0.01;
    if (column.visible) {
      glass.material.opacity = 0.16 * s.column;
      const level = Math.max(0.02, s.columnFill) * COL_H * 0.95;
      fillMesh.scale.y = level;
      fillMesh.position.y = level / 2;
      fillMat.opacity = s.column;
      fillMat.color.copy(COLORS.mint).lerp(COLORS.amber, smooth(0.35, 0.62, s.columnFill)).lerp(COLORS.coral, smooth(0.63, 0.68, s.columnFill));
      threshMat.opacity = 0.9 * s.column;
      threshPlaneMat.opacity = 0.07 * s.column;
    }

    // model core and data streams
    core.visible = s.core > 0.01;
    if (core.visible) {
      coreWireMat.opacity = 0.7 * s.core;
      coreInnerMat.opacity = 0.9 * s.core;
      coreInner.material.emissiveIntensity = 0.7 + Math.sin(t * 3) * 0.25;
      coreWire.rotation.y = t * 0.3;
      coreWire.rotation.x = t * 0.15;
      outCls.material.opacity = outReg.material.opacity = s.core;
      streamMat.opacity = 0.95 * s.core;
      for (let i = 0; i < STREAM; i++) {
        const u = (t * 0.22 + i / STREAM) % 1;
        const toCls = i % 2 === 0;
        const out = toCls ? outCls.position : outReg.position;
        if (u < 0.5) {
          tmp.lerpVectors(streamStart, coreInner.position, u * 2);
          streamCol.set([0.65, 0.92, 1], i * 3);
        } else {
          tmp.lerpVectors(coreInner.position, out, (u - 0.5) * 2);
          streamCol.set(toCls ? [0.22, 0.84, 0.96] : [0.97, 0.73, 0.33], i * 3);
        }
        tmp.y += Math.sin(u * Math.PI * 2 + i) * 0.06;
        streamPos.set([tmp.x, tmp.y, tmp.z], i * 3);
      }
      streamGeo.attributes.position.needsUpdate = true;
      streamGeo.attributes.color.needsUpdate = true;
    }

    // confusion matrix
    matrix.visible = s.matrix > 0.01 && bars.length > 0;
    if (matrix.visible) {
      for (const b of bars) {
        b.mesh.material.opacity = (b.isPlate ? 0.8 : 0.92) * s.matrix;
        if (!b.isPlate) {
          const h = Math.max(0.001, b.height * s.matrix);
          b.mesh.scale.set(0.9, h, 0.9);
          b.mesh.position.y = 0.05 + h / 2;
        }
      }
      matrix.rotation.y = -0.6 + Math.sin(t * 0.2) * 0.12;
    }

    renderer.render(scene, camera);
  }

  function start() { if (!running) { running = true; clock.getDelta(); frame(); } }
  function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = null; }

  resize();
  window.addEventListener("resize", resize);
  const onVisibility = () => (document.hidden ? stop() : start());
  document.addEventListener("visibilitychange", onVisibility);
  start();

  return {
    state,
    goal,
    mobile,
    setDamping(on) { damping = on; },
    setPointer(x, y) { pointer.x = x; pointer.y = y; },
    setDisplay(label, value) { displayText = { label, value }; displayColorKey = ""; },
    onContextLost(cb) { canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); stop(); cb(); }, { once: true }); },
    dispose() {
      stop();
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
      disposables.forEach((d) => d.dispose && d.dispose());
      renderer.dispose();
    },
  };
}

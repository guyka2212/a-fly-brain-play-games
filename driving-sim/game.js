/* ============================================================================
 * driving-sim/game.js — three.js driving game for the connectome-seeded fly brain.
 *
 * Two modes (start screen, or ?mode=human|fly in the URL):
 *   Human Play — arrow keys / WASD steer and control speed.
 *   Watch the Fly Brain Play — shared/fly-brain.js drives, learning live via
 *   REINFORCE: +0.02/frame on road, +0.03 centred, +1 per checkpoint, −0.05 off
 *   road, −1 and episode end on crash.
 *
 * Rendering is three.js from CDN (no bundler); the track is procedural: the
 * centreline x-offset is a fixed function of distance travelled, so both the
 * ribbon mesh and the agent's "curve ahead / curve sign" sensors derive from
 * the same curve and can never disagree.
 * ========================================================================== */
(function () {
  "use strict";

  /* Bail out cleanly if a CDN script failed (offline preview etc.). */
  if (typeof THREE === "undefined") {
    const note = document.getElementById("load-note");
    if (note) note.textContent = "three.js failed to load from CDN — check your connection.";
    return;
  }

  /* ============================ track geometry ============================ */
  const ROAD = 90;
  /* Centreline x-offset from road centre as a function of distance travelled.
     Max |slope| = 90/260 + 50/97 ≈ 0.86 px/px — the agent's steer speed (260)
     must beat worst-case drift 0.86 × vmax 240 ≈ 206 px/s. It does. */
  function centreX(d) { return 60 + 90 * Math.sin(d / 260) + 50 * Math.sin(d / 97); }
  function centreXPrime(d) { return (90 / 260) * Math.cos(d / 260) + (50 / 97) * Math.cos(d / 97); }

  function carState(s) {
    const cx = centreX(s.y);
    return { cx, laneOffset: s.x - cx, onRoad: Math.abs(s.x - cx) < ROAD / 2 - 14 };
  }
  function newSim() {
    return { x: centreX(0), y: 0, v: 130, lane: 0, checkpoint: 0, alive: true, t: 0 };
  }
  let sim = newSim();

  /* ============================ fly-brain wiring ========================== */
  const FEATURES = ["laneOffset", "vNorm", "curveAhead", "curveSign"];
  const ACTIONS = ["steerL", "hold", "steerR"];

  let mode = null;             // null | 'human' | 'fly'
  let started = false, paused = false;
  let brainReady = false;
  let lastProbs = [1 / 3, 1 / 3, 1 / 3];
  let lastActionName = null;

  function stateVector(s) {
    const st = carState(s);
    const curve0 = centreX(s.y + 40), curve1 = centreX(s.y + 110);
    const d1 = curve0 - st.cx, d2 = curve1 - st.cx;
    return [
      Math.max(-1, Math.min(1, st.laneOffset / (ROAD / 2))),
      s.v / 260,
      Math.max(0, Math.min(1, (d2 - d1) / 60 + 0.5)),
      Math.max(-1, Math.min(1, d1 / 60)),
    ];
  }

  async function ensureBrain() {
    if (brainReady) return true;
    if (typeof window.flyBrain === "undefined") return false;
    try {
      const pools = await window.flyBrain.init({ features: FEATURES, actions: ACTIONS });
      console.log("fly-brain ready:", pools);
      buildProbRows(); buildNeuronRows();
      brainReady = true;
      return true;
    } catch (e) {
      console.error("fly-brain init failed:", e);
      return false;
    }
  }

  /* ================================ input ================================= */
  const keys = {};
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    keys[k] = true;
    if (["arrowleft", "arrowright", "arrowup", "arrowdown", " "].includes(k)) e.preventDefault();
    if (k === "r" && started) restart();
    if (k === "escape") toggleMenu();
  });
  addEventListener("keyup", (e) => { keys[e.key.toLowerCase()] = false; });

  function applyAction(name, dt) {
    if (name === "steerL") sim.lane = -1;
    else if (name === "steerR") sim.lane = 1;
    else sim.lane = 0;
    const up = keys["arrowup"] || keys["w"], down = keys["arrowdown"] || keys["s"];
    if (up) sim.v = Math.min(240, sim.v + 160 * dt);
    else if (down) sim.v = Math.max(70, sim.v - 220 * dt);
  }

  function stepSim(dt) {
    sim.x += sim.lane * 260 * dt;
    sim.y += sim.v * dt;
    if (sim.y > (sim.checkpoint + 1) * 600) { sim.checkpoint++; reward(1); }
    sim.t += dt;
  }

  /* ============================ rewards (fly) ============================= */
  function reward(v) { if (mode === "fly" && brainReady) window.flyBrain.reward(v); }

  function crash() {
    sim.alive = false;
    if (mode === "fly" && brainReady) { reward(-1); window.flyBrain.endEpisode(); }
    const el = document.getElementById("status");
    el.textContent = mode === "fly"
      ? "Crashed — episode ended, the fly brain just learned something."
      : "Crashed! Press R to try again.";
    el.classList.add("crashed");
    setTimeout(() => { if (!sim.alive && started && !paused) restart(); }, mode === "fly" ? 400 : 1200);
  }

  function restart() {
    if (mode === "fly" && brainReady && sim.alive && sim.t > 0.5) window.flyBrain.endEpisode();
    sim = newSim();
    const el = document.getElementById("status");
    el.textContent = ""; el.classList.remove("crashed");
  }

  /* ============================== three.js ================================ */
  const wrap = document.getElementById("canvas-wrap");
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10151c);
  scene.fog = new THREE.Fog(0x10151c, 260, 950);

  const camera = new THREE.PerspectiveCamera(60, 3 / 2, 0.5, 2200);

  scene.add(new THREE.HemisphereLight(0xbfd4e8, 0x0c0f13, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7);
  sun.position.set(120, 300, 80);
  scene.add(sun);

  /* ground + grid that follows the car, so the world reads as infinite */
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(4000, 4000),
    new THREE.MeshLambertMaterial({ color: 0x0b0f14 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.2;
  scene.add(ground);
  const grid = new THREE.GridHelper(3000, 60, 0x1c2733, 0x16202b);
  scene.add(grid);

  /* road ribbon: two flat strips (shoulder + asphalt), rebuilt as the car
     advances. Fixed vertex/index counts; only positions update. */
  const AHEAD = 820, BEHIND = 160, STEP = 10;
  const NSEG = Math.ceil((AHEAD + BEHIND) / STEP);
  function makeRibbon(halfW, color, y) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array((NSEG + 1) * 6), 3));
    const nor = new Float32Array((NSEG + 1) * 6);
    for (let i = 0; i <= NSEG; i++) { nor[i * 6 + 1] = 1; nor[i * 6 + 4] = 1; }
    geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    const idx = [];
    for (let i = 0; i < NSEG; i++) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    geo.setIndex(idx);
    const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide }));
    mesh.position.y = y;
    mesh.frustumCulled = false;
    mesh.userData.halfW = halfW;
    scene.add(mesh);
    return mesh;
  }
  const shoulder = makeRibbon(ROAD / 2 + 9, 0x1a2230, -0.06);
  const asphalt = makeRibbon(ROAD / 2, 0x232d3a, 0);

  function rebuildRibbon(mesh) {
    const posAttr = mesh.geometry.getAttribute("position");
    const d0 = sim.y - BEHIND, hw = mesh.userData.halfW;
    for (let i = 0; i <= NSEG; i++) {
      const d = d0 + i * STEP, cx = centreX(d);
      posAttr.setXYZ(i * 2, cx - hw, 0, -d);
      posAttr.setXYZ(i * 2 + 1, cx + hw, 0, -d);
    }
    posAttr.needsUpdate = true;
  }

  /* centre-line dashes */
  const dashes = [];
  const dashGeo = new THREE.BoxGeometry(2.5, 0.1, 16);
  const dashMat = new THREE.MeshBasicMaterial({ color: 0x4a5a6e });
  for (let i = 0; i < 16; i++) {
    const m = new THREE.Mesh(dashGeo, dashMat);
    scene.add(m); dashes.push(m);
  }
  function layoutDashes() {
    const first = Math.floor((sim.y - BEHIND) / 55) * 55;
    for (let i = 0; i < dashes.length; i++) {
      const d = first + i * 55;
      dashes[i].position.set(centreX(d), 0.05, -d);
      dashes[i].rotation.y = Math.atan2(centreXPrime(d), -1);
    }
  }

  /* next-checkpoint gate */
  const gateMat = new THREE.MeshBasicMaterial({ color: 0x3fb950 });
  const gate = new THREE.Group();
  const pL = new THREE.Mesh(new THREE.BoxGeometry(4, 22, 4), gateMat);
  const pR = new THREE.Mesh(new THREE.BoxGeometry(4, 22, 4), gateMat);
  const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 2.5, 1), gateMat);
  pL.position.set(-ROAD / 2, 11, 0);
  pR.position.set(ROAD / 2, 11, 0);
  bar.scale.set(ROAD + 4, 1, 1);
  bar.position.set(0, 21, 0);
  gate.add(pL, pR, bar);
  scene.add(gate);

  /* the car */
  const car = new THREE.Group();
  const bodyMat = new THREE.MeshLambertMaterial({ color: 0x58a6ff });
  const body = new THREE.Mesh(new THREE.BoxGeometry(16, 7, 30), bodyMat);
  body.position.y = 6;
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(10, 5, 13),
    new THREE.MeshLambertMaterial({ color: 0x0d1117 })
  );
  cabin.position.set(0, 11.5, -2);
  car.add(body, cabin);
  scene.add(car);

  /* fly mode: perceived lane-offset line from car to road centre */
  const laneLine = new THREE.Mesh(
    new THREE.BoxGeometry(1, 0.15, 2.5),
    new THREE.MeshBasicMaterial({ color: 0xa371f7, transparent: true, opacity: 0.55 })
  );
  laneLine.position.y = 0.15;
  scene.add(laneLine);

  /* yaw that points a -Z-forward object along the track tangent */
  function trackYaw(d) { return Math.atan2(-centreXPrime(d), 1); }

  function layoutWorld() {
    const cs = carState(sim);
    car.position.set(sim.x, 0, -sim.y);
    car.rotation.y = trackYaw(sim.y) + (sim.lane || 0) * -0.08;
    bodyMat.color.set(sim.alive ? (mode === "fly" ? 0xa371f7 : 0x58a6ff) : 0xf85149);

    ground.position.z = -sim.y;
    grid.position.z = Math.round(-sim.y / 50) * 50;

    rebuildRibbon(shoulder);
    rebuildRibbon(asphalt);
    layoutDashes();

    const dcp = (sim.checkpoint + 1) * 600;
    gate.position.set(centreX(dcp), 0, -dcp);
    gate.rotation.y = trackYaw(dcp);

    laneLine.visible = mode === "fly" && brainReady && sim.alive;
    if (laneLine.visible) {
      laneLine.scale.set(Math.max(0.5, Math.abs(sim.x - cs.cx)), 1, 1);
      laneLine.position.x = (sim.x + cs.cx) / 2;
      laneLine.position.z = -sim.y;
    }
  }

  function chaseCamera() {
    camera.position.set(sim.x * 0.55, 78, -sim.y + 150);
    camera.lookAt(sim.x * 0.8, 4, -sim.y - 90);
  }
  function attractCamera(t) {
    const a = t * 0.25;
    camera.position.set(sim.x + Math.sin(a) * 130, 60, -sim.y + Math.cos(a) * 130);
    camera.lookAt(sim.x, 6, -sim.y);
  }

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
  }
  addEventListener("resize", resize);
  resize();

  /* ================================= HUD ================================== */
  function buildProbRows() {
    const el = document.getElementById("probs");
    if (!el) return;
    el.innerHTML = "";
    for (const a of ACTIONS) {
      const row = document.createElement("div");
      row.className = "prob"; row.id = "prob-" + a;
      row.innerHTML = `<span class="name">${a}</span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
      el.appendChild(row);
    }
  }
  function buildNeuronRows() {
    const el = document.getElementById("neurons");
    if (!el) return;
    el.innerHTML = "";
    for (let i = 0; i < 10; i++) {
      const row = document.createElement("div");
      row.className = "neuron"; row.id = "neu-" + i;
      row.innerHTML = `<span class="name"></span><span class="role"></span><span class="track"><span class="fill"></span></span><span class="val"></span>`;
      el.appendChild(row);
    }
  }

  function updateHUD() {
    const stats = brainReady && mode === "fly" ? window.flyBrain.getStats() : null;
    const rows = [
      ["mode", mode === "fly" ? "🧠 fly brain" : mode === "human" ? "🎮 human" : "—"],
      ["episode", stats ? stats.episode : "—"],
      ["avg score", stats ? stats.avgScore : "—"],
      ["last score", stats ? stats.lastScore : "—"],
      ["distance", Math.round(sim.y) + " m"],
      ["checkpoints", sim.checkpoint],
      ["speed", Math.round(sim.v)],
    ];
    document.getElementById("stats").innerHTML =
      rows.map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");

    if (brainReady && mode === "fly") {
      const probs = lastProbs;
      const best = probs.indexOf(Math.max.apply(null, probs));
      ACTIONS.forEach((a, i) => {
        const row = document.getElementById("prob-" + a);
        if (!row) return;
        row.classList.toggle("best", i === best);
        row.querySelector(".fill").style.width = (probs[i] * 100).toFixed(1) + "%";
        row.querySelector(".pct").textContent = (probs[i] * 100).toFixed(0) + "%";
      });
      window.flyBrain.getActivity(10).forEach((n, i) => {
        const row = document.getElementById("neu-" + i);
        if (!row) return;
        row.className = "neuron " + n.role;
        row.querySelector(".name").textContent = n.name;
        row.querySelector(".role").textContent = n.role;
        row.querySelector(".fill").style.width = Math.min(100, Math.abs(n.value) * 60) + "%";
        row.querySelector(".val").textContent = n.value.toFixed(2);
      });
    }
  }

  /* learning curve: episode dots + EWA average line (the spec's "score /
     learning-curve chart" for Watch mode) */
  const chartCv = document.getElementById("chart");
  function drawChart() {
    if (!chartCv) return;
    const ctx2 = chartCv.getContext("2d");
    const W = chartCv.width, H = chartCv.height;
    ctx2.clearRect(0, 0, W, H);
    if (!brainReady || mode !== "fly") return;
    const hist = window.flyBrain.getStats().history;
    if (!hist.length) return;
    const show = hist.slice(-120);
    let lo = Math.min(0, Math.min.apply(null, show.map((h) => h.score)));
    let hi = Math.max(1, Math.max.apply(null, show.map((h) => h.score)));
    const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
    const X = (i) => 8 + (W - 16) * (i / Math.max(1, show.length - 1));
    const Y = (v) => H - 8 - (H - 16) * ((v - lo) / (hi - lo));
    if (lo < 0 && hi > 0) {
      ctx2.strokeStyle = "#30363d"; ctx2.lineWidth = 1;
      ctx2.beginPath(); ctx2.moveTo(8, Y(0)); ctx2.lineTo(W - 8, Y(0)); ctx2.stroke();
    }
    ctx2.fillStyle = "#8b949e";
    show.forEach((h, i) => { ctx2.beginPath(); ctx2.arc(X(i), Y(h.score), 1.8, 0, Math.PI * 2); ctx2.fill(); });
    let avg = 0;
    ctx2.strokeStyle = "#58a6ff"; ctx2.lineWidth = 2; ctx2.beginPath();
    show.forEach((h, i) => {
      avg += (h.score - avg) * 0.15;
      if (i === 0) ctx2.moveTo(X(i), Y(avg)); else ctx2.lineTo(X(i), Y(avg));
    });
    ctx2.stroke();
  }

  /* ============================ mode switch =============================== */
  const bHuman = document.getElementById("btn-human"), bFly = document.getElementById("btn-fly");
  const startScreen = document.getElementById("start-screen");
  const note = document.getElementById("load-note");

  async function startMode(m) {
    if (m === "fly") {
      note.textContent = "loading fly brain…";
      const ok = await ensureBrain();
      if (!ok) { note.textContent = "fly brain failed to load — Human Play still works."; return; }
      window.flyBrain.setMode("train");
    }
    if (mode === "fly" && brainReady && sim.alive && sim.t > 0.5) window.flyBrain.endEpisode();
    mode = m; started = true; paused = false;
    sim = newSim();
    startScreen.style.display = "none";
    bHuman.classList.toggle("on", m === "human");
    bFly.classList.toggle("on", m === "fly");
    document.getElementById("hint").style.visibility = m === "human" ? "visible" : "hidden";
    document.getElementById("status").textContent = m === "fly"
      ? "The fly is driving. It learns from every crash and checkpoint."
      : "";
    note.textContent = "";
  }

  function toggleMenu() {
    if (!started) return;
    paused = !paused;
    startScreen.style.display = paused ? "flex" : "none";
    if (paused) note.textContent = "paused — pick a mode to continue";
  }

  bHuman.onclick = () => startMode("human");
  bFly.onclick = () => startMode("fly");
  document.getElementById("btn-restart").onclick = restart;
  /* Honest reset = fresh page: weights live in page memory by design. */
  document.getElementById("btn-reset").onclick = () => location.reload();
  document.getElementById("start-human").onclick = () => startMode("human");
  document.getElementById("start-fly").onclick = () => startMode("fly");

  /* Spec: read a URL query param OR the in-page toggle for mode. */
  const qmode = new URLSearchParams(location.search).get("mode");
  if (qmode === "fly" || qmode === "ai") startMode("fly");
  else if (qmode === "human") startMode("human");

  /* ============================== main loop =============================== */
  let lastTs = 0, hudLast = 0, chartLast = 0;
  function frame(ts) {
    const dt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;

    if (started && !paused && sim.alive) {
      if (mode === "fly" && brainReady) {
        const a = window.flyBrain.act(stateVector(sim));
        lastActionName = a;
        lastProbs = window.flyBrain.getActionProbs() || lastProbs;
        applyAction(a, dt);
        /* Cap episodes so the REINFORCE replay stays fast when the fly gets good. */
        if (sim.t > 60) { reward(2); window.flyBrain.endEpisode(); sim = newSim(); }
      } else if (mode === "human") {
        const left = keys["arrowleft"] || keys["a"], right = keys["arrowright"] || keys["d"];
        applyAction(left ? "steerL" : right ? "steerR" : "hold", dt);
        lastActionName = null;
      }
      stepSim(dt);

      const st = carState(sim);
      if (!st.onRoad) { reward(-0.05); }
      else {
        reward(0.02);
        if (Math.abs(st.laneOffset) < ROAD * 0.15) reward(0.03);
      }
      if (!st.onRoad && Math.abs(st.laneOffset) > ROAD / 2 + 4) crash();
    }

    layoutWorld();
    if (started && !paused) chaseCamera(); else attractCamera(ts / 1000);
    renderer.render(scene, camera);

    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

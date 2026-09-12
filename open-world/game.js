/* ============================================================================
 * open-world/game.js — three.js foraging sandbox for the fly brain.
 *
 * AI-ONLY: there is no human control mode. The fly brain (shared/fly-brain.js)
 * forages and learns live via REINFORCE; you watch, fast-forward, and reset.
 * The camera is still draggable — that's spectating, not control.
 *
 * The agent samples turn/forward actions each step (10 Hz) and learns via
 * REINFORCE: reward for closing distance to the nearest orb, +1 per orb
 * collected, small exploration bonus, small step cost. Episode = 45 s, then
 * the world re-seeds.
 *
 * World: 240×240 arena, walls at the edges, 8 orbs per episode, fog + grid for
 * depth. Grid-visit tracking gives the "area explored" signal from the spec.
 * ========================================================================== */
(function () {
  "use strict";

  if (typeof THREE === "undefined") {
    const note = document.getElementById("load-note");
    if (note) note.textContent = "three.js failed to load from CDN — check your connection.";
    return;
  }

  /* ============================ game constants ============================ */
  const HALF = 120;            // arena half-size
  const STEP_TIME = 0.1;       // agent decision period (s) -> 10 Hz of sim time
  const EPISODE_TIME = 45;     // episode length (s of sim time)
  const N_ORBS = 8;
  const WALK = 26;             // world units/s
  const TURN = 2.6;            // rad/s

  /* ============================ fly-brain wiring ========================== */
  const FEATURES = [
    "orbDist",     // distance to nearest orb (1 near .. 0 far)
    "orbBearing",  // sin of bearing to nearest orb relative to heading
    "orbBearing2", // cos of bearing (sign disambiguation)
    "wallDist",    // distance to nearest wall ahead (1 clear .. 0 close)
  ];
  const ACTIONS = ["forward", "turnL", "turnR", "forwardLeft", "forwardRight"];

  /* Speed multipliers for the simulation clock (1x / 4x / 16x). */
  const SPEEDS = [1, 4, 16];
  let speedIdx = 0;
  let started = false, paused = false;
  let brainReady = false;
  let lastProbs = [0.2, 0.2, 0.2, 0.2, 0.2];
  let lastActionName = null;
  let bestScore = -Infinity;

  /* ============================== game state ============================== */
  let agent, orbs, visited, visitedCount, epClock, collected;
  const GRID = 40;             // visitation grid resolution per axis
  const CELL = (HALF * 2) / GRID;

  function resetWorld() {
    agent = { x: 0, z: 0, h: Math.random() * Math.PI * 2, bumpT: 0 };
    orbs = [];
    for (let i = 0; i < N_ORBS; i++) {
      /* keep orbs away from spawn centre */
      let ox, oz;
      do {
        ox = (Math.random() * 2 - 1) * (HALF - 12);
        oz = (Math.random() * 2 - 1) * (HALF - 12);
      } while (ox * ox + oz * oz < 30 * 30);
      orbs.push({ x: ox, z: oz, taken: false, t: Math.random() * Math.PI * 2 });
    }
    visited = new Uint8Array(GRID * GRID);
    visitedCount = 0;
    epClock = 0;
    collected = 0;
  }

  function nearestOrb() {
    let best = null, bd = 1e9;
    for (const o of orbs) {
      if (o.taken) continue;
      const dx = o.x - agent.x, dz = o.z - agent.z;
      const d2 = dx * dx + dz * dz;
      if (d2 < bd) { bd = d2; best = o; }
    }
    return best === null ? null : { orb: best, dist: Math.sqrt(bd) };
  }

  function stateVector() {
    const near = nearestOrb();
    let orbDist = 0, bearingSin = 0, bearingCos = 0;
    if (near) {
      orbDist = Math.max(0, Math.min(1, 1 - near.dist / (HALF * 1.5)));
      const dx = near.orb.x - agent.x, dz = near.orb.z - agent.z;
      /* world angle of the orb, relative to heading h. Agent forward is
         (sin h, -cos h) matching three.js default -Z forward on yaw h. */
      const worldAng = Math.atan2(dx, -dz);
      const rel = worldAng - agent.h;
      bearingSin = Math.max(-1, Math.min(1, Math.sin(rel)));
      bearingCos = Math.max(-1, Math.min(1, Math.cos(rel)));
    }
    /* wall distance along heading */
    const fx = Math.sin(agent.h), fz = -Math.cos(agent.h);
    const tWallX = fx > 0 ? (HALF - agent.x) / fx : fx < 0 ? (-HALF - agent.x) / fx : 1e9;
    const tWallZ = fz > 0 ? (HALF - agent.z) / fz : fz < 0 ? (-HALF - agent.z) / fz : 1e9;
    const tWall = Math.max(0, Math.min(1, Math.min(tWallX, tWallZ) / HALF));
    return [orbDist, bearingSin, bearingCos, tWall];
  }

  function moveAgent(turn, forwardFrac, dt) {
    agent.h += turn * TURN * dt;
    const fx = Math.sin(agent.h), fz = -Math.cos(agent.h);
    const sp = WALK * forwardFrac;
    let nx = agent.x + fx * sp * dt, nz = agent.z + fz * sp * dt;
    let bumped = false;
    if (nx < -HALF || nx > HALF) { nx = Math.max(-HALF, Math.min(HALF, nx)); bumped = true; }
    if (nz < -HALF || nz > HALF) { nz = Math.max(-HALF, Math.min(HALF, nz)); bumped = true; }
    agent.x = nx; agent.z = nz;
    if (bumped) agent.bumpT = 0.2;

    /* exploration tracking */
    const gx = Math.floor((agent.x + HALF) / CELL), gz = Math.floor((agent.z + HALF) / CELL);
    const gi = Math.min(GRID - 1, Math.max(0, gz)) * GRID + Math.min(GRID - 1, Math.max(0, gx));
    if (!visited[gi]) { visited[gi] = 1; visitedCount++; return bumped ? 2 : 1; }
    return bumped ? 3 : 0;   // 1 = new cell, 3 = bumped, 2 = both, 0 = plain step
  }

  function collectCheck() {
    for (const o of orbs) {
      if (o.taken) continue;
      const dx = o.x - agent.x, dz = o.z - agent.z;
      if (dx * dx + dz * dz < 7 * 7) {
        o.taken = true;
        collected++;
        return true;
      }
    }
    return false;
  }

  async function ensureBrain() {
    if (brainReady) return true;
    if (typeof window.flyBrain === "undefined") return false;
    try {
      const pools = await window.flyBrain.init({ features: FEATURES, actions: ACTIONS });
      console.log("fly-brain ready:", pools);
      buildProbRows(); buildNeuronRows();
      updateBrainBadge();
      brainReady = true;
      return true;
    } catch (e) {
      console.error("fly-brain init failed:", e);
      const note = document.getElementById("load-note");
      if (note) note.textContent = "fly brain failed to load — see console.";
      return false;
    }
  }

  /* On-screen honest data-source badge, wired from flyBrain.getDataSource(). */
  function updateBrainBadge() {
    const el = document.getElementById("brain-badge");
    if (!el) return;
    const src = window.flyBrain.getDataSource();
    if (!src) { el.textContent = "brain: source unknown"; el.className = "badge warn"; return; }
    if (src.synthetic) {
      el.textContent = `brain: ⚠ synthetic fallback — connectome-data.json failed to load (${src.nNeurons} neurons)`;
      el.className = "badge warn";
    } else {
      el.textContent = `brain: ${src.dataset} (${src.nNeurons} neurons, ${src.nEdges} edges)`;
      el.className = "badge ok";
    }
  }

  /* ============================== three.js ================================ */
  const wrap = document.getElementById("canvas-wrap");
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e13);
  scene.fog = new THREE.Fog(0x0b0e13, 80, 340);

  const camera = new THREE.PerspectiveCamera(60, 3 / 2, 0.5, 1200);

  scene.add(new THREE.HemisphereLight(0xbfd4e8, 0x0c0f13, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 0.6);
  sun.position.set(120, 260, 80);
  scene.add(sun);

  /* ground + follow grid */
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(HALF * 2, HALF * 2),
    new THREE.MeshLambertMaterial({ color: 0x10161f })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const grid = new THREE.GridHelper(HALF * 2, GRID, 0x1c2733, 0x16202b);
  grid.position.y = 0.05;
  scene.add(grid);

  /* walls */
  const wallMat = new THREE.MeshLambertMaterial({ color: 0x1a2230 });
  const wallH = 8, wallT = 3;
  [[0, -HALF - wallT / 2, HALF * 2 + wallT * 2, wallT],
   [0, HALF + wallT / 2, HALF * 2 + wallT * 2, wallT],
   [-HALF - wallT / 2, 0, wallT, HALF * 2 + wallT * 2],
   [HALF + wallT / 2, 0, wallT, HALF * 2 + wallT * 2]].forEach(([x, z, sx, sz]) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, wallH, sz), wallMat);
    m.position.set(x, wallH / 2, z);
    scene.add(m);
  });

  /* agent: a little "fly rover". Blender-built body (fly-rover.glb, nose at
     -Z, ~2.5 units) with a primitive sphere+whisker fallback so the page
     always runs. The whisker stays visible in both cases — it makes heading
     legible at orbit-camera distance. */
  const agentGroup = new THREE.Group();
  const bodyMat = new THREE.MeshLambertMaterial({ color: 0xa371f7 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(3.2, 20, 14), bodyMat);
  body.scale.set(1, 0.7, 1.3);
  body.position.y = 3;
  const whisker = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.5, 7),
    new THREE.MeshBasicMaterial({ color: 0xe6edf3 })
  );
  whisker.position.set(0, 3, -5);
  agentGroup.add(body, whisker);
  scene.add(agentGroup);
  if (typeof flyAssets !== "undefined") {
    flyAssets.load("fly-rover").then((g) => {
      g.scale.setScalar(2.6);
      g.position.y = 3;
      agentGroup.add(g);
      body.visible = false;                    // primitive fallback hidden
    }).catch((e) => console.warn("fly-rover.glb unavailable — using primitives", e));
  }

  /* orbs (pooled meshes) */
  const orbMat = new THREE.MeshBasicMaterial({ color: 0x3fb950, transparent: true, opacity: 0.85 });
  const orbMeshes = [];
  for (let i = 0; i < N_ORBS; i++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 12), orbMat);
    m.visible = false;
    scene.add(m);
    orbMeshes.push(m);
  }
  /* soft rings under orbs so they read at distance */
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x3fb950, transparent: true, opacity: 0.25 });
  const ringMeshes = [];
  for (let i = 0; i < N_ORBS; i++) {
    const m = new THREE.Mesh(new THREE.RingGeometry(4.2, 5.4, 24), ringMat);
    m.rotation.x = -Math.PI / 2;
    m.position.y = 0.15;
    m.visible = false;
    scene.add(m);
    ringMeshes.push(m);
  }

  /* collect burst: pooled point cloud, spawned where an orb is taken */
  const BURST_N = 20;
  const bursts = [];
  function spawnBurst(x, z) {
    let b = bursts.find((b) => b.t <= 0);
    if (!b) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
      const mat = new THREE.PointsMaterial({ color: 0x7ee787, size: 2.2, transparent: true, opacity: 1 });
      const pts = new THREE.Points(geo, mat);
      pts.visible = false;
      scene.add(pts);
      b = { pts, geo, mat, t: 0 };
      bursts.push(b);
    }
    b.t = 0.6;
    b.mat.color.set(0x7ee787);
    const p = b.geo.getAttribute("position");
    b.vel = new Array(BURST_N).fill(0).map(() => {
      const a = Math.random() * Math.PI * 2, r = 8 + Math.random() * 14;
      return { vx: Math.cos(a) * r, vy: 10 + Math.random() * 12, vz: Math.sin(a) * r };
    });
    for (let i = 0; i < BURST_N; i++) p.setXYZ(i, x, 6, z);
    p.needsUpdate = true;
  }
  function layoutBursts(dt) {
    for (const b of bursts) {
      if (b.t <= 0) { b.pts.visible = false; continue; }
      b.t -= dt;
      b.pts.visible = true;
      b.mat.opacity = Math.max(0, b.t / 0.6);
      const p = b.geo.getAttribute("position");
      for (let i = 0; i < BURST_N; i++) {
        const v = b.vel[i];
        p.setXYZ(i, p.getX(i) + v.vx * dt, p.getY(i) + v.vy * dt, p.getZ(i) + v.vz * dt);
        v.vy -= 30 * dt;
      }
      p.needsUpdate = true;
    }
  }

  /* trail: breadcrumbs so the foraging path reads at a glance */
  const TRAIL_N = 60;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
  const trailMat = new THREE.PointsMaterial({ color: 0xa371f7, size: 1.4, transparent: true, opacity: 0.4 });
  const trail = new THREE.Points(trailGeo, trailMat);
  trail.frustumCulled = false;
  scene.add(trail);
  let trailIdx = 0, trailAcc = 0;

  function setAgentColor(hex) {
    bodyMat.color.set(hex);
    agentGroup.traverse((o) => {
      if (o.isMesh && o.material && o.material.color && o !== whisker) o.material.color.set(hex);
    });
  }

  function layoutScene(t, dt) {
    agentGroup.position.set(agent.x, 0, agent.z);
    agentGroup.rotation.y = agent.h;
    setAgentColor(agent.bumpT > 0 ? 0xf85149 : 0xa371f7);
    if (agent.bumpT > 0) agent.bumpT -= dt;

    for (let i = 0; i < orbs.length; i++) {
      const o = orbs[i];
      orbMeshes[i].visible = !o.taken;
      ringMeshes[i].visible = !o.taken;
      if (!o.taken) {
        const bob = Math.sin(t * 2 + o.t) * 0.8;
        orbMeshes[i].position.set(o.x, 6 + bob, o.z);
        orbMeshes[i].scale.setScalar(1 + Math.sin(t * 3 + o.t) * 0.08);
        ringMeshes[i].position.x = o.x;
        ringMeshes[i].position.z = o.z;
      }
    }

    /* breadcrumb trail (sim-time spaced) */
    trailAcc += dt;
    if (trailAcc > 0.25) {
      trailAcc = 0;
      const p = trailGeo.getAttribute("position");
      p.setXYZ(trailIdx % TRAIL_N, agent.x, 1.2, agent.z);
      trailIdx++;
      p.needsUpdate = true;
    }

    layoutBursts(dt);
  }

  /* orbit camera (drag to spectate) + follow */
  let camYaw = 0, camPitch = 0.55, camDist = 90, dragging = false, lastX = 0, lastY = 0;
  wrap.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  addEventListener("pointerup", () => { dragging = false; });
  addEventListener("pointermove", (e) => {
    if (!dragging) return;
    camYaw -= (e.clientX - lastX) * 0.005;
    camPitch = Math.max(0.15, Math.min(1.25, camPitch + (e.clientY - lastY) * 0.004));
    lastX = e.clientX; lastY = e.clientY;
  });
  function updateCamera() {
    const cx = agent.x + Math.sin(camYaw) * Math.cos(camPitch) * camDist;
    const cz = agent.z + Math.cos(camYaw) * Math.cos(camPitch) * camDist;
    const cy = 6 + Math.sin(camPitch) * camDist;
    camera.position.set(cx, cy, cz);
    camera.lookAt(agent.x, 4, agent.z);
  }

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
  }
  addEventListener("resize", resize);
  resize();

  /* ============================ rewards (fly) ============================= */
  function flyStep() {
    const st = stateVector();
    const a = window.flyBrain.act(st);
    lastActionName = a;
    lastProbs = window.flyBrain.getActionProbs() || lastProbs;

    let turn = 0, fwd = 0;
    if (a === "turnL") turn = 1;
    else if (a === "turnR") turn = -1;
    else if (a === "forward") fwd = 1;
    else if (a === "forwardLeft") { turn = 0.6; fwd = 0.8; }
    else if (a === "forwardRight") { turn = -0.6; fwd = 0.8; }

    const near = nearestOrb();
    const ev = moveAgent(turn, fwd, STEP_TIME);
    const gotOrb = collectCheck();
    const near2 = nearestOrb();

    /* reward shaping */
    let r = -0.01;                                   // step cost
    if (near && near2) {
      const closer = (near.dist - near2.dist);       // + if closing in
      r += Math.max(-0.05, Math.min(0.05, closer * 0.08));
    }
    if (ev === 1 || ev === 2) r += 0.02;             // explored new ground
    if (ev >= 2) r -= 0.05;                          // wall bump
    if (gotOrb) r += 1;

    window.flyBrain.reward(r);
    if (gotOrb) spawnBurst(agent.x, agent.z);
  }

  function endEpisodeIfDue(force) {
    const done = force || epClock >= EPISODE_TIME || collected >= N_ORBS;
    if (done) {
      if (brainReady) window.flyBrain.endEpisode();
      if (brainReady) {
        const s = window.flyBrain.getStats().lastScore;
        if (s > bestScore) bestScore = s;
      }
      resetWorld();
    }
    return done;
  }

  /* Honest reset: rebuild the network from the seeded initialization. */
  function resetBrain() {
    window.flyBrain.reset();
    bestScore = -Infinity;
    resetWorld();
    const p = trailGeo.getAttribute("position");
    for (let i = 0; i < TRAIL_N; i++) p.setXYZ(i, 0, -50, 0);
    p.needsUpdate = true;
    const el = document.getElementById("status");
    el.textContent = "Brain reset — training starts from scratch.";
    drawChart(true);
  }

  function restart() {
    if (brainReady && started) window.flyBrain.endEpisode();
    resetWorld();
    document.getElementById("status").textContent = "";
  }

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
    const stats = brainReady ? window.flyBrain.getStats() : null;
    const explored = Math.round(100 * visitedCount / (GRID * GRID));
    const rows = [
      ["episode", stats ? stats.episode : "—"],
      ["last score", stats ? stats.lastScore : "—"],
      ["avg score", stats ? stats.avgScore : "—"],
      ["best score", bestScore > -Infinity ? bestScore : "—"],
      ["orbs", collected + " / " + N_ORBS],
      ["explored", explored + "%"],
      ["episode time", Math.max(0, Math.ceil(EPISODE_TIME - epClock)) + "s"],
    ];
    document.getElementById("stats").innerHTML =
      rows.map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");

    if (brainReady) {
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

  /* learning-curve chart */
  const chartCv = document.getElementById("chart");
  function drawChart(force) {
    if (!chartCv) return;
    const ctx2 = chartCv.getContext("2d");
    const W = chartCv.width, H = chartCv.height;
    ctx2.clearRect(0, 0, W, H);
    if (!brainReady) return;
    const hist = window.flyBrain.getStats().history;
    const show = force ? [] : hist.slice(-120);
    if (!show.length) return;
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

  /* ============================ playback controls ========================= */
  const bSpeed = document.getElementById("btn-speed");
  const bPause = document.getElementById("btn-pause");
  const bRestart = document.getElementById("btn-restart");
  const bReset = document.getElementById("btn-reset");
  const startScreen = document.getElementById("start-screen");
  const note = document.getElementById("load-note");

  async function start() {
    note.textContent = "loading fly brain…";
    const ok = await ensureBrain();
    if (!ok) { note.textContent = "fly brain failed to load — see console."; return; }
    window.flyBrain.setMode("train");
    started = true; paused = false;
    resetWorld();
    startScreen.style.display = "none";
    note.textContent = "";
    document.getElementById("status").textContent =
      "The fly is foraging. Episodes are 45s of sim time; it learns each one.";
  }

  bSpeed.onclick = () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    bSpeed.textContent = `⏩ ${SPEEDS[speedIdx]}x`;
    bSpeed.classList.toggle("on", speedIdx > 0);
  };
  bPause.onclick = () => {
    paused = !paused;
    bPause.textContent = paused ? "▶ Resume" : "⏸ Pause";
    bPause.classList.toggle("on", paused);
  };
  bRestart.onclick = restart;
  bReset.onclick = resetBrain;
  document.getElementById("start-fly").onclick = start;

  /* ============================== main loop =============================== */
  /* The agent decides every STEP_TIME of SIMULATED time (10 Hz of sim time,
     regardless of the speed multiplier) — fast-forward yields more episodes,
     not a different decision cadence. */
  let lastTs = 0, hudLast = 0, chartLast = 0, stepAcc = 0;
  function frame(ts) {
    const rdt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;
    const dt = rdt * SPEEDS[speedIdx];
    const t = ts / 1000;

    if (started && !paused) {
      epClock += dt;
      stepAcc += dt;
      while (stepAcc >= STEP_TIME) {
        stepAcc -= STEP_TIME;
        flyStep();
      }
      endEpisodeIfDue(false);
    }

    layoutScene(t, dt);
    updateCamera();
    renderer.render(scene, camera);
    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(false); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

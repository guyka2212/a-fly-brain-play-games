/* ============================================================================
 * driving-sim/game.js — three.js driving game for the connectome-seeded fly brain.
 *
 * AI-ONLY: there is no human control mode. The fly brain (shared/fly-brain.js)
 * drives and learns live via REINFORCE; you watch, fast-forward, and reset.
 *
 * Rewards (fly): +0.02/frame on road, +0.03 centred, +1 per checkpoint, +2 for
 * surviving the 60s time cap, −0.05 off road, −1 and episode end on crash.
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
    return { x: centreX(0), y: 0, lane: 0, checkpoint: 0, alive: true, t: 0 };
  }
  let sim = newSim();

  /* ============================ fly-brain wiring ========================== */
  /* Sensor-visibility encoding: all four features are signed and zero-centred
     so the tuned sensors (sign*gain*x + bias, relu) respond continuously in
     both directions. The two curve probes (40u and 110u ahead) replace the old
     0..1 curveAhead, which sat near 0.5 and left its sensors permanently
     silent. */
  const FEATURES = ["laneOffset", "vNorm", "curveAheadNear", "curveAheadFar"];
  const ACTIONS = ["steerL", "hold", "steerR"];

  /* Speed multipliers for the simulation clock (1x / 4x / 16x). */
  const SPEEDS = [1, 4, 16];
  let speedIdx = 0;
  let started = false, paused = false;
  let brainReady = false;
  let lastProbs = [1 / 3, 1 / 3, 1 / 3];
  let lastActionName = null;
  let bestScore = -Infinity;

  function stateVector(s) {
    const st = carState(s);
    const curve0 = centreX(s.y + 40), curve1 = centreX(s.y + 110);
    const d1 = curve0 - st.cx, d2 = curve1 - st.cx;
    return [
      Math.max(-1, Math.min(1, st.laneOffset / (ROAD / 2))),
      130 / 260,                                  // constant cruise speed, normalised
      Math.max(-1, Math.min(1, (d2 - d1) / 60)),  // signed: >0 = curving left ahead
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

  /* On-screen honest data-source badge, wired from flyBrain.getDataSource()
     (never hardcoded): curated connectome vs ⚠ synthetic fallback. */
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

  /* ============================ fly actions =============================== */
  function applyAction(name) {
    if (name === "steerL") sim.lane = -1;
    else if (name === "steerR") sim.lane = 1;
    else sim.lane = 0;
  }

  function stepSim(dt) {
    sim.x += sim.lane * 260 * dt;
    sim.y += 130 * dt;                 // constant cruise speed (AI drives only)
    sim.t += dt;
  }

  /* ============================ rewards (fly) ============================= */
  function reward(v) { if (brainReady) window.flyBrain.reward(v); }

  /* Near-miss: road edge is ROAD/2 − 14; record when the fly runs within 6
     units of it so the watch loop can flash the shoulder (no reward effect). */
  function nearMiss() {
    const off = Math.abs(sim.x - centreX(sim.y));
    return off > ROAD / 2 - 20;
  }

  function crash() {
    sim.alive = false;
    reward(-1);
    window.flyBrain.endEpisode();
    updateBest();
    const el = document.getElementById("status");
    el.textContent = "Crashed — episode ended, the fly brain just learned something.";
    el.classList.add("crashed");
    setTimeout(() => { if (!sim.alive && started && !paused) restart(); }, 300);
  }

  function updateBest() {
    if (!brainReady) return;
    const s = window.flyBrain.getStats().lastScore;
    if (s > bestScore) bestScore = s;
  }

  /* Restart = end the current episode honestly (if one is open) and respawn.
     New road gets a fresh random phase so the fly can't overfit one shape. */
  function restart() {
    if (brainReady && sim.alive && sim.t > 0.5) window.flyBrain.endEpisode();
    sim = newSim();
    const el = document.getElementById("status");
    el.textContent = ""; el.classList.remove("crashed");
  }

  /* Honest reset: rebuild the network from the seeded initialization and wipe
     all training history (no page reload needed). */
  function resetBrain() {
    window.flyBrain.reset();
    bestScore = -Infinity;
    sim = newSim();
    const el = document.getElementById("status");
    el.textContent = "Brain reset — training starts from scratch.";
    el.classList.remove("crashed");
    drawChart(true);
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

  /* shoulder flash for near-misses: amber edge strips that light up when the
     car runs close to the road edge, red on a crash. */
  const flashRibbon = makeRibbon(ROAD / 2 + 9, 0xf85149, -0.05);
  flashRibbon.material.transparent = true;
  flashRibbon.material.opacity = 0;
  let flashT = 0;

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

  /* checkpoint pulse: the gate flashes green and a brief particle burst when
     a checkpoint is crossed. */
  let gatePulse = 0;
  const burstGeo = new THREE.BufferGeometry();
  const BURST_N = 26;
  burstGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
  const burstMat = new THREE.PointsMaterial({ color: 0x3fb950, size: 3.2, transparent: true, opacity: 0.9 });
  const burst = new THREE.Points(burstGeo, burstMat);
  burst.visible = false;
  scene.add(burst);
  let burstT = 0;
  const burstVel = new Array(BURST_N).fill(0).map(() => {
    const a = Math.random() * Math.PI * 2, r = 24 + Math.random() * 30;
    return { vx: Math.cos(a) * r, vz: Math.sin(a) * r, vy: 30 + Math.random() * 26 };
  });

  /* the car */
  const car = new THREE.Group();
  const bodyMat = new THREE.MeshLambertMaterial({ color: 0xa371f7 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(16, 7, 30), bodyMat);
  body.position.y = 6;
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(10, 5, 13),
    new THREE.MeshLambertMaterial({ color: 0x0d1117 })
  );
  cabin.position.set(0, 11.5, -2);
  car.add(body, cabin);
  scene.add(car);

  /* Blender-built car (tools/blender-assets): loads asynchronously and swaps
     in over the primitive placeholder; the placeholder stays if the asset
     fails to load (offline etc.). Model units: nose at -Z, ~4 long, so scale
     to the game's 30-unit car length. */
  if (typeof flyAssets !== "undefined") {
    flyAssets.load("car").then((g) => {
      g.scale.setScalar(7.5);
      g.position.y = 1;
      car.add(g);
      body.visible = false;
      cabin.visible = false;
    }).catch((e) => console.warn("car.glb unavailable — using primitives", e));
  }

  /* perceived lane-offset line from car to road centre */
  const laneLine = new THREE.Mesh(
    new THREE.BoxGeometry(1, 0.15, 2.5),
    new THREE.MeshBasicMaterial({ color: 0xa371f7, transparent: true, opacity: 0.55 })
  );
  laneLine.position.y = 0.15;
  scene.add(laneLine);

  /* yaw that points a -Z-forward object along the track tangent */
  function trackYaw(d) { return Math.atan2(-centreXPrime(d), 1); }

  /* crash flash: whole-body material swap works for both the primitive and
     the Blender mesh (tints the first mesh child). */
  function setCarColor(hex) {
    bodyMat.color.set(hex);
    car.traverse((o) => {
      if (o.isMesh && o.material && o.material.color) o.material.color.set(hex);
    });
  }

  function layoutWorld() {
    const cs = carState(sim);
    car.position.set(sim.x, 0, -sim.y);
    car.rotation.y = trackYaw(sim.y) + (sim.lane || 0) * -0.08;
    setCarColor(sim.alive ? 0xa371f7 : 0xf85149);

    ground.position.z = -sim.y;
    grid.position.z = Math.round(-sim.y / 50) * 50;

    rebuildRibbon(shoulder);
    rebuildRibbon(asphalt);
    rebuildRibbon(flashRibbon);
    layoutDashes();

    const dcp = (sim.checkpoint + 1) * 600;
    gate.position.set(centreX(dcp), 0, -dcp);
    gate.rotation.y = trackYaw(dcp);
    const gm = 1 + Math.max(0, gatePulse) * 0.5;
    gate.scale.set(gm, gm, 1);
    gateMat.color.set(gatePulse > 0 ? 0x7ee787 : 0x3fb950);
    if (gatePulse > 0) gatePulse -= 0.05;

    /* near-miss / crash shoulder flash */
    if (flashT > 0) {
      flashT -= 0.06;
      flashRibbon.material.opacity = Math.max(0, flashT) * 0.55;
    } else flashRibbon.material.opacity = 0;

    laneLine.visible = brainReady && sim.alive;
    if (laneLine.visible) {
      laneLine.scale.set(Math.max(0.5, Math.abs(sim.x - cs.cx)), 1, 1);
      laneLine.position.x = (sim.x + cs.cx) / 2;
      laneLine.position.z = -sim.y;
    }

    /* checkpoint particle burst */
    if (burstT > 0) {
      burstT -= 0.03;
      burst.visible = true;
      burstMat.opacity = Math.max(0, burstT) * 1.2;
      const p = burstGeo.getAttribute("position");
      for (let i = 0; i < BURST_N; i++) {
        const v = burstVel[i];
        p.setXYZ(i,
          gate.position.x + v.vx * (1 - burstT),
          4 + v.vy * (1 - burstT) - 40 * (1 - burstT) * (1 - burstT),
          gate.position.z + v.vz * (1 - burstT));
      }
      p.needsUpdate = true;
    } else burst.visible = false;
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
    const stats = brainReady ? window.flyBrain.getStats() : null;
    const rows = [
      ["episode", stats ? stats.episode : "—"],
      ["last score", stats ? stats.lastScore : "—"],
      ["avg score", stats ? stats.avgScore : "—"],
      ["best score", bestScore > -Infinity ? bestScore : "—"],
      ["distance", Math.round(sim.y) + " m"],
      ["checkpoints", sim.checkpoint],
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

  /* learning curve: episode dots + EWA average line, drawn prominently. */
  const chartCv = document.getElementById("chart");
  function drawChart(force) {
    if (!chartCv) return;
    const ctx2 = chartCv.getContext("2d");
    const W = chartCv.width, H = chartCv.height;
    ctx2.clearRect(0, 0, W, H);
    if (!brainReady) return;
    const hist = window.flyBrain.getStats().history;
    if (!hist.length) return;
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
    ctx2.strokeStyle = "#a371f7"; ctx2.lineWidth = 2; ctx2.beginPath();
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
    sim = newSim();
    startScreen.style.display = "none";
    note.textContent = "";
    document.getElementById("status").textContent =
      "The fly is driving. It learns from every crash and checkpoint.";
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
  let lastTs = 0, hudLast = 0, chartLast = 0, acc = 0;
  function frame(ts) {
    const rdt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;
    const dt = rdt * SPEEDS[speedIdx];

    if (started && !paused && sim.alive) {
      /* One brain decision per simulated tick (act stays ≤ 20 Hz per contract;
         at high speed multipliers we sub-step the brain in fixed 1/30 slices so
         episode length in decisions stays identical across speeds). */
      const H = 1 / 30;
      acc += dt;
      while (acc >= H) {
        acc -= H;
        const a = window.flyBrain.act(stateVector(sim));
        lastActionName = a;
        lastProbs = window.flyBrain.getActionProbs() || lastProbs;
        applyAction(a);
        stepSim(H);
        const st = carState(sim);
        if (!st.onRoad) reward(-0.05);
        else { reward(0.02); if (Math.abs(st.laneOffset) < ROAD * 0.15) reward(0.03); }
        if (sim.y > (sim.checkpoint + 1) * 600) { sim.checkpoint++; reward(1); gatePulse = 1; burstT = 1; }
        if (nearMiss()) flashT = Math.max(flashT, 0.35); else flashT = Math.max(0, flashT - 0.06);
        if (!st.onRoad && Math.abs(st.laneOffset) > ROAD / 2 + 4) { crash(); break; }
        if (sim.t > 60) { reward(2); window.flyBrain.endEpisode(); updateBest(); sim = newSim(); break; }
      }
    }

    layoutWorld();
    if (started && !paused) chaseCamera(); else attractCamera(ts / 1000);
    renderer.render(scene, camera);

    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(false); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

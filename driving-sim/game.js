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
  /* Each drive starts at a random point along the endless road (START_SPAN
     covers both curve periods). Always starting at distance 0 — the track's
     steepest bend — made a beginner crash within ~0.2 s every episode, so
     early training had almost no signal and often never took off.
     Checkpoints count distance from the start point (y0). */
  const START_SPAN = 3300;
  function newSim() {
    const y0 = Math.random() * START_SPAN;
    return { x: centreX(y0), y: y0, y0, lane: 0, checkpoint: 0, alive: true, t: 0 };
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
  let brainViz = null;
  const GAME_ID = "driving-sim";
  const SAVE_KEY = "fly-brain:" + GAME_ID + ":v1";       // this browser's saved fly

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
      if (typeof flyBrainViz !== "undefined") {
        brainViz = flyBrainViz.create({ container: document.getElementById("brain-panel") });
      }
      brainReady = true;
      window.flyBrain.enableAutosave(SAVE_KEY, 5);
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
     Each new drive starts at a random point on the road (see newSim), so
     the fly can't overfit one stretch. */
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
    window.flyBrain.clearSaved(SAVE_KEY);
    setBestPlay(false);
    if (brainViz) brainViz.rebuild();
    bestScore = -Infinity;
    sim = newSim();
    const el = document.getElementById("status");
    el.textContent = "Brain reset — training starts from scratch.";
    el.classList.remove("crashed");
    drawChart(true);
  }

  /* ============================== three.js ================================ */
  /* Look: dusk highway. Low sun ahead on a gradient sky, textured grass,
     painted asphalt with kerbs, roadside trees + reflector posts, a glowing
     checkpoint arch, car head/tail lights and soft shadows. Cosmetic only —
     nothing below feeds back into the sim, sensors or rewards. */
  const wrap = document.getElementById("canvas-wrap");
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const fx = window.flyFx;
  if (fx) fx.setupRenderer(renderer, { exposure: 1.05 });

  const SKY = { top: 0x1b2a6b, horizon: 0xff8e5e, bottom: 0x1a1622, curve: 0.28,
    sunDir: new THREE.Vector3(0.25, 0.06, -1), sunColor: 0xffb070, sunGlow: 0.9 };
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2236);
  scene.fog = new THREE.Fog(0x4a3448, 240, 1150);
  const sky = fx ? fx.skyDome(Object.assign({ radius: 1500 }, SKY)) : null;
  if (sky) scene.add(sky);
  const starField = fx ? fx.stars(500, 1400, 0.3) : null;
  if (starField) { starField.material.opacity = 0.55; scene.add(starField); }
  if (fx) scene.environment = fx.envFromSky(renderer, SKY);

  const camera = new THREE.PerspectiveCamera(58, 3 / 2, 0.5, 3200);

  scene.add(new THREE.HemisphereLight(0xa8b8ff, 0x3a2a20, 1.1));
  /* warm low sun from ahead-right; its shadow box follows the car */
  const sun = new THREE.DirectionalLight(0xffc08a, 2.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -140, right: 140, top: 140, bottom: -140, near: 10, far: 900 });
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.6;
  scene.add(sun, sun.target);
  /* cool fill from behind the camera so the backlit car still reads */
  const fill = new THREE.DirectionalLight(0xb8c8ff, 1.3);
  scene.add(fill, fill.target);

  /* deterministic per-slot hash for scenery placement (same every load) */
  const hash = (k) => { const x = Math.sin(k * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

  /* grass ground that follows the car, snapped to the texture tile so the
     pattern stays glued to the world instead of sliding with the camera */
  const TILE = 40;
  const grassTex = fx ? fx.noiseTexture({ base: 0x31452a, spread: 0.12, size: 256, density: 0.9, repeat: [100, 100], seed: 3 }) : null;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(TILE * 100, TILE * 100),
    new THREE.MeshStandardMaterial({ color: 0xffffff, map: grassTex, roughness: 0.95 })
  );
  if (!grassTex) ground.material.color.set(0x23301f);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.2;
  ground.receiveShadow = true;
  scene.add(ground);

  /* far mountain silhouettes: two layered ridges of unfogged low-poly peaks
     on a ring around the car (back ridge lighter = aerial perspective) */
  const mountains = new THREE.Group();
  const ridges = [
    { r: 1380, n: 64, h: [70, 210], color: 0x4a3a62 },
    { r: 1260, n: 52, h: [40, 140], color: 0x2a2542 },
  ];
  ridges.forEach((rg, ri) => {
    const mat = new THREE.MeshBasicMaterial({ color: rg.color, fog: false });
    for (let i = 0; i < rg.n; i++) {
      const a = (i / rg.n) * Math.PI * 2 + hash(i * 3.7 + ri) * 0.08;
      const h = rg.h[0] + hash(i * 1.3 + ri * 17) * (rg.h[1] - rg.h[0]);
      const m = new THREE.Mesh(new THREE.ConeGeometry(h * (1.6 + hash(i + 9) * 1.4), h, 4 + (i % 3)), mat);
      m.position.set(Math.cos(a) * rg.r, h / 2 - 14, Math.sin(a) * rg.r);
      m.rotation.y = hash(i * 5.1) * 6.28;
      mountains.add(m);
    }
  });
  scene.add(mountains);

  /* road ribbons (shoulder/kerb + asphalt + crash flash), rebuilt as the car
     advances. Fixed vertex/index counts; positions and UVs update; UV.v is
     world distance so the painted texture is glued to the road. */
  const AHEAD = 900, BEHIND = 160, STEP = 10;
  const NSEG = Math.ceil((AHEAD + BEHIND) / STEP);
  const V_PER = 60;                                  // world units per texture repeat
  function makeRibbon(halfW, mat, y) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array((NSEG + 1) * 6), 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array((NSEG + 1) * 4), 2));
    const nor = new Float32Array((NSEG + 1) * 6);
    for (let i = 0; i <= NSEG; i++) { nor[i * 6 + 1] = 1; nor[i * 6 + 4] = 1; }
    geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    const idx = [];
    for (let i = 0; i < NSEG; i++) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    geo.setIndex(idx);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = y;
    mesh.frustumCulled = false;
    mesh.receiveShadow = true;
    mesh.userData.halfW = halfW;
    scene.add(mesh);
    return mesh;
  }
  /* asphalt: grain + white edge lines + dashed centre line, painted once */
  const asphaltTex = fx ? fx.noiseTexture({
    base: 0x2c2f36, spread: 0.07, size: 256, density: 0.8, seed: 5,
    draw(g, n) {
      g.fillStyle = "#e9e4d8";
      g.fillRect(n * 0.035, 0, n * 0.022, n);
      g.fillRect(n * (1 - 0.057), 0, n * 0.022, n);
      g.fillStyle = "#f2c14e";
      g.fillRect(n * 0.49, 0, n * 0.02, n * 0.55);
    },
  }) : null;
  /* kerb: red/white bands along the road */
  const kerbTex = fx ? fx.noiseTexture({
    base: 0xd8d4cc, spread: 0.05, size: 128, density: 0.4, seed: 9,
    draw(g, n) { g.fillStyle = "#c9302c"; g.fillRect(0, 0, n, n / 4); g.fillRect(0, n / 2, n, n / 4); },
  }) : null;
  const shoulder = makeRibbon(ROAD / 2 + 7, new THREE.MeshStandardMaterial({
    color: kerbTex ? 0xffffff : 0x1a2230, map: kerbTex, roughness: 0.7, side: THREE.DoubleSide }), -0.06);
  const asphalt = makeRibbon(ROAD / 2, new THREE.MeshStandardMaterial({
    color: asphaltTex ? 0xffffff : 0x232d3a, map: asphaltTex, roughness: 0.82, metalness: 0.05,
    side: THREE.DoubleSide }), 0);
  shoulder.userData.vScale = 0.25;

  /* shoulder flash for near-misses (amber) and crashes (red) */
  const flashRibbon = makeRibbon(ROAD / 2 + 9, new THREE.MeshBasicMaterial({
    color: 0xff5a3c, side: THREE.DoubleSide, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false }), 0.05);
  flashRibbon.receiveShadow = false;
  let flashT = 0;

  function rebuildRibbon(mesh) {
    const posAttr = mesh.geometry.getAttribute("position");
    const uvAttr = mesh.geometry.getAttribute("uv");
    const d0 = sim.y - BEHIND, hw = mesh.userData.halfW;
    const vs = mesh.userData.vScale || 1;
    for (let i = 0; i <= NSEG; i++) {
      const d = d0 + i * STEP, cx = centreX(d);
      posAttr.setXYZ(i * 2, cx - hw, 0, -d);
      posAttr.setXYZ(i * 2 + 1, cx + hw, 0, -d);
      const v = (d / V_PER) * (1 / vs);
      uvAttr.setXY(i * 2, 0, v);
      uvAttr.setXY(i * 2 + 1, 1, v);
    }
    posAttr.needsUpdate = true;
    uvAttr.needsUpdate = true;
  }

  /* roadside props: instanced pines + reflector posts, placed per 40-unit
     slot along the track with a per-slot hash, so they stay put in the
     world as the window of visible road slides forward */
  const SLOT = 40, NSLOT = Math.ceil((AHEAD + BEHIND) / SLOT) + 1;
  const pineGeo = new THREE.ConeGeometry(9, 30, 7);
  pineGeo.translate(0, 21, 0);
  const trunkGeo = new THREE.CylinderGeometry(1.4, 1.8, 7, 6);
  trunkGeo.translate(0, 3.5, 0);
  const pines = new THREE.InstancedMesh(pineGeo, new THREE.MeshStandardMaterial({ color: 0x2f6b45, roughness: 0.85, flatShading: true }), NSLOT * 4);
  const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x3b2a1f, roughness: 1 }), NSLOT * 4);
  pines.castShadow = trunks.castShadow = true;
  pines.frustumCulled = trunks.frustumCulled = false;
  scene.add(pines, trunks);
  const postGeo = new THREE.BoxGeometry(1.2, 6, 1.2);
  postGeo.translate(0, 3, 0);
  const posts = new THREE.InstancedMesh(postGeo, new THREE.MeshStandardMaterial({ color: 0xe8e8e8, roughness: 0.6 }), NSLOT * 2);
  const reflGeo = new THREE.BoxGeometry(1.3, 1.1, 1.3);
  reflGeo.translate(0, 5.2, 0);
  const refls = new THREE.InstancedMesh(reflGeo, new THREE.MeshBasicMaterial({ color: 0xffa040 }), NSLOT * 2);
  posts.castShadow = true;
  posts.frustumCulled = refls.frustumCulled = false;
  scene.add(posts, refls);
  const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);
  function layoutProps() {
    const k0 = Math.floor((sim.y - BEHIND) / SLOT);
    let pi = 0, qi = 0;
    for (let j = 0; j < NSLOT; j++) {
      const k = k0 + j, d = k * SLOT, cx = centreX(d);
      for (let side = -1; side <= 1; side += 2) {
        /* reflector post right on the shoulder */
        tmpP.set(cx + side * (ROAD / 2 + 11), 0, -d);
        tmpM.compose(tmpP, tmpQ.identity(), tmpS.set(1, 1, 1));
        posts.setMatrixAt(qi, tmpM); refls.setMatrixAt(qi, tmpM); qi++;
        /* two pines per side per slot, scattered back from the road */
        for (let t = 0; t < 2; t++) {
          const h = hash(k * 4 + (side + 1) + t * 2);
          const off = ROAD / 2 + 32 + h * 120 + t * 70;
          const dz = (hash(k * 9 + t + side) - 0.5) * SLOT;
          const sc = 0.7 + hash(k * 3 + t - side) * 0.9;
          tmpP.set(centreX(d + dz) + side * off, 0, -(d + dz));
          tmpQ.setFromAxisAngle(UP, h * 6.28);
          tmpM.compose(tmpP, tmpQ, tmpS.set(sc, sc * (0.9 + h * 0.4), sc));
          pines.setMatrixAt(pi, tmpM); trunks.setMatrixAt(pi, tmpM); pi++;
        }
      }
    }
    pines.count = trunks.count = pi;
    posts.count = refls.count = qi;
    for (const im of [pines, trunks, posts, refls]) im.instanceMatrix.needsUpdate = true;
  }

  /* next-checkpoint arch: emissive green pylons + banner, with glow halos */
  const gateMat = new THREE.MeshStandardMaterial({ color: 0x1d6b33, emissive: 0x3fb950, emissiveIntensity: 1.4, roughness: 0.4 });
  const gate = new THREE.Group();
  const pL = new THREE.Mesh(new THREE.BoxGeometry(4, 26, 4), gateMat);
  const pR = new THREE.Mesh(new THREE.BoxGeometry(4, 26, 4), gateMat);
  const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 4, 1.5), gateMat);
  pL.position.set(-ROAD / 2 - 2, 13, 0);
  pR.position.set(ROAD / 2 + 2, 13, 0);
  bar.scale.set(ROAD + 8, 1, 1);
  bar.position.set(0, 25, 0);
  pL.castShadow = pR.castShadow = bar.castShadow = true;
  gate.add(pL, pR, bar);
  const gateGlows = [];
  if (fx) {
    for (const x of [-ROAD / 2 - 2, 0, ROAD / 2 + 2]) {
      const g = fx.glowSprite(0x5dff8a, x === 0 ? 70 : 34, 0.55);
      g.position.set(x, 26, 0);
      gate.add(g); gateGlows.push(g);
    }
  }
  scene.add(gate);

  /* checkpoint pulse: gate flares and a glowing particle burst */
  let gatePulse = 0;
  const burstGeo = new THREE.BufferGeometry();
  const BURST_N = 40;
  burstGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
  const burstMat = new THREE.PointsMaterial({ color: 0x7dff9e, size: 5, transparent: true, opacity: 0.9,
    map: fx ? fx.glowTexture() : null, blending: THREE.AdditiveBlending, depthWrite: false });
  const burst = new THREE.Points(burstGeo, burstMat);
  burst.visible = false;
  burst.frustumCulled = false;
  scene.add(burst);
  let burstT = 0;
  const burstVel = new Array(BURST_N).fill(0).map(() => {
    const a = Math.random() * Math.PI * 2, r = 24 + Math.random() * 40;
    return { vx: Math.cos(a) * r, vz: Math.sin(a) * r, vy: 30 + Math.random() * 36 };
  });

  /* the car (primitive placeholder until car.glb loads) */
  const car = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xa371f7, metalness: 0.3, roughness: 0.35 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(16, 7, 30), bodyMat);
  body.position.y = 6;
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(10, 5, 13),
    new THREE.MeshStandardMaterial({ color: 0x0d1117, metalness: 0.5, roughness: 0.15 })
  );
  cabin.position.set(0, 11.5, -2);
  body.castShadow = cabin.castShadow = true;
  car.add(body, cabin);
  scene.add(car);
  /* body-paint materials the crash tint may recolour (never glass/trim/lights) */
  const paintMats = [bodyMat];

  /* head + tail lights: glow halos plus one forward spotlight on the road */
  const lampGlows = [];
  if (fx) {
    for (const x of [-5.5, 5.5]) {
      const h = fx.glowSprite(0xfff1c8, 16, 0.9); h.position.set(x, 6, -16); car.add(h); lampGlows.push(h);
      const t = fx.glowSprite(0xff3030, 4.5, 0.6); t.position.set(x * 0.6, 5.5, 14); car.add(t);
    }
  }
  const headlight = new THREE.SpotLight(0xfff1d0, 900, 260, 0.42, 0.6, 1.6);
  headlight.position.set(0, 8, -12);
  headlight.target.position.set(0, 0, -120);
  car.add(headlight, headlight.target);

  /* Blender-built car (tools/blender-assets): loads asynchronously and swaps
     in over the primitive placeholder; the placeholder stays if the asset
     fails to load (offline etc.). Model units: nose at -Z, ~4 long, so scale
     to the game's 30-unit car length. */
  if (typeof flyAssets !== "undefined") {
    flyAssets.load("car").then((g) => {
      g.scale.setScalar(7.5);
      g.position.y = 0;
      car.add(g);
      body.visible = false;
      cabin.visible = false;
      g.traverse((o) => { if (o.isMesh && o.material.name === "car_body") paintMats.push(o.material); });
    }).catch((e) => console.warn("car.glb unavailable — using primitives", e));
  }

  /* soft contact shadow blob under the car (reads even outside the sun box) */
  let carBlob = null;
  if (fx) {
    carBlob = new THREE.Mesh(new THREE.PlaneGeometry(30, 44), new THREE.MeshBasicMaterial({
      map: fx.glowTexture(), color: 0x000000, transparent: true, opacity: 0.55, depthWrite: false }));
    carBlob.rotation.x = -Math.PI / 2;
    carBlob.position.y = 0.12;
    car.add(carBlob);
  }

  /* perceived lane-offset line from car to road centre */
  const laneLine = new THREE.Mesh(
    new THREE.BoxGeometry(1, 0.15, 2.5),
    new THREE.MeshBasicMaterial({ color: 0xc8a8ff, transparent: true, opacity: 0.6,
      blending: THREE.AdditiveBlending, depthWrite: false })
  );
  laneLine.position.y = 0.3;
  scene.add(laneLine);

  /* yaw that points a -Z-forward object along the track tangent */
  function trackYaw(d) { return Math.atan2(-centreXPrime(d), 1); }

  /* crash flash: tints only the body paint (glass, trim and lights keep
     their own materials) */
  const PAINT = new THREE.Color(0xa371f7), CRASH = new THREE.Color(0xf85149);
  function setCarColor(crashed) {
    for (const m of paintMats) m.color.copy(crashed ? CRASH : PAINT);
  }

  function layoutWorld() {
    const cs = carState(sim);
    car.position.set(sim.x, 0, -sim.y);
    car.rotation.y = trackYaw(sim.y) + (sim.lane || 0) * -0.08;
    car.rotation.z = (sim.lane || 0) * 0.03;
    setCarColor(!sim.alive);

    /* world-anchored ground; sky, stars, mountains and sun ride with the car */
    ground.position.set(Math.round(sim.x / TILE) * TILE, -0.2, Math.round(-sim.y / TILE) * TILE);
    if (sky) sky.position.set(sim.x, 0, -sim.y);
    if (starField) starField.position.set(sim.x, 0, -sim.y);
    mountains.position.set(sim.x, 0, -sim.y);
    sun.position.set(sim.x + 160, 260, -sim.y - 420);
    sun.target.position.set(sim.x, 0, -sim.y - 60);
    fill.position.set(sim.x - 40, 120, -sim.y + 200);
    fill.target.position.set(sim.x, 0, -sim.y);

    rebuildRibbon(shoulder);
    rebuildRibbon(asphalt);
    rebuildRibbon(flashRibbon);
    layoutProps();

    const dcp = sim.y0 + (sim.checkpoint + 1) * 600;
    gate.position.set(centreX(dcp), 0, -dcp);
    gate.rotation.y = trackYaw(dcp);
    const gm = 1 + Math.max(0, gatePulse) * 0.35;
    gate.scale.set(gm, gm, 1);
    gateMat.emissiveIntensity = 1.4 + Math.max(0, gatePulse) * 4;
    for (const g of gateGlows) g.material.opacity = 0.5 + Math.max(0, gatePulse) * 0.5;
    if (gatePulse > 0) gatePulse -= 0.05;

    /* near-miss / crash shoulder flash */
    if (flashT > 0) {
      flashT -= 0.06;
      flashRibbon.material.opacity = Math.max(0, flashT) * 0.6;
    } else flashRibbon.material.opacity = 0;
    flashRibbon.visible = flashRibbon.material.opacity > 0.01;

    for (const g of lampGlows) g.material.opacity = sim.alive ? 0.9 : 0.25;
    headlight.intensity = sim.alive ? 900 : 120;

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
          10 + v.vy * (1 - burstT) - 40 * (1 - burstT) * (1 - burstT),
          gate.position.z + v.vz * (1 - burstT));
      }
      p.needsUpdate = true;
    } else burst.visible = false;
  }

  /* chase camera: behind the car along the track tangent, eased so the
     agent's 30 Hz steering jitter doesn't shake the view */
  const camPos = new THREE.Vector3(), camLook = new THREE.Vector3();
  let camInit = false;
  function chaseCamera() {
    const tx = centreXPrime(sim.y), len = Math.hypot(tx, 1);
    const fx_ = tx / len, fz = -1 / len;                  // forward unit (x, z)
    const want = new THREE.Vector3(sim.x - fx_ * 82, 36, -sim.y - fz * 82);
    const look = new THREE.Vector3(sim.x + fx_ * 130, 6, -sim.y + fz * 130);
    /* snap (don't sweep) after a respawn somewhere else on the road */
    if (!camInit || camPos.distanceTo(want) > 300) { camPos.copy(want); camLook.copy(look); camInit = true; }
    camPos.lerp(want, 0.12);
    camLook.lerp(look, 0.18);
    camera.position.copy(camPos);
    camera.lookAt(camLook);
  }
  function attractCamera(t) {
    const a = t * 0.18;
    camera.position.set(sim.x + Math.sin(a) * 120, 34, -sim.y + Math.cos(a) * 120);
    camera.lookAt(sim.x, 8, -sim.y);
    camInit = false;
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
      ["episodes trained", stats ? stats.episodesTrained : "—"],
      ["this session", stats ? stats.episode : "—"],
      ["last score", stats ? stats.lastScore : "—"],
      ["avg score", stats ? stats.avgScore : "—"],
      ["best score", bestScore > -Infinity ? bestScore : "—"],
      ["distance", Math.round(sim.y - sim.y0) + " m"],
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

  /* ========================= memory, best play ========================= */
  /* Your fly's brain auto-saves to this browser every 5 episodes and is
     restored next visit, so it keeps learning from its own mistakes across
     sessions. Best Play = greedy: always take the top action (its current
     skill); Learning = sample from the policy so it keeps exploring. */
  let bestPlay = false;
  function setBestPlay(on) {
    bestPlay = !!on;
    if (brainReady) window.flyBrain.setMode(bestPlay ? "greedy" : "train");
    const b = document.getElementById("btn-best");
    if (b) { b.textContent = bestPlay ? "🎯 Best Play: on" : "📚 Learning (exploring)"; b.classList.toggle("on", bestPlay); }
  }
  /* start-screen hint: offer to continue a saved fly */
  (function labelContinue() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(SAVE_KEY) || "null");
      const btn = document.getElementById("start-fly");
      if (saved && btn && saved.label !== "pro") btn.textContent = `▶ Continue Your Fly (${saved.episodesTrained} episodes trained)`;
    } catch (e) { /* no storage: keep the default label */ }
  })();

  async function start() {
    note.textContent = "loading fly brain…";
    const ok = await ensureBrain();
    if (!ok) { note.textContent = "fly brain failed to load — see console."; return; }
    window.flyBrain.setMode("train");
    let hello = null;
    const saved = window.flyBrain.restoreSaved(SAVE_KEY);
    if (saved && saved.label === "pro") {
      /* a brain saved while the (removed) offline "pro" was loaded is not
         this fly's own learning: start it from scratch instead */
      window.flyBrain.reset();
      window.flyBrain.clearSaved(SAVE_KEY);
    } else if (saved) {
      hello = `Welcome back — your fly remembers ${saved.episodesTrained} episodes of training.`;
    }
    setBestPlay(false);
    started = true; paused = false;
    sim = newSim();
    startScreen.style.display = "none";
    if (hello) setTimeout(() => { document.getElementById("status").textContent = hello; }, 0);
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
    /* A crash's delayed auto-restart is skipped while paused; respawn on
       resume so the episode loop can't stall on a dead car. */
    if (!paused && started && !sim.alive) restart();
  };
  bRestart.onclick = restart;
  bReset.onclick = resetBrain;
  document.getElementById("btn-viz").onclick = (e) => {
    const on = e.target.textContent === "hide";
    e.target.textContent = on ? "show" : "hide";
    if (brainViz) brainViz.setVisible(!on);
  };
  document.getElementById("start-fly").onclick = () => start();
  const bBest = document.getElementById("btn-best");
  if (bBest) bBest.onclick = () => setBestPlay(!bestPlay);

  /* ============================== main loop =============================== */
  let lastTs = 0, hudLast = 0, chartLast = 0, acc = 0;
  function frame(ts) {
    const rdt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;
    const dt = rdt * SPEEDS[speedIdx];

    if (started && !paused && sim.alive) {
      /* One brain decision per simulated tick, in fixed 1/30 s slices of sim
         time (30 Hz — the cadence verify.js learning-checks), so episode length
         in decisions stays identical across speed multipliers. A full 60 s
         episode is 1800 steps; fly-brain.js replays it as one batched graph. */
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
        if (sim.y - sim.y0 > (sim.checkpoint + 1) * 600) { sim.checkpoint++; reward(1); gatePulse = 1; burstT = 1; }
        if (nearMiss()) flashT = Math.max(flashT, 0.35); else flashT = Math.max(0, flashT - 0.06);
        if (!st.onRoad && Math.abs(st.laneOffset) > ROAD / 2 + 4) { crash(); break; }
        if (sim.t > 60) { reward(2); window.flyBrain.endEpisode(); updateBest(); sim = newSim(); break; }
      }
    }

    layoutWorld();
    if (started && !paused) chaseCamera(); else attractCamera(ts / 1000);
    renderer.render(scene, camera);
    if (brainViz) { brainViz.update(rdt); brainViz.render(); }

    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(false); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

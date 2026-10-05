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
  let brainViz = null;
  const GAME_ID = "open-world";
  const SAVE_KEY = "fly-brain:" + GAME_ID + ":v1";       // this browser's saved fly
  const PRO_URL = "../shared/trained/" + GAME_ID + ".json"; // offline-trained pro

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
  /* The render loop draws the attract view from page load, before Start:
     the world must exist by then or the first frame throws and kills it. */
  resetWorld();

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
  /* Look: twilight meadow. Gradient dusk sky, grassy arena with tufts and
     stone walls trimmed in light, a forest and hills beyond the walls,
     glowing crystal orbs with light beams, fireflies, and soft shadows.
     Cosmetic only — nothing here feeds the sim, sensors or rewards (all
     decorations are visual; collisions are still just the arena walls). */
  const wrap = document.getElementById("canvas-wrap");
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const fx = window.flyFx;
  if (fx) fx.setupRenderer(renderer, { exposure: 1.0 });

  const SKY = { top: 0x13224f, horizon: 0xf09a74, bottom: 0x141a22, curve: 0.32,
    sunDir: new THREE.Vector3(-0.8, 0.1, -0.6), sunColor: 0xffb27a, sunGlow: 0.8 };
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1b2236);
  scene.fog = new THREE.Fog(0x4a4058, 180, 760);
  if (fx) {
    scene.add(fx.skyDome(Object.assign({ radius: 900 }, SKY)));
    const st = fx.stars(500, 860, 0.35);
    st.material.opacity = 0.5;
    scene.add(st);
    scene.environment = fx.envFromSky(renderer, SKY);
  }

  const camera = new THREE.PerspectiveCamera(55, 3 / 2, 0.5, 2000);

  scene.add(new THREE.HemisphereLight(0xa9bcff, 0x3a3020, 1.0));
  /* low warm sun from the sky's sun direction; shadow box covers the arena */
  const sun = new THREE.DirectionalLight(0xffc48f, 2.4);
  sun.position.copy(SKY.sunDir).normalize().multiplyScalar(320).setY(150);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -150, right: 150, top: 150, bottom: -150, near: 10, far: 700 });
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.5;
  scene.add(sun);

  const hash = (k) => { const x = Math.sin(k * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

  /* arena grass + a wider darker meadow beyond the walls */
  const grassTex = fx ? fx.noiseTexture({ base: 0x3d5a2e, spread: 0.12, size: 256, density: 0.9, repeat: [12, 12], seed: 21 }) : null;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(HALF * 2, HALF * 2),
    new THREE.MeshStandardMaterial({ color: grassTex ? 0xffffff : 0x10161f, map: grassTex, roughness: 0.95 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  const meadowTex = fx ? fx.noiseTexture({ base: 0x2c4224, spread: 0.1, size: 256, density: 0.8, repeat: [40, 40], seed: 22 }) : null;
  const meadow = new THREE.Mesh(
    new THREE.PlaneGeometry(1600, 1600),
    new THREE.MeshStandardMaterial({ color: meadowTex ? 0xffffff : 0x0b0f14, map: meadowTex, roughness: 1 })
  );
  meadow.rotation.x = -Math.PI / 2;
  meadow.position.y = -0.3;
  meadow.receiveShadow = true;
  scene.add(meadow);
  /* faint exploration grid, so coverage still reads */
  const grid = new THREE.GridHelper(HALF * 2, GRID, 0x9fd18a, 0x9fd18a);
  grid.material.transparent = true;
  grid.material.opacity = 0.07;
  grid.position.y = 0.05;
  scene.add(grid);

  /* stone walls with a glowing cyan trim along the top */
  const stoneTex = fx ? fx.noiseTexture({ base: 0x6b6a72, spread: 0.14, size: 128, density: 0.9, seed: 31,
    draw(g, n) { g.strokeStyle = "rgba(30,30,40,0.55)"; g.lineWidth = 2;
      for (let y = 0; y < n; y += n / 4) { g.beginPath(); g.moveTo(0, y); g.lineTo(n, y); g.stroke();
        for (let x = (y / (n / 4)) % 2 ? n / 4 : 0; x < n; x += n / 2) { g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + n / 4); g.stroke(); } } } }) : null;
  const wallMat = new THREE.MeshStandardMaterial({ color: stoneTex ? 0xffffff : 0x1a2230, map: stoneTex, roughness: 0.9 });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0x0f3340, emissive: 0x4fe3ff, emissiveIntensity: 1.6 });
  const wallH = 8, wallT = 3;
  [[0, -HALF - wallT / 2, HALF * 2 + wallT * 2, wallT],
   [0, HALF + wallT / 2, HALF * 2 + wallT * 2, wallT],
   [-HALF - wallT / 2, 0, wallT, HALF * 2 + wallT * 2],
   [HALF + wallT / 2, 0, wallT, HALF * 2 + wallT * 2]].forEach(([x, z, sx, sz]) => {
    const geo = new THREE.BoxGeometry(sx, wallH, sz);
    /* tile the stone texture by wall length instead of stretching it */
    const uv = geo.getAttribute("uv"), len = Math.max(sx, sz) / 8;
    for (let i = 0; i < uv.count; i++) uv.setX(i, uv.getX(i) * len);
    if (stoneTex) wallMat.map.wrapS = THREE.RepeatWrapping;
    const m = new THREE.Mesh(geo, wallMat);
    m.position.set(x, wallH / 2, z);
    m.castShadow = m.receiveShadow = true;
    scene.add(m);
    const trim = new THREE.Mesh(new THREE.BoxGeometry(sx + 0.2, 0.5, sz + 0.2), trimMat);
    trim.position.set(x, wallH + 0.25, z);
    scene.add(trim);
  });

  /* grass tufts inside the arena (decor only, not obstacles) */
  const TUFTS = 700;
  const tuftGeo = new THREE.ConeGeometry(1.1, 1.5, 3);
  tuftGeo.translate(0, 0.75, 0);
  const tufts = new THREE.InstancedMesh(tuftGeo,
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, flatShading: true }), TUFTS);
  const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < TUFTS; i++) {
    const x = (hash(i * 2.1) * 2 - 1) * (HALF - 4), z = (hash(i * 3.7 + 1) * 2 - 1) * (HALF - 4);
    const sc = 0.5 + hash(i * 5.3) * 1.1;
    tmpQ.setFromAxisAngle(UP, hash(i * 7.9) * 6.28);
    tmpM.compose(tmpP.set(x, 0, z), tmpQ, tmpS.set(sc, sc * (0.8 + hash(i) * 0.8), sc));
    tufts.setMatrixAt(i, tmpM);
    tufts.setColorAt(i, new THREE.Color().setHSL(0.27 + hash(i * 1.7) * 0.06, 0.55, 0.13 + hash(i * 2.9) * 0.1));
  }
  scene.add(tufts);

  /* forest ring + rolling hills beyond the walls */
  const TREES = 260;
  const pineGeo = new THREE.ConeGeometry(7, 24, 7);
  pineGeo.translate(0, 17, 0);
  const trunkGeo = new THREE.CylinderGeometry(1.1, 1.5, 6, 6);
  trunkGeo.translate(0, 3, 0);
  const pines = new THREE.InstancedMesh(pineGeo,
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, flatShading: true }), TREES);
  const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x3b2a1f, roughness: 1 }), TREES);
  for (let i = 0; i < TREES; i++) {
    const a = hash(i * 1.31) * Math.PI * 2;
    const r = HALF + 18 + hash(i * 2.77) * 150;
    const sq = Math.max(Math.abs(Math.cos(a)), Math.abs(Math.sin(a)));  // push out of the square arena
    const sc = 0.7 + hash(i * 4.4) * 1.0;
    tmpQ.setFromAxisAngle(UP, hash(i) * 6.28);
    tmpM.compose(tmpP.set(Math.cos(a) * r / sq, 0, Math.sin(a) * r / sq), tmpQ, tmpS.set(sc, sc * (0.8 + hash(i * 9) * 0.5), sc));
    pines.setMatrixAt(i, tmpM); trunks.setMatrixAt(i, tmpM);
    pines.setColorAt(i, new THREE.Color().setHSL(0.36 + hash(i * 3.3) * 0.06, 0.5, 0.1 + hash(i * 6.1) * 0.08));
  }
  pines.castShadow = trunks.castShadow = true;
  scene.add(pines, trunks);
  const hillMat = new THREE.MeshStandardMaterial({ color: 0x2b3d2c, roughness: 1, flatShading: true });
  for (let i = 0; i < 22; i++) {
    const a = (i / 22) * Math.PI * 2 + hash(i) * 0.2, r = 520 + hash(i * 3) * 160;
    const h = 40 + hash(i * 7) * 90;
    const hill = new THREE.Mesh(new THREE.SphereGeometry(1, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2), hillMat);
    hill.scale.set(h * 2.2, h, h * 2.2);
    hill.position.set(Math.cos(a) * r, -2, Math.sin(a) * r);
    scene.add(hill);
  }

  /* agent: a little "fly rover". Blender-built body (fly-rover.glb, nose at
     -Z, ~2.5 units) with a primitive sphere+whisker fallback so the page
     always runs. The whisker stays visible in both cases — it makes heading
     legible at orbit-camera distance. */
  const agentGroup = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xa371f7, metalness: 0.2, roughness: 0.4 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(3.2, 20, 14), bodyMat);
  body.scale.set(1, 0.7, 1.3);
  body.position.y = 3;
  body.castShadow = true;
  const whisker = new THREE.Mesh(
    new THREE.BoxGeometry(0.35, 0.35, 7),
    new THREE.MeshBasicMaterial({ color: 0xbff4ff, transparent: true, opacity: 0.8 })
  );
  whisker.position.set(0, 3, -5);
  agentGroup.add(body, whisker);
  const paintMats = [bodyMat];
  if (fx) {
    const tip = fx.glowSprite(0x7fe9ff, 5, 0.9);
    tip.position.set(0, 3, -8.5);
    agentGroup.add(tip);
  }
  scene.add(agentGroup);
  if (typeof flyAssets !== "undefined") {
    flyAssets.load("fly-rover").then((g) => {
      g.scale.setScalar(2.6);
      g.position.y = 3;
      agentGroup.add(g);
      body.visible = false;                    // primitive fallback hidden
      g.traverse((o) => { if (o.isMesh && o.material.name === "rover_body") paintMats.push(o.material); });
    }).catch((e) => console.warn("fly-rover.glb unavailable — using primitives", e));
  }

  /* orbs (pooled): spinning crystal + halo + light beam + ground ring */
  const ORB_COLOR = 0x7dffb0;
  const orbMat = new THREE.MeshStandardMaterial({ color: 0x1d6b3e, emissive: ORB_COLOR, emissiveIntensity: 1.3,
    metalness: 0.1, roughness: 0.2, flatShading: true });
  const beamMat = new THREE.MeshBasicMaterial({ color: ORB_COLOR, transparent: true, opacity: 0.07,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
  const orbMeshes = [];
  for (let i = 0; i < N_ORBS; i++) {
    const m = new THREE.Group();
    const crystal = new THREE.Mesh(new THREE.OctahedronGeometry(2.6, 0), orbMat);
    crystal.scale.set(1, 1.5, 1);
    crystal.castShadow = true;
    m.add(crystal);
    m.userData.crystal = crystal;
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 1.5, 60, 12, 1, true), beamMat);
    beam.position.y = 24;
    m.add(beam);
    if (fx) { const h = fx.glowSprite(ORB_COLOR, 16, 0.7); m.add(h); }
    m.visible = false;
    scene.add(m);
    orbMeshes.push(m);
  }
  /* soft rings under orbs so they read at distance */
  const ringMat = new THREE.MeshBasicMaterial({ color: ORB_COLOR, transparent: true, opacity: 0.35,
    blending: THREE.AdditiveBlending, depthWrite: false });
  const ringMeshes = [];
  for (let i = 0; i < N_ORBS; i++) {
    const m = new THREE.Mesh(new THREE.RingGeometry(4.2, 5.0, 32), ringMat);
    m.rotation.x = -Math.PI / 2;
    m.position.y = 0.15;
    m.visible = false;
    scene.add(m);
    ringMeshes.push(m);
  }

  /* fireflies: slow-drifting additive specks over the arena (decor) */
  const FLIES = 70;
  const flyGeo = new THREE.BufferGeometry();
  flyGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(FLIES * 3), 3));
  const fireflies = new THREE.Points(flyGeo, new THREE.PointsMaterial({ color: 0xfff3a0, size: 2.2,
    map: fx ? fx.glowTexture() : null, transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  fireflies.frustumCulled = false;
  scene.add(fireflies);

  /* collect burst: pooled additive point cloud, spawned where an orb is taken */
  const BURST_N = 36;
  const bursts = [];
  function spawnBurst(x, z) {
    let b = bursts.find((b) => b.t <= 0);
    if (!b) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
      const mat = new THREE.PointsMaterial({ color: 0x7ee787, size: 3.2, transparent: true, opacity: 1,
        map: fx ? fx.glowTexture() : null, blending: THREE.AdditiveBlending, depthWrite: false });
      const pts = new THREE.Points(geo, mat);
      pts.visible = false;
      pts.frustumCulled = false;
      scene.add(pts);
      b = { pts, geo, mat, t: 0 };
      bursts.push(b);
    }
    b.t = 0.6;
    b.mat.color.set(0x9dffc4);
    const p = b.geo.getAttribute("position");
    b.vel = new Array(BURST_N).fill(0).map(() => {
      const a = Math.random() * Math.PI * 2, r = 8 + Math.random() * 16;
      return { vx: Math.cos(a) * r, vy: 10 + Math.random() * 16, vz: Math.sin(a) * r };
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

  /* trail: glowing breadcrumbs so the foraging path reads at a glance */
  const TRAIL_N = 60;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_N * 3), 3));
  const trailMat = new THREE.PointsMaterial({ color: 0xc9a8ff, size: 2.4, transparent: true, opacity: 0.7,
    map: fx ? fx.glowTexture() : null, blending: THREE.AdditiveBlending, depthWrite: false });
  const trail = new THREE.Points(trailGeo, trailMat);
  trail.frustumCulled = false;
  scene.add(trail);
  let trailIdx = 0, trailAcc = 0;

  /* wall-bump flash tints only the body paint (eyes, wings keep theirs) */
  const PAINT = new THREE.Color(0xa371f7), BUMP = new THREE.Color(0xf85149);
  function setAgentColor(bumped) {
    for (const m of paintMats) m.color.copy(bumped ? BUMP : PAINT);
  }

  function layoutScene(t, dt) {
    const vdt = Math.min(dt, 1 / 30);      // effects stay visible at 16x
    agentGroup.position.set(agent.x, 0, agent.z);
    agentGroup.rotation.y = agent.h;
    setAgentColor(agent.bumpT > 0);
    if (agent.bumpT > 0) agent.bumpT -= dt;

    for (let i = 0; i < orbs.length; i++) {
      const o = orbs[i];
      orbMeshes[i].visible = !o.taken;
      ringMeshes[i].visible = !o.taken;
      if (!o.taken) {
        const bob = Math.sin(t * 2 + o.t) * 0.8;
        orbMeshes[i].position.set(o.x, 6 + bob, o.z);
        orbMeshes[i].userData.crystal.rotation.y = t * 1.2 + o.t;
        ringMeshes[i].position.x = o.x;
        ringMeshes[i].position.z = o.z;
        const s = 1 + Math.sin(t * 3 + o.t) * 0.12;
        ringMeshes[i].scale.set(s, s, s);
      }
    }
    orbMat.emissiveIntensity = 1.2 + Math.sin(t * 3) * 0.25;

    /* fireflies wander on slow deterministic loops */
    const fp = flyGeo.getAttribute("position");
    for (let i = 0; i < FLIES; i++) {
      const a = t * (0.15 + hash(i) * 0.2) + hash(i * 3.1) * 6.28;
      const r = 20 + hash(i * 5.7) * (HALF - 25);
      fp.setXYZ(i, Math.cos(a + i) * r, 3 + hash(i * 8.3) * 10 + Math.sin(t * 1.3 + i) * 1.5,
        Math.sin(a * 0.8 + i * 2) * r);
    }
    fp.needsUpdate = true;
    fireflies.material.opacity = 0.55 + Math.sin(t * 2.3) * 0.3;

    /* breadcrumb trail (sim-time spaced) */
    trailAcc += dt;
    if (trailAcc > 0.25) {
      trailAcc = 0;
      const p = trailGeo.getAttribute("position");
      p.setXYZ(trailIdx % TRAIL_N, agent.x, 1.2, agent.z);
      trailIdx++;
      p.needsUpdate = true;
    }

    layoutBursts(vdt);
  }

  /* orbit camera (drag to spectate) + follow */
  let camYaw = 0, camPitch = 0.36, camDist = 84, dragging = false, lastX = 0, lastY = 0;
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
    window.flyBrain.clearSaved(SAVE_KEY);
    setBestPlay(false);
    if (brainViz) brainViz.rebuild();
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
      ["brain", brainName()],
      ["episodes trained", stats ? stats.episodesTrained : "—"],
      ["this session", stats ? stats.episode : "—"],
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

  /* ===================== memory, pro brain, best play ===================== */
  /* Your fly's brain auto-saves to this browser every 5 episodes and is
     restored next visit, so learning accumulates across sessions. "Pro" is a
     brain trained offline (tools/fly-brain-test/train.js, thousands of
     episodes on exact mirrors of this game's rules); it keeps learning live.
     Best Play = greedy: always take the top action (its real skill level);
     Learning = sample from the policy so it keeps exploring and improving. */
  let bestPlay = false;
  function setBestPlay(on) {
    bestPlay = !!on;
    if (brainReady) window.flyBrain.setMode(bestPlay ? "greedy" : "train");
    const b = document.getElementById("btn-best");
    if (b) { b.textContent = bestPlay ? "🎯 Best Play: on" : "📚 Learning (exploring)"; b.classList.toggle("on", bestPlay); }
  }
  function proSkill(ev) {
    if (!ev) return "";
    if (ev.finishedPct != null) return `; exam on 100 new roads: finishes ${Math.round(ev.finishedPct)}% of 60 s drives`;
    if (ev.hitRatePct != null) return `; exam on 100 new songs: hits ${Math.round(ev.hitRatePct)}% of notes`;
    if (ev.allOrbsPct != null) return `; exam on 100 new arenas: clears all 8 orbs in ${Math.round(ev.allOrbsPct)}% of runs`;
    return "";
  }
  async function loadPro() {
    try {
      const res = await fetch(PRO_URL);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const pro = await res.json();
      const info = window.flyBrain.importBrain(pro);
      window.flyBrain.setBrainLabel("pro");
      /* play the pro the way it tested best (driving steers by sampling) */
      setBestPlay(pro.playMode === "greedy");
      return `🏆 Pro brain loaded — ${info.episodesTrained} episodes of training${proSkill(pro.trainedOffline && (pro.trainedOffline.exam || pro.trainedOffline.eval))}. It keeps learning live.`;
    } catch (e) {
      console.warn("pro brain unavailable:", e);
      return "Pro brain unavailable (" + e.message + ") — this fly keeps training from where it is.";
    }
  }
  function brainName() {
    if (!brainReady) return "—";
    return window.flyBrain.getBrainLabel() === "pro" ? "🏆 pro" : "your fly";
  }
  /* start-screen hint: offer to continue a saved fly */
  (function labelContinue() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(SAVE_KEY) || "null");
      const btn = document.getElementById("start-fly");
      if (saved && btn) btn.textContent = `▶ Continue Your Fly (${saved.episodesTrained} episodes trained)`;
    } catch (e) { /* no storage: keep the default label */ }
  })();

  async function start(withPro) {
    note.textContent = "loading fly brain…";
    const ok = await ensureBrain();
    if (!ok) { note.textContent = "fly brain failed to load — see console."; return; }
    window.flyBrain.setMode("train");
    let hello = null;
    if (withPro) hello = await loadPro();
    else {
      const saved = window.flyBrain.restoreSaved(SAVE_KEY);
      if (saved) hello = `Welcome back — your fly remembers ${saved.episodesTrained} episodes of training.`;
      setBestPlay(false);
    }
    started = true; paused = false;
    resetWorld();
    startScreen.style.display = "none";
    if (hello) setTimeout(() => { document.getElementById("status").textContent = hello; }, 0);
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
  document.getElementById("btn-viz").onclick = (e) => {
    const on = e.target.textContent === "hide";
    e.target.textContent = on ? "show" : "hide";
    if (brainViz) brainViz.setVisible(!on);
  };
  document.getElementById("start-fly").onclick = () => start(false);
  document.getElementById("start-pro").onclick = () => start(true);
  document.getElementById("btn-pro").onclick = async () => {
    if (!brainReady) return;
    document.getElementById("status").textContent = await loadPro();
  };
  const bBest = document.getElementById("btn-best");
  if (bBest) bBest.onclick = () => setBestPlay(!bestPlay);

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
    if (brainViz) { brainViz.update(rdt); brainViz.render(); }
    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(false); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

/* ============================================================================
 * beat-saber/game.js — three.js beat-slicing game for the fly brain.
 *
 * AI-ONLY: there is no human control mode. The fly brain (shared/fly-brain.js)
 * plays and learns live via REINFORCE; you watch, fast-forward, and reset.
 *
 * Notes spawn on a generated metronome (WebAudio, no licensed music): a 110 BPM
 * click plus a bass note every bar. Notes fall toward a beat line; the ideal
 * swing moment is when the note crosses it.
 *
 * Reward shaping (tuned so learning is visible within a few dozen episodes —
 * see tools/fly-brain-test/RESULTS.md): hit +0.4 (+1.0 perfect, within 40% of
 * the hit window), miss −0.3, wrong-lane swing −0.05, wait = 0. Decisions land
 * every 0.1s of song time (10 Hz) to keep replay variance low.
 *
 * The key learnability fix was feature ENCODING, not reward magnitude: the
 * sensor model fires on high |sign*gain*x + bias|, so cos-phase features
 * (signed, sweeping ±1, +1 exactly at the hit moment) make the hit opportunity
 * visible to the whole sensor population, where urgency/time-to-line kept
 * most sensors silent and collapsed the policy to a constant distribution.
 * ========================================================================== */
(function () {
  "use strict";

  if (typeof THREE === "undefined") {
    const note = document.getElementById("load-note");
    if (note) note.textContent = "three.js failed to load from CDN — check your connection.";
    return;
  }

  /* ============================ game constants ============================ */
  const BPM = 110;
  const BEAT = 60 / BPM;                 // seconds per beat
  const SPAWN_LEAD = 2.0;                // notes spawn this many seconds ahead
  const APPROACH = 14;                   // z units per approach unit
  const LANES = [-4, 0, 4];              // x position of the three lanes
  const HIT_WINDOW = 0.12;               // ±s around the beat line
  const SONG_START_DELAY = 1.0;
  /* 16-beat songs: shorter episodes => tighter Monte-Carlo returns and 2x the
     gradient steps per unit experience vs 32 beats (verified in the harness). */
  const EPISODE_BEATS = 16;
  /* This game needs a faster optimizer than the default: the per-decision
     credit signal is small, so 0.04 (vs the shared default 0.02) is what moves
     the policy within a few hundred episodes (see RESULTS.md). */
  const LR_GAME = 0.04;

  /* ============================ fly-brain wiring ========================== */
  /* Features are COS PHASES (see ph() in stateVector): signed values sweeping
     [−1, 1] with +1 exactly at the beat-line crossing. The sensor model fires
     on high |sign*gain*x + bias|, so (a) the critical moment is the feature
     MAXIMUM and (b) sensors respond continuously — urgency-in-[0,1] or
     time-to-line encodings left most sensors permanently silent, which
     collapsed the policy to a state-independent distribution (the fly could
     not even see the hit opportunity). */
  const FEATURES = [
    "phLeft",     // cos phase of nearest left-lane note (+1 = at beat line)
    "phCentre",
    "phRight",
    "beatPhase",  // cos of the metronome phase (+1 on the beat)
  ];
  const ACTIONS = ["swingL", "swingC", "swingR", "wait"];

  /* Speed multipliers for the simulation clock (1x / 4x / 16x). */
  const SPEEDS = [1, 4, 16];
  let speedIdx = 0;
  let started = false, paused = false;
  let brainReady = false;
  let lastProbs = [0.25, 0.25, 0.25, 0.25];
  let lastActionName = null;
  let bestScore = -Infinity;
  let brainViz = null;
  const GAME_ID = "beat-saber";
  const SAVE_KEY = "fly-brain:" + GAME_ID + ":v1";       // this browser's saved fly

  /* ============================ audio (metronome) ========================= */
  let audioCtx = null, musicTimer = null;
  function audioInit() {
    if (audioCtx) return;
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) { audioCtx = null; }
  }
  function click(t, freq, gainVal, dur) {
    if (!audioCtx) return;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = "square"; o.frequency.value = freq;
    g.gain.setValueAtTime(gainVal, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(t); o.stop(t + dur + 0.02);
  }
  /* schedule metronome ticks ~200ms ahead, forever */
  function startMusic() {
    audioInit();
    if (!audioCtx) return;
    if (audioCtx.state === "suspended") audioCtx.resume();
    let nextBeat = audioCtx.currentTime + 0.1;
    let beatCount = 0;
    musicTimer = setInterval(() => {
      while (nextBeat < audioCtx.currentTime + 0.2) {
        const isBar = beatCount % 4 === 0;
        click(nextBeat, isBar ? 880 : 660, isBar ? 0.18 : 0.1, 0.06);
        if (isBar) click(nextBeat, 110, 0.25, 0.3);   // bass note each bar
        nextBeat += BEAT; beatCount++;
      }
    }, 50);
  }
  function stopMusic() { if (musicTimer) { clearInterval(musicTimer); musicTimer = null; } }

  /* ============================== game state ============================== */
  /* songClock = seconds since song start. Notes are pre-rolled: each beat
     index b gets a note with a random lane (seeded per game, not per note). */
  let notes = [];          // {lane, hit:false, missed:false, beat}
  let songClock = 0, songBeat = 0;
  let score = 0, streak = 0, bestStreak = 0, hits = 0, misses = 0;
  let swingFlash = null;   // {lane, t}
  let laneNote = [null, null, null];  // nearest unhit note per lane (for features)
  /* per-lane hit flash for the saber strike effect */
  let laneFlash = [0, 0, 0];

  function resetSong() {
    notes = [];
    songClock = -SONG_START_DELAY;
    songBeat = 0;
    score = 0; streak = 0; bestStreak = 0; hits = 0; misses = 0;
    swingFlash = null;
    laneFlash = [0, 0, 0];
  }

  function spawnUpTo() {
    const targetBeat = Math.floor((songClock + SPAWN_LEAD) / BEAT);
    while (songBeat <= targetBeat) {
      if (songBeat >= 0) {
        const lane = Math.floor(Math.random() * 3);
        notes.push({ lane, hit: false, missed: false, beat: songBeat });
      }
      songBeat++;
    }
  }

  function timeToBeatLine(beat) { return beat * BEAT - songClock; }

  /* ============================ fly-brain wiring ========================== */
  function stateVector() {
    /* nearest unhit, in-window note per lane */
    laneNote = [null, null, null];
    for (const n of notes) {
      if (n.hit || n.missed) continue;
      const t = timeToBeatLine(n.beat);
      if (t < -HIT_WINDOW || t > SPAWN_LEAD) continue;
      if (laneNote[n.lane] === null || t < timeToBeatLine(laneNote[n.lane].beat)) laneNote[n.lane] = n;
    }
    /* +1 at the line, −1 a full beat away, 0 when nothing is inbound */
    const ph = (n) => (n === null ? 0 : Math.max(-1, Math.min(1, Math.cos(Math.PI * timeToBeatLine(n.beat) / BEAT))));
    const phase = ((songClock % BEAT) + BEAT) % BEAT / BEAT;
    return [ph(laneNote[0]), ph(laneNote[1]), ph(laneNote[2]), Math.cos(2 * Math.PI * phase)];
  }

  async function ensureBrain() {
    if (brainReady) return true;
    if (typeof window.flyBrain === "undefined") return false;
    try {
      const pools = await window.flyBrain.init({ features: FEATURES, actions: ACTIONS, learningRate: LR_GAME });
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

  /* ============================== gameplay ================================ */
  function trySwing(lane) {
    /* find the nearest swingable note in this lane */
    let best = null, bestDt = 1e9;
    for (const n of notes) {
      if (n.hit || n.missed) continue;
      if (n.lane !== lane) continue;
      const dt = timeToBeatLine(n.beat);
      if (Math.abs(dt) < Math.abs(bestDt)) { best = n; bestDt = dt; }
    }
    const inWindow = best !== null && Math.abs(bestDt) <= HIT_WINDOW;
    if (inWindow) {
      const perfect = Math.abs(bestDt) < HIT_WINDOW * 0.4;
      best.hit = true;
      hits++; streak++;
      bestStreak = Math.max(bestStreak, streak);
      score += streak * (perfect ? 1 : 0.3);
      reward(perfect ? 1 : 0.4);
      laneFlash[lane] = perfect ? 0.25 : 0.18;   // clean hits flash brighter
      flashSwing(lane, perfect);
      spawnBurst(LANES[lane], perfect ? 0x7ee787 : 0x3fb950);
    } else {
      streak = 0;
      score -= 0.05;
      reward(-0.05);          // wrong lane / no note to hit (mild)
      flashSwing(lane, false);
    }
  }

  function missNotes() {
    for (const n of notes) {
      if (!n.hit && !n.missed && timeToBeatLine(n.beat) < -HIT_WINDOW) {
        n.missed = true;
        misses++; streak = 0;
        score -= 0.3;
        reward(-0.3);
      }
    }
  }

  function reward(v) { if (brainReady) window.flyBrain.reward(v); }

  /* Is a note hittable RIGHT NOW (any lane)? Used for the wait opportunity
     cost: holding still while a note is inside the hit window is a small
     penalty, giving the act-vs-wait contrast a direct gradient. */
  function hittableNow() {
    for (const n of notes) {
      if (n.hit || n.missed) continue;
      if (Math.abs(timeToBeatLine(n.beat)) <= HIT_WINDOW) return true;
    }
    return false;
  }

  /* Episode = one 32-beat "song". endEpisode on completion. */
  function endSong() {
    if (!brainReady) return;
    window.flyBrain.endEpisode();
    const s = window.flyBrain.getStats().lastScore;
    if (s > bestScore) bestScore = s;
  }

  /* Honest reset: rebuild the network from the seeded initialization. */
  function resetBrain() {
    stopMusic();
    window.flyBrain.reset();
    window.flyBrain.clearSaved(SAVE_KEY);
    setBestPlay(false);
    if (brainViz) brainViz.rebuild();
    bestScore = -Infinity;
    resetSong();
    const el = document.getElementById("status");
    el.textContent = "Brain reset — training starts from scratch.";
    drawChart(true);
  }

  function restart() {
    if (brainReady && started) endSong();
    resetSong();
    document.getElementById("status").textContent = "";
  }

  /* ============================== three.js =============================== */
  /* Look: neon synthwave stage. Magenta horizon with a striped retro sun, a
     scrolling neon grid floor, light frames rushing toward the camera on the
     beat, glowing gems, and a saber that slashes the lane the fly picks.
     Cosmetic only — nothing here feeds the sim, sensors or rewards. */
  const wrap = document.getElementById("canvas-wrap");
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const fx = window.flyFx;
  if (fx) fx.setupRenderer(renderer, { exposure: 1.1, shadows: false });

  const LANE_COLORS = [0xff4fa8, 0xb07cff, 0x35e0ff];   // left pink · centre violet · right cyan
  const SKY = { top: 0x07021c, horizon: 0xc0247a, bottom: 0x05010d, curve: 0.35,
    sunDir: new THREE.Vector3(0, 0.12, -1), sunColor: 0xff6a3d, sunGlow: 0.5, sunSize: 2 };
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07021c);
  scene.fog = new THREE.Fog(0x1a0628, 26, 110);
  const sky = fx ? fx.skyDome(Object.assign({ radius: 300 }, SKY)) : null;
  if (sky) scene.add(sky);
  if (fx) {
    const st = fx.stars(700, 280, 0.12);
    st.material.size = 1.4;
    scene.add(st);
    scene.environment = fx.envFromSky(renderer, SKY);
  }

  const camera = new THREE.PerspectiveCamera(66, 3 / 2, 0.1, 400);
  const CAM0 = new THREE.Vector3(0, 6.2, 11.5);
  camera.position.copy(CAM0);
  camera.lookAt(0, 2, -8);

  scene.add(new THREE.HemisphereLight(0x8a7cff, 0x2a0830, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(6, 14, 12);
  scene.add(key);

  /* retro sun: striped gradient disc on the horizon (canvas texture) */
  if (fx) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 256;
    const g = cv.getContext("2d");
    const grd = g.createLinearGradient(0, 0, 0, 256);
    grd.addColorStop(0, "#ffe76a"); grd.addColorStop(0.55, "#ff7a3c"); grd.addColorStop(1, "#ff2e88");
    g.fillStyle = grd;
    g.beginPath(); g.arc(128, 128, 126, 0, Math.PI * 2); g.fill();
    g.globalCompositeOperation = "destination-out";
    for (let i = 0; i < 7; i++) {
      const y = 140 + i * 17, hgt = 2 + i * 1.6;
      g.fillRect(0, y, 256, hgt);
    }
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sunDisc = new THREE.Mesh(new THREE.PlaneGeometry(90, 90),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, fog: false, depthWrite: false }));
    sunDisc.position.set(0, 20, -220);
    sunDisc.renderOrder = -0.5;
    scene.add(sunDisc);
    const halo = fx.glowSprite(0xff4f8a, 260, 0.55);
    halo.material.fog = false;
    halo.position.set(0, 22, -225);
    scene.add(halo);
  }

  /* neon grid floor: procedural lines that scroll toward the camera at the
     note approach speed, fading into the fog */
  const gridMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { scroll: { value: 0 }, pulse: { value: 0 },
      colA: { value: new THREE.Color(0xff3fa4) }, colB: { value: new THREE.Color(0x35e0ff) } },
    vertexShader: `
      varying vec2 vW; varying float vDist;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xz;
        vec4 mv = viewMatrix * w;
        vDist = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float scroll, pulse; uniform vec3 colA, colB;
      varying vec2 vW; varying float vDist;
      float line(float c, float w) {
        float f = abs(fract(c) - 0.5);
        float d = fwidth(c);
        return 1.0 - smoothstep(w - d, w + d, 0.5 - f);
      }
      void main() {
        float lx = line(vW.x / 4.0 + 0.5, 0.03);
        float lz = line((vW.y - scroll) / 4.0, 0.03);
        float g = max(lx, lz);
        float fade = 1.0 - smoothstep(18.0, 120.0, vDist);
        vec3 c = mix(colB, colA, smoothstep(10.0, 70.0, vDist));
        vec3 base = vec3(0.02, 0.0, 0.05);
        gl_FragColor = vec4(base + c * g * (1.1 + pulse * 1.4) * fade, 0.96);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  gridMat.extensions = { derivatives: true };
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(220, 260), gridMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, -0.5, -100);
  scene.add(floor);

  /* lane strips + glowing rails between and beside the lanes */
  const laneMats = LANE_COLORS.map((c) => new THREE.MeshBasicMaterial({
    color: c, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false }));
  LANES.forEach((lx, i) => {
    const g = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 60), laneMats[i]);
    g.rotation.x = -Math.PI / 2;
    g.position.set(lx, -0.44, -26);
    scene.add(g);
  });
  const railMat = new THREE.MeshBasicMaterial({ color: 0xc8b8ff });
  for (const rx of [-6, -2, 2, 6]) {
    const r = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.06, 60), railMat);
    r.position.set(rx, -0.4, -26);
    scene.add(r);
  }

  /* light tunnel: rectangular neon frames that travel toward the camera in
     step with the beat and flare on each beat */
  const FRAME_N = 10, FRAME_GAP = 9;
  const frameMat = new THREE.MeshBasicMaterial({ color: 0xff3fa4, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false });
  const frameMat2 = frameMat.clone();
  frameMat2.color.set(0x35e0ff);
  const frames = [];
  for (let i = 0; i < FRAME_N; i++) {
    const f = new THREE.Group();
    const m = i % 2 ? frameMat2 : frameMat;
    const W = 17, H = 10, T = 0.14;
    const top = new THREE.Mesh(new THREE.BoxGeometry(W, T, T), m); top.position.y = H;
    const l = new THREE.Mesh(new THREE.BoxGeometry(T, H, T), m); l.position.set(-W / 2, H / 2, 0);
    const r = new THREE.Mesh(new THREE.BoxGeometry(T, H, T), m); r.position.set(W / 2, H / 2, 0);
    f.add(top, l, r);
    f.position.y = -0.5;
    scene.add(f); frames.push(f);
  }

  /* beat line: emissive bar with a row of halos that pulse on the beat */
  const beatMat = new THREE.MeshStandardMaterial({ color: 0x1a2a55, emissive: 0x58a6ff, emissiveIntensity: 2 });
  const beatLine = new THREE.Mesh(new THREE.BoxGeometry(13.5, 0.12, 0.3), beatMat);
  beatLine.position.set(0, 0, 0);
  scene.add(beatLine);
  const beatGlows = [];
  if (fx) {
    for (const x of [-6, -3, 0, 3, 6]) {
      const g = fx.glowSprite(0x58a6ff, 4, 0.5);
      g.position.set(x, 0.1, 0);
      scene.add(g); beatGlows.push(g);
    }
  }

  /* notes: Blender-built gems (shared/assets/note-gem.glb) with a primitive
     cube fallback. Only the gem body is tinted per lane (stem/collar keep
     their metal); each pooled note carries a lane-coloured halo. */
  const noteMats = LANE_COLORS.map((c) => new THREE.MeshStandardMaterial({
    color: new THREE.Color(c).multiplyScalar(0.3), emissive: c, emissiveIntensity: 0.9,
    metalness: 0.1, roughness: 0.3, envMapIntensity: 0.4 }));
  const primitiveGem = new THREE.BoxGeometry(2.0, 2.0, 2.0);
  let gemProto = null;
  if (typeof flyAssets !== "undefined") {
    flyAssets.load("note-gem").then((g) => {
      g.scale.setScalar(1.9);     // model is ~1.2 units; match the 2.2-unit footprint
      gemProto = g;
      for (const m of noteMeshes) attachGem(m);
    }).catch((e) => console.warn("note-gem.glb unavailable — using primitives", e));
  }
  function attachGem(group) {
    if (!gemProto || group.userData.gem) return;
    const g = gemProto.clone();
    g.traverse((o) => { if (o.isMesh) o.material = o.material.clone(); });
    group.add(g);
    group.userData.gem = g;
    group.userData.prim.visible = false;
    group.userData.lane = -1;   // force a retint
  }
  function retint(group, lane) {
    if (group.userData.lane === lane) return;
    group.userData.lane = lane;
    group.userData.prim.material = noteMats[lane];
    if (group.userData.gem) {
      group.userData.gem.traverse((o) => {
        if (o.isMesh && o.material.name === "gem_body") o.material = noteMats[lane];
      });
    }
    if (group.userData.glow) group.userData.glow.material.color.set(LANE_COLORS[lane]);
  }
  const noteMeshes = [];  // pooled groups
  function noteMesh() {
    const g = new THREE.Group();
    const prim = new THREE.Mesh(primitiveGem, noteMats[0]);
    g.add(prim);
    g.userData.prim = prim;
    if (fx) {
      const glow = fx.glowSprite(LANE_COLORS[0], 6.5, 0.5);
      glow.material.depthTest = false;      // halo sits at the gem centre
      glow.renderOrder = 2;
      g.add(glow);
      g.userData.glow = glow;
    }
    attachGem(g);
    scene.add(g); noteMeshes.push(g);
    return g;
  }

  /* the saber: handle + glowing blade. It hovers over the last lane and,
     on a swing, sweeps through that lane leaving a slash arc (green = hit,
     red = miss). */
  const saber = new THREE.Group();
  const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 1.2, 12),
    new THREE.MeshStandardMaterial({ color: 0x2a2f3a, metalness: 0.9, roughness: 0.25 }));
  const blade = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 5, 10),
    new THREE.MeshBasicMaterial({ color: 0xeaf6ff }));
  blade.position.y = 3.1;
  const bladeGlowMat = new THREE.MeshBasicMaterial({ color: 0x35e0ff, transparent: true, opacity: 0.45,
    blending: THREE.AdditiveBlending, depthWrite: false });
  const bladeGlow = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 5.2, 12), bladeGlowMat);
  bladeGlow.position.y = 3.1;
  saber.add(handle, blade, bladeGlow);
  if (fx) {
    const tip = fx.glowSprite(0x35e0ff, 3.5, 0.8);
    tip.position.y = 5.6;
    saber.add(tip);
  }
  saber.position.set(0, 0.6, 2.2);
  scene.add(saber);
  let saberX = 0;

  const flashMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0,
    side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false });
  const flashMesh = new THREE.Mesh(new THREE.RingGeometry(2.2, 3.4, 40, 1, Math.PI * 0.15, Math.PI * 0.7), flashMat);
  flashMesh.position.set(0, 1.0, 0.6);
  scene.add(flashMesh);
  function flashSwing(lane, good) {
    swingFlash = { lane, t: 0.22 };
    flashMat.color.set(good ? 0x5dff8a : 0xff4a4a);
  }

  /* hit particles: pooled additive point clouds */
  const BURST_N = 34;
  const bursts = [];
  function spawnBurst(x, color) {
    let b = bursts.find((b) => b.t <= 0);
    if (!b) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
      const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.9, transparent: true, opacity: 1,
        map: fx ? fx.glowTexture() : null, blending: THREE.AdditiveBlending, depthWrite: false });
      const pts = new THREE.Points(geo, mat);
      pts.visible = false;
      pts.frustumCulled = false;
      scene.add(pts);
      b = { pts, geo, mat, t: 0 };
      bursts.push(b);
    }
    b.t = 0.5;
    b.x = x; b.y = 2.2; b.z = 0;
    b.mat.color.set(color);
    const p = b.geo.getAttribute("position");
    for (let i = 0; i < BURST_N; i++) {
      p.setXYZ(i, b.x, b.y, b.z + (Math.random() - 0.5) * 0.6);
    }
    p.needsUpdate = true;
    b.vel = new Array(BURST_N).fill(0).map(() => ({
      vx: (Math.random() - 0.5) * 9, vy: 2 + Math.random() * 7, vz: -1 - Math.random() * 6,
    }));
  }
  function layoutBursts(dt) {
    for (const b of bursts) {
      if (b.t <= 0) { b.pts.visible = false; continue; }
      b.t -= dt;
      b.pts.visible = true;
      b.mat.opacity = Math.max(0, b.t / 0.5);
      const p = b.geo.getAttribute("position");
      for (let i = 0; i < BURST_N; i++) {
        const v = b.vel[i];
        v.vy -= 9 * dt;
        p.setXYZ(i, p.getX(i) + v.vx * dt, p.getY(i) + v.vy * dt, p.getZ(i) + v.vz * dt);
      }
      p.needsUpdate = true;
    }
  }

  let wallClock = 0;
  function layoutScene(dt) {
    /* effects run on a capped clock so they stay visible at 16x */
    const vdt = Math.min(dt, 1 / 30);
    wallClock += vdt;

    /* notes */
    let mi = 0;
    for (const n of notes) {
      if (n.hit || n.missed) continue;
      const t = timeToBeatLine(n.beat);           // seconds until beat line
      if (t > SPAWN_LEAD || t < -HIT_WINDOW) continue;
      const z = -t * (APPROACH / SPAWN_LEAD);     // 0 at beat line, -APPROACH at spawn
      const g = noteMeshes[mi] || noteMesh();
      retint(g, n.lane);
      g.position.set(LANES[n.lane], 2.2, z);
      g.rotation.x = songClock * 1.5;
      g.visible = true;
      mi++;
    }
    for (let i = mi; i < noteMeshes.length; i++) noteMeshes[i].visible = false;

    /* beat pulse drives the line, the floor and the tunnel */
    const phase = ((songClock % BEAT) + BEAT) % BEAT / BEAT;
    const pulse = Math.max(0, 1 - phase * 3);
    beatMat.emissiveIntensity = 2 + pulse * 5;
    beatLine.scale.y = 1 + pulse * 2;
    for (const g of beatGlows) { g.material.opacity = 0.35 + pulse * 0.6; g.scale.setScalar(4 + pulse * 3); }
    gridMat.uniforms.scroll.value = songClock * (APPROACH / SPAWN_LEAD);
    gridMat.uniforms.pulse.value = pulse;

    const travel = (songClock / BEAT) * FRAME_GAP * 0.5;
    for (let i = 0; i < FRAME_N; i++) {
      let z = -((i * FRAME_GAP - travel) % (FRAME_N * FRAME_GAP));
      if (z > 0) z -= FRAME_N * FRAME_GAP;
      frames[i].position.z = z - 6;
    }
    frameMat.opacity = frameMat2.opacity = 0.45 + pulse * 0.55;

    /* saber: drift to the last swung lane, sweep on a swing */
    if (swingFlash) {
      swingFlash.t -= vdt;
      const k = 1 - Math.max(0, swingFlash.t / 0.22);
      saberX = LANES[swingFlash.lane];
      saber.position.x += (saberX - saber.position.x) * 0.6;
      saber.rotation.z = 1.2 - k * 2.4;
      saber.rotation.x = -0.5 + k * 0.3;
      flashMat.opacity = Math.max(0, swingFlash.t / 0.22) * 0.85;
      flashMesh.position.x = LANES[swingFlash.lane];
      flashMesh.rotation.z = -k * 0.6;
      if (swingFlash.t <= 0) { swingFlash = null; flashMat.opacity = 0; }
    } else {
      flashMat.opacity = 0;
      saber.position.x += (saberX - saber.position.x) * 0.1;
      saber.rotation.z += (0.35 + Math.sin(wallClock * 1.7) * 0.08 - saber.rotation.z) * 0.12;
      saber.rotation.x += (-0.25 - saber.rotation.x) * 0.12;
    }
    bladeGlowMat.opacity = 0.35 + pulse * 0.25;

    /* gentle camera sway on the beat */
    camera.position.set(CAM0.x + Math.sin(wallClock * 0.4) * 0.6, CAM0.y + pulse * 0.08, CAM0.z);
    camera.lookAt(0, 2, -8);

    layoutBursts(vdt);
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
      ["song score", score.toFixed(1)],
      ["hits", hits + " / " + (hits + misses)],
      ["streak", streak + (bestStreak ? " (best " + bestStreak + ")" : "")],
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
      window.flyBrain.getActivity(10).forEach((n, i) => card(n, i));
    }
  }
  function card(n, i) {
    const row = document.getElementById("neu-" + i);
    if (!row) return;
    row.className = "neuron " + n.role;
    row.querySelector(".name").textContent = n.name;
    row.querySelector(".role").textContent = n.role;
    row.querySelector(".fill").style.width = Math.min(100, Math.abs(n.value) * 60) + "%";
    row.querySelector(".val").textContent = n.value.toFixed(2);
  }

  /* learning curve chart */
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
    resetSong();
    startScreen.style.display = "none";
    if (hello) setTimeout(() => { document.getElementById("status").textContent = hello; }, 0);
    note.textContent = "";
    document.getElementById("status").textContent =
      "The fly is playing. Episodes are 16 beats; it learns each song.";
    startMusic();
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
    if (paused) stopMusic(); else if (started) startMusic();
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
  /* Brain acts every 100 ms of SONG time (10 decisions/s regardless of the
     speed multiplier), so fast-forward yields more episodes, not a different
     policy-observation cadence. The song itself advances in 1/60 s ticks. */
  let lastTs = 0, hudLast = 0, chartLast = 0, actAcc = 0, simAcc = 0;
  const SIM_H = 1 / 60;               // song tick, as in the headless mirror
  function frame(ts) {
    const rdt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;
    const dt = rdt * SPEEDS[speedIdx];

    if (started && !paused) {
      /* Advance the song in fixed 1/60 s ticks (exactly like the headless
         mirror in tools/fly-brain-test/envs.js). Stepping the clock once per
         frame and then taking several decisions at that frozen moment made
         timing impossible at 4x/16x or on slow frames — notes never moved
         between decisions, so even a trained fly missed everything. */
      simAcc += dt;
      while (simAcc >= SIM_H) {
        simAcc -= SIM_H;
        songClock += SIM_H;
        spawnUpTo();
        missNotes();
        /* cull consumed notes */
        notes = notes.filter((n) => !n.hit && !n.missed && timeToBeatLine(n.beat) > -HIT_WINDOW);

        actAcc += SIM_H;
        if (actAcc < 0.1 - 1e-9) continue;   // 10 decisions per song-second
        actAcc = 0;
        const a = window.flyBrain.act(stateVector());
        lastActionName = a;
        lastProbs = window.flyBrain.getActionProbs() || lastProbs;
        if (a === "wait") { if (hittableNow()) reward(-0.15); }
        else if (a === "swingL") trySwing(0);
        else if (a === "swingC") trySwing(1);
        else if (a === "swingR") trySwing(2);

        if (songClock > EPISODE_BEATS * BEAT) {
          endSong();
          resetSong();
          simAcc = 0; actAcc = 0;
          break;
        }
      }
    }

    layoutScene(dt);
    renderer.render(scene, camera);
    if (brainViz) { brainViz.update(rdt); brainViz.render(); }
    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(false); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

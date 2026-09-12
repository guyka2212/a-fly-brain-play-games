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
  const wrap = document.getElementById("canvas-wrap");
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e13);
  scene.fog = new THREE.Fog(0x0b0e13, 30, 70);

  const camera = new THREE.PerspectiveCamera(70, 3 / 2, 0.1, 200);
  camera.position.set(0, 6.5, 11);
  camera.lookAt(0, 1.5, -6);

  scene.add(new THREE.HemisphereLight(0xbfd4e8, 0x0c0f13, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.6);
  sun.position.set(10, 30, 10);
  scene.add(sun);

  /* floor grid + lane guides */
  const grid = new THREE.GridHelper(80, 40, 0x1c2733, 0x141c26);
  grid.position.y = -0.5;
  scene.add(grid);
  const laneGuideMat = new THREE.MeshBasicMaterial({ color: 0x1c2733, transparent: true, opacity: 0.7 });
  for (const lx of LANES) {
    const g = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 80), laneGuideMat);
    g.rotation.x = -Math.PI / 2;
    g.position.set(lx, -0.45, -28);
    scene.add(g);
  }

  /* beat line across the lanes */
  const beatLine = new THREE.Mesh(
    new THREE.BoxGeometry(13.5, 0.1, 0.3),
    new THREE.MeshBasicMaterial({ color: 0x58a6ff })
  );
  beatLine.position.set(0, 0, 0);
  scene.add(beatLine);

  /* note cubes per lane */
  const noteGeos = LANES.map((lx, i) => new THREE.BoxGeometry(2.2, 2.2, 2.2));
  const noteMats = [
    new THREE.MeshLambertMaterial({ color: 0xd29922 }),   // left  — amber
    new THREE.MeshLambertMaterial({ color: 0xa371f7 }),   // centre— violet
    new THREE.MeshLambertMaterial({ color: 0x3fb950 }),   // right — green
  ];
  const noteMeshes = [];  // pooled meshes
  function noteMesh() {
    for (const m of noteMeshes) if (!m.visible) return m;
    const m = new THREE.Mesh(noteGeos[0], noteMats[0]);
    scene.add(m); noteMeshes.push(m);
    return m;
  }

  /* saber arm flash: a colored plane in the swung lane */
  const flashMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 });
  const flashMesh = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 3), flashMat);
  flashMesh.position.set(0, 2.2, 0.5);
  scene.add(flashMesh);
  function flashSwing(lane, good) {
    swingFlash = { lane, t: 0.18 };
    flashMat.color.set(good ? 0x3fb950 : 0xf85149);
  }

  /* hit particles: one pooled point cloud per effect, world-positioned */
  const BURST_N = 18;
  const bursts = [];       // {pts, geo, mat, t, x, y, z, color}
  function spawnBurst(x, color) {
    let b = bursts.find((b) => b.t <= 0);
    if (!b) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
      const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.5, transparent: true, opacity: 1 });
      const pts = new THREE.Points(geo, mat);
      pts.visible = false;
      scene.add(pts);
      b = { pts, geo, mat, t: 0 };
      bursts.push(b);
    }
    b.t = 0.4;
    b.x = x; b.y = 2.2; b.z = 0;
    b.mat.color.set(color);
    const p = b.geo.getAttribute("position");
    for (let i = 0; i < BURST_N; i++) {
      p.setXYZ(i, b.x, b.y, b.z + (Math.random() - 0.5) * 0.6);
    }
    p.needsUpdate = true;
    b.vel = new Array(BURST_N).fill(0).map(() => ({
      vx: (Math.random() - 0.5) * 6, vy: 3 + Math.random() * 4, vz: -2 - Math.random() * 4,
    }));
  }
  function layoutBursts(dt) {
    for (const b of bursts) {
      if (b.t <= 0) { b.pts.visible = false; continue; }
      b.t -= dt;
      b.pts.visible = true;
      b.mat.opacity = Math.max(0, b.t / 0.4);
      const p = b.geo.getAttribute("position");
      for (let i = 0; i < BURST_N; i++) {
        const v = b.vel[i];
        p.setXYZ(i, p.getX(i) + v.vx * dt, p.getY(i) + v.vy * dt, p.getZ(i) + v.vz * dt);
      }
      p.needsUpdate = true;
    }
  }

  function layoutScene(dt) {
    /* notes */
    let mi = 0;
    for (const n of notes) {
      if (n.hit || n.missed) continue;
      const t = timeToBeatLine(n.beat);           // seconds until beat line
      if (t > SPAWN_LEAD || t < -HIT_WINDOW) continue;
      const z = -t * (APPROACH / SPAWN_LEAD);     // 0 at beat line, -APPROACH at spawn
      const m = noteMeshes[mi] || noteMesh();
      m.geometry = noteGeos[n.lane];
      m.material = noteMats[n.lane];
      m.position.set(LANES[n.lane], 2.2, z);
      m.rotation.x = songClock * 1.5;
      m.visible = true;
      mi++;
    }
    for (let i = mi; i < noteMeshes.length; i++) noteMeshes[i].visible = false;

    /* beat-line pulse on the beat */
    const phase = ((songClock % BEAT) + BEAT) % BEAT / BEAT;
    const pulse = Math.max(0, 1 - phase * 3);
    beatLine.material.color.setHex(pulse > 0 ? 0x9cd3ff : 0x58a6ff);
    beatLine.scale.y = 1 + pulse * 2;

    /* swing flash */
    if (swingFlash) {
      swingFlash.t -= dt;
      flashMat.opacity = Math.max(0, swingFlash.t / 0.18) * 0.7;
      flashMesh.position.x = LANES[swingFlash.lane];
      flashMesh.position.z = 0.5;
      if (swingFlash.t <= 0) { swingFlash = null; flashMat.opacity = 0; }
    } else flashMat.opacity = 0;

    layoutBursts(dt);
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

  async function start() {
    note.textContent = "loading fly brain…";
    const ok = await ensureBrain();
    if (!ok) { note.textContent = "fly brain failed to load — see console."; return; }
    window.flyBrain.setMode("train");
    started = true; paused = false;
    resetSong();
    startScreen.style.display = "none";
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
  document.getElementById("start-fly").onclick = start;

  /* ============================== main loop =============================== */
  /* Brain acts every 50 ms of SONG time (≤ 20 decisions/s regardless of the
     speed multiplier), so fast-forward yields more episodes, not a different
     policy-observation cadence. */
  let lastTs = 0, hudLast = 0, chartLast = 0, actAcc = 0;
  function frame(ts) {
    const rdt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;
    const dt = rdt * SPEEDS[speedIdx];

    if (started && !paused) {
      songClock += dt;
      spawnUpTo();

      actAcc += dt;
      while (actAcc >= 0.1) {        // 10 decisions per song-second
        actAcc -= 0.1;
        const a = window.flyBrain.act(stateVector());
        lastActionName = a;
        lastProbs = window.flyBrain.getActionProbs() || lastProbs;
        if (a === "wait") { if (hittableNow()) reward(-0.15); }
        else if (a === "swingL") trySwing(0);
        else if (a === "swingC") trySwing(1);
        else if (a === "swingR") trySwing(2);
      }

      missNotes();

      /* cull consumed notes */
      notes = notes.filter((n) => !n.hit && !n.missed && timeToBeatLine(n.beat) > -HIT_WINDOW);

      if (songClock > EPISODE_BEATS * BEAT) {
        endSong();
        resetSong();
      }
    }

    layoutScene(dt);
    renderer.render(scene, camera);
    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(false); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

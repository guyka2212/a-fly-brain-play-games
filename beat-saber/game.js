/* ============================================================================
 * beat-saber/game.js — three.js beat-slicing game for the fly brain.
 *
 * Modes (start screen or ?mode=human|fly):
 *   Human Play — A/S/D swings the saber in the left/centre/right lane.
 *   Watch the Fly Brain Play — flyBrain samples a policy every frame and learns
 *   via REINFORCE from timing + lane-matching rewards.
 *
 * Notes spawn on a generated metronome (WebAudio, no licensed music): a 110 BPM
 * click plus a bass note every bar. Notes fall toward a beat line; the ideal
 * swing moment is when the note crosses it.
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
  const NOTE_INTERVAL = 1;               // spawn every beat
  const SONG_START_DELAY = 1.0;

  /* ============================ fly-brain wiring ========================== */
  const FEATURES = [
    "tLeft",      // time to beat line for nearest left-lane note (1..0, 1 = far)
    "tCentre",
    "tRight",
    "beatPhase",  // 0..1 position within the current beat
  ];
  const ACTIONS = ["swingL", "swingC", "swingR", "wait"];

  let mode = null;             // null | 'human' | 'fly'
  let started = false, paused = false;
  let brainReady = false;
  let lastProbs = [0.25, 0.25, 0.25, 0.25];
  let lastActionName = null;

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

  function resetSong() {
    notes = [];
    songClock = -SONG_START_DELAY;
    songBeat = 0;
    score = 0; streak = 0; bestStreak = 0; hits = 0; misses = 0;
    swingFlash = null;
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
    const tt = (n) => (n === null ? 1 : Math.max(0, Math.min(1, timeToBeatLine(n.beat) / SPAWN_LEAD)));
    const phase = ((songClock % BEAT) + BEAT) % BEAT / BEAT;
    return [tt(laneNote[0]), tt(laneNote[1]), tt(laneNote[2]), phase];
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

  /* ================================ actions =============================== */
  function trySwing(lane, isPerfect) {
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
      const perfect = isPerfect === true || Math.abs(bestDt) < 0.05;
      best.hit = true;
      hits++; streak++;
      bestStreak = Math.max(bestStreak, streak);
      score += streak * (perfect ? 1 : 0.3);
      reward(perfect ? 1 : 0.3);
      flashSwing(lane, true);
    } else {
      streak = 0;
      score -= 0.1;
      reward(-0.1);           // wrong lane / no note to hit
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

  function reward(v) { if (mode === "fly" && brainReady) window.flyBrain.reward(v); }

  /* Episode = one 32-beat "song". endEpisode on completion (fly mode). */
  const EPISODE_BEATS = 32;
  function endSong() {
    if (mode === "fly" && brainReady) window.flyBrain.endEpisode();
  }

  /* ============================== input ================================== */
  const keys = {};
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    keys[k] = true;
    if (k === "a" && started && !paused && mode === "human") trySwing(0);
    if (k === "s" && started && !paused && mode === "human") trySwing(1);
    if (k === "d" && started && !paused && mode === "human") trySwing(2);
    if (k === "r" && started) restart();
    if (k === "escape") toggleMenu();
  });

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

  function layoutScene() {
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

    /* swing flash */
    if (swingFlash) {
      swingFlash.t -= 0.016;
      flashMat.opacity = Math.max(0, swingFlash.t / 0.18) * 0.7;
      flashMesh.position.x = LANES[swingFlash.lane];
      flashMesh.position.z = 0.5;
      if (swingFlash.t <= 0) { swingFlash = null; flashMat.opacity = 0; }
    } else flashMat.opacity = 0;
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
      ["score", score.toFixed(1)],
      ["hits", hits + " / " + (hits + misses)],
      ["streak", streak + (bestStreak ? " (best " + bestStreak + ")" : "")],
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
    if (mode === "fly" && brainReady && started) endSong();
    mode = m; started = true; paused = false;
    resetSong();
    startScreen.style.display = "none";
    bHuman.classList.toggle("on", m === "human");
    bFly.classList.toggle("on", m === "fly");
    document.getElementById("hint").style.visibility = m === "human" ? "visible" : "hidden";
    document.getElementById("status").textContent = m === "fly"
      ? "The fly is playing. Episodes are 32 beats; it learns each song."
      : "";
    note.textContent = "";
    startMusic();
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
  document.getElementById("btn-reset").onclick = () => location.reload();
  document.getElementById("start-human").onclick = () => startMode("human");
  document.getElementById("start-fly").onclick = () => startMode("fly");

  const qmode = new URLSearchParams(location.search).get("mode");
  if (qmode === "fly" || qmode === "ai") startMode("fly");
  else if (qmode === "human") startMode("human");

  function restart() {
    if (mode === "fly" && brainReady && started) endSong();
    resetSong();
    document.getElementById("status").textContent = "";
  }

  /* ============================== main loop =============================== */
  let lastTs = 0, hudLast = 0, chartLast = 0, lastActT = 0;
  function frame(ts) {
    const dt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
    lastTs = ts;

    if (started && !paused) {
      songClock += dt;
      spawnUpTo();      if (mode === "fly" && brainReady && ts - lastActT >= 50) {  // ~20 decisions/s
        lastActT = ts;
        const a = window.flyBrain.act(stateVector());
        lastActionName = a;
        lastProbs = window.flyBrain.getActionProbs() || lastProbs;
        if (a === "swingL") trySwing(0);
        else if (a === "swingC") trySwing(1);
        else if (a === "swingR") trySwing(2);
        /* "wait" = no swing this step */
      }

      missNotes();

      /* cull consumed notes */
      notes = notes.filter((n) => !n.hit && !n.missed && timeToBeatLine(n.beat) > -HIT_WINDOW);

      if (songClock > EPISODE_BEATS * BEAT) {
        endSong();
        resetSong();
      }
    }

    layoutScene();
    renderer.render(scene, camera);
    if (ts - hudLast > 200) { hudLast = ts; updateHUD(); }
    if (ts - chartLast > 600) { chartLast = ts; drawChart(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

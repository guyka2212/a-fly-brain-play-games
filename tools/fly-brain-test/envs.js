/* envs.js — headless mirrors of the three games, shared by verify.js
 * (learning check) and train.js (long offline training of "pro" brains).
 *
 * Each mirror reproduces its game.js EXACTLY: same sensor features, same
 * reward shaping, same action semantics, same timing. If you change a game's
 * rules, change its mirror here in the same commit.
 */
"use strict";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* These mirror each game.js's dynamics EXACTLY (same sensors, same reward
   shaping, same action semantics) so the learning trend measured here is the
   one a browser would see. Contract: step(action) -> done; the DRIVER (not
   the env) calls flyBrain.endEpisode() exactly once per episode. Each factory
   takes (rand, reward): `reward(v)` is how the env pays the agent. Continuous
   envs (driving, beat-saber) expose DT + ACT_EVERY; open-world is fixed-step
   (one action per STEP_TIME, advanced inside step()). */

function drivingEnv(rand, reward) {
  const ROAD = 90, V = 130, DT = 1 / 30, CAP_S = 60;
  const centreX = (d) => 60 + 90 * Math.sin(d / 260) + 50 * Math.sin(d / 97);
  const actions = ["steerL", "hold", "steerR"];
  let s;
  return {
    actions,
    /* Sensor-visibility encoding: signed, zero-centred features so the tuned
       sensors (sign*gain*x + bias, relu) respond continuously in BOTH
       directions. laneOffset/curveSign already are; curveAhead was a 0..1
       sigmoid-ish value that sat at ~0.5 with a negative bias — a permanently
       silent sensor — so it is re-centred to ±1 (0.5 + 0.5*curveAhead). */
    features: ["laneOffset", "vNorm", "curveAheadNear", "curveAheadFar"],
    DT, ACT_EVERY: DT,          // act every tick, like game.js does per frame
    newEpisode() {
      /* random start point along the road, as game.js newSim() */
      const y0 = rand() * 3300;
      s = { x: centreX(y0), y: y0, y0, lane: 0, cp: 0, t: 0, alive: true };
    },
    /* skill readout (not a reward): how long it survived, checkpoints, and
       whether it drove the full 60 s without crashing */
    metrics() {
      return { survivedS: s.t, checkpoints: s.cp, finished: s.alive && s.t > CAP_S };
    },
    state() {
      const cx = centreX(s.y);
      const d1 = centreX(s.y + 40) - cx, d2 = centreX(s.y + 110) - cx;
      const curveAheadSigned = clamp((d2 - d1) / 60, -1, 1);
      return [
        clamp((s.x - cx) / (ROAD / 2), -1, 1),
        V / 260,
        curveAheadSigned,
        clamp(d1 / 60, -1, 1),
      ];
    },
    step(action) {
      if (!s.alive) return true;
      s.lane = action === "steerL" ? -1 : action === "steerR" ? 1 : 0;
      s.x += s.lane * 260 * DT;
      s.y += V * DT;
      s.t += DT;
      if (s.y - s.y0 > (s.cp + 1) * 600) { s.cp++; reward(1); }
      const cx = centreX(s.y), off = s.x - cx;
      if (Math.abs(off) >= ROAD / 2 - 14) reward(-0.05);
      else { reward(0.02); if (Math.abs(off) < ROAD * 0.15) reward(0.03); }
      if (Math.abs(off) > ROAD / 2 + 4) {
        s.alive = false;
        reward(-1);
        return true; // crash — driver ends the episode
      }
      if (s.t > CAP_S) { reward(2); return true; } // time cap — driver ends it
      return false;
    },
  };
}

function beatSaberEnv(rand, reward) {
  const BPM = 110, BEAT = 60 / BPM, SPAWN_LEAD = 2.0, HIT_WINDOW = 0.12;
  /* 16-beat songs: shorter episodes => tighter Monte-Carlo returns and 2x the
     gradient steps per unit experience vs 32 beats. */
  const EPISODE_BEATS = 16, START_DELAY = 1.0, DT = 1 / 60, ACT_EVERY = 0.1;
  /* this game needs a faster optimizer than the default: the per-decision
     credit signal is small, so 0.04 (vs 0.02) is what moves the policy within
     a few hundred episodes (oracle ceiling ≈ +8.5/song, control ≈ −1.2). */
  const LR_GAME = 0.04;
  const actions = ["swingL", "swingC", "swingR", "wait"];
  let notes, songClock, songBeat, m;
  const t2line = (beat) => beat * BEAT - songClock;
  return {
    learningRate: LR_GAME,
    /* Features are COS PHASES (see ph() below): signed values sweeping [−1, 1]
       with +1 exactly at the beat-line crossing. The sensor model fires on
       high |sign*gain*x + bias|, so (a) the critical moment is the feature
       MAXIMUM and (b) sensors respond continuously — urgency-in-[0,1] or
       time-to-line encodings left most sensors permanently silent, which
       collapsed the policy to a state-independent distribution. */
    actions,
    features: ["phLeft", "phCentre", "phRight", "beatPhase"],
    DT, ACT_EVERY,
    newEpisode() {
      notes = []; songClock = -START_DELAY; songBeat = 0;
      m = { hits: 0, perfect: 0, missed: 0, wrongSwings: 0 };
    },
    /* skill readout (not a reward): notes hit / perfect / missed */
    metrics() {
      const total = m.hits + m.missed;
      return { ...m, hitRate: total ? m.hits / total : 0 };
    },
    state() {
      const laneNote = [null, null, null];
      for (const n of notes) {
        if (n.hit || n.missed) continue;
        const t = t2line(n.beat);
        if (t < -HIT_WINDOW || t > SPAWN_LEAD) continue;
        if (laneNote[n.lane] === null || t < t2line(laneNote[n.lane].beat)) laneNote[n.lane] = n;
      }
      /* +1 at the line, −1 a full beat away, 0 when nothing is inbound */
      const ph = (n) => (n === null ? 0 : Math.max(-1, Math.min(1, Math.cos(Math.PI * t2line(n.beat) / BEAT))));
      const phase = (((songClock % BEAT) + BEAT) % BEAT) / BEAT;
      return [ph(laneNote[0]), ph(laneNote[1]), ph(laneNote[2]), Math.cos(2 * Math.PI * phase)];
    },
    tick(dt) { songClock += dt; },
    spawn() {
      const target = Math.floor((songClock + SPAWN_LEAD) / BEAT);
      while (songBeat <= target) {
        if (songBeat >= 0) notes.push({ lane: Math.floor(rand() * 3), hit: false, missed: false, beat: songBeat });
        songBeat++;
      }
    },
    swing(lane) {
      let best = null, bestDt = 1e9;
      for (const n of notes) {
        if (n.hit || n.missed || n.lane !== lane) continue;
        const dt = t2line(n.beat);
        if (Math.abs(dt) < Math.abs(bestDt)) { best = n; bestDt = dt; }
      }
      if (best !== null && Math.abs(bestDt) <= HIT_WINDOW) {
        /* clean hit beats a sloppy one; perfect (within 40% of window) best.
           best.hit = true exactly as game.js trySwing() does — the mirror
           used to omit it, so one note could score twice and then also
           count as a miss. */
        const perfect = Math.abs(bestDt) < HIT_WINDOW * 0.4;
        best.hit = true;
        m.hits++; if (perfect) m.perfect++;
        reward(perfect ? 1 : 0.4);
      } else {
        m.wrongSwings++;
        reward(-0.05);   // wrong lane / no note to hit (mild)
      }
    },
    /* reward for a hittable note RIGHT NOW (wait opportunity cost) */
    hittableNow() {
      for (const n of notes) {
        if (n.hit || n.missed) continue;
        if (Math.abs(t2line(n.beat)) <= HIT_WINDOW) return true;
      }
      return false;
    },
    step(action) {
      if (action === "wait" && this.hittableNow()) reward(-0.15);
      else if (action !== "wait") this.swing({ swingL: 0, swingC: 1, swingR: 2 }[action]);
      return songClock > EPISODE_BEATS * BEAT;    // 16-beat song — driver ends it
    },
    missPass() {
      for (const n of notes) {
        if (!n.hit && !n.missed && t2line(n.beat) < -HIT_WINDOW) { n.missed = true; m.missed++; reward(-0.3); }
      }
      notes = notes.filter((n) => !n.hit && !n.missed && t2line(n.beat) > -HIT_WINDOW);
    },
  };
}

function openWorldEnv(rand, reward) {
  const HALF = 120, STEP_TIME = 0.1, EPISODE_TIME = 45, N_ORBS = 8, WALK = 26, TURN = 2.6;
  const GRID = 40, CELL = (HALF * 2) / GRID;
  const actions = ["forward", "turnL", "turnR", "forwardLeft", "forwardRight"];
  let agent, orbs, visited, epClock, collected;
  const nearest = () => {
    let best = null, bd = 1e9;
    for (const o of orbs) {
      if (o.taken) continue;
      const dx = o.x - agent.x, dz = o.z - agent.z, d2 = dx * dx + dz * dz;
      if (d2 < bd) { bd = d2; best = o; }
    }
    return best === null ? null : { orb: best, dist: Math.sqrt(bd) };
  };
  return {
    actions,
    features: ["orbDist", "orbBearing", "orbBearing2", "wallDist"],
    newEpisode() {
      agent = { x: 0, z: 0, h: rand() * Math.PI * 2 };
      orbs = [];
      for (let i = 0; i < N_ORBS; i++) {
        let ox, oz;
        do { ox = (rand() * 2 - 1) * (HALF - 12); oz = (rand() * 2 - 1) * (HALF - 12); }
        while (ox * ox + oz * oz < 900);
        orbs.push({ x: ox, z: oz, taken: false });
      }
      visited = new Uint8Array(GRID * GRID); epClock = 0; collected = 0;
    },
    /* skill readout (not a reward): orbs collected and time used */
    metrics() { return { orbs: collected, of: N_ORBS, timeS: epClock }; },
    state() {
      const near = nearest();
      let od = 0, bs = 0, bc = 0;
      if (near) {
        od = clamp(1 - near.dist / (HALF * 1.5), 0, 1);
        const rel = Math.atan2(near.orb.x - agent.x, -(near.orb.z - agent.z)) - agent.h;
        bs = clamp(Math.sin(rel), -1, 1); bc = clamp(Math.cos(rel), -1, 1);
      }
      const fx = Math.sin(agent.h), fz = -Math.cos(agent.h);
      const twx = fx > 0 ? (HALF - agent.x) / fx : fx < 0 ? (-HALF - agent.x) / fx : 1e9;
      const twz = fz > 0 ? (HALF - agent.z) / fz : fz < 0 ? (-HALF - agent.z) / fz : 1e9;
      return [od, bs, bc, clamp(Math.min(twx, twz) / HALF, 0, 1)];
    },
    step(action) {
      epClock += STEP_TIME;
      let turn = 0, fwd = 0;
      if (action === "turnL") turn = 1;
      else if (action === "turnR") turn = -1;
      else if (action === "forward") fwd = 1;
      else if (action === "forwardLeft") { turn = 0.6; fwd = 0.8; }
      else if (action === "forwardRight") { turn = -0.6; fwd = 0.8; }
      const near = nearest();
      agent.h += turn * TURN * STEP_TIME;
      const fx = Math.sin(agent.h), fz = -Math.cos(agent.h);
      const sp = WALK * fwd * STEP_TIME;
      let nx = agent.x + fx * sp, nz = agent.z + fz * sp, bumped = false;
      if (nx < -HALF || nx > HALF) { nx = clamp(nx, -HALF, HALF); bumped = true; }
      if (nz < -HALF || nz > HALF) { nz = clamp(nz, -HALF, HALF); bumped = true; }
      agent.x = nx; agent.z = nz;
      const gx = clamp(Math.floor((agent.x + HALF) / CELL), 0, GRID - 1);
      const gz = clamp(Math.floor((agent.z + HALF) / CELL), 0, GRID - 1);
      const gi = gz * GRID + gx;
      let ev = 0;
      if (!visited[gi]) { visited[gi] = 1; ev = bumped ? 2 : 1; } else if (bumped) ev = 3;
      let got = false;
      for (const o of orbs) {
        if (o.taken) continue;
        const dx = o.x - agent.x, dz = o.z - agent.z;
        if (dx * dx + dz * dz < 49) { o.taken = true; collected++; got = true; }
      }
      const near2 = nearest();
      let r = -0.01;
      if (near && near2) r += clamp((near.dist - near2.dist) * 0.08, -0.05, 0.05);
      if (ev === 1 || ev === 2) r += 0.02;
      if (ev >= 2) r -= 0.05;
      if (got) r += 1;
      reward(r);
      return epClock >= EPISODE_TIME || collected >= N_ORBS;
    },
  };
}


/* Play one episode. `policy(state) -> action name`; the caller decides what
   happens at the end (flyBrain.endEpisode() for learning, nothing for an
   evaluation run). Returns true when the episode finished. */
function runEpisode(env, policy) {
  env.newEpisode();
  let actAcc = 0;
  for (;;) {
    if (env.tick) env.tick(env.DT);
    if (env.spawn) env.spawn();
    if (env.missPass) env.missPass();   // beat-saber: misses resolve every tick
    let actDue = true;
    if (env.DT) {
      actAcc += env.DT;
      actDue = actAcc >= env.ACT_EVERY - 1e-9;
      if (actDue) actAcc = 0;
    }
    if (actDue && env.step(policy(env.state()))) return true;
  }
}

const GAMES = [
  { name: "driving-sim", make: drivingEnv },
  { name: "beat-saber", make: beatSaberEnv },
  { name: "open-world", make: openWorldEnv },
];

module.exports = { drivingEnv, beatSaberEnv, openWorldEnv, runEpisode, GAMES, clamp };

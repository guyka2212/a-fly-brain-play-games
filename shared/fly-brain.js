/* ============================================================================
 * fly-brain.js — a connectome-seeded neural network + policy-gradient learner.
 *
 * IMPORTANT FRAMING (please read before using):
 *   This is a *connectome-inspired / seeded* network, NOT a literal biological
 *   simulation of the Drosophila brain. shared/connectome-data.json supplies the
 *   ARCHITECTURE and INITIAL WEIGHTS: each connectome neuron becomes a hidden
 *   node, each weighted edge becomes an initial connection weight, and each
 *   neuron's bias carries over as a resting threshold. Everything after that is
 *   learned live in the browser with a lightweight Monte-Carlo policy-gradient
 *   trainer.
 *
 * Topology (built once at init):
 *   inputs (game features)
 *      └───> relu  sensors (each tuned to a specific input feature)
 *               ├───> relu  interneurons ──────┐
 *               └───> relu  motor pool  <──────┤   (weights seeded from the
 *                                └──> softmax logits   connectome edges)
 *   Motor→action readout weights *and* all seeded weights are trainable, so the
 *   fly brain adapts to each game instead of doing fixed graph propagation.
 *
 * Training: Monte-Carlo policy gradient (REINFORCE) with an average-reward
 * baseline and a small entropy bonus. At episode end the stored states are
 * replayed through the (unchanged) network and the loss
 *     Σ_t [ (G_t - baseline) · -log π(a_t|s_t) ] − β·Σ_t entropy
 * is back-propagated and applied with Adam.
 *
 * API for games:
 *   await flyBrain.init(gameConfig)   -> {sensors, interneurons, motors}
 *   flyBrain.act(stateArray)          -> action name
 *   flyBrain.reward(scalar)           -> fractional reward for the current step
 *   flyBrain.endEpisode()             -> gradient update + history push
 *   flyBrain.getStats()               -> {episode, avgScore, lastScore, history, ...}
 *   flyBrain.getActivity()            -> top activated neurons {id,name,role,value}
 *   flyBrain.getActionProbs()         -> probability array over actions
 *   flyBrain.getHidden()              -> {sensors, interneurons, motors} activations
 *   flyBrain.setMode('train'|'greedy')
 *
 * Load after TensorFlow.js (UMD build exposes global `tf`):
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
 *   <script src="../shared/fly-brain.js"></script>
 * ========================================================================== */
(function () {
  "use strict";

  const DATA_PATH = "../shared/connectome-data.json";
  const ENTROPY_BETA = 0.02;
  const TEMPERATURE = 1.0;
  const LR = 0.02;
  const BASELINE_ALPHA = 0.15;
  const GAMMA = 0.98;

  /* mulberry32 — deterministic PRNG for reproducible initial readout weights. */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const st = {
    config: null,
    sensorNeurons: [],
    interNeurons: [],
    motorNeurons: [],
    vars: [],
    optimizer: null,
    episode: 0,
    epSteps: [],       // [{state, actionIdx, reward, pending}]
    pendingReward: 0,
    baseline: 0,
    lastScore: 0,
    history: [],       // [{ep, score}]
    mode: "train",
    lastProbs: null,
    lastAction: 0,
    lastHidden: { sensors: [], interneurons: [], motors: [] },
  };

  /* --------------------------------------------------------------- data load */

  async function loadConnectome() {
    const res = await fetch(DATA_PATH);
    if (!res.ok) throw new Error("fetch " + DATA_PATH + " -> " + res.status);
    const data = await res.json();
    if (!Array.isArray(data.neurons) || data.neurons.length === 0) {
      throw new Error("connectome-data.json has no neurons");
    }
    return data;
  }

  /* Synthetic fallback so games still run from file:// when the data file can't
     be fetched. Clearly synthetic — only used if the fetch fails. */
  function synthConnectome(cfg) {
    const neurons = [];
    let id = 0;
    for (let f = 0; f < cfg.features.length; f++) {
      for (let s = 0; s < 4; s++) {
        neurons.push({ id: "S" + id, name: "S" + id, role: "sensor", bias: -0.5,
          tuning: { input: f, sign: (s % 2 === 0 ? 1 : -1), gain: 1 },
          connections: [] });
        id++;
      }
    }
    const inter = [];
    for (let i = 0; i < 10; i++) {
      const it = { id: "I" + id, name: "I" + id, role: "interneuron", bias: -1.5, tuning: null, connections: [] };
      for (const s of neurons) {
        if ((s.connections.length + it.connections.length) % 7 === 0) {
          s.connections.push({ target: it.id, weight: 0.8 + (s.connections.length % 5) / 10 });
        }
      }
      inter.push(it); id++;
    }
    const motor = [];
    for (let m = 0; m < 6; m++) {
      const mo = { id: "M" + id, name: "M" + id, role: "motor", bias: -2, tuning: null, connections: [] };
      for (const it of inter) {
        if (it.connections.length % 3 === 0) {
          it.connections.push({ target: mo.id, weight: 0.7 + (it.connections.length % 5) / 10 });
        }
      }
      motor.push(mo); id++;
    }
    return { metadata: { notes: "synthetic fallback (connectome-data.json not available)" },
             neurons: neurons.concat(inter, motor) };
  }

  /* ------------------------------------------------------------- build model */

  function buildModel(data) {
    const S = data.neurons.filter((n) => n.role === "sensor");
    const I = data.neurons.filter((n) => n.role === "interneuron");
    const M = data.neurons.filter((n) => n.role === "motor");
    if (!S.length || !I.length || !M.length) {
      throw new Error("connectome needs sensor + interneuron + motor neurons");
    }
    st.sensorNeurons = S; st.interNeurons = I; st.motorNeurons = M;

    const iIdx = new Map(I.map((n, i) => [n.id, i]));
    const mIdx = new Map(M.map((n, mi) => [n.id, mi]));

    /* W1 [S, I] sensor -> interneuron, W2 [I, M] inter -> motor,
       W3 [S, M] direct sensor -> motor shortcuts (giant-fibre path). */
    const w1 = S.map(() => new Array(I.length).fill(0));
    for (const s of S) for (const c of s.connections) {
      const t = iIdx.get(c.target);
      if (t !== undefined) w1[S.indexOf(s)][t] += Number(c.weight);
    }
    const w2 = I.map(() => new Array(M.length).fill(0));
    for (const it of I) for (const c of it.connections) {
      const t = mIdx.get(c.target);
      if (t !== undefined) w2[I.indexOf(it)][t] += Number(c.weight);
    }
    const w3 = S.map(() => new Array(M.length).fill(0));
    for (const s of S) for (const c of s.connections) {
      const t = mIdx.get(c.target);
      if (t !== undefined) w3[S.indexOf(s)][t] += Number(c.weight);
    }
    const b2 = I.map((n) => Number(n.bias) || 0);
    const b3 = M.map((n) => Number(n.bias) || 0);

    /* readout [M, A] — seeded, trainable */
    const rand = mulberry32(1234);
    const A = st.config.actions.length;
    const w4 = M.map(() => new Array(A).fill(0).map(() =>
      (rand() - 0.5) * 2 / Math.sqrt(M.length)));
    const b4 = new Array(A).fill(0).map(() => (rand() - 0.5) * 0.3);

    const v = {
      W1: tf.variable(tf.tensor2d(w1)),
      W2: tf.variable(tf.tensor2d(w2)),
      W3: tf.variable(tf.tensor2d(w3)),
      b2: tf.variable(tf.tensor1d(b2)),
      b3: tf.variable(tf.tensor1d(b3)),
      W4: tf.variable(tf.tensor2d(w4)),
      b4: tf.variable(tf.tensor1d(b4)),
    };
    st.vars = [v.W1, v.W2, v.W3, v.b2, v.b3, v.W4, v.b4];
    st.optimizer = tf.train.adam(st.config.learningRate || LR);
    return { sensors: S.length, interneurons: I.length, motors: M.length };
  }

  /* ------------------------------------------------------------ forward pass */

  /* Replays one state through the network as a tf graph of the variables.
     Returns { probs, logits } as tensors. With capture=true (default) also
     stores the sensor/interneuron/motor activations as plain arrays in
     st.lastHidden for the overlay. Training replay passes capture=false. */
  function forward(stateArr, capture = true) {
    return tf.tidy(() => {
      const ns = st.sensorNeurons.length, ni = st.interNeurons.length, nm = st.motorNeurons.length;
      const A = st.config.actions.length;
      /* st.vars is the flat trainable array [W1, W2, W3, b2, b3, W4, b4]
         (the shape optimizer.minimize() and getWeights() expect). */
      const [W1, W2, W3, b2, b3, W4, b4] = st.vars;

      /* L1 sensors: value = sign*gain*x[feature] + bias, then relu */
      const sActs = st.sensorNeurons.map((n) => {
        const t = n.tuning || {};
        const idx = Math.min(Math.max(Number.isFinite(t.input) ? t.input : 0, 0),
          st.config.features.length - 1);
        const xv = Number.isFinite(stateArr[idx]) ? stateArr[idx] : 0;
        return Math.max(0, (t.sign || 1) * (t.gain || 1) * xv + (Number(n.bias) || 0));
      });
      const s = tf.tensor1d(sActs).reshape([1, ns]);

      const h2 = tf.relu(tf.add(tf.matMul(s, W1).reshape([ni]), b2));
      const h2r = h2.reshape([1, ni]);
      const h3m = tf.matMul(h2r, W2);                   // [1, M]
      const h3s = tf.matMul(s, W3);                     // [1, M]
      const h3 = tf.relu(tf.add(tf.add(h3m, h3s), b3)); // [1, M]

      const logits = tf.add(tf.matMul(h3, W4).reshape([A]), b4);
      const probs = tf.softmax(tf.div(logits, TEMPERATURE));

      if (capture) {
        st.lastHidden.sensors = sActs;
        st.lastHidden.interneurons = Array.from(h2.dataSync());
        st.lastHidden.motors = Array.from(h3.dataSync());
      }

      return { probs, logits };
    });
  }

  /* ------------------------------------------------------------- public API */

  async function init(gameConfig) {
    st.config = gameConfig;
    let data;
    try {
      data = await loadConnectome();
    } catch (e) {
      console.warn("flyBrain: connectome-data not loadable, using synthetic fallback.", e);
      data = synthConnectome(gameConfig);
    }
    st.epSteps = []; st.pendingReward = 0;
    return buildModel(data);
  }

  /* Sample an action from the policy (train mode) or take argmax (greedy). */
  function act(stateArr) {
    flushPending();
    const f = forward(stateArr);
    const p = Array.from(f.probs.dataSync());
    st.lastProbs = p;

    let actionIdx = 0;
    if (st.mode === "greedy") {
      actionIdx = p.indexOf(Math.max.apply(null, p));
    } else {
      let r = Math.random(), acc = 0;
      for (let i = 0; i < p.length; i++) {
        acc += p[i];
        if (r <= acc || i === p.length - 1) { actionIdx = i; break; }
      }
    }
    st.lastAction = actionIdx;

    st.epSteps.push({
      state: stateArr.slice(),
      actionIdx,
      reward: 0,
      pending: true,
    });
    return st.config.actions[actionIdx];
  }

  /* Accumulate reward for the current open step. Fractional amounts are fine. */
  function reward(value) { st.pendingReward += Number(value) || 0; }

  function flushPending() {
    if (st.epSteps.length && st.epSteps[st.epSteps.length - 1].pending) {
      st.epSteps[st.epSteps.length - 1].reward += st.pendingReward;
      st.epSteps[st.epSteps.length - 1].pending = false;
      st.pendingReward = 0;
    }
  }

  /* End the episode: compute discounted returns, build the REINFORCE loss as a
     tf graph of the (unchanged) variables, and apply the gradient via Adam. */
  function endEpisode() {
    flushPending();
    let score = 0;
    for (const s of st.epSteps) score += s.reward;
    st.baseline += BASELINE_ALPHA * (score - st.baseline);
    st.lastScore = score;
    st.history.push({ ep: st.episode, score: Math.round(score * 100) / 100 });
    st.episode++;

    const steps = st.epSteps;
    if (steps.length) {
      const T = steps.length;
      const G = new Array(T).fill(0);
      let run = 0;
      for (let t = T - 1; t >= 0; t--) { run = steps[t].reward + GAMMA * run; G[t] = run; }
      const baseline = st.baseline;

      const lossFn = () => tf.tidy(() => {
        let loss = tf.scalar(0);
        for (let t = 0; t < T; t++) {
          const f = forward(steps[t].state, false);
          const p = f.probs;                                  // [A]
          const logP = tf.log(tf.gather(p, tf.scalar(steps[t].actionIdx, "int32")));
          const entropy = tf.neg(tf.sum(tf.mul(p, tf.log(tf.add(p, 1e-9)))));
          const adv = G[t] - baseline;
          const term = tf.sub(tf.mul(tf.neg(logP), adv), tf.mul(entropy, ENTROPY_BETA));
          loss = tf.add(loss, term);
        }
        return tf.div(loss, Math.max(T, 1));
      });

      st.optimizer.minimize(lossFn, true, st.vars);
    }
    st.epSteps = [];
  }

  function getStats() {
    return {
      episode: st.episode,
      lastScore: Math.round(st.lastScore * 100) / 100,
      avgScore: st.history.length
        ? Math.round(st.history.reduce((a, h) => a + h.score, 0) / st.history.length * 100) / 100
        : 0,
      baseline: Math.round(st.baseline * 100) / 100,
      history: st.history.slice(-200),
      sensors: st.sensorNeurons.length,
      interneurons: st.interNeurons.length,
      motors: st.motorNeurons.length,
      actions: st.config ? st.config.actions : [],
      features: st.config ? st.config.features : [],
    };
  }

  /* Top activated neurons across the three pools — the fun "who is firing" view. */
  function getActivity(topN = 16) {
    const pools = [
      { role: "sensor", list: st.sensorNeurons, acts: st.lastHidden.sensors },
      { role: "interneuron", list: st.interNeurons, acts: st.lastHidden.interneurons },
      { role: "motor", list: st.motorNeurons, acts: st.lastHidden.motors },
    ];
    const all = [];
    for (const p of pools) {
      for (let i = 0; i < p.list.length; i++) {
        all.push({ id: p.list[i].id, name: p.list[i].name, role: p.role, value: p.acts[i] || 0 });
      }
    }
    all.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return all.slice(0, topN);
  }

  window.flyBrain = {
    init, act, reward, endEpisode, getStats, getActivity,
    getHidden() { return st.lastHidden; },
    /* Debug/persistence: flat copies of every trainable tensor, in
       [W1, W2, W3, b2, b3, W4, b4] order. Used by the test harness to prove
       gradients are applied; also the basis for save/load of trained brains. */
    getWeights() {
      return st.vars.length ? st.vars.map((v) => Array.from(v.dataSync())) : null;
    },
    getActionProbs() { return st.lastProbs; },
    getLastAction() { return st.config ? st.config.actions[st.lastAction] : null; },
    setMode(m) { st.mode = m === "greedy" ? "greedy" : "train"; },
    getMode() { return st.mode; },
  };
})();
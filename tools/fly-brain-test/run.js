#!/usr/bin/env node
/* run.js — headless regression harness for shared/fly-brain.js.
 *
 * Proves (not just eyeballs) that the connectome-seeded network + REINFORCE
 * trainer actually executes in Node:
 *   1. loads the REAL shared/connectome-data.json from disk (fetch shim),
 *   2. runs 50 fake episodes against a stub game config,
 *   3. asserts no exceptions, episode/history growth, weight movement
 *      (checksum before vs after -> gradients applied), and that
 *      getHidden()/getActivity() now see real activations.
 *
 * Run:  npm test        (or)   node tools/fly-brain-test/run.js
 * Exit code 0 = all checks passed.
 */
"use strict";

const path = require("path");
const fs = require("fs");
const assert = require("assert/strict");

const ROOT = path.resolve(__dirname, "..", "..");

/* ---- environment shims: fly-brain.js expects a browser ------------------ */
global.window = {}; // flyBrain attaches here
/* fly-brain.js fetches "../shared/connectome-data.json" relative to the *page*.
   A page sits one level deep (e.g. /driving-sim/), so simulate that here:
   resolve against a virtual page dir directly under ROOT. */
global.fetch = (rel) =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve(JSON.parse(fs.readFileSync(path.resolve(ROOT, "virtual-page", rel), "utf8"))),
  });

/* Browser: the UMD build sets window.tf. CommonJS require() does not, so
   publish the export as the global tf the IIFE reads at call time. */
global.tf = require("@tensorflow/tfjs");
require(path.join(ROOT, "shared", "fly-brain.js"));
const flyBrain = global.window.flyBrain;

/* ---- deterministic fake data -------------------------------------------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);
const ACTIONS = ["left", "right", "stay"];
const N_STEPS = 5;

/* Reward scheme: mild signal so REINFORCE has something to find. */
function rewardFor(action, step) {
  let r = action === "right" ? 0.06 : action === "left" ? -0.05 : 0;
  if (step === N_STEPS - 1) r += 1; // goal bonus
  return r;
}

/* ---- checks -------------------------------------------------------------- */
function checksum() {
  const w = flyBrain.getWeights();
  assert.ok(Array.isArray(w) && w.length === 7, "getWeights() -> [W1,W2,W3,b2,b3,W4,b4]");
  /* Sum of |w|: a signed sum can cancel out across ~7k params and hide real
     gradient movement; absolute sum cannot. */
  return w.reduce((s, t) => s + t.reduce((a, x) => a + Math.abs(x), 0), 0);
}

async function main() {
  const cfg = {
    features: ["laneOffset", "speed", "curveAhead", "obstacleDist"],
    actions: ACTIONS.slice(),
  };
  const pools = await flyBrain.init(cfg);
  console.log("init:", pools, "| dataset loaded from disk (no synthetic fallback)");

  assert.equal(flyBrain.getStats().episode, 0, "starts at episode 0");
  const before = checksum();
  console.log("weight checksum (episode 0):", before.toFixed(4));

  const scoreByEp = [];
  for (let ep = 1; ep <= 50; ep++) {
    let score = 0;
    for (let step = 0; step < N_STEPS; step++) {
      const state = [rand(), rand(), rand(), rand()]; // aligned with features
      const a = flyBrain.act(state); // returns action NAME
      assert.ok(ACTIONS.includes(a), `act() returns a known action, got ${a}`);
      const r = rewardFor(a, step);
      flyBrain.reward(r);
      score += r;
    }
    flyBrain.endEpisode();
    scoreByEp.push(score);
  }

  const s = flyBrain.getStats();
  console.log("\nstats after 50 episodes:", {
    episode: s.episode,
    lastScore: s.lastScore,
    avgScore: s.avgScore,
    history: s.history.length,
    baseline: s.baseline,
  });

  assert.equal(s.episode, 50, "episode count increments to 50");
  assert.equal(s.history.length, 50, "history populated");
  assert.ok(s.history.every((h) => Number.isFinite(h.score)), "scores finite");

  const after = checksum();
  console.log("weight checksum (episode 50):", after.toFixed(4));
  assert.ok(Math.abs(after - before) > 0.01, "weights actually changed (gradients applied)");
  assert.ok(flyBrain.getMode() === "train", "default mode is train");

  /* activation plumbing (the getHidden/getActivity fix). Sensor biases are
     strongly negative, so weak inputs legitimately silence the network —
     probe several states and require at least one to fire. */
  let hidden = null, active = null;
  for (let i = 0; i < 20; i++) {
    flyBrain.act([rand(), rand(), rand(), rand()]);
    hidden = flyBrain.getHidden();
    active = flyBrain.getActivity(8);
    if (active.some((n) => n.value > 0)) break;
  }
  assert.equal(hidden.sensors.length, pools.sensors, "lastHidden.sensors populated");
  assert.equal(hidden.interneurons.length, pools.interneurons, "lastHidden.interneurons populated");
  assert.equal(hidden.motors.length, pools.motors, "lastHidden.motors populated");
  assert.ok(active.length === 8 && active.some((n) => n.value > 0),
    "getActivity sees non-zero activations for at least one probe state");
  console.log("activations ok — top neuron:", active[0].name, active[0].role, active[0].value.toFixed(3));

  const probs = flyBrain.getActionProbs();
  assert.equal(probs.length, ACTIONS.length, "action probs per action");
  assert.ok(probs.every((p) => p >= 0 && p <= 1), "probs are probabilities");
  const dist = ACTIONS.map((_, i) => (probs[i] * 100).toFixed(1) + "%").join(" / ");
  console.log("action probs:", dist);

  const first5 = scoreByEp.slice(0, 5).map((x) => x.toFixed(2)).join(", ");
  const last5 = scoreByEp.slice(-5).map((x) => x.toFixed(2)).join(", ");
  console.log("\nscores ep 1-5:", first5);
  console.log("scores ep 46-50:", last5);

  console.log("\nALL CHECKS PASSED ✅");
}

main().catch((e) => {
  console.error("\nHARNESS FAILED ❌\n", e);
  process.exit(1);
});

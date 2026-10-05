#!/usr/bin/env node
/* train.js — long offline training of a "pro" fly brain for one game.
 *
 * Same agent (shared/fly-brain.js), same connectome, same REINFORCE trainer
 * and the same game rules (envs.js mirrors game.js exactly) as the browser —
 * just thousands of episodes instead of the few hundred you'd watch live.
 * Every EVAL_EVERY episodes the brain is tested on fresh tracks/songs/orb
 * layouts it has not trained on, both greedy (top action) and sampled; the
 * most skilled brain (real skill metrics, not shaped reward) is written to
 * shared/trained/<game>.json with the play mode it does best in. The game
 * page offers it as "Watch the Pro"; the fly keeps learning live from there.
 *
 * Run:  node tools/fly-brain-test/train.js <game> [episodes] [seed]
 *       games: driving-sim | beat-saber | open-world
 */
"use strict";

const path = require("path");
const fs = require("fs");
const { GAMES, runEpisode } = require("./envs.js");

const ROOT = path.resolve(__dirname, "..", "..");
const DATA_FILE = path.join(ROOT, "shared", "connectome-data.json");
const OUT_DIR = path.join(ROOT, "shared", "trained");

global.window = {};
global.fetch = () => Promise.resolve({
  ok: true, json: () => Promise.resolve(JSON.parse(fs.readFileSync(DATA_FILE, "utf8"))),
});
global.tf = require("@tensorflow/tfjs");
require(path.join(ROOT, "shared", "fly-brain.js"));
const flyBrain = global.window.flyBrain;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);

const EVAL_EVERY = 100, EVAL_EPISODES = 20, EXAM_EPISODES = 100;

/* summarise a list of per-episode metrics into one human readout */
function skillSummary(game, ms, scores) {
  const s = { score: mean(scores) };
  if (game === "driving-sim") {
    s.finishedPct = 100 * mean(ms.map((m) => (m.finished ? 1 : 0)));
    s.survivedS = mean(ms.map((m) => m.survivedS));
    s.checkpoints = mean(ms.map((m) => m.checkpoints));
  } else if (game === "beat-saber") {
    s.hitRatePct = 100 * mean(ms.map((m) => m.hitRate));
    s.perfectPct = 100 * mean(ms.map((m) => (m.hits ? m.perfect / m.hits : 0)));
    s.wrongSwings = mean(ms.map((m) => m.wrongSwings));
  } else {
    s.orbs = mean(ms.map((m) => m.orbs));
    s.allOrbsPct = 100 * mean(ms.map((m) => (m.orbs === m.of ? 1 : 0)));
    s.timeS = mean(ms.map((m) => m.timeS));
  }
  return s;
}
const fmt = (s) => Object.entries(s).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ");

/* Evaluation on its own random stream (unseen layouts), in one play mode:
   "greedy" = always the top action; "sample" = draw from the policy, as in
   training. Some skills only exist as a mix: 30 Hz driving with three
   buttons steers by dithering steerL/hold/steerR, so greedy drifts off the
   road while sampling drives the full course. Both are measured. */
function evaluate(game, make, evalSeed, mode, episodes = EVAL_EPISODES) {
  const rand = mulberry32(evalSeed);
  let score = 0;
  const env = make(rand, (v) => { score += v; });
  const ms = [], scores = [];
  flyBrain.setMode(mode === "greedy" ? "greedy" : "train");
  for (let i = 0; i < episodes; i++) {
    score = 0;
    runEpisode(env, (st) => flyBrain.act(st));
    flyBrain.discardEpisode();
    ms.push(env.metrics()); scores.push(score);
  }
  flyBrain.setMode("train");
  return Object.assign({ mode }, skillSummary(game, ms, scores));
}

/* one number to rank brains by real skill (not shaped reward) */
function skillRank(game, ev) {
  /* ties (e.g. many brains finishing every drive) go to the cleaner player */
  if (game === "driving-sim") return ev.finishedPct * 1000 + ev.survivedS * 10 + ev.checkpoints + ev.score * 0.01;
  if (game === "open-world") return ev.allOrbsPct * 1000 + ev.orbs * 100 - ev.timeS + ev.score * 0.01;
  /* beat-saber: notes hit, then clean timing, and a pro doesn't flail —
     wasted swings cost only −0.05 reward each, too little to rank by score */
  return ev.hitRatePct * 2 + ev.perfectPct - ev.wrongSwings;
}

async function main() {
  const game = process.argv[2];
  const episodes = Number(process.argv[3]) || 3000;
  const seed = Number(process.argv[4]) || 7;
  const g = GAMES.find((x) => x.name === game);
  if (!g) { console.error("usage: train.js <" + GAMES.map((x) => x.name).join("|") + "> [episodes] [seed]"); process.exit(2); }

  const rand = mulberry32(seed);
  const env = g.make(rand, (v) => flyBrain.reward(v));
  await flyBrain.init({ features: env.features, actions: env.actions, learningRate: env.learningRate });
  flyBrain.setMode("train");

  let best = null;
  const log = [];
  const t0 = Date.now();
  for (let ep = 1; ep <= episodes; ep++) {
    runEpisode(env, (st) => flyBrain.act(st));
    flyBrain.endEpisode();
    if (ep % EVAL_EVERY === 0 || ep === episodes) {
      const evs = ["greedy", "sample"].map((m) => evaluate(game, g.make, 999, m));
      const ev = skillRank(game, evs[0]) >= skillRank(game, evs[1]) ? evs[0] : evs[1];
      log.push({ ep, ...ev });
      const isBest = !best || skillRank(game, ev) > skillRank(game, best.eval);
      if (isBest) best = { ep, eval: ev, brain: flyBrain.exportBrain() };
      const show = (e) => fmt(Object.fromEntries(Object.entries(e).filter(([k]) => k !== "mode")));
      console.log(`${game} ep ${ep} · ${((Date.now() - t0) / 1000).toFixed(0)}s · greedy: ${show(evs[0])} | sample: ${show(evs[1])}${isBest ? "  ★ best (" + ev.mode + ")" : ""}`);
    }
  }

  /* final exam: the chosen brain, 100 fresh runs on a different seed */
  flyBrain.importBrain(best.brain);
  const exam = evaluate(game, g.make, 4242, best.eval.mode, EXAM_EPISODES);
  console.log(`exam (${EXAM_EPISODES} unseen runs, ${exam.mode}): ${fmt(Object.fromEntries(Object.entries(exam).filter(([k]) => k !== "mode")))}`);

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const out = Object.assign(best.brain, {
    game,
    playMode: best.eval.mode,          // how this brain plays best: greedy | sample
    trainedOffline: { episodes: best.ep, seed, evalEpisodes: EVAL_EPISODES, eval: best.eval,
      examEpisodes: EXAM_EPISODES, exam,
      note: "Trained headless in Node with tools/fly-brain-test/train.js on exact mirrors of the game rules (envs.js); the brain with the best skill evaluation on unseen layouts was kept." },
    generated_at: new Date().toISOString(),
  });
  const file = path.join(OUT_DIR, game + ".json");
  fs.writeFileSync(file, JSON.stringify(out) + "\n");
  console.log(`\nbest at episode ${best.ep} (${best.eval.mode}): ${fmt(Object.fromEntries(Object.entries(best.eval).filter(([k]) => k !== "mode")))}\nwrote ${path.relative(ROOT, file)}`);
}

main().catch((e) => { console.error(e); process.exit(1); });

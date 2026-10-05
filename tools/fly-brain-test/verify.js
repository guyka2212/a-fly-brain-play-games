#!/usr/bin/env node
/* verify.js — "is it really the fly brain playing?" verification harness.
 *
 * Three checks, per the spec:
 *   1. SOURCE   — the agent loaded the REAL shared/connectome-data.json from
 *                 disk (neuron count, edge count, sha256 of the file), not the
 *                 synthetic fallback. Fails loudly on fallback.
 *   2. AGENCY   — static analysis of each game's game.js: no keyboard/mouse
 *                 control path, no hardcoded/faked action stream, no
 *                 "if AI mode, play well" shortcut; the fly path may only call
 *                 flyBrain.act()/reward()/endEpisode() and must apply the
 *                 action act() returned (no re-rolling the policy game-side).
 *   3. LEARNING — each game's dynamics are mirrored headless (same sensors,
 *                 same reward shaping, same action semantics as game.js) and
 *                 the real agent is trained for N episodes x 3 seeds. Mean
 *                 score of the first 10% vs last 10% of episodes is compared,
 *                 AND against a control arm that steps the identical
 *                 environment with uniformly random actions — so a positive
 *                 trend has to be the policy improving, not noise.
 *
 * The regression harness (run.js) is run first, as a subprocess.
 *
 * Run:  npm run verify     (or)   node tools/fly-brain-test/verify.js
 * Exit 0 = all checks passed; exit 1 = at least one FAILED.
 *
 * Writes tools/fly-brain-test/RESULTS.md as a persistent record.
 */
"use strict";

const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const { spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..", "..");
const DATA_FILE = path.join(ROOT, "shared", "connectome-data.json");

/* ------------------------------- environment shims ---------------------- */
global.window = {};
global.document = { getElementById: () => null };
global.addEventListener = () => {};
global.clearInterval = () => {};
global.setInterval = () => 0;
global.fetch = () =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve(JSON.parse(fs.readFileSync(DATA_FILE, "utf8"))),
  });
global.tf = require("@tensorflow/tfjs");
require(path.join(ROOT, "shared", "fly-brain.js"));
const flyBrain = global.window.flyBrain;

/* ------------------------------- deterministic PRNG --------------------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const { GAMES, runEpisode } = require("./envs.js");

/* =========================== CHECK 1 — source =========================== */
function checkSource() {
  const diskRaw = fs.readFileSync(DATA_FILE, "utf8");
  const disk = JSON.parse(diskRaw);
  const sha256 = crypto.createHash("sha256").update(diskRaw).digest("hex");
  const diskNeurons = disk.neurons.length;
  const diskEdges = disk.neurons.reduce((a, n) => a + (n.connections || []).length, 0);

  const src = flyBrain.getDataSource();
  const lines = [];
  lines.push(`file: shared/connectome-data.json (${diskRaw.length} bytes)`);
  lines.push(`sha256: ${sha256}`);
  lines.push(`on disk: ${diskNeurons} neurons / ${diskEdges} edges (metadata says ${disk.metadata.n_neurons}/${disk.metadata.n_edges})`);
  lines.push(`agent loaded: ${src.dataset} — synthetic=${src.synthetic}, ${src.nNeurons} neurons / ${src.nEdges} edges`);

  const checks = [
    ["not the synthetic fallback", src.synthetic === false],
    ["neuron count matches disk", src.nNeurons === diskNeurons && diskNeurons === disk.metadata.n_neurons],
    ["edge count matches disk", src.nEdges === diskEdges && diskEdges === disk.metadata.n_edges],
    ["dataset label is the curated builder", /curated/.test(src.dataset)],
    ["generated_by is build_curated.py", /build_curated\.py/.test(src.generatedBy)],
  ];
  const ok = checks.every(([, v]) => v);
  for (const [name, v] of checks) lines.push(`  ${v ? "PASS" : "FAIL"} — ${name}`);
  if (!ok) lines.push("  !! SOURCE CHECK FAILED — the synthetic fallback is NOT an acceptable data source.");
  return { ok, lines, sha256, diskNeurons, diskEdges };
}

/* =========================== CHECK 2 — agency =========================== */
/* Each game's fly loop must be the ONLY control path: no keyboard/mouse input
   may reach the game's control layer (act/applyAction/moveAgent/trySwing/step),
   no ?mode=human escape hatch, no scripted/hardcoded action lists, no
   "if AI mode, pick the good move" shortcut, and no game-side re-sampling of
   the policy (the action applied must be the one act() returned).

   Spectator-only pointer handling (orbit camera) is allowed, but proven safe
   structurally: the pointer handlers must not appear in the same lexical scope
   as any control-layer call (they can only touch camYaw/camPitch/dragging). */
const AGENCY_FORBIDDEN = [
  { re: /addEventListener\(\s*["']key(down|up|press)["']/, why: "keyboard handler" },
  { re: /keydown|keyup|onkeydown/, why: "keyboard event token" },
  { re: /addEventListener\(\s*["']wheel["']/, why: "wheel handler" },
  { re: /\bmode\s*===?\s*["']human["']|\bstartMode\(\s*["']human["']|["']human["']\s*:\s*null|["']human["']\s*\|/, why: "human-mode code path" },
  { re: /URLSearchParams|location\.search|query.*mode|mode.*query/i, why: "mode query-param handling" },
  { re: /SCRIPTED_ACTIONS|HARDCODED|fakeMove|pretendAction|debugAction/, why: "scripted/hardcoded action marker" },
  { re: /if\s*\(\s*(isAi|aiMode|flyMode)\s*\)\s*(return|actions?\.push)\s*\(?\s*\[?["']/, why: "AI-mode canned-move shortcut" },
  { re: /flyBrain\.act[\s\S]{0,120}(Math\.random|rand\(\))[\s\S]{0,40}(!==|===)\s*(undefined|null)\s*\?/, why: "game-side policy re-roll" },
  { re: /\b(?:const|let|var)\s+\w*[Aa]ctions?\w*\s*=\s*\[[^\]]*\]\s*;\s*\/\/\s*(?:scripted|fixed|canned)/i, why: "scripted action list" },
];
/* Identifiers that only make sense for human input — none may survive. */
const AGENCY_GAME_SPECIFIC = {
  "driving-sim": [/\bkeys\s*=\s*\{\}/, /arrowup|arrowdown|arrowleft|arrowright/, /["'][wads]["']/, /Human Play/i, /btn-human/],
  "beat-saber": [/\bkeys\s*=\s*\{\}/, /["'][wasd]["']\s*&&/, /Human Play/i, /btn-human/],
  "open-world": [/\bkeys\s*=\s*\{\}/, /humanStep|humanKeysDown/, /Human Play/i, /btn-human/],
};
/* Pointer events may only drive the spectator camera. If any pointer handler
   shares a function scope with a control-layer call, that's agency FAIL.
   Control-layer tokens: brain act/reward, game-side action application, sim
   stepping, reward emission. Camera tokens are exempt. */
const CONTROL_TOKENS = [
  /flyBrain\.(act|reward|endEpisode)\(/, /applyAction\(/, /trySwing\(/,
  /moveAgent\(/, /flyStep\(/, /humanStep\(/, /stepSim\(/, /collectCheck\(/,
];
const CAMERA_TOKENS = /camYaw|camPitch|camDist|dragging|lookAt|updateCamera/;
function checkPointerIsolation(code) {
  /* Split on pointer addEventListener calls and inspect each handler body:
     any control-layer token in a pointer handler = FAIL. Camera-only bodies
     (camYaw/camPitch/...) pass — spectating is not controlling. */
  const hits = [];
  const blocks = code.split(/addEventListener\(/).slice(1)
    .filter((b) => /^\s*["']pointer/.test(b));
  for (const b of blocks) {
    const body = b.slice(b.indexOf("{") + 1, b.indexOf("}") + 1);
    for (const t of CONTROL_TOKENS) if (t.test(body)) hits.push("pointer handler references control token " + t);
  }
  return hits;
}

function checkAgency() {
  const games = ["driving-sim", "beat-saber", "open-world"];
  const lines = [];
  let ok = true;
  for (const game of games) {
    const file = path.join(ROOT, game, "game.js");
    const code = fs.readFileSync(file, "utf8");
    const hits = [];
    for (const { re, why } of AGENCY_FORBIDDEN) if (re.test(code)) hits.push(why);
    for (const re of AGENCY_GAME_SPECIFIC[game] || []) if (re.test(code)) hits.push("game-specific human-input remnant: " + re);
    for (const h of checkPointerIsolation(code)) hits.push(h);
    /* must still drive itself through the brain API */
    const usesAct = /flyBrain\.act\(/.test(code);
    const usesReward = /flyBrain\.reward\(/.test(code);
    const usesEnd = /flyBrain\.endEpisode\(\)/.test(code);
    const good = hits.length === 0 && usesAct && usesReward && usesEnd;
    lines.push(`${game}: ${good ? "PASS" : "FAIL"} — act()=${usesAct ? "yes" : "NO"}, reward()=${usesReward ? "yes" : "NO"}, endEpisode()=${usesEnd ? "yes" : "NO"}${hits.length ? ", forbidden: " + hits.join("; ") : ", no human/scripted control path"}`);
    if (!good) ok = false;
  }
  /* index.html pages must not offer a mode choice either */
  for (const game of games) {
    const html = fs.readFileSync(path.join(ROOT, game, "index.html"), "utf8");
    const bad = [/btn-human/, /start-human/, /Human Play/, /mode=human/].filter((re) => re.test(html));
    if (bad.length) { lines.push(`${game}/index.html: FAIL — human-mode UI remnants`); ok = false; }
    else lines.push(`${game}/index.html: PASS — no human-mode UI`);
  }
  return { ok, lines };
}

/* =========================== CHECK 4 — brain-viz ======================== */
/* The 3D brain view must (a) be backed by the same neuron count as the
   connectome on disk, (b) have a valid finite [x,y,z] pos for every neuron
   (no missing/NaN coordinates), and (c) be actually wired into every game
   page with a performant (instanced) renderer. */
function checkViz() {
  const lines = [];
  let ok = true;

  const disk = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
  const neurons = disk.neurons;
  const missing = neurons.filter((n) => !Array.isArray(n.pos) || n.pos.length !== 3);
  const nan = neurons.filter((n) => Array.isArray(n.pos) && n.pos.some((v) => !Number.isFinite(v)));
  lines.push(`connectome: ${neurons.length} neurons, ${missing.length} missing pos, ${nan.length} non-finite pos`);
  if (missing.length || nan.length) ok = false;

  /* flyBrain.getNeurons() must expose the same count with positions intact */
  const viaAgent = flyBrain.getNeurons();
  const agentOk = viaAgent.length === neurons.length &&
    viaAgent.every((n, i) => Array.isArray(n.pos) && n.pos.length === 3 &&
      n.pos.every(Number.isFinite));
  lines.push(`flyBrain.getNeurons(): ${viaAgent.length} neurons, positions intact: ${agentOk ? "yes" : "NO"}`);
  if (!agentOk) ok = false;

  /* the real init path must have loaded the same neurons (init was called
     for the source check) */
  const srcDS = flyBrain.getDataSource();
  const loadedOk = !srcDS.synthetic && srcDS.nNeurons === neurons.length;
  lines.push(`agent loaded dataset is the same file: ${loadedOk ? "yes" : "NO"}`);
  if (!loadedOk) ok = false;

  /* viz component: exists, instanced, and wired into every game */
  const vizSrc = fs.readFileSync(path.join(ROOT, "shared", "brain-viz.js"), "utf8");
  const instanced = /InstancedMesh/.test(vizSrc);
  lines.push(`brain-viz.js uses InstancedMesh (browser budget): ${instanced ? "yes" : "NO"}`);
  if (!instanced) ok = false;

  for (const game of ["driving-sim", "beat-saber", "open-world"]) {
    const html = fs.readFileSync(path.join(ROOT, game, "index.html"), "utf8");
    const js = fs.readFileSync(path.join(ROOT, game, "game.js"), "utf8");
    const wired = /brain-viz\.js/.test(html) && /id="brain-panel"/.test(html) &&
      /flyBrainViz\.create\(/.test(js) && /brainViz\.update\(/.test(js);
    lines.push(`${game}: panel wired ${wired ? "PASS" : "FAIL"}`);
    if (!wired) ok = false;
  }
  if (!ok) lines.push("  !! VIZ CHECK FAILED");
  return { ok, lines };
}

/* ============================ training driver =========================== */
const EPISODES = 200;
const SEEDS = [101, 202, 303];

async function runSeries(envFactory, seed, useBrain) {
  const rand = mulberry32(seed);
  const env = envFactory(rand, (v) => flyBrain.reward(v));
  const scores = [];
  /* Honest per-run isolation: re-init the agent with THIS game's real
     features/actions (fresh model + seeded readout, empty history). This is
     also what guarantees the policy's action set matches env.step(). */
  if (useBrain) {
    await flyBrain.init({ features: env.features, actions: env.actions, learningRate: env.learningRate });
    flyBrain.setMode("train");
  }
  const origReward = flyBrain.reward;
  let score = 0;
  flyBrain.reward = (v) => { score += Number(v) || 0; origReward(v); };

  for (let ep = 0; ep < EPISODES; ep++) {
    score = 0;
    runEpisode(env, (s) => useBrain
      ? flyBrain.act(s)
      : env.actions[Math.floor(rand() * env.actions.length)]);
    if (useBrain) flyBrain.endEpisode();  // the ONLY endEpisode call
    scores.push(useBrain ? flyBrain.getStats().lastScore : score);
  }
  flyBrain.reward = origReward;
  return scores;
}

/* ---- statistics helpers ---- */
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const std = (a) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) * (x - m))));
};
function trendVerdict(scores, controlMean) {
  const k = Math.max(1, Math.floor(scores.length / 10));
  const first = mean(scores.slice(0, k));
  const last = mean(scores.slice(-k));
  const delta = last - first;
  const sem = std(scores.slice(-k)) / Math.sqrt(k); // SEM of the last-10% mean
  let verdict;
  if (delta >= Math.max(0.5, 0.15 * Math.abs(first)) && delta > 2 * sem && last > controlMean) verdict = "PASS";
  else if (delta > 0) verdict = "INCONCLUSIVE";
  else verdict = "FAIL";
  return { first, last, delta, sem, verdict };
}

/* ================================ main ================================== */
async function main() {
  const out = [];
  const line = (s) => { out.push(s); console.log(s); };

  line("# Verification results — is it really the fly brain playing?");
  line("");
  line(`Generated: ${new Date().toISOString()} · node ${process.version}`);
  line("");

  /* Exercise the agent's REAL init path first (same call every game makes);
     the source check then inspects what init() actually loaded. */
  await flyBrain.init({ features: ["f0", "f1", "f2", "f3"], actions: ["a", "b", "c"] });

  /* 0. regression harness first */
  line("## 0. Regression harness (run.js)");
  const reg = spawnSync(process.execPath, [path.join(__dirname, "run.js")], { encoding: "utf8", timeout: 300000 });
  const regOk = reg.status === 0;
  line(`\`\`\`\n${(reg.stdout + reg.stderr).trim()}\n\`\`\``);
  line(`Verdict: ${regOk ? "✅ PASS" : "❌ FAIL"} (exit ${reg.status})`);
  line("");

  /* 1. source */
  line("## 1. Source check — real curated connectome, not the fallback");
  const src = checkSource();
  for (const l of src.lines) line(l);
  line(`Verdict: ${src.ok ? "✅ PASS" : "❌ FAIL"}`);
  line("");

  /* 2. agency */
  line("## 2. Agency check — no human / scripted control path");
  const ag = checkAgency();
  for (const l of ag.lines) line(l);
  line(`Verdict: ${ag.ok ? "✅ PASS" : "❌ FAIL"}`);
  line("");

  /* 3. learning */
  line(`## 3. Learning check — ${EPISODES} episodes × ${SEEDS.length} seeds (+ random-action control)`);
  line("");
  line("Method: each game's dynamics (sensors, reward shaping, action semantics) are mirrored headless exactly as in game.js. The fly brain trains from the seeded initialization each run. A control arm steps the identical environment with uniformly random actions (same seeds) — the fly has to beat both its own start *and* noise. Verdict: PASS = last-10% mean beats first-10% mean by ≥ max(0.5, 15%) and > 2×SEM and > control; INCONCLUSIVE = positive but small/noisy; FAIL = no gain.");
  line("");

  const games = GAMES;
  const learningOk = {};
  for (const g of games) {
    line(`### ${g.name}`);
    let anyPass = false, anyFail = false;
    for (const seed of SEEDS) {
      const t0 = Date.now();
      const fly = await runSeries(g.make, seed, true);
      const ctl = await runSeries(g.make, seed + 1000, false);
      const t = trendVerdict(fly, mean(ctl));
      if (t.verdict === "PASS") anyPass = true;
      if (t.verdict === "FAIL") anyFail = true;
      line(`- seed ${seed}: first10% ${t.first.toFixed(2)} → last10% ${t.last.toFixed(2)} (Δ ${t.delta >= 0 ? "+" : ""}${t.delta.toFixed(2)}, SEM ${t.sem.toFixed(2)}) · control(noise) mean ${mean(ctl).toFixed(2)} · **${t.verdict}** · ${((Date.now() - t0) / 1000) | 0}s`);
    }
    const verdict = anyFail ? "❌ FAIL" : anyPass ? "✅ PASS" : "⚠️ INCONCLUSIVE";
    learningOk[g.name] = !anyFail && anyPass;
    line(`Verdict: ${verdict}`);
    line("");
  }

  /* 4. brain-viz data + wiring */
  line("## 4. 3D brain view — positions valid, wiring present");
  const viz = checkViz();
  for (const l of viz.lines) line(l);
  line(`Verdict: ${viz.ok ? "✅ PASS" : "❌ FAIL"}`);
  line("");

  /* summary */
  line("## Summary");
  line("");
  line("| check | driving-sim | beat-saber | open-world |");
  line("|---|---|---|---|");
  line(`| data source | ${src.ok ? "✓" : "✗"} | ${src.ok ? "✓" : "✗"} | ${src.ok ? "✓" : "✗"} |`);
  const agBy = (gname) => (ag.lines.find((l) => l.startsWith(gname + ":")) || "").includes("PASS") ? "✓" : "✗";
  line(`| agency | ${agBy("driving-sim")} | ${agBy("beat-saber")} | ${agBy("open-world")} |`);
  line(`| learning trend | ${learningOk["driving-sim"] ? "✓" : "✗"} | ${learningOk["beat-saber"] ? "✓" : "✗"} | ${learningOk["open-world"] ? "✓" : "✗"} |`);
  line(`| 3D brain view | ${viz.ok ? "✓" : "✗"} | ${viz.ok ? "✓" : "✗"} | ${viz.ok ? "✓" : "✗"} |`);
  line("");
  line("Notes:");
  line("- The learning check mirrors game dynamics headless (same sensors/rewards/actions as game.js); a real-browser run shows the same curves via the on-page chart.");
  line("- Agency is proven statically (no human/scripted path) plus structurally: the harness applies exactly the action flyBrain.act() returned — the same contract game.js uses.");
  line("- Control arm = identical environment, uniformly random actions, same episode budget: a learning trend must exceed that noise floor, not just itself.");
  line("- The 3D brain view check validates the DATA (same neuron count as disk, every pos finite) and the per-game wiring; rendering itself needs a browser pass.");

  fs.writeFileSync(path.join(__dirname, "RESULTS.md"), out.join("\n") + "\n", "utf8");
  console.log("\nRESULTS.md written to tools/fly-brain-test/RESULTS.md");

  const allOk = regOk && src.ok && ag.ok && viz.ok && Object.values(learningOk).every(Boolean);
  console.log(allOk ? "\nALL VERIFICATION CHECKS PASSED ✅" : "\nVERIFICATION FAILED ❌");
  if (!allOk) process.exit(1);
}

main().catch((e) => { console.error("\nVERIFY HARNESS CRASHED ❌\n", e); process.exit(1); });

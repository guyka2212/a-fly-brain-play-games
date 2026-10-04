# CLAUDE.md

Guidance for AI agents and humans working in this repo.

## What this is

**a-fly-brain-play-games** — a static, browser-based playground where an artificial
fly brain, seeded from a curated *Drosophila* escape/leg motor connectome, learns to
play games via reinforcement learning.

There is **no build step and no bundler**. Games are vanilla JS (IIFE + globals,
TF.js from CDN). The repo has one npm manifest, `package.json`, whose only purpose
is the **Node regression harness** (`npm test`, dev-dep `@tensorflow/tfjs`) — the
site itself still loads TF.js from CDN. Data tooling is Python 3.10+.

Framing that must be preserved everywhere: this is a **connectome-inspired / seeded**
network, **not** a literal biological simulation of the fly brain. Keep the honesty
notes in file headers and READMEs when you edit them.

## Repo layout

```
shared/fly-brain.js            # core agent: connectome-seeded network + REINFORCE trainer
                               # (IIFE exposing global `flyBrain`)
shared/connectome-data.json    # committed static data: 92 neurons / 966 edges.
                               # GENERATED — do not hand-edit.
tools/fly-brain-test/run.js    # headless Node regression harness (npm test)
tools/fly-brain-test/verify.js # source / agency / learning checks (npm run verify)
                               # -> rewrites tools/fly-brain-test/RESULTS.md
tools/neuron-fetch/
  build_curated.py             # offline, deterministic builder (SEED 20260911), no network
  fetch_connectome.py          # live pull from Janelia neuPrint hemibrain (needs token)
  requirements.txt             # only needed for the live fetch
open-world/ beat-saber/ driving-sim/   # game folders: index.html (page) + game.js
                                        # (logic); three.js + TF.js from CDN
.github/workflows/deploy.yml   # GitHub Pages deploy (static upload of repo root)
index.html                     # hub page linking the 3 games
README.md
```

There is no separate vestigial package-lock: `package.json` + lockfile belong to
the harness. Do not add npm tooling for the site itself.

## Commands

```bash
# Regression harness — run after ANY change to shared/fly-brain.js
npm install        # first time only (installs @tensorflow/tfjs for Node)
npm test           # 50 fake episodes; exit 0 = training loop + API verified
npm run verify     # ~5 min: data source, agency, multi-seed learning checks

# Serve locally — MUST run from repo root (see "Relative paths" gotcha below)
python -m http.server 8000
# then open http://localhost:8000/<game>/

# Regenerate connectome data (offline, deterministic)
python tools/neuron-fetch/build_curated.py    # writes shared/connectome-data.json

# Live connectome fetch (rarely needed)
pip install -r tools/neuron-fetch/requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS="<token>"
python tools/neuron-fetch/fetch_connectome.py
```

**Verification:** `npm test` for the agent/trainer (it loads the real connectome
from disk, runs 50 episodes, and asserts weights move). For game pages, also serve
the site and exercise them in the browser: console clean, scores/episodes advance,
activations render.

## Architecture (`shared/fly-brain.js`)

- **Data:** fetches `../shared/connectome-data.json` at init; falls back to a clearly
  labeled synthetic connectome if the fetch fails (so games run from `file://`).
- **Topology from the data file:** sensor neurons (each tuned to one game feature via
  `tuning {input, sign, gain}`) → interneurons → motor pool, plus direct sensor→motor
  shortcut edges (the "giant fibre" path). Motor→action readout weights are randomly
  seeded (mulberry32, seed 1234) and trainable.
- **Training:** Monte-Carlo policy gradient (REINFORCE) with an average-reward
  baseline, small entropy bonus, and Adam. Per episode: states are replayed through
  the *frozen* network, discounted returns computed, one gradient step applied.
  Tuning constants live at the top of the file: `ENTROPY_BETA 0.02`,
  `TEMPERATURE 1.0`, `LR 0.02`, `BASELINE_ALPHA 0.15`, `GAMMA 0.98`.
- **Dependency:** TF.js UMD build exposing global `tf` (CDN script tag, e.g.
  `@tensorflow/tfjs@4.20.0`) must load **before** `shared/fly-brain.js`.

## The flyBrain API contract (what every game must follow)

Game config passed to `init`:

```js
await flyBrain.init({
  features: ["playerY", "obstacleDist", ...],  // names, max 6 (N_INPUT_SLOTS)
  actions:  ["flap", "idle", ...],             // discrete action names
  learningRate: 0.02,                          // optional, defaults to LR above
});  // -> { sensors, interneurons, motors } counts
```

Per-step / per-episode lifecycle:

```js
const action = flyBrain.act(stateArray);  // state values aligned with `features`,
                                          // roughly normalized (0..1-ish ranges)
flyBrain.reward(x);                       // fractional rewards accumulate for the
                                          // current step until the next act()/endEpisode()
flyBrain.endEpisode();                    // on death / goal: gradient update + history push
```

Read APIs for HUD/overlays: `getStats()` (episode, avgScore, lastScore, history,
baseline, pool sizes), `getActionProbs()`, `getHidden()`, `getActivity(topN)`,
`getLastAction()`. Control: `setMode('train' | 'greedy')` — train samples from the
policy, greedy takes argmax (use for demo/playback).

## Adding a game — checklist

1. Create `<game>/index.html` (page must sit exactly one level deep; the data path
   `../shared/connectome-data.json` is hardcoded relative in `fly-brain.js`).
2. Load TF.js UMD from CDN, then `../shared/fly-brain.js` — in that order.
3. Define ≤ 6 features and the action set; call `init` after the Start button
   (or on load). Games are **AI-only** — no keyboard/mouse control path and no
   `?mode=` query handling; `npm run verify` (agency check) fails if one appears.
4. Each frame/step: build the state array, call `act`, call `reward` (small, frequent,
   fractional rewards work best), and call `endEpisode` when the episode ends.
   Throttle `act()` to ≤ ~20 Hz in fast games so the REINFORCE replay stays cheap.
5. Show stats / neuron-activity overlay via the read APIs.
6. Keep everything self-contained in the folder; no bundler, no imports.

## Conventions & gotchas

- **Tensor hygiene:** `act()` runs every step, so nothing it allocates may
  outlive the call (`forward()` reads results with `dataSync()` inside a
  `tf.tidy`). The episode replay is one batched `[T, …]` graph, not T per-step
  graphs. `tf.memory().numTensors` must stay flat across episodes.

- **Variable shape trap (fixed once, don't reintroduce):** `st.vars` is a flat
  array `[W1, W2, W3, b2, b3, W4, b4]` — the shape `optimizer.minimize()`,
  `getWeights()` and the destructure at the top of `forward()` all expect. Never
  index it as `st.vars.W1`; `npm test` exists precisely to catch that class of bug.

- **Relative paths:** `DATA_PATH = "../shared/connectome-data.json"` — game pages must
  be served from the repo root (`/open-world/`, not `file://`-opened from inside the
  folder if you want real data; the synthetic fallback hides nothing but works).
- **Never hand-edit `shared/connectome-data.json`.** Regenerate it with the Python
  builders. Both cap fan-out (~18–24 edges/neuron) to keep the browser model small —
  preserve that.
- **Determinism:** `build_curated.py` must stay deterministic (SEED 20260911) so the
  committed JSON diffs cleanly between regenerations (only `generated_at` changes).
- **Browser budget:** neuron/edge counts are deliberately small; don't grow the
  dataset or model without a reason.
- **JS style:** match `shared/fly-brain.js` — IIFE, `"use strict"`, globals, no
  modules or bundler; async/await is fine.
- **Python style:** 3.10+ (PEP 604 unions), `from __future__ import annotations`,
  `pathlib`, stdlib-only for the offline builder (`neuprint`/`navis` only in the live
  fetcher).
- **Sensors:** each feature gets 4 tuned sensor neurons (2 excitatory, 2 inhibitory);
  `tuning.input` indexes into the game's `features` array. Keep feature values in
  comparable ranges so sensor biases (drawn from ~[-1.5, -0.5]) behave sensibly.

## Known gaps (fix, don't replicate)

- The bonus tools (`tools/neuron-to-stl/`, `tools/brain-cli/`,
  `tools/microcontroller/`) and the live `fetch_connectome.py` path have never
  been executed against the real APIs in this workspace (they need tokens /
  hardware); they are written to spec but unverified end-to-end.
- Game pages are only smoke-tested in headless Chromium (Playwright, CDN
  scripts served from local npm copies of the same versions): console clean,
  episodes advance, overlay populates, `tf.memory()` flat. That smoke test
  caught open-world dying on its first frame. Still do a pass in a real
  GPU browser when one is available.
- Every game duplicates a small HUD/chart/probs helper block (deliberately —
  no shared game-side JS by convention). Keep them in sync by hand.

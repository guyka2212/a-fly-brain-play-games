# CLAUDE.md

Guidance for AI agents and humans working in this repo.

## What this is

**a-fly-brain-play-games** — a static, browser-based playground where an artificial
fly brain, seeded from a curated *Drosophila* escape/leg motor connectome, learns to
play games via reinforcement learning.

There is **no build step, no npm dependencies, no test framework**. The repo is
served as-is (e.g. GitHub Pages). JS is vanilla (IIFE + globals, TF.js from CDN);
data tooling is Python 3.10+.

Framing that must be preserved everywhere: this is a **connectome-inspired / seeded**
network, **not** a literal biological simulation of the fly brain. Keep the honesty
notes in file headers and READMEs when you edit them.

## Repo layout

```
shared/fly-brain.js            # core agent: connectome-seeded network + REINFORCE trainer
                               # (IIFE exposing global `flyBrain`)
shared/connectome-data.json    # committed static data: 92 neurons / 966 edges.
                               # GENERATED — do not hand-edit.
tools/neuron-fetch/
  build_curated.py             # offline, deterministic builder (SEED 20260911), no network
  fetch_connectome.py          # live pull from Janelia neuPrint hemibrain (needs token)
  requirements.txt             # only needed for the live fetch
open-world/ beat-saber/ driving-sim/   # game folders, one self-contained page each
                                        # (currently empty scaffolding)
.github/workflows/             # reserved for a GitHub Pages deploy workflow
README.md                      # currently just the title
package-lock.json              # vestigial — there is no package.json; don't add npm tooling
```

## Commands

```bash
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

**Verification:** there is no typecheck or test suite. Verify changes by serving the
site and exercising it in the browser: check the console for errors, confirm scores /
episode counts advance and that network activations render.

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
3. Define ≤ 6 features and the action set; call `init` after user gesture or on load.
4. Each frame/step: build the state array, call `act`, call `reward` (small, frequent,
   fractional rewards work best), and call `endEpisode` when the episode ends.
5. Show stats / neuron-activity overlay via the read APIs.
6. Keep everything self-contained in the folder; no bundler, no imports.

## Conventions & gotchas

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

- `forward()` computes sensor/interneuron activations but never stores them, so
  `getHidden()` returns empty arrays and `getActivity()` reports all zeros. Fix by
  storing `sActs` / `niArr` / motor acts into `st.lastHidden` inside `forward()`
  (careful: do it after `tf.tidy` disposal — `dataSync()` inside tidy is fine).
- The three game folders and the Pages workflow are empty scaffolding; nothing
  renders at the site root yet (no root `index.html`).

# a-fly-brain-play-games

A static, browser-based playground where an artificial fly brain — seeded from a
curated *Drosophila* escape/leg-motor connectome — learns to play games via
reinforcement learning, live in front of you.

**Live (GitHub Pages):** https://guyka2212.github.io/a-fly-brain-play-games/

| URL | What |
|---|---|
| `/` | Hub page with the 3 games |
| `/driving-sim/` | Steer a car on a winding road; checkpoints + crashes |
| `/beat-saber/` | Slice falling notes on the beat (generated metronome audio) |
| `/open-world/` | Free-roam arena; collect orbs, explore terrain |

Every game has a start screen with two modes: **Human Play** (keyboard/mouse) and
**Watch the Fly Brain Play** (the agent controls the game and learns live).
`?mode=human` / `?mode=fly` in the URL also works.

## Honesty note (please read)

This is a **connectome-inspired / seeded neural network**, **not** a literal
biological simulation of a fly brain. `shared/connectome-data.json` supplies the
architecture and initial weights (neurons → hidden nodes, synapse counts → edge
weights); everything after that is learned client-side with a lightweight
Monte-Carlo policy-gradient (REINFORCE) trainer. The fun "who is firing" neuron
overlays are a legible visualization, not rigorous neuroscience.

## How it fits together

```
shared/connectome-data.json     generated once (Python, offline), committed
shared/fly-brain.js             the agent: connectome-seeded network + REINFORCE
<game>/index.html + game.js     three.js (CDN) + TF.js (CDN) + the agent
tools/                          data pipeline + bonus tooling
```

- Rendering: **three.js** from CDN (`three@0.160.0` UMD build).
- ML: **TensorFlow.js** from CDN (`@tensorflow/tfjs@4.20.0` UMD build).
- No build step, no bundler — each game folder is self-contained and runs
  unmodified from GitHub Pages (or any static file server).
- Games reward the agent with small, frequent, fractional signals (stay on the
  road, hit the beat, close in on an orb) and call `flyBrain.endEpisode()` when
  an episode ends; the agent applies one REINFORCE gradient step per episode.

## The flyBrain API (used by all games)

```js
await flyBrain.init({ features: [...], actions: [...] });  // -> pool counts
flyBrain.act(stateArray);      // -> action name (samples policy in train mode)
flyBrain.reward(x);            // fractional reward for the current step
flyBrain.endEpisode();         // gradient update + history push
flyBrain.getStats();           // episode, avgScore, history, pool sizes ...
flyBrain.getActivity(n);       // top-N active neurons (for the overlay)
flyBrain.getActionProbs();     // current policy distribution
flyBrain.setMode("train" | "greedy");
```

## Local development

```bash
# serve from the repo root (paths are root-relative)
python -m http.server 8000
# then open http://localhost:8000/
```

`npm test` runs a headless Node regression harness for the agent
(`tools/fly-brain-test/`): it loads the real connectome from disk, runs 50 fake
episodes, and asserts weights actually move. Run it after touching
`shared/fly-brain.js`.

## The connectome data pipeline

`shared/connectome-data.json` is **generated** — never hand-edit it.

```bash
# offline, deterministic (no network, no deps):
python tools/neuron-fetch/build_curated.py

# live pull from Janelia neuPrint (rarely needed; requires a token):
pip install -r tools/neuron-fetch/requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS="<token>"
python tools/neuron-fetch/fetch_connectome.py
```

See `tools/neuron-fetch/README.md` for the schema and curation notes.

## Bonus tools (in `tools/`)

- `tools/neuron-to-stl/` — fetch a neuron mesh (navis/neuPrint) and export an
  `.stl` for Bambu Studio / OrcaSlicer. Pipeline documented in its README.
- `tools/brain-cli/` — `brain-cli --neuron <id> --stats`: formatted synapse /
  connectivity info from the Virtual Fly Brain API.
- `tools/microcontroller/` — ESP32 + MicroPython + SSD1306 OLED that fetches a
  random neuron's stats over Wi-Fi and displays them.

## Deployment

`.github/workflows/deploy.yml` deploys the repo root to GitHub Pages on every
push to `main` (static upload, no build). Enable *Settings → Pages → Source:
GitHub Actions* if it isn't already.

## Repo layout

```
index.html                     hub page
shared/fly-brain.js            connectome-seeded agent + REINFORCE trainer
shared/connectome-data.json    generated static connectome (92 neurons / 966 edges)
beat-saber/ driving-sim/ open-world/   one self-contained game per folder
tools/fly-brain-test/          headless Node regression harness (npm test)
tools/neuron-fetch/            connectome data pipeline (Python)
tools/neuron-to-stl/ brain-cli/ microcontroller/   bonus tools
.github/workflows/deploy.yml   GitHub Pages deploy
```

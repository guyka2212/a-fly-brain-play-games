# Verification results — is it really the fly brain playing?

Generated: 2026-10-05T07:31:16.723Z · node v22.22.0

## 0. Regression harness (run.js)
```
init: { sensors: 24, interneurons: 44, motors: 24 } | dataset loaded from disk (no synthetic fallback)
weight checksum (episode 0): 631.1232

stats after 50 episodes: {
  episode: 50,
  lastScore: 1.3,
  avgScore: 1.18,
  history: 50,
  baseline: 1.26
}
weight checksum (episode 50): 717.9721
live tensors after ep 1 / ep 50: 23 / 23
activations ok — top neuron: TA0_P1 sensor 0.061
hidden layers alive — states with interneuron firing: 108/200, motor firing: 95/200
action probs: 33.3% / 44.2% / 22.5%

scores ep 1-5: 1.07, 1.13, 1.07, 1.24, 1.02
scores ep 46-50: 1.24, 1.24, 1.30, 1.30, 1.30

ALL CHECKS PASSED ✅

============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================
```
Verdict: ✅ PASS (exit 0)

## 1. Source check — real curated connectome, not the fallback
file: shared/connectome-data.json (97120 bytes)
sha256: 70496beec3e06dcd91bc523a57cb04b93e1c68528e166d44e1e0f1e81df404c5
on disk: 92 neurons / 966 edges (metadata says 92/966)
agent loaded: curated model — Drosophila escape / leg motor circuit (offline builder) — synthetic=false, 92 neurons / 966 edges
  PASS — not the synthetic fallback
  PASS — neuron count matches disk
  PASS — edge count matches disk
  PASS — dataset label is the curated builder
  PASS — generated_by is build_curated.py
Verdict: ✅ PASS

## 2. Agency check — no human / scripted control path
driving-sim: PASS — act()=yes, reward()=yes, endEpisode()=yes, no human/scripted control path
beat-saber: PASS — act()=yes, reward()=yes, endEpisode()=yes, no human/scripted control path
open-world: PASS — act()=yes, reward()=yes, endEpisode()=yes, no human/scripted control path
driving-sim/index.html: PASS — no human-mode UI
beat-saber/index.html: PASS — no human-mode UI
open-world/index.html: PASS — no human-mode UI
Verdict: ✅ PASS

## 3. Learning check — 200 episodes × 3 seeds (+ random-action control)

Method: each game's dynamics (sensors, reward shaping, action semantics) are mirrored headless exactly as in game.js. The fly brain trains from the seeded initialization each run. A control arm steps the identical environment with uniformly random actions (same seeds) — the fly has to beat both its own start *and* noise. Verdict: PASS = last-10% mean beats first-10% mean by ≥ max(0.5, 15%) and > 2×SEM and > control; INCONCLUSIVE = positive but small/noisy; FAIL = no gain.

### driving-sim
- seed 101: first10% -0.35 → last10% 37.43 (Δ +37.78, SEM 3.73) · control(noise) mean -0.45 · **PASS** · 13s
- seed 202: first10% -0.27 → last10% 26.72 (Δ +26.99, SEM 3.71) · control(noise) mean -0.48 · **PASS** · 8s
- seed 303: first10% -0.54 → last10% 2.58 (Δ +3.13, SEM 0.71) · control(noise) mean -0.43 · **PASS** · 1s
Verdict: ✅ PASS

### beat-saber
- seed 101: first10% -1.04 → last10% 5.02 (Δ +6.06, SEM 0.62) · control(noise) mean -1.00 · **PASS** · 2s
- seed 202: first10% 0.55 → last10% 6.94 (Δ +6.39, SEM 0.42) · control(noise) mean -1.24 · **PASS** · 2s
- seed 303: first10% -0.30 → last10% 7.21 (Δ +7.52, SEM 0.36) · control(noise) mean -1.42 · **PASS** · 2s
Verdict: ✅ PASS

### open-world
- seed 101: first10% -2.87 → last10% 18.74 (Δ +21.61, SEM 0.36) · control(noise) mean -7.63 · **PASS** · 9s
- seed 202: first10% 3.37 → last10% 19.42 (Δ +16.05, SEM 0.65) · control(noise) mean -7.06 · **PASS** · 9s
- seed 303: first10% -3.24 → last10% 17.83 (Δ +21.07, SEM 1.16) · control(noise) mean -7.70 · **PASS** · 9s
Verdict: ✅ PASS

## 4. 3D brain view — positions valid, wiring present
connectome: 92 neurons, 0 missing pos, 0 non-finite pos
flyBrain.getNeurons(): 92 neurons, positions intact: yes
agent loaded dataset is the same file: yes
brain-viz.js uses InstancedMesh (browser budget): yes
driving-sim: panel wired PASS
beat-saber: panel wired PASS
open-world: panel wired PASS
Verdict: ✅ PASS

## Summary

| check | driving-sim | beat-saber | open-world |
|---|---|---|---|
| data source | ✓ | ✓ | ✓ |
| agency | ✓ | ✓ | ✓ |
| learning trend | ✓ | ✓ | ✓ |
| 3D brain view | ✓ | ✓ | ✓ |

Notes:
- The learning check mirrors game dynamics headless (same sensors/rewards/actions as game.js); a real-browser run shows the same curves via the on-page chart.
- Agency is proven statically (no human/scripted path) plus structurally: the harness applies exactly the action flyBrain.act() returned — the same contract game.js uses.
- Control arm = identical environment, uniformly random actions, same episode budget: a learning trend must exceed that noise floor, not just itself.
- The 3D brain view check validates the DATA (same neuron count as disk, every pos finite) and the per-game wiring; rendering itself needs a browser pass.

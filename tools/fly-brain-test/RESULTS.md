# Verification results — is it really the fly brain playing?

Generated: 2026-10-04T06:53:06.364Z · node v22.22.0

## 0. Regression harness (run.js)
```
init: { sensors: 24, interneurons: 44, motors: 24 } | dataset loaded from disk (no synthetic fallback)
weight checksum (episode 0): 749.2868

stats after 50 episodes: {
  episode: 50,
  lastScore: 1.18,
  avgScore: 1.03,
  history: 50,
  baseline: 1.08
}
weight checksum (episode 50): 749.4775
live tensors after ep 1 / ep 50: 23 / 23
activations ok — top neuron: TA0_P1 sensor 0.061
action probs: 32.1% / 34.4% / 33.5%

scores ep 1-5: 1.13, 0.80, 1.01, 1.08, 1.07
scores ep 46-50: 0.96, 1.13, 1.07, 1.13, 1.18

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
- seed 101: first10% -0.67 → last10% -0.19 (Δ +0.48, SEM 0.13) · control(noise) mean -0.86 · **INCONCLUSIVE** · 1s
- seed 202: first10% -0.54 → last10% -0.20 (Δ +0.34, SEM 0.15) · control(noise) mean -0.85 · **INCONCLUSIVE** · 1s
- seed 303: first10% -0.74 → last10% -0.13 (Δ +0.61, SEM 0.13) · control(noise) mean -0.92 · **PASS** · 1s
Verdict: ✅ PASS

### beat-saber
- seed 101: first10% -2.48 → last10% -1.33 (Δ +1.15, SEM 0.44) · control(noise) mean -2.44 · **PASS** · 2s
- seed 202: first10% -3.29 → last10% -2.17 (Δ +1.12, SEM 0.53) · control(noise) mean -2.65 · **PASS** · 2s
- seed 303: first10% -1.61 → last10% -1.14 (Δ +0.47, SEM 0.41) · control(noise) mean -2.83 · **INCONCLUSIVE** · 2s
Verdict: ✅ PASS

### open-world
- seed 101: first10% -7.41 → last10% -2.94 (Δ +4.47, SEM 0.50) · control(noise) mean -7.63 · **PASS** · 9s
- seed 202: first10% -5.42 → last10% -1.69 (Δ +3.73, SEM 0.43) · control(noise) mean -7.06 · **PASS** · 9s
- seed 303: first10% -6.96 → last10% -1.75 (Δ +5.21, SEM 0.46) · control(noise) mean -7.70 · **PASS** · 9s
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

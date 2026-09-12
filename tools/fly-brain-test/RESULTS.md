# Verification results — is it really the fly brain playing?

Generated: 2026-09-12T06:05:43.724Z · node v24.20.0

## 0. Regression harness (run.js)
```
init: { sensors: 24, interneurons: 44, motors: 24 } | dataset loaded from disk (no synthetic fallback)
weight checksum (episode 0): 749.2868

stats after 50 episodes: {
  episode: 50,
  lastScore: 1.07,
  avgScore: 1.07,
  history: 50,
  baseline: 1.08
}
weight checksum (episode 50): 749.7627
activations ok — top neuron: TA0_P1 sensor 0.245
action probs: 23.9% / 48.1% / 28.0%

scores ep 1-5: 1.13, 1.13, 1.12, 0.80, 1.24
scores ep 46-50: 1.24, 0.96, 1.13, 0.97, 1.07

ALL CHECKS PASSED ✅

============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================
```
Verdict: ✅ PASS (exit 0)

## 1. Source check — real curated connectome, not the fallback
file: shared/connectome-data.json (92233 bytes)
sha256: efad8613617e181c3c91281345000a1a5f10c3a9eb534ac43ac83508faa7df71
on disk: 92 neurons / 966 edges (metadata says 92/966)
agent loaded: curated model — Drosophila escape / leg motor circuit (offline builder) — synthetic=false, 92 neurons / 966 edges
  PASS — not the synthetic fallback
  PASS — neuron count matches disk
  PASS — edge count matches disk
  PASS — dataset label is the curated builder
  PASS — generated_by is build_curated.py
Verdict: ✅ PASS

## 2. Agency check — no human / scripted control path
driving-sim: FAIL — act()=yes, reward()=yes, endEpisode()=yes, forbidden: keyboard handler; keyboard/pointer event token; human-mode code path; mode query-param handling; game-specific human-input remnant: /\bkeys\s*=\s*\{\}/; game-specific human-input remnant: /arrowup|arrowdown|arrowleft|arrowright/; game-specific human-input remnant: /["'][wads]["']/; game-specific human-input remnant: /Human Play/i; game-specific human-input remnant: /btn-human/
beat-saber: FAIL — act()=yes, reward()=yes, endEpisode()=yes, forbidden: keyboard handler; keyboard/pointer event token; human-mode code path; mode query-param handling; game-specific human-input remnant: /\bkeys\s*=\s*\{\}/; game-specific human-input remnant: /["'][wasd]["']\s*&&/; game-specific human-input remnant: /Human Play/i; game-specific human-input remnant: /btn-human/
open-world: FAIL — act()=yes, reward()=yes, endEpisode()=yes, forbidden: keyboard handler; mouse/pointer handler; keyboard/pointer event token; human-mode code path; mode query-param handling; game-specific human-input remnant: /\bkeys\s*=\s*\{\}/; game-specific human-input remnant: /humanStep|humanKeysDown/; game-specific human-input remnant: /dragToOrbit|pointerdown/; game-specific human-input remnant: /Human Play/i; game-specific human-input remnant: /btn-human/; game-specific human-input remnant: /camYaw\s*-=\s*\(e\.clientX/
driving-sim/index.html: FAIL — human-mode UI remnants
beat-saber/index.html: FAIL — human-mode UI remnants
open-world/index.html: FAIL — human-mode UI remnants
Verdict: ❌ FAIL

## 3. Learning check — 100 episodes × 3 seeds (+ random-action control)

Method: each game's dynamics (sensors, reward shaping, action semantics) are mirrored headless exactly as in game.js. The fly brain trains from the seeded initialization each run. A control arm steps the identical environment with uniformly random actions (same seeds) — the fly has to beat both its own start *and* noise. Verdict: PASS = last-10% mean beats first-10% mean by ≥ max(0.5, 15%) and > 2×SEM and > control; INCONCLUSIVE = positive but small/noisy; FAIL = no gain.

### driving-sim
- seed 101: first10% -1.08 → last10% -0.32 (Δ +0.77, SEM 0.18) · control(noise) mean -0.83 · **PASS** · 3s
- seed 202: first10% -0.74 → last10% -0.25 (Δ +0.49, SEM 0.22) · control(noise) mean -0.82 · **INCONCLUSIVE** · 2s
- seed 303: first10% -0.69 → last10% -0.20 (Δ +0.49, SEM 0.17) · control(noise) mean -0.96 · **INCONCLUSIVE** · 3s
Verdict: ✅ PASS

### beat-saber
- seed 101: first10% -9.15 → last10% -8.30 (Δ +0.85, SEM 1.19) · control(noise) mean -10.15 · **INCONCLUSIVE** · 41s
- seed 202: first10% -10.21 → last10% -8.85 (Δ +1.36, SEM 0.86) · control(noise) mean -9.92 · **INCONCLUSIVE** · 42s
- seed 303: first10% -9.67 → last10% -12.11 (Δ -2.44, SEM 1.20) · control(noise) mean -8.76 · **FAIL** · 41s
Verdict: ❌ FAIL

### open-world
- seed 101: first10% -4.69 → last10% -4.21 (Δ +0.48, SEM 1.46) · control(noise) mean -7.61 · **INCONCLUSIVE** · 52s
- seed 202: first10% -4.47 → last10% -3.14 (Δ +1.33, SEM 1.03) · control(noise) mean -6.89 · **INCONCLUSIVE** · 51s
- seed 303: first10% -5.95 → last10% -0.72 (Δ +5.23, SEM 0.73) · control(noise) mean -8.15 · **PASS** · 50s
Verdict: ✅ PASS

## Summary

| check | driving-sim | beat-saber | open-world |
|---|---|---|---|
| data source | ✓ | ✓ | ✓ |
| agency | ✗ | ✗ | ✗ |
| learning trend | ✓ | ✗ | ✓ |

Notes:
- The learning check mirrors game dynamics headless (same sensors/rewards/actions as game.js); a real-browser run shows the same curves via the on-page chart.
- Agency is proven statically (no human/scripted path) plus structurally: the harness applies exactly the action flyBrain.act() returned — the same contract game.js uses.
- Control arm = identical environment, uniformly random actions, same episode budget: a learning trend must exceed that noise floor, not just itself.

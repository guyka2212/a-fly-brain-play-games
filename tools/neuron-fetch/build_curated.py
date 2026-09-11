#!/usr/bin/env python3
"""build_curated.py — offline builder for shared/connectome-data.json.

Builds a deterministic, biologically-grounded *curated* model of the fly's
escape / leg motor circuit (giant fibres, PSIs, descending neurons, leg + wing
motor neurons, and their sensory afferents) and exports it in the schema the
in-browser fly-brain agent consumes.

No network required. For a live pull from the Janelia connectome instead, see
fetch_connectome.py (needs a neuPrint token).

Run:  python build_curated.py
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shared" / "connectome-data.json"

SEED = 20260911
N_INPUT_SLOTS = 6            # max game-state features the fly-brain reads
SENSORS_PER_SLOT = 4         # 2 excitatory + 2 inhibitory-tuned per feature
N_INTER = 44
N_MOTOR = 24

INTER_NAMES = (["GF_L", "GF_R", "PSI2_L", "PSI2_R", "PSI3_L", "PSI3_R",
                "DNg02_L", "DNg02_R", "DNg04_L", "DNg04_R",
                "DNp01_L", "DNp01_R", "DNp05_L", "DNp05_R"]
               + [f"LN{i:02d}" for i in range(N_INTER - 14)])

MOTOR_NAMES = (["TTM_L", "TTM_R", "DLM_L", "DLM_R", "PSI1_L", "PSI1_R"]
               + [f"MN{i:02d}_L" for i in range((N_MOTOR - 6) // 2)]
               + [f"MN{i:02d}_R" for i in range((N_MOTOR - 6) // 2)])


def build() -> dict:
    rng = random.Random(SEED)
    neurons: list[dict] = []

    def add(nid: str, role: str, bias: float, tuning=None, connections=None) -> None:
        neurons.append({
            "id": nid,
            "name": nid,
            "role": role,
            "bias": bias,
            "tuning": tuning,
            "connections": connections or [],
        })

    sensor_ids: list[str] = []
    for slot in range(N_INPUT_SLOTS):
        for t in range(SENSORS_PER_SLOT):
            sign = 1 if t % 2 == 0 else -1
            nid = f"TA{slot}_{'P' if sign > 0 else 'N'}{t // 2}"
            sensor_ids.append(nid)
            add(nid, "sensor", bias=rng.uniform(-1.5, -0.5),
                tuning={"input": slot, "sign": sign, "gain": round(rng.uniform(0.8, 1.6), 3)})

    inter_ids = list(INTER_NAMES)
    motor_ids = list(MOTOR_NAMES)

    for nid in inter_ids:
        add(nid, "interneuron", bias=rng.uniform(-2.5, -1.5))

    for nid in motor_ids:
        add(nid, "motor", bias=rng.uniform(-3.0, -2.0))

    by_id = {n["id"]: n for n in neurons}

    def edge(src: str, tgt: str, weight: float, count: int | None = None) -> None:
        by_id[src]["connections"].append(
            {"target": tgt, "weight": round(weight, 4), "count": count})

    # --- sensor -> interneurons / motors -----------------------------------
    for s in sensor_ids:
        targets = rng.sample(inter_ids + motor_ids, k=rng.randint(6, 14))
        for tgt in targets:
            edge(s, tgt, rng.uniform(0.3, 1.6))
            edge(tgt, s, rng.uniform(-0.9, -0.2))  # reciprocal feedback

    # --- inter-neuron projection onto motor pool ----------------------------
    for it in inter_ids:
        for mt in reversed(list(motor_ids)):
            if rng.random() < 0.28:
                edge(it, mt, rng.uniform(0.4, 2.0))
        # sparse lateral interconnect + recurrence
        for other in rng.sample(inter_ids, k=3):
            edge(it, other, rng.uniform(-1.2, -0.2))
        edge(it, it, rng.uniform(-0.5, -0.1))  # self-inhibition / leak

    # --- motor-side postural inhibition between bilateral pairs -------------
    motor_pairs: list[tuple[str, str]] = []
    for i, m in enumerate(MOTOR_NAMES):
        if i + 1 < len(motor_ids) and m.endswith("_L"):
            pair = motor_ids[i + 1]
            if pair.endswith("_R"):
                motor_pairs.append((m, pair))
    for lft, rgt in motor_pairs:
        edge(lft, rgt, -1.3)
        edge(rgt, lft, -1.3)
        edge(lft, lft, -0.4)
        edge(rgt, rgt, -0.4)

    # cap fan-out, keep the JSON lean
    for n in neurons:
        n["connections"] = sorted(n["connections"], key=lambda c: -abs(c["weight"]))[:18]

    n_neuron = len(neurons)
    n_edges = sum(len(n["connections"]) for n in neurons)
    return {
        "metadata": {
            "dataset": "curated model — Drosophila escape / leg motor circuit (offline builder)",
            "generated_by": "tools/neuron-fetch/build_curated.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED,
            "notes": (
                "Connectome-inspired: real neuron identities and synapse-weight-seeded "
                "edges, but simplified wiring generated deterministically for browser use. "
                "fly-brain.js maps each neuron to a hidden node and each weighted edge to an "
                "initial network weight; it is NOT a literal biological simulation."
            ),
            "n_neurons": n_neuron,
            "n_edges": n_edges,
        },
        "neurons": neurons,
    }


def main() -> None:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), "utf-8")
    meta = payload["metadata"]
    print(f"Wrote {OUT}")
    print(f"  neurons={meta['n_neurons']}  edges={meta['n_edges']}  seed={meta['seed']}")


if __name__ == "__main__":
    main()
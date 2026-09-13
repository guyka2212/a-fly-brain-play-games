#!/usr/bin/env python3
"""fetch_connectome.py — pull a curated motor/sensorimotor circuit from neuPrint
(Janelia hemibrain) and export it to shared/connectome-data.json.

Requires a Janelia neuPrint token (see README.md):
    export NEUPRINT_APPLICATION_CREDENTIALS="your-token"

Run:  python fetch_connectome.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shared" / "connectome-data.json"

# Curated seed list: identities of the known escape / leg motor circuit.
# (name, role, base_bias, tuning_input) — tuning is only meaningful for 'sensor'.
CURATED_ROOTS = [
    ("GF_L", "interneuron", -1.0, None),             # Giant Fibre (escape trigger)
    ("GF_R", "interneuron", -1.0, None),
    ("PSI2_L", "interneuron", -2.0, None),
    ("PSI2_R", "interneuron", -2.0, None),
    ("PSI3_L", "interneuron", -2.0, None),
    ("PSI3_R", "interneuron", -2.0, None),
    ("TTM_L", "motor", -3.0, None),                  # tergotrochanteral muscle MN
    ("TTM_R", "motor", -3.0, None),
    ("DNg02_L", "interneuron", -2.0, None),          # descending neurons
    ("DNg02_R", "interneuron", -2.0, None),
    ("DNg04_L", "interneuron", -2.0, None),
    ("DNg04_R", "interneuron", -2.0, None),
    ("DNp01_L", "interneuron", -2.0, None),
    ("DNp01_R", "interneuron", -2.0, None),
]

NEUPRINT_SERVER = os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org")


def neuron_id(name: str) -> str:
    """Map a readable circuit name to a neuPrint body id via a criteria query.

    neuPrint (hemibrain v1.2.1) uses numeric body IDs. In the v2 FlyWire
    datasets instance names are used directly. We make a best-effort lookup
    with a criteria LIKE query and take the first hit.
    """
    from neuprint import NeuronCriteria, fetch_neurons

    crit = NeuronCriteria(name=f"^{name}$", client=CLIENT)
    try:
        df = fetch_neurons(crit)
    except Exception as exc:  # pragma: no cover - depends on dataset quirks
        print(f"  ! lookup failed for {name}: {exc}")
        return None
    if df is None or df.empty:
        return None
    return df.iloc[0]["bodyId"]


def fetch_soma_position(bid) -> list[float] | None:
    """Real soma/mesh centroid (x, y, z) in neuropil space, via navis.
    Used for the 3D brain view: real anatomy when available (the offline
    builder's role-shell layout is illustrative instead). Returns None on
    failure so the neuron simply gets no pos (the viz falls back to a
    deterministic role-shell layout in flyBrain.buildModel)."""
    try:
        import navis
        soma = navis.fetch_soma(bid, client=CLIENT)
        if soma is None:
            return None
        x, y, z = soma
        return [round(float(x), 1), round(float(y), 1), round(float(z), 1)]
    except Exception as exc:  # pragma: no cover - depends on dataset quirks
        print(f"  ! soma lookup failed for body {bid}: {exc}")
        return None


def main() -> None:
    global CLIENT  # noqa: PLW0603
    try:
        from neuprint import Client
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"neuprint-python missing: pip install -r requirements.txt ({exc})")

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS") or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit(
            "No neuPrint token found. Set NEUPRINT_APPLICATION_CREDENTIALS=<token>. "
            "Without one, use the offline builder: python build_curated.py"
        )
    CLIENT = Client(NEUPRINT_SERVER, token)

    print(f"Curating {len(CURATED_ROOTS)} seed neurons from {NEUPRINT_SERVER} ...")
    neurons: list[dict] = []
    for name, role, bias, tuning in CURATED_ROOTS:
        bid = neuron_id(name)
        if bid is None:
            print(f"  ! {name}: not found, skipping")
            continue
        conns = CLIENT.fetch_synapse_connectivity(bid)
        # DataFrame: columns ~ ['bodyId', 'type', 'connector', 'synapses', ...]
        postsyn = conns[conns["connector"] == "postsynaptic"] if conns is not None else None
        connections: list[dict] = []
        if postsyn is not None and not postsyn.empty:
            total = float(postsyn["synapses"].sum()) or 1.0
            for _, row in postsyn.iterrows():
                target = row["type"] if row.get("type") else str(row["bodyId"])
                count = int(row["synapses"])
                weight = max(-3.0, min(3.0, 3.0 * (count / total)))  # normalize
                connections.append({"target": target, "weight": round(weight, 4), "count": count})
        connections.sort(key=lambda c: c.get("count", 0), reverse=True)
        connections = connections[:24]  # cap fan-out so the browser model stays small
        pos = fetch_soma_position(bid)
        entry = {
            "id": name,
            "name": name,
            "role": role,
            "bias": bias,
            "tuning": tuning,
            "connections": connections,
        }
        if pos is not None:
            entry["pos"] = pos
        neurons.append(entry)
        print(f"  + {name}: {bid} -> {len(connections)} targets")

    payload = {
        "metadata": {
            "dataset": "neuPrint hemibrain v1.2.1 (curated motor/sensorimotor subset)",
            "generated_by": "tools/neuron-fetch/fetch_connectome.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": (
                "Connectome-inspired: real neuron identities and synapse-weight seeded "
                "edges; simplified wiring. Used to seed the in-browser fly-brain network."
                " This file is a standalone static asset (no live API calls from GitHub Pages)."
                " Per-neuron 'pos' (when present) is the real soma/mesh centroid from"
                " neuPrint+navis in raw neuropil coordinates; neurons without one get a"
                " deterministic illustrative role-shell layout in the 3D view."
            ),
        },
        "neurons": neurons,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), "utf-8")
    print(f"Wrote {OUT} ({len(neurons)} neurons)")


if __name__ == "__main__":
    main()
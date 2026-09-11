#!/usr/bin/env python3
"""neuron-to-stl — fetch a Drosophila neuron mesh and export it as an STL.

Pipeline: neuPrint (Janelia hemibrain) via navis -> neuron mesh (navis
Volume) -> clean/repair -> scale nm -> mm -> trimesh STL -> ready for
Bambu Studio / OrcaSlicer.

Usage:
    export NEUPRINT_APPLICATION_CREDENTIALS="<your-token>"
    python neuron_to_stl.py --neuron 511271748
    python neuron_to_stl.py --neuron 511271748 --scale 0.00035 --out my_neuron.stl

Requires: navis, trimesh (see requirements.txt) and a neuPrint token.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_SCALE = 0.0005  # navis hemibrain units are ~nm; 0.0005 => 1 unit = 0.5 um


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch a neuron mesh and save it as an STL.")
    ap.add_argument("--neuron", required=True, help="neuPrint body ID of the neuron")
    ap.add_argument("--out", default=None, help="output .stl path (default: <bodyid>.stl)")
    ap.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"scale factor applied to the raw mesh (default {DEFAULT_SCALE})",
    )
    ap.add_argument(
        "--server",
        default="https://neuprint.janelia.org",
        help="neuPrint server (default: Janelia hemibrain)",
    )
    args = ap.parse_args()

    try:
        import navis  # type: ignore
        import trimesh  # type: ignore
    except ImportError:
        print("Missing deps. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        print("Set NEUPRINT_APPLICATION_CREDENTIALS=<your neuPrint token>", file=sys.stderr)
        return 1

    import navis.interfaces.neuprint as neu  # type: ignore

    client = neu.Client(args.server, token=token)
    print(f"[1/4] Fetching mesh for body {args.neuron} from {args.server} ...")
    mesh = neu.fetch_mesh(int(args.neuron), client=client)

    # navis returns a navis.Volume (trimesh-backed). Pull out verts/faces.
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    else:  # already a trimesh object
        tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)

    print(f"[2/4] Mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")
    print("[3/4] Cleaning (merge vertices, fix normals, fill holes) ...")
    tm.merge_vertices()
    tm.fix_normals()
    trimesh.repair.fix_inversion(tm)
    trimesh.repair.fill_holes(tm)
    trimesh.repair.fix_winding(tm)

    tm.apply_scale(args.scale)
    print(
        f"[4/4] Scaled x{args.scale} -> {tm.extents[0]:.1f} x "
        f"{tm.extents[1]:.1f} x {tm.extents[2]:.1f} mm"
    )

    out = Path(args.out) if args.out else Path(f"{args.neuron}.stl")
    tm.export(out)
    print(f"Wrote {out} — import it into Bambu Studio / OrcaSlicer and slice.")
    print("Tip: print with supports; neuron arbors are all overhangs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

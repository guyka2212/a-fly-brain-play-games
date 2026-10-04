#!/usr/bin/env python3
"""build_models.py — deterministic, offline Blender asset builder.

Run headless (see run.sh):
    blender -b -P tools/blender-assets/build_models.py

Models three low-poly, stylized meshes used by the game pages and exports
each as a self-contained binary .glb (geometry + PBR materials embedded):

    shared/assets/car.glb         driving-sim player car (rally body, light
                                  lenses, tail lights, hubs, skirts, exhausts)
    shared/assets/note-gem.glb    beat-saber note cube (spikes, collar, inlay,
                                  base ring — reads from the approach face)
    shared/assets/fly-rover.glb   open-world forager rover (head, antennae,
                                  stinger, collar ring, abdomen bands)

Determinism notes
-----------------
* No randomness is used anywhere; every parameter below is a literal.
* Blender's glTF exporter, object-join and modifier internals are threaded /
  hash-ordered and NOT run-to-run deterministic, so none of them are used:
  parts are merged in THIS script's fixed part order, vertices are transformed
  by each part's matrix with plain mathutils ops, and the GLB is written by
  our own minimal glTF 2.0 writer from the merged triangle lists.
* Vertices/normals are snapped to a 1e-4 grid (sub-visual, ~0.05 mm on a
  3-unit car) to erase last-bit float drift.
* Verified byte-identical across repeated runs (sha256 in manifest.json).

The games load these with THREE.GLTFLoader and fall back to plain primitives
if the fetch fails, so an offline/asset-less deployment still runs.
"""
from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shared" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

GRID = 10_000  # snap floats to 1/1e4 — far below visual significance


def snap(v: float) -> float:
    """Deterministic float cleanup: grid-snap, kill -0.0 and denormals."""
    if v == 0.0:
        return 0.0
    nv = round(v * GRID) / GRID
    return 0.0 if nv == 0.0 else nv


# ------------------------------------------------------------- GLB writer --
def write_glb(path: Path, prims: list[dict], name: str) -> None:
    """Write a minimal glTF 2.0 GLB: one node, one mesh, one primitive per
    material. prims: [{mat:{name,baseColor,roughness,metallic},
    verts:[(x,y,z)], normals:[(x,y,z)], indices:[int]}] in glTF Y-up space."""
    bin_chunks = bytearray()
    accessors: list[dict] = []
    buffer_views: list[dict] = []
    materials: list[dict] = []
    meshes_prims: list[dict] = []
    seen_mats: dict[str, int] = {}

    def add_view(data: bytes, target: int) -> int:
        off = len(bin_chunks)
        bin_chunks.extend(data)
        while len(bin_chunks) % 4:
            bin_chunks.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": off,
                             "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    for p in prims:
        mname = p["mat"]["name"]
        if mname not in seen_mats:
            materials.append(p["mat"])
            seen_mats[mname] = len(materials) - 1

        pos = b"".join(struct.pack("<3f", *(snap(c) for c in v)) for v in p["verts"])
        nor = b"".join(struct.pack("<3f", *(snap(c) for c in v)) for v in p["normals"])
        idx = b"".join(struct.pack("<H", i) for i in p["indices"])

        v_pos = add_view(pos, 34962)
        v_nor = add_view(nor, 34962)
        v_idx = add_view(idx, 34963)

        mn = [min(v[i] for v in p["verts"]) for i in range(3)]
        mx = [max(v[i] for v in p["verts"]) for i in range(3)]
        accessors.append({"bufferView": v_pos, "componentType": 5126,
                          "count": len(p["verts"]), "type": "VEC3",
                          "min": [snap(m) for m in mn], "max": [snap(m) for m in mx]})
        accessors.append({"bufferView": v_nor, "componentType": 5126,
                          "count": len(p["normals"]), "type": "VEC3"})
        accessors.append({"bufferView": v_idx, "componentType": 5123,
                          "count": len(p["indices"]), "type": "SCALAR"})

        meshes_prims.append({
            "attributes": {"POSITION": len(accessors) - 3,
                           "NORMAL": len(accessors) - 2},
            "indices": len(accessors) - 1,
            "material": seen_mats[mname],
        })

    gltf = {
        "asset": {"version": "2.0",
                  "generator": "a-fly-brain-play-games/tools/blender-assets"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"primitives": meshes_prims}],
        "materials": [
            {"name": m["name"], "pbrMetallicRoughness": {
                "baseColorFactor": m["baseColor"],
                "metallicFactor": m["metallic"],
                "roughnessFactor": m["roughness"]}}
            for m in materials
        ],
        "buffers": [{"byteLength": len(bin_chunks)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_chunks)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(bin_chunks), b"BIN\x00"))
        f.write(bytes(bin_chunks))


# ------------------------------------------------------------- extraction --
def compose_matrix(loc, rot=None, scale=None) -> Matrix:
    """TRS matrix built in Python from literals. We deliberately do NOT read
    obj.matrix_world: it is a depsgraph-evaluated value that can be stale or
    evaluation-order dependent in background mode (observed), which would leak
    nondeterminism into the exported vertices."""
    m = Matrix.Translation(Vector(loc))
    if rot:
        m = m @ Matrix.Rotation(rot[2], 4, "Z") @ \
            Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
    if scale:
        m = m @ Matrix.Diagonal(Vector(scale).to_4d())
    return m


def merge_parts(parts: list[tuple[bpy.types.Object, Matrix, dict]],
                post: Matrix | None = None) -> list[dict]:
    """Merge (object, matrix, material-spec) triples into per-material
    triangle buckets, in the caller's fixed part order.

    Determinism: Blender 4.0 emits the polygon LIST of a freshly created
    primitive in a run-to-run varying order (same vertices, same counts —
    only the sequence differs; also true for calc_loop_triangles). Therefore
    triangles are canonicalized: each triangle is keyed by its corner
    (position, normal) tuples rotated to start at the lexicographically
    smallest corner (winding preserved), and the triangle list is sorted by
    that key before vertices are deduped in first-use order. The emitted
    buffer is thus independent of Blender's internal ordering."""
    buckets: dict[str, dict] = {}

    for obj, mw, mat in parts:
        me = obj.data
        # Blender 4.0 only fills loop.normal after calc_normals_split(); without
        # it every exported normal was (0, 0, 0) and lit materials rendered
        # black. 4.1+ computes corner normals automatically (method removed).
        if hasattr(me, "calc_normals_split"):
            me.calc_normals_split()
        mwi = mw.inverted().transposed()          # normal matrix
        b = buckets.setdefault(mat["name"], {"mat": mat, "verts": [],
                                             "normals": [], "indices": [],
                                             "map": {}, "tris": []})
        for poly in me.polygons:
            ls, lt = poly.loop_start, poly.loop_total
            for k in range(1, lt - 1):            # fan: (0, k, k+1), convex prims
                corners = []
                for li in (ls, ls + k, ls + k + 1):
                    loop = me.loops[li]
                    co = mw @ Vector(me.vertices[loop.vertex_index].co)
                    ln = Vector(loop.normal)
                    if ln.length < 1e-6:              # never emit a zero normal
                        ln = Vector(poly.normal)
                    no = (mwi @ ln).normalized()
                    if post is not None:
                        co = post @ co
                        no = post @ no
                    key = (round(co.x, 5), round(co.y, 5), round(co.z, 5),
                           round(no.x, 4), round(no.y, 4), round(no.z, 4))
                    corners.append((key, (co.x, co.y, co.z), (no.x, no.y, no.z)))
                # rotate so the smallest corner key leads (cyclic, keeps winding)
                start = min(range(3), key=lambda i: corners[i][0])
                corners = corners[start:] + corners[:start]
                b["tris"].append(tuple(c[0] for c in corners))
                b.setdefault("corners", {})[tuple(c[0] for c in corners)] = \
                    tuple((c[1], c[2]) for c in corners)

    prims = []
    for name in buckets:                          # dict order = first-use order
        b = buckets[name]
        corners_by_key = b.pop("corners")
        tris = sorted(b.pop("tris"))
        seen: dict[tuple, int] = {}
        for tri in tris:
            for key in tri:
                if key not in seen:
                    co, no = corners_by_key[tri][tri.index(key)]
                    seen[key] = len(b["verts"])
                    b["verts"].append(co)
                    b["normals"].append(no)
                b["indices"].append(seen[key])
        b.pop("map", None)
        prims.append(b)
    return prims


def export(parts: list[tuple[bpy.types.Object, Matrix, dict]], name: str,
           post: Matrix | None = None) -> None:
    prims = merge_parts(parts, post)
    write_glb(OUT / f"{name}.glb", prims, name)


def material(name: str, color: tuple[float, float, float],
             rough: float, metal: float) -> dict:
    return {"name": name, "baseColor": [*color, 1.0],
            "roughness": rough, "metallic": metal}


# ----------------------------------------------------------------- helpers --
def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


# Blender is Z-up, glTF is Y-up: (x, y, z) -> (x, z, -y) is a -90 deg turn
# about X. (+90 was used originally and shipped both vehicles upside down,
# the car also facing backwards; the committed GLBs were corrected with the
# equivalent exact 180-degree sign flips.)
Z_UP_TO_Y_UP = Matrix.Rotation(-3.141592653589793 / 2, 4, "X")
ROT_Y_180 = Matrix.Rotation(3.141592653589793, 4, "Y")


def box(loc, scale, rot=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    o = bpy.context.active_object
    return o, compose_matrix(loc, rot, scale)


def cyl(loc, radius, depth, rot=None, vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
                                        depth=depth, location=(0, 0, 0))
    return bpy.context.active_object, compose_matrix(loc, rot, (1, 1, 1))


def sphere(loc, radius, segments=16, ring_count=10, scale=None, rot=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=ring_count,
                                         radius=radius, location=(0, 0, 0))
    s = scale if scale else (1, 1, 1)
    return bpy.context.active_object, compose_matrix(loc, rot, s)


def cone(loc, radius, depth, rot):
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=radius, depth=depth,
                                    location=(0, 0, 0))
    return bpy.context.active_object, compose_matrix(loc, rot, (1, 1, 1))


def torus(loc, major_radius, minor_radius, rot=None, segments=24):
    """Ring (torus) in the XY plane, Z-thin — used for rover collar/bands."""
    bpy.ops.mesh.primitive_torus_add(major_radius=major_radius,
                                     minor_radius=minor_radius,
                                     major_segments=segments,
                                     minor_segments=segments // 2,
                                     location=(0, 0, 0))
    return bpy.context.active_object, compose_matrix(loc, rot, (1, 1, 1))


# -------------------------------------------------------------- car model --
def build_car() -> None:
    reset_scene()
    body = material("car_body", (0.64, 0.44, 0.97), 0.35, 0.25)
    dark = material("car_dark", (0.05, 0.07, 0.09), 0.5, 0.2)
    glass = material("car_glass", (0.15, 0.25, 0.35), 0.15, 0.4)
    trim = material("car_trim", (0.90, 0.93, 0.96), 0.3, 0.6)
    amber = material("car_light_amber", (1.00, 0.72, 0.18), 0.25, 0.3)
    red = material("car_light_red", (0.97, 0.15, 0.17), 0.25, 0.3)

    parts: list[tuple[bpy.types.Object, Matrix, dict]] = []
    parts.append((*box((0, 0, 0.55), (0.8, 1.5, 0.45)), body))          # hull
    parts.append((*box((0, -0.18, 1.15), (0.62, 0.85, 0.4)), glass))    # cabin
    parts.append((*box((0, 1.45, 0.45), (0.68, 0.55, 0.3)), body))      # nose
    parts.append((*box((0, -1.35, 1.25), (0.85, 0.18, 0.06)), dark))    # wing
    for sx in (-0.55, 0.55):                                           # struts
        parts.append((*box((sx, -1.35, 1.02), (0.07, 0.1, 0.28)), dark))
    for sx in (-0.78, 0.78):                                           # wheels
        for sy in (-0.85, 0.95):
            parts.append((*cyl((sx, sy, 0.34), 0.34, 0.22,
                               rot=(0, 3.141592653589793 / 2, 0)), dark))
    for sx in (-0.4, 0.4):                                             # headlights
        parts.append((*sphere((sx, 1.72, 0.55), 0.09, 12, 8), trim))

    # --- new detail (same fixed-order style) ---
    for sy in (-0.85, 0.95):                                           # wheel hubs
        for sx in (-0.78, 0.78):
            parts.append((*cyl((sx, sy, 0.34), 0.14, 0.24,
                               rot=(0, 3.141592653589793 / 2, 0)), trim))
    for sx in (-0.4, 0.4):                                             # headlight lenses
        parts.append((*cyl((sx, 1.78, 0.55), 0.085, 0.12,
                           rot=(3.141592653589793 / 2, 0, 0)), amber))
    for sx in (-0.36, 0.36):                                           # tail lights
        parts.append((*box((sx, -1.68, 0.72), (0.16, 0.05, 0.1)), red))
    for sx in (-0.9, 0.9):                                             # side skirts
        parts.append((*box((sx, 0.05, 0.18), (0.06, 2.3, 0.16)), dark))
    for sx in (-0.34, 0.34):                                           # exhausts
        parts.append((*cyl((sx, -1.72, 0.28), 0.05, 0.14,
                           rot=(3.141592653589793 / 2, 0, 0)), trim))
    for sx in (-0.3, 0.3):                                             # bumper lip
        parts.append((*box((sx, 1.74, 0.26), (0.3, 0.1, 0.12)), dark))
    parts.append((*box((0, -0.42, 0.9), (0.3, 0.55, 0.1)), dark))       # roof scoop
    for sx in (-0.26, 0.26):                                           # windshield wipers
        parts.append((*box((sx, 0.62, 0.98), (0.03, 0.02, 0.34)), dark))
    parts.append((*cyl((0, 0.3, 1.38), 0.12, 0.03, vertices=12), trim))  # mirror base

    export(parts, "car", post=Z_UP_TO_Y_UP)   # nose +Y (Blender) -> -Z (glTF)


# -------------------------------------------------------- note-gem model ---
def build_note_gem() -> None:
    """Beat note gem: beveled cube body, corner spikes, stem + orb, plus a
    metal collar, base ring and dark inlay panel on the front face (the face
    the player sees approaching down the lane)."""
    reset_scene()
    gemm = material("gem_body", (0.65, 0.45, 0.97), 0.25, 0.35)
    stem = material("gem_stem", (0.90, 0.93, 0.96), 0.35, 0.7)
    collar = material("gem_collar", (0.55, 0.58, 0.62), 0.3, 0.85)
    dark = material("gem_dark", (0.05, 0.06, 0.08), 0.5, 0.2)

    parts: list[tuple[bpy.types.Object, Matrix, dict]] = []
    parts.append((*box((0, 0, 0), (1, 1, 1)), gemm))
    for sx in (-0.62, 0.62):               # corner spikes
        parts.append((*cone((sx, 0, 0), 0.22, 0.3,
                            rot=(0, 3.141592653589793 / 2 * (1 if sx > 0 else -1), 0)), stem))
    parts.append((*cyl((0, 0, 0.72), 0.12, 0.5, vertices=12), stem))
    parts.append((*sphere((0, 0, 1.0), 0.16, 12, 8), gemm))

    # --- new detail ---
    for sz in (-0.62, 0.62):               # top/bottom spikes (full 4-corner set)
        parts.append((*cone((0, 0, sz), 0.22, 0.3,
                            rot=((3.141592653589793 / 2) * (-1 if sz > 0 else 1), 0, 0)), stem))
    parts.append((*cyl((0, 0, 0.52), 0.16, 0.1, vertices=12), collar))  # stem collar
    parts.append((*cyl((0, 0, 0.04), 0.3, 0.05, vertices=12), collar))  # front inlay ring
    parts.append((*box((0, 0, 0.09), (0.36, 0.36, 0.04)), dark))        # inlay panel
    for sx in (-0.5, 0.5):                 # base ring feet
        parts.append((*sphere((sx, 0, -0.5), 0.09, 10, 8), stem))
    for sy in (-0.5, 0.5):
        parts.append((*sphere((0, sy, -0.5), 0.09, 10, 8), stem))

    export(parts, "note-gem")


# --------------------------------------------------------- fly-rover model -
def build_fly_rover() -> None:
    """Forager rover: ellipsoid thorax, compound eyes, swept wings — plus a
    separate head sphere with antennae, a stinger, a ring collar and wing
    stripes. Built long on +Y (nose at −Y), exported nose at −Z."""
    reset_scene()
    body = material("rover_body", (0.64, 0.44, 0.97), 0.4, 0.2)
    wing = material("rover_wing", (0.90, 0.93, 0.96), 0.25, 0.5)
    eye = material("rover_eye", (0.06, 0.66, 0.35), 0.2, 0.3)
    dark = material("rover_dark", (0.05, 0.06, 0.08), 0.5, 0.2)

    parts: list[tuple[bpy.types.Object, Matrix, dict]] = []
    parts.append((*sphere((0, 0, 0), 1, 20, 14, scale=(1, 1.25, 0.8)), body))
    for sx in (-0.45, 0.45):               # compound eyes
        parts.append((*sphere((sx, -0.95, 0.18), 0.34, 14, 10,
                              scale=(0.8, 0.7, 0.9)), eye))
    for sx in (-1, 1):                     # swept wings
        parts.append((*sphere((sx * 1.15, 0.35, 0.45), 1, 16, 10,
                              scale=(0.55, 1.05, 0.08),
                              rot=(0.12 * sx, -0.3, 0.28 * sx)), wing))

    # --- new detail ---
    parts.append((*sphere((0, -1.35, 0.15), 0.32, 14, 10), body))       # head
    for sx in (-0.14, 0.14):               # antennae (long thin boxes)
        parts.append((*box((sx, -1.98, 0.42), (0.035, 0.75, 0.035),
                           rot=(0.5, 0, -0.25 * sx * 10)), dark))
    for sx in (-0.3, 0.3):                 # wing stripes (leading edge)
        parts.append((*box((sx * 1.6, 0.55, 0.5), (0.05, 0.85, 0.03),
                           rot=(0.1, -0.25, 0.3 * sx)), dark))
    parts.append((*cone((0, 1.5, 0.15), 0.14, 0.5,
                        rot=(3.141592653589793 / 2, 0, 0)), dark))      # stinger
    parts.append((*torus((0, -0.55, 0.1), 0.42, 0.055), dark))          # collar ring
    for sy in (-0.15, 0.15):               # abdomen bands
        parts.append((*torus((0, sy, 0.0), 0.88, 0.035), dark))

    # nose -Y (Blender) -> +Z after the up-axis fix, so turn it to face -Z
    export(parts, "fly-rover", post=ROT_Y_180 @ Z_UP_TO_Y_UP)


# -------------------------------------------------------------------- main --
def main() -> None:
    built: dict[str, str] = {}
    for builder, name in ((build_car, "car"),
                          (build_note_gem, "note-gem"),
                          (build_fly_rover, "fly-rover")):
        builder()
        path = OUT / f"{name}.glb"
        data = path.read_bytes()
        built[f"{name}.glb"] = hashlib.sha256(data).hexdigest()
        print(f"built {path.relative_to(ROOT)} ({len(data)} bytes)")

    manifest = {
        "generated_by": "tools/blender-assets/build_models.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Deterministic meshes for the game pages; games fall back to "
                "primitives if these fail to load. sha256 for reproducibility.",
        "files": built,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print("wrote shared/assets/manifest.json")


main()

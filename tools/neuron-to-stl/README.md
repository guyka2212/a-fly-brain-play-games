# neuron-to-stl — "Neuron to Desk" 3D printing pipeline

Turn a single *Drosophila* neuron from the Janelia hemibrain connectome into a
watertight `.stl` you can slice in **Bambu Studio** or **OrcaSlicer** and print.

## Pipeline

```
neuPrint API (hemibrain) → neuron mesh (navis Volume) → trimesh
    → merge vertices + fix normals + fill holes → scale nm → mm → .stl → slicer
```

1. **API request** — `navis.interfaces.neuprint` authenticates with your
   `NEUPRINT_APPLICATION_CREDENTIALS` token and fetches the mesh for one body ID
   (the same mesh neuPrint's "neuron mesh" view shows).
2. **Mesh** — the navis Volume's vertices/faces are wrapped in a `trimesh.Trimesh`.
3. **Repair** — merge duplicate vertices, fix winding/normals, fill small holes so
   slicers see a watertight shell.
4. **Scale** — hemibrain units are ~nanometres; the default `--scale 0.0005`
   brings a typical neuron to a good desk size (~5–15 cm). Tune to taste.
5. **Export** — binary STL, ready to slice.

## Usage

```bash
pip install -r requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS="<your neuPrint token>"

python neuron_to_stl.py --neuron 511271748                 # -> 511271748.stl
python neuron_to_stl.py --neuron 511271748 --scale 0.001 --out big.stl
```

Get a token: sign in at <https://neuprint.janelia.org> (free account), then
Account → New Token. Find body IDs with neuPrint's search, or use names from
`shared/connectome-data.json` looked up in neuPrint.

## Slicer tips

- Import the STL directly (mm units).
- **Supports are mandatory** — neuronal arbors are all overhangs; use tree supports.
- A small brim helps; the mesh often contacts the bed at a single arbor tip.
- 0.2 mm layers is fine; the interesting detail is in the branch topology, not
  fine surface texture.
- Scale up more (10–20 cm) if you want neurite thickness to survive at 0.4 mm
  nozzle — thin processes can drop below printable width.

## Honesty note

This prints the **morphology of one neuron** (or a small group if you pass a
group ID) — a real reconstruction from the hemibrain dataset, simplified by the
meshing pipeline upstream of neuPrint. It is not a printout of "the fly brain"
and not related to the game agent's network beyond sharing the source dataset.

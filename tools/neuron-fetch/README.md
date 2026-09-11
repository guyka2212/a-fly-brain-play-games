# neuron-fetch — connectome data pipeline

Builds `shared/connectome-data.json`, the static neuron/connectivity file that the
in-browser fly-brain agent loads. The file is generated **once** and committed to the
repo — GitHub Pages never hits the connectome API live (avoids CORS / rate limits).

## What it does

1. Pulls a curated set of neurons from a motor / sensorimotor circuit of the fruit-fly
   connectome (the escape-and-leg motor circuit: giant fibres, descending neurons,
   leg motor neurons and their sensory afferents — kept to ~40–120 neurons so the file
   is practical for a browser).

   - **Live path** (`fetch_connectome.py`, default): queries the Janelia **neuPrint**
     hemibrain server for the curated neuron list, then extracts weighted connectivity
     with `Client.fetch_synapse_connectivity` (postsynaptic partners + synapse counts).
   - **Offline path** (`build_curated.py`): no network required. Rebuilds the same
     schema deterministically with a biologically-grounded, curated connectivity model
     so the repo works out of the box.

2. Exports `shared/connectome-data.json`:

   ```json
   {
     "metadata": { "dataset": "...", "generated_by": "...", "generated_at": "...", "notes": "..." },
     "neurons": [
       {
         "id": "DNg02",
         "name": "DNg02",
         "role": "sensor | sensorimotor | motor | interneuron",
         "bias": 0,                       // resting threshold offset
         "tuning": { "input": 0, "gain": 1, "sign": 1 },  // only for sensor role
         "connections": [ { "target": "TTMn_R", "weight": 0.9, "count": 84 } ]
       }
     ]
   }
   ```

3. The in-browser `shared/fly-brain.js` maps each neuron to a hidden node and each
   `connection` to an initial edge weight — topology and initial weights come from this
   file; everything else is learned client-side.

## Re-running live

You need a Janelia neuPrint token:

```bash
cd tools/neuron-fetch
pip install -r requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS="YOUR_TOKEN"
python fetch_connectome.py            # -> ../../shared/connectome-data.json
```

The token is only needed for the live fetch. The offline builder needs nothing:

```bash
python build_curated.py               # -> ../../shared/connectome-data.json
```

## Honesty note

The committed dataset is a **curated, representative subset** of the fly connectome:
real neuron identities and biologically-plausible synapse counts, but simplified wiring
dictated by what is practical to run in a browser tab. The games use it to *seed* a
neural network's architecture and initial weights — it is a connectome-inspired / seeded
network, not a literal simulation of the biological fly brain.
# brain-cli (a.k.a. fly-fetch)

A tiny stdlib-only Python CLI that prints stats and connectivity for a
*Drosophila* neuron, pulled live from the **Virtual Fly Brain** (VFB) API.

## Usage

```bash
# neuron summary
python brain_cli.py --neuron FBbt:00003644 --stats

# connectivity partners
python brain_cli.py --neuron FBbt:00003644 --connections

# both, as raw JSON
python brain_cli.py --neuron FBbt:00003644 --stats --connections --json

# search by name fragment
python brain_cli.py --search "giant fibre" --limit 5
```

With no flags, `--stats` is assumed. `--neuron` takes a VFB **short form** ID
(`FBbt:00003644`) or a VFB report ID — the same IDs neuPrint names map onto.

## Example output

```
neuron:  FBbt:00003644
label    giant fiber of成年 brain  (example label from VFB)
connections (12 shown):
  partner                                             weight
  TTMn (tb)                                                84
  PSI                                                      61
  ...
```

## How it works

- `--stats`: `GET https://virtualflybrain.org/api/v2/find/neurons/...`
- `--connections`: a small SPARQL query against VFB's public SPARQL endpoint
  (`/nsparql`), asking for overlap partners (`RO_0002131`) and weights.
- Stdlib only (`urllib`, `json`, `argparse`) — no `pip install` needed.

## Notes & honesty

- VFB's API is a living service; response shapes can change. `--json` always
  gives you the raw payload so you can adapt the printer.
- The SPARQL connection query shows *overlapping/connected* partners as VFB
  models them — it is not the same weighted synapse-count table neuPrint gives
  (see `tools/neuron-fetch/` for that). For the exact per-synapse numbers the
  games' connectome file is built from, use the neuPrint-based fetcher.

#!/usr/bin/env python3
"""brain-cli — a tiny CLI for fly-brain neuron info.

Pulls neuron stats and connectivity from the Virtual Fly Brain (VFB) API and
prints them in a compact, human-readable form. Also supports searching by name.

Examples:
    python brain_cli.py --neuron FBbt:00003644 --stats
    python brain_cli.py --neuron FBbt:00003644 --connections
    python brain_cli.py --search "giant fibre" --limit 5
    python brain_cli.py --neuron FBbt:00003644 --stats --connections --json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

VFB_API = "https://virtualflybrain.org/api/v2/find/neurons"
VFB_SPARQL = "https://virtualflybrain.org/nsparql"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def search_neurons(term: str, limit: int) -> list[dict]:
    q = urllib.parse.quote(term)
    url = f"{VFB_API}?q={q}&rows={limit}"
    data = _get_json(url)
    return data.get("results", []) if isinstance(data, dict) else data


def fetch_neuron_stats(short_form: str) -> dict:
    """Core neuron info: label, classification, lineage, images."""
    url = f"{VFB_API}/{urllib.parse.quote(short_form)}"
    return _get_json(url)


def fetch_neuron_connections(short_form: str) -> list[dict]:
    """Known upstream/downstream partners via VFB's SPARQL endpoint."""
    query = f"""
    PREFIX obo: <http://purl.obolibrary.org/obo/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?partner ?partnerLabel ?weight WHERE {{
      <http://virtualflybrain.org/reports/{short_form}> obo:RO_0002131 ?partner .
      ?partner rdfs:label ?partnerLabel .
      OPTIONAL {{ <http://virtualflybrain.org/reports/{short_form}>
                  <http://purl.org/vfb/fact/weight> ?weight . }}
    }} LIMIT 50
    """
    url = VFB_SPARQL + "?query=" + urllib.parse.quote(query)
    data = _get_json(url)
    bindings = data.get("results", {}).get("bindings", [])
    out = []
    for b in bindings:
        out.append({
            "partner": b.get("partner", {}).get("value", ""),
            "label": b.get("partnerLabel", {}).get("value", ""),
            "weight": b.get("weight", {}).get("value", "?"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="brain-cli",
        description="Print fly-brain neuron stats / connectivity from Virtual Fly Brain.",
    )
    ap.add_argument("--neuron", help="neuron ID (VFB short form, e.g. FBbt:00003644)")
    ap.add_argument("--stats", action="store_true", help="print neuron summary stats")
    ap.add_argument("--connections", action="store_true", help="print connectivity")
    ap.add_argument("--search", help="search neurons by name fragment")
    ap.add_argument("--limit", type=int, default=10, help="search result limit")
    ap.add_argument("--json", action="store_true", help="dump raw JSON instead of a table")
    args = ap.parse_args()

    if args.search:
        results = search_neurons(args.search, args.limit)
        if args.json:
            print(json.dumps(results, indent=2))
            return 0
        for r in results:
            sid = r.get("short_form") or r.get("id") or "?"
            label = r.get("label") or r.get("name") or "?"
            print(f"{sid:24} {label}")
        if not results:
            print("no results")
        return 0

    if not args.neuron:
        ap.print_help()
        return 0

    nid = args.neuron.strip()
    want_stats = args.stats or not args.connections
    want_conn = args.connections or not args.stats

    if want_stats:
        try:
            data = fetch_neuron_stats(nid)
        except Exception as e:
            print(f"error fetching {nid}: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"neuron:  {nid}")
            for key in ("label", "name", "classification", "lineage", "note"):
                if isinstance(data, dict) and data.get(key):
                    print(f"{key:8} {data[key]}")
            if isinstance(data, dict) and "results" in data and data["results"]:
                r0 = data["results"][0] if isinstance(data["results"], list) else data["results"]
                for key in ("label", "name"):
                    if r0.get(key):
                        print(f"{key:8} {r0[key]}")

    if want_conn:
        try:
            conns = fetch_neuron_connections(nid)
        except Exception as e:
            print(f"error fetching connections: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(conns, indent=2))
        elif not conns:
            print("connections: none found via SPARQL (try --json)")
        else:
            print(f"connections ({len(conns)} shown):")
            print(f"  {'partner':44} {'weight':>8}")
            for c in conns:
                print(f"  {c['label'][:44]:44} {c['weight']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

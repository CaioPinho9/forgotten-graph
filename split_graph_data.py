import json
from pathlib import Path

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "graph_data.json"
OUTPUT_DIR = Path("graph_chunks")

NODES_PER_CHUNK = 5000
EDGES_PER_CHUNK = 20000


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {INPUT_FILE} ...")
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")

    node_files = []
    edge_files = []

    for i, batch in enumerate(chunked(nodes, NODES_PER_CHUNK)):
        filename = f"nodes_{i:05d}.json"
        path = OUTPUT_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False)
        node_files.append(filename)
        print(f"Wrote {filename} ({len(batch)} nodes)")

    for i, batch in enumerate(chunked(edges, EDGES_PER_CHUNK)):
        filename = f"edges_{i:05d}.json"
        path = OUTPUT_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False)
        edge_files.append(filename)
        print(f"Wrote {filename} ({len(batch)} edges)")

    manifest = {
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "nodes_per_chunk": NODES_PER_CHUNK,
        "edges_per_chunk": EDGES_PER_CHUNK,
        "node_files": node_files,
        "edge_files": edge_files,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

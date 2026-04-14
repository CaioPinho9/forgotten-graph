import csv
import json
import time
import math
from pathlib import Path

import igraph as ig

DATA_DIR = Path("data")

DISCOVERED_FILE = DATA_DIR / "discovered_titles.txt"
EDGES_FILE = DATA_DIR / "edges.csv"
OUTPUT_JSON = DATA_DIR / "graph_data.json"

# Progress print frequency while reading edges
PRINT_EVERY_EDGES = 100_000


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def count_lines(path: Path) -> int:
    """
    Counts lines in a file. For CSV with header, total data rows ~= lines - 1.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f)


def load_discovered_titles(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def read_edges_filtered(
        edges_file: Path,
        discovered_titles: set[str],
        print_every: int = PRINT_EVERY_EDGES,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Reads the edges CSV and keeps only edges where both source and target
    are in discovered_titles.
    """
    total_lines = count_lines(edges_file)
    estimated_total_rows = max(0, total_lines - 1)

    print(f"Estimated edge rows to read: {estimated_total_rows}")

    started_at = time.time()

    edges: list[tuple[str, str]] = []
    nodes: set[str] = set()
    removed = 0
    kept = 0

    with edges_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            source = (row.get("source") or "").strip()
            target = (row.get("target") or "").strip()

            if not source or not target:
                removed += 1
            elif source not in discovered_titles or target not in discovered_titles:
                removed += 1
            else:
                edges.append((source, target))
                nodes.add(source)
                nodes.add(target)
                kept += 1

            if i % print_every == 0:
                elapsed = time.time() - started_at
                avg = elapsed / i if i else 0
                remaining = estimated_total_rows - i
                eta = avg * remaining if remaining > 0 else 0

                print(
                    f"[read_edges] {i}/{estimated_total_rows} "
                    f"({(i / estimated_total_rows) * 100:.2f}%) | "
                    f"kept={kept} removed={removed} | "
                    f"elapsed={format_seconds(elapsed)} | "
                    f"eta={format_seconds(eta)}"
                )

    elapsed = time.time() - started_at
    print(
        f"[read_edges] Done | kept={kept} removed={removed} | "
        f"nodes={len(nodes)} | elapsed={format_seconds(elapsed)}"
    )

    return sorted(nodes), edges


def build_igraph(nodes: list[str], edges: list[tuple[str, str]]) -> ig.Graph:
    started_at = time.time()
    print("Building igraph index...")

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    indexed_edges = [(node_to_idx[s], node_to_idx[t]) for s, t in edges]

    g = ig.Graph(n=len(nodes), edges=indexed_edges, directed=True)
    g.vs["label"] = nodes

    elapsed = time.time() - started_at
    print(
        f"Built igraph | nodes={g.vcount()} edges={g.ecount()} | "
        f"elapsed={format_seconds(elapsed)}"
    )
    return g


def normalize_coords(coords):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)

    normalized = []
    for x, y in coords:
        nx = ((x - min_x) / width) * 2000 - 1000
        ny = ((y - min_y) / height) * 2000 - 1000
        normalized.append((nx, ny))

    return normalized


def compute_layout(g: ig.Graph) -> list[tuple[float, float]]:
    """
    Fast clustered layout:
    1. Detect communities
    2. Place communities globally
    3. Place nodes radially inside each community

    This is much faster than a full force layout and still groups related nodes.
    """
    print(
        f"Computing clustered layout for {g.vcount()} nodes and {g.ecount()} edges..."
    )
    started_at = time.time()

    # Undirected graph is better for community detection / coarse layout
    ug = g.as_undirected(combine_edges="ignore")

    print("Detecting communities...")
    comm_started_at = time.time()
    communities = ug.community_leiden(objective_function="modularity")
    membership = communities.membership
    comm_elapsed = time.time() - comm_started_at
    print(
        f"Detected {len(communities)} communities | "
        f"elapsed={format_seconds(comm_elapsed)}"
    )

    # Build community -> node list
    community_nodes: dict[int, list[int]] = {}
    for node_idx, comm_id in enumerate(membership):
        community_nodes.setdefault(comm_id, []).append(node_idx)

    community_ids = sorted(community_nodes.keys())
    community_sizes = [len(community_nodes[cid]) for cid in community_ids]

    # Build a coarse community graph using inter-community edge counts
    print("Building community graph...")
    edge_weights: dict[tuple[int, int], int] = {}
    for e in ug.es:
        s = e.source
        t = e.target
        cs = membership[s]
        ct = membership[t]

        if cs == ct:
            continue

        a, b = sorted((cs, ct))
        edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1

    comm_index = {cid: i for i, cid in enumerate(community_ids)}
    coarse_edges = []
    coarse_weights = []

    for (ca, cb), w in edge_weights.items():
        coarse_edges.append((comm_index[ca], comm_index[cb]))
        coarse_weights.append(w)

    cg = ig.Graph(
        n=len(community_ids),
        edges=coarse_edges,
        directed=False,
    )
    if coarse_weights:
        cg.es["weight"] = coarse_weights

    # Global layout for communities
    print("Computing community layout...")
    coarse_started_at = time.time()
    if cg.vcount() == 1:
        coarse_coords = [(0.0, 0.0)]
    elif cg.ecount() > 0:
        try:
            coarse_layout = cg.layout_graphopt()
        except Exception:
            coarse_layout = cg.layout_fruchterman_reingold()
        coarse_coords = coarse_layout.coords
    else:
        # if communities are disconnected, place them in a ring
        coarse_coords = []
        n = len(community_ids)
        big_radius = max(200.0, math.sqrt(max(1, n)) * 80.0)
        for i in range(n):
            angle = 2 * math.pi * i / max(1, n)
            coarse_coords.append((
                big_radius * math.cos(angle),
                big_radius * math.sin(angle),
            ))

    coarse_elapsed = time.time() - coarse_started_at
    print(f"Community layout done | elapsed={format_seconds(coarse_elapsed)}")

    # Final node coordinates
    coords = [(0.0, 0.0)] * g.vcount()

    print("Placing nodes inside communities...")
    place_started_at = time.time()

    for coarse_i, comm_id in enumerate(community_ids):
        node_indices = community_nodes[comm_id]
        cx, cy = coarse_coords[coarse_i]

        size = len(node_indices)

        # community radius grows sublinearly
        base_radius = max(8.0, math.sqrt(size) * 6.0)

        if size == 1:
            coords[node_indices[0]] = (cx, cy)
            continue

        # sort by degree so high-degree nodes stay closer to center
        node_indices_sorted = sorted(
            node_indices,
            key=lambda idx: ug.degree(idx),
            reverse=True
        )

        # place nodes in concentric rings
        ring_capacity = 8
        placed = 0
        ring = 0

        while placed < size:
            ring += 1
            remaining = size - placed
            current_ring_count = min(remaining, ring_capacity * ring)
            radius = base_radius * ring

            for j in range(current_ring_count):
                idx = node_indices_sorted[placed + j]
                angle = 2 * math.pi * j / max(1, current_ring_count)

                # slight deterministic offset based on degree
                deg = ug.degree(idx)
                radial_boost = min(10.0, deg * 0.15)

                x = cx + (radius + radial_boost) * math.cos(angle)
                y = cy + (radius + radial_boost) * math.sin(angle)
                coords[idx] = (x, y)

            placed += current_ring_count

    place_elapsed = time.time() - place_started_at
    print(f"Placed nodes in communities | elapsed={format_seconds(place_elapsed)}")

    coords = normalize_coords(coords)

    elapsed = time.time() - started_at
    print(f"Clustered layout done | elapsed={format_seconds(elapsed)}")
    return coords


def build_graph_json(g: ig.Graph, coords: list[tuple[float, float]]) -> dict:
    print("Building JSON payload...")
    started_at = time.time()

    indegrees = g.indegree()
    outdegrees = g.outdegree()
    degrees = g.degree()

    nodes_json = []
    for v in g.vs:
        idx = v.index
        label = v["label"]
        x, y = coords[idx]

        # lightweight size scaling
        size = max(1.5, min(10, 2 + (degrees[idx] ** 0.35)))

        nodes_json.append({
            "key": label,
            "label": label,
            "x": x,
            "y": y,
            "size": size,
            "color": "#4f46e5",
            "attributes": {
                "degree": degrees[idx],
                "in_degree": indegrees[idx],
                "out_degree": outdegrees[idx],
            }
        })

    edges_json = []
    for i, e in enumerate(g.es):
        source = g.vs[e.source]["label"]
        target = g.vs[e.target]["label"]

        edges_json.append({
            "key": f"e{i}",
            "source": source,
            "target": target,
            "color": "rgba(120,120,120,0.18)",
            "size": 0.4,
        })

    payload = {
        "nodes": nodes_json,
        "edges": edges_json,
    }

    elapsed = time.time() - started_at
    print(
        f"JSON payload ready | nodes={len(nodes_json)} edges={len(edges_json)} | "
        f"elapsed={format_seconds(elapsed)}"
    )
    return payload


def write_graph_json(output_file: Path, graph_data: dict) -> None:
    print(f"Writing {output_file} ...")
    started_at = time.time()

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False)

    elapsed = time.time() - started_at
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_file.resolve()} | {size_mb:.2f} MB | elapsed={format_seconds(elapsed)}")


def main():
    total_started_at = time.time()

    if not DISCOVERED_FILE.exists():
        raise FileNotFoundError(f"Could not find {DISCOVERED_FILE.resolve()}")

    if not EDGES_FILE.exists():
        raise FileNotFoundError(f"Could not find {EDGES_FILE.resolve()}")

    print("Loading discovered titles...")
    discovered_started_at = time.time()
    discovered_titles = load_discovered_titles(DISCOVERED_FILE)
    discovered_elapsed = time.time() - discovered_started_at
    print(
        f"Loaded discovered titles: {len(discovered_titles)} | "
        f"elapsed={format_seconds(discovered_elapsed)}"
    )

    nodes, edges = read_edges_filtered(EDGES_FILE, discovered_titles)
    print(f"Filtered graph: {len(nodes)} nodes, {len(edges)} edges")

    g = build_igraph(nodes, edges)
    coords = compute_layout(g)
    graph_data = build_graph_json(g, coords)
    write_graph_json(OUTPUT_JSON, graph_data)

    total_elapsed = time.time() - total_started_at
    print(f"Done | total elapsed={format_seconds(total_elapsed)}")


if __name__ == "__main__":
    main()

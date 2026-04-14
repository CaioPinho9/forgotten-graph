import colorsys
import os
import random
import time
from collections import Counter

import igraph as ig

from db import init_db, load_page_categories, read_filtered_edges, save_cluster_assignments

PRINT_EVERY_EDGES = 100_000
DEFAULT_CLUSTER_SEED = 42


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_igraph(nodes: list[str], edges: list[tuple[str, str]]) -> ig.Graph:
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    indexed_edges = [(node_to_idx[s], node_to_idx[t]) for s, t in edges]

    g = ig.Graph(n=len(nodes), edges=indexed_edges, directed=True)
    g.vs["label"] = nodes
    return g


def color_for_cluster(cluster_id: int, total_clusters: int) -> str:
    if total_clusters <= 0:
        total_clusters = 1

    hue = (cluster_id % total_clusters) / total_clusters
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def choose_cluster_name(
    cluster_id: int,
    cluster_titles: list[str],
    page_categories: dict[str, list[str]],
) -> str:
    counts: Counter[str] = Counter()

    for title in cluster_titles:
        for category in page_categories.get(title, []):
            category_name = (category or "").strip()
            if category_name:
                counts[category_name] += 1

    if not counts:
        return f"Cluster {cluster_id}"

    return min(
        counts.items(),
        key=lambda item: (-item[1], item[0].lower(), item[0]),
    )[0]


def build_clusters():
    init_db()

    print("Loading page categories from DB...")
    page_categories = load_page_categories()
    print(f"Loaded categories for {len(page_categories)} pages")

    print("Reading filtered edges from DB...")
    started_at = time.time()
    nodes, edges = read_filtered_edges(discovered_only=True)
    elapsed = time.time() - started_at
    print(
        f"Filtered graph: {len(nodes)} nodes, {len(edges)} edges | "
        f"elapsed {format_eta(elapsed)}"
    )

    print("Building graph...")
    g = build_igraph(nodes, edges)

    print("Converting to undirected for clustering...")
    ug = g.as_undirected(combine_edges="ignore")

    cluster_seed = int(os.environ.get("CLUSTER_SEED", DEFAULT_CLUSTER_SEED))
    ig.set_random_number_generator(random.Random(cluster_seed))

    print(f"Running Leiden clustering with seed={cluster_seed}...")
    cluster_started_at = time.time()
    communities = ug.community_leiden(
        objective_function="modularity",
        # Optional tuning:
        # resolution_parameter=2.0,
    )
    cluster_elapsed = time.time() - cluster_started_at

    membership = communities.membership
    total_clusters = len(communities)
    cluster_colors = {
        cluster_id: color_for_cluster(cluster_id, total_clusters)
        for cluster_id in range(total_clusters)
    }

    print(
        f"Detected {total_clusters} clusters | "
        f"elapsed {format_eta(cluster_elapsed)}"
    )

    assignments = []
    cluster_titles: dict[int, list[str]] = {}
    for idx, vertex in enumerate(g.vs):
        title = vertex["label"]
        cluster_id = membership[idx]
        assignments.append((title, cluster_id))
        cluster_titles.setdefault(cluster_id, []).append(title)

    cluster_metadata = {}
    for cluster_id in range(total_clusters):
        cluster_name = choose_cluster_name(
            cluster_id,
            cluster_titles.get(cluster_id, []),
            page_categories,
        )
        cluster_metadata[cluster_id] = {
            "cluster_color": color_for_cluster(cluster_id, total_clusters),
            "cluster_name": cluster_name,
        }

    print("Saving cluster assignments to DB...")
    save_started_at = time.time()
    save_cluster_assignments(assignments, cluster_metadata)
    save_elapsed = time.time() - save_started_at

    print(
        f"Saved {len(assignments)} cluster assignments and {len(cluster_metadata)} cluster metadata rows | "
        f"elapsed {format_eta(save_elapsed)}"
    )


def main():
    build_clusters()


if __name__ == "__main__":
    main()

import colorsys
import os
import random
import re
import time
from collections import Counter

import igraph as ig

from db import (
    init_db,
    load_page_categories,
    read_filtered_edges,
    save_cluster_assignments,
    clear_cluster_and_layout_data,
)

PRINT_EVERY_EDGES = 100_000
DEFAULT_CLUSTER_SEED = 42

GENERIC_CATEGORY_PATTERNS = [
    r"^articles? ",
    r"^pages? ",
    r"^images? ",
    r"^browse ",
    r"^templates? ",
    r"^stubs?$",
    r"^wiki ",
    r"^maintenance",
    r"^cleanup",
    r"^candidates? ",
    r"^all ",
    r"^years?$",
    r"^year of ",
    r"^\d{1,4}$",
    r"^\d{1,4}s$",
    r"^births?$",
    r"^deaths?$",
    r"^inhabitants?$",
    r"^locations?$",
    r"^items?$",
    r"^creatures?$",
    r"^males?$",
    r"^females?$",
    r"^humans?$",
    r"^events?$",
    r"^books?$",
    r"^comics?$",
]


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
    golden_ratio_conjugate = 0.618033988749895
    hue = (cluster_id * golden_ratio_conjugate) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def normalize_category_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def is_generic_category(name: str) -> bool:
    lowered = name.lower()
    return any(re.search(pattern, lowered) for pattern in GENERIC_CATEGORY_PATTERNS)


def choose_cluster_name_candidates(
    cluster_titles: list[str],
    page_categories: dict[str, list[str]],
) -> list[str]:
    raw_counts: Counter[str] = Counter()
    page_coverage: Counter[str] = Counter()

    for title in cluster_titles:
        seen_for_page = set()

        for category in page_categories.get(title, []):
            category_name = normalize_category_name(category)
            if not category_name:
                continue
            if is_generic_category(category_name):
                continue

            raw_counts[category_name] += 1

            key = category_name.lower()
            if key not in seen_for_page:
                page_coverage[category_name] += 1
                seen_for_page.add(key)

    if not raw_counts:
        return []

    ranked = sorted(
        raw_counts.keys(),
        key=lambda name: (
            -page_coverage[name],
            -raw_counts[name],
            len(name),
            name.lower(),
            name,
        ),
    )

    return ranked


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
        resolution=2,
    )
    cluster_elapsed = time.time() - cluster_started_at

    membership = communities.membership
    total_clusters = len(communities)

    print(
        f"Detected {total_clusters} clusters | "
        f"elapsed {format_eta(cluster_elapsed)}"
    )

    sizes = communities.sizes()
    if sizes:
        print(
            f"Cluster size stats | min={min(sizes)} max={max(sizes)} "
            f"avg={sum(sizes)/len(sizes):.2f}"
        )
        print(f"Singletons: {sum(1 for s in sizes if s == 1)}")
        print(f"Clusters < 5 nodes: {sum(1 for s in sizes if s < 5)}")
        print(f"Clusters < 10 nodes: {sum(1 for s in sizes if s < 10)}")

    assignments = []
    cluster_titles: dict[int, list[str]] = {}
    for idx, vertex in enumerate(g.vs):
        title = vertex["label"]
        cluster_id = membership[idx]
        assignments.append((title, cluster_id))
        cluster_titles.setdefault(cluster_id, []).append(title)

    cluster_metadata = {}
    used_names: set[str] = set()

    for cluster_id in range(total_clusters):
        candidates = choose_cluster_name_candidates(
            cluster_titles.get(cluster_id, []),
            page_categories,
        )

        cluster_name = None

        for candidate in candidates:
            normalized = candidate.strip().lower()
            if normalized not in used_names:
                cluster_name = candidate
                used_names.add(normalized)
                break

        if not cluster_name:
            sample_titles = cluster_titles.get(cluster_id, [])[:2]
            if sample_titles:
                cluster_name = " / ".join(sample_titles)
            else:
                cluster_name = f"Cluster {cluster_id}"

        cluster_metadata[cluster_id] = {
            "cluster_color": color_for_cluster(cluster_id, total_clusters),
            "cluster_name": cluster_name,
        }

    print("Clearing old cluster data...")
    clear_cluster_and_layout_data()

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

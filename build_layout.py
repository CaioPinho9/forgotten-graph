import math
import time

import igraph as ig

from db import (
    init_db,
    read_filtered_edges,
    load_cluster_assignments,
    load_cluster_colors,
    save_nodes_layout,
)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ConsoleLogger:
    def __init__(self):
        self._transient_width = 0
        self._transient_active = False

    def finish_transient(self) -> None:
        if self._transient_active:
            print()
        self._transient_width = 0
        self._transient_active = False

    def line(self, message: str) -> None:
        self.finish_transient()
        print(message)

    def status(self, message: str) -> None:
        self._transient_width = max(self._transient_width, len(message))
        self._transient_active = True
        print(message.ljust(self._transient_width), end="\r", flush=True)


LOGGER = ConsoleLogger()


def build_igraph(nodes: list[str], edges: list[tuple[str, str]]) -> ig.Graph:
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    indexed_edges = [(node_to_idx[s], node_to_idx[t]) for s, t in edges]

    g = ig.Graph(n=len(nodes), edges=indexed_edges, directed=True)
    g.vs["label"] = nodes
    return g


def normalize_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not coords:
        return coords

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


def repel_points_grid(
    points: list[tuple[float, float]],
    iterations: int = 6,
    min_dist: float = 0.03,
    step: float = 0.01,
    progress_prefix: str = "",
    community_started_at: float | None = None,
    layout_started_at: float | None = None,
    layout_progress_start: float = 0.0,
    layout_progress_end: float = 1.0,
) -> list[tuple[float, float]]:
    """
    Spatial-grid repel:
    instead of checking all pairs, only checks points in neighboring cells.
    """
    if len(points) <= 1:
        return points

    pts = [list(p) for p in points]
    n = len(pts)
    cell_size = max(min_dist, 1e-6)
    started_at = time.time()

    for iteration in range(iterations):
        grid: dict[tuple[int, int], list[int]] = {}

        for i, (x, y) in enumerate(pts):
            cx = int(math.floor(x / cell_size))
            cy = int(math.floor(y / cell_size))
            grid.setdefault((cx, cy), []).append(i)

        deltas = [[0.0, 0.0] for _ in range(n)]

        for i, (xi, yi) in enumerate(pts):
            cx = int(math.floor(xi / cell_size))
            cy = int(math.floor(yi / cell_size))

            dx_total = 0.0
            dy_total = 0.0

            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    for j in grid.get((nx, ny), []):
                        if i == j:
                            continue

                        xj, yj = pts[j]
                        dx = xi - xj
                        dy = yi - yj
                        dist2 = dx * dx + dy * dy

                        if dist2 < 1e-12:
                            dx += 1e-4
                            dy += 1e-4
                            dist2 = dx * dx + dy * dy

                        dist = dist2 ** 0.5
                        if dist < min_dist:
                            force = (min_dist - dist) / min_dist
                            dx_total += (dx / dist) * force
                            dy_total += (dy / dist) * force

            deltas[i][0] = dx_total * step
            deltas[i][1] = dy_total * step

        for i in range(n):
            pts[i][0] += deltas[i][0]
            pts[i][1] += deltas[i][1]

        if progress_prefix:
            elapsed = time.time() - started_at
            completed = iteration + 1
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = iterations - completed
            step_eta = remaining / rate if rate > 0 else 0.0

            layout_eta_text = "--:--"
            if layout_started_at is not None and community_started_at is not None and rate > 0:
                community_elapsed = time.time() - community_started_at
                community_total_estimate = community_elapsed + step_eta
                community_progress = (
                    community_elapsed / community_total_estimate
                    if community_total_estimate > 0 else 0.0
                )
                layout_progress = layout_progress_start + (
                    (layout_progress_end - layout_progress_start) * community_progress
                )
                if layout_progress > 0:
                    layout_elapsed = time.time() - layout_started_at
                    layout_total_estimate = layout_elapsed / layout_progress
                    layout_eta = max(0.0, layout_total_estimate - layout_elapsed)
                    layout_eta_text = format_eta(layout_eta)

            LOGGER.status(
                f"{progress_prefix} repel {completed}/{iterations} | "
                f"elapsed {format_eta(elapsed)} | "
                f"step eta {format_eta(step_eta) if rate > 0 else '--:--'} | "
                f"layout eta {layout_eta_text}"
            )

    LOGGER.finish_transient()
    return [(x, y) for x, y in pts]


def compute_clustered_layout(g: ig.Graph, cluster_assignments: dict[str, dict]) -> list[tuple[float, float]]:
    LOGGER.line(f"Computing clustered layout for {g.vcount()} nodes and {g.ecount()} edges...")

    ug = g.as_undirected(combine_edges="ignore")

    community_nodes: dict[int, list[int]] = {}
    for idx, vertex in enumerate(g.vs):
        title = vertex["label"]
        cluster_id = cluster_assignments[title]["cluster_id"]
        community_nodes.setdefault(cluster_id, []).append(idx)

    community_ids = sorted(community_nodes.keys())

    edge_weights: dict[tuple[int, int], int] = {}
    for e in ug.es:
        s = e.source
        t = e.target

        s_title = g.vs[s]["label"]
        t_title = g.vs[t]["label"]

        cs = cluster_assignments[s_title]["cluster_id"]
        ct = cluster_assignments[t_title]["cluster_id"]

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

    cg = ig.Graph(n=len(community_ids), edges=coarse_edges, directed=False)
    if coarse_weights:
        cg.es["weight"] = coarse_weights

    LOGGER.line("Computing community layout...")
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
        coarse_coords = [(float(i), 0.0) for i in range(len(community_ids))]
    LOGGER.line(f"Community layout done | elapsed {format_eta(time.time() - coarse_started_at)}")

    coords = [(0.0, 0.0)] * g.vcount()

    def compute_local_layout(
        subgraph: ig.Graph,
        size: int,
        progress_prefix: str = "",
        layout_started_at: float | None = None,
        layout_progress_start: float = 0.0,
        layout_progress_end: float = 1.0,
    ):
        if size == 1:
            return [(0.0, 0.0)]

        community_started_at = time.time()

        if size <= 1500:
            if progress_prefix:
                LOGGER.line(f"{progress_prefix} local layout=fruchterman_reingold size={size}")
            try:
                layout = subgraph.layout_fruchterman_reingold(niter=250)
                local_coords = layout.coords
            except Exception:
                if progress_prefix:
                    LOGGER.line(f"{progress_prefix} fallback layout=graphopt size={size}")
                layout = subgraph.layout_graphopt()
                local_coords = layout.coords

            local_coords = normalize_coords(local_coords)
            local_coords = [(x / 1000.0, y / 1000.0) for x, y in local_coords]

            if progress_prefix:
                LOGGER.line(f"{progress_prefix} repel pass iterations=6")
            local_coords = repel_points_grid(
                local_coords,
                iterations=6,
                min_dist=0.03,
                step=0.02,
                progress_prefix=progress_prefix,
                community_started_at=community_started_at,
                layout_started_at=layout_started_at,
                layout_progress_start=layout_progress_start,
                layout_progress_end=layout_progress_end,
            )
            return local_coords

        if progress_prefix:
            LOGGER.line(f"{progress_prefix} large-cluster mode=anchor-cloud size={size}")

        degrees = subgraph.degree()
        order = sorted(range(size), key=lambda i: degrees[i], reverse=True)

        anchor_count = min(600, max(120, int(size ** 0.5) * 10))
        anchor_ids = set(order[:anchor_count])

        if progress_prefix:
            LOGGER.line(f"{progress_prefix} selected anchors={anchor_count}")

        anchor_vertices = sorted(anchor_ids)
        anchor_sub = subgraph.subgraph(anchor_vertices)

        try:
            if progress_prefix:
                LOGGER.line(f"{progress_prefix} anchor layout=fruchterman_reingold")
            anchor_layout = anchor_sub.layout_fruchterman_reingold(niter=250)
            anchor_coords = anchor_layout.coords
        except Exception:
            if progress_prefix:
                LOGGER.line(f"{progress_prefix} fallback anchor layout=graphopt")
            anchor_layout = anchor_sub.layout_graphopt()
            anchor_coords = anchor_layout.coords

        anchor_coords = normalize_coords(anchor_coords)
        anchor_coords = [(x / 1000.0, y / 1000.0) for x, y in anchor_coords]

        anchor_map = {}
        for i, original_local_idx in enumerate(anchor_vertices):
            anchor_map[original_local_idx] = anchor_coords[i]

        local_coords = [(0.0, 0.0)] * size
        progress_every = max(1000, size // 10)
        started_at = time.time()

        if progress_prefix:
            LOGGER.line(f"{progress_prefix} positioning non-anchor nodes={size - len(anchor_map)}")

        for i in range(size):
            if i in anchor_map:
                local_coords[i] = anchor_map[i]
                continue

            nbrs = subgraph.neighbors(i)
            anchor_nbrs = [n for n in nbrs if n in anchor_map]

            if anchor_nbrs:
                weighted = sorted(
                    anchor_nbrs,
                    key=lambda n: degrees[n],
                    reverse=True,
                )[:8]
                x = sum(anchor_map[n][0] * max(1, degrees[n]) for n in weighted)
                y = sum(anchor_map[n][1] * max(1, degrees[n]) for n in weighted)
                weight_total = sum(max(1, degrees[n]) for n in weighted)
                x /= weight_total
                y /= weight_total

                # pull gently toward a second local center to avoid radial-looking shells
                if len(weighted) > 1:
                    alt = weighted[(i + len(weighted)) % len(weighted)]
                    x = (x * 0.82) + (anchor_map[alt][0] * 0.18)
                    y = (y * 0.82) + (anchor_map[alt][1] * 0.18)
            else:
                base = anchor_vertices[i % len(anchor_vertices)]
                x, y = anchor_map[base]

            # irregular deterministic jitter, not angular / circular
            j1 = (((i * 92821) % 1000) / 1000.0) - 0.5
            j2 = (((i * 68917) % 1000) / 1000.0) - 0.5
            j3 = (((i * 39119) % 1000) / 1000.0) - 0.5
            density_scale = 0.06 + (0.05 * min(1.0, len(anchor_nbrs) / 4.0))
            x += (j1 * density_scale) + (j3 * 0.025)
            y += (j2 * density_scale) - (j3 * 0.02)

            local_coords[i] = (x, y)

            if progress_prefix and (i + 1) % progress_every == 0:
                pct = ((i + 1) / size) * 100
                elapsed = time.time() - started_at
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining = size - (i + 1)
                eta = remaining / rate if rate > 0 else 0.0
                layout_eta_text = "--:--"
                if layout_started_at is not None and rate > 0:
                    layout_elapsed = time.time() - layout_started_at
                    position_progress = 0.12 + (0.43 * ((i + 1) / size))
                    layout_progress = layout_progress_start + (
                        (layout_progress_end - layout_progress_start) * position_progress
                    )
                    if layout_progress > 0:
                        layout_total_estimate = layout_elapsed / layout_progress
                        layout_eta_text = format_eta(max(0.0, layout_total_estimate - layout_elapsed))
                LOGGER.status(
                    f"{progress_prefix} positioned {i + 1}/{size} nodes ({pct:.1f}%) | "
                    f"step eta {format_eta(eta) if rate > 0 else '--:--'} | "
                    f"layout eta {layout_eta_text}"
                )

        LOGGER.finish_transient()

        # stronger repel so nodes spread into clouds instead of rings
        if progress_prefix:
            LOGGER.line(f"{progress_prefix} repel pass iterations=12")
        local_coords = repel_points_grid(
            local_coords,
            iterations=10,
            min_dist=0.045,
            step=0.024,
            progress_prefix=progress_prefix,
            community_started_at=community_started_at,
            layout_started_at=layout_started_at,
            layout_progress_start=layout_progress_start,
            layout_progress_end=layout_progress_end,
        )

        return local_coords

    community_sizes = {cid: len(community_nodes[cid]) for cid in community_ids}
    total_nodes_in_communities = sum(community_sizes.values())

    def estimated_cost(size: int) -> float:
        if size <= 1500:
            return size ** 1.25
        return (1500 ** 1.25) + ((size - 1500) ** 1.08) * 3.0

    community_costs = {cid: estimated_cost(sz) for cid, sz in community_sizes.items()}
    total_cost = sum(community_costs.values())
    completed_cost = 0.0
    completed_nodes = 0

    communities_started_at = time.time()
    total_communities = len(community_ids)

    LOGGER.line(
        f"Placing nodes inside communities... 0/{total_communities} (0.00%) | "
        f"elapsed {format_eta(0)} | eta --:--"
    )

    for coarse_i, comm_id in enumerate(community_ids, start=1):
        node_indices = community_nodes[comm_id]
        cx, cy = coarse_coords[coarse_i - 1]
        size = len(node_indices)
        cost = community_costs[comm_id]

        elapsed_before = time.time() - communities_started_at
        rate_cost = completed_cost / elapsed_before if elapsed_before > 0 else 0.0
        remaining_cost = total_cost - completed_cost
        eta_before = remaining_cost / rate_cost if rate_cost > 0 else 0.0
        completed_nodes_pct = (
            (completed_nodes / total_nodes_in_communities) * 100
            if total_nodes_in_communities else 100.0
        )
        completed_cost_pct = (completed_cost / total_cost) * 100 if total_cost else 100.0

        LOGGER.line(
            f"[community {coarse_i}/{total_communities}] cluster_id={comm_id} "
            f"size={size} | nodes {completed_nodes}/{total_nodes_in_communities} "
            f"({completed_nodes_pct:.2f}%) | weighted {completed_cost_pct:.2f}% | "
            f"elapsed {format_eta(elapsed_before)} | "
            f"eta {format_eta(eta_before) if rate_cost > 0 else '--:--'}"
        )

        community_started_at = time.time()

        if size == 1:
            coords[node_indices[0]] = (cx, cy)
        else:
            subgraph = ug.subgraph(node_indices)
            progress_prefix = (
                f"[community {coarse_i}/{total_communities}] "
                f"cluster_id={comm_id} progress |"
            )
            local_coords = compute_local_layout(
                subgraph,
                size,
                progress_prefix=progress_prefix,
                layout_started_at=communities_started_at,
                layout_progress_start=(completed_cost / total_cost) if total_cost else 1.0,
                layout_progress_end=((completed_cost + cost) / total_cost) if total_cost else 1.0,
            )

            spread = max(60.0, math.sqrt(size) * 42.0)

            for local_i, node_idx in enumerate(node_indices):
                lx, ly = local_coords[local_i]
                coords[node_idx] = (
                    cx + lx * spread,
                    cy + ly * spread,
                )

        completed_cost += cost
        completed_nodes += size

        community_elapsed = time.time() - community_started_at
        total_elapsed = time.time() - communities_started_at
        rate_cost = completed_cost / total_elapsed if total_elapsed > 0 else 0.0
        remaining_cost = total_cost - completed_cost
        eta_after = remaining_cost / rate_cost if rate_cost > 0 else 0.0
        completed_nodes_pct = (
            (completed_nodes / total_nodes_in_communities) * 100
            if total_nodes_in_communities else 100.0
        )
        completed_cost_pct = (completed_cost / total_cost) * 100 if total_cost else 100.0

        LOGGER.line(
            f"[community {coarse_i}/{total_communities}] done | "
            f"nodes {completed_nodes}/{total_nodes_in_communities} "
            f"({completed_nodes_pct:.2f}%) | weighted {completed_cost_pct:.2f}% | "
            f"community elapsed {format_eta(community_elapsed)} | "
            f"total elapsed {format_eta(total_elapsed)} | "
            f"eta {format_eta(eta_after) if rate_cost > 0 else '--:--'}"
        )

    return normalize_coords(coords)


def compute_node_size_from_in_degree(in_degree: int) -> float:
    size = 1.5 + 1.4 * math.log1p(in_degree)
    return max(1.5, min(18.0, size))


def build_layout():
    init_db()

    LOGGER.line("Loading cluster assignments from DB...")
    raw_cluster_assignments = load_cluster_assignments()
    cluster_assignments = {
        title: {"cluster_id": cid}
        for title, cid in raw_cluster_assignments.items()
    }
    LOGGER.line(f"Loaded {len(cluster_assignments)} cluster assignments")

    LOGGER.line("Loading cluster colors from DB...")
    cluster_colors = load_cluster_colors()
    LOGGER.line(f"Loaded {len(cluster_colors)} cluster colors")

    LOGGER.line("Reading filtered edges from DB...")
    started_at = time.time()
    nodes, edges = read_filtered_edges(discovered_only=True)
    elapsed = time.time() - started_at
    LOGGER.line(
        f"Filtered graph: {len(nodes)} nodes, {len(edges)} edges | "
        f"elapsed {format_eta(elapsed)}"
    )

    LOGGER.line("Building graph...")
    g = build_igraph(nodes, edges)

    LOGGER.line("Computing layout...")
    coords = compute_clustered_layout(g, cluster_assignments)

    indegrees = g.indegree()
    outdegrees = g.outdegree()

    LOGGER.line("Saving nodes layout to DB...")
    rows = []
    for idx, vertex in enumerate(g.vs):
        title = vertex["label"]
        x, y = coords[idx]
        cluster_id = cluster_assignments[title]["cluster_id"]
        cluster_color = cluster_colors[cluster_id]
        in_degree = indegrees[idx]
        out_degree = outdegrees[idx]
        node_size = compute_node_size_from_in_degree(in_degree)

        rows.append((
            title,
            x,
            y,
            cluster_id,
            cluster_color,
            in_degree,
            out_degree,
            node_size,
        ))

    save_nodes_layout(rows)
    LOGGER.line(f"Saved {len(rows)} layout rows")


def main():
    build_layout()


if __name__ == "__main__":
    main()

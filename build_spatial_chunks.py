import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from db import init_db, load_cluster_metadata, load_edges_for_chunks, load_nodes_layout, load_page_categories

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "graph_chunks"

ZOOM_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
INTERACTIVE_MIN_ZOOM = 7
LABEL_MIN_ZOOM = 8

MIN_IN_DEGREE_BY_ZOOM = {
    0: 999_999,
    1: 999_999,
    2: 999_999,
    3: 999_999,
    4: 999_999,
    5: 999_999,
    6: 999_999,
    7: 6,
    8: 0,
}


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_bounds(nodes: dict[str, dict]) -> tuple[float, float, float, float]:
    xs = [n["x"] for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()]
    return min(xs), max(xs), min(ys), max(ys)


def get_tile_xy(x: float, y: float, bounds: tuple[float, float, float, float], zoom: int) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)

    tiles_per_axis = 2 ** zoom

    nx = (x - min_x) / width
    ny = (y - min_y) / height

    tx = min(tiles_per_axis - 1, max(0, int(nx * tiles_per_axis)))
    ty = min(tiles_per_axis - 1, max(0, int(ny * tiles_per_axis)))
    return tx, ty


def make_node_payload(
    node: dict,
    page_categories: dict[str, list[str]],
    cluster_metadata: dict[int, dict[str, str | None]],
    include_label: bool,
) -> dict:
    categories = page_categories.get(node["page_title"], [])
    cluster_info = cluster_metadata.get(node["cluster_id"], {})
    payload = {
        "id": node["page_title"],
        "x": node["x"],
        "y": node["y"],
        "size": node["node_size"],
        "color": node["cluster_color"],
        "cluster_id": node["cluster_id"],
        "cluster_name": cluster_info.get("cluster_name"),
        "in_degree": node["in_degree"],
        "out_degree": node["out_degree"],
        "categories": categories,
    }
    if include_label:
        payload["label"] = node["page_title"]
    return payload


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_included_node_ids_by_zoom(nodes: dict[str, dict]) -> dict[int, set[str]]:
    included_by_zoom = {}

    for zoom in ZOOM_LEVELS:
        min_in_degree = MIN_IN_DEGREE_BY_ZOOM.get(zoom, 0)
        included_by_zoom[zoom] = {
            node["page_title"]
            for node in nodes.values()
            if node["in_degree"] >= min_in_degree
        }

    return included_by_zoom


def build_spatial_chunks():
    init_db()
    reset_output_dir(OUTPUT_DIR)

    print("Loading node layout from DB...")
    node_rows = load_nodes_layout()
    nodes = {row["page_title"]: row for row in node_rows}
    print(f"Nodes with layout: {len(nodes)}")

    print("Loading page categories from DB...")
    page_categories = load_page_categories()
    print(f"Loaded categories for {len(page_categories)} pages")

    print("Loading cluster metadata from DB...")
    cluster_metadata = load_cluster_metadata()
    print(f"Loaded metadata for {len(cluster_metadata)} clusters")

    print("Loading edges from DB...")
    edges = load_edges_for_chunks(discovered_only=True)
    print(f"Filtered edges: {len(edges)}")

    bounds = get_bounds(nodes)
    print(f"Bounds: {bounds}")
    included_node_ids_by_zoom = get_included_node_ids_by_zoom(nodes)

    manifest = {
        "bounds": {
            "min_x": bounds[0],
            "max_x": bounds[1],
            "min_y": bounds[2],
            "max_y": bounds[3],
        },
        "zoom_levels": {},
        "label_min_zoom": LABEL_MIN_ZOOM,
    }

    node_zoom_started_at = time.time()
    total_node_zooms = len(ZOOM_LEVELS)

    for zoom_index, zoom in enumerate(ZOOM_LEVELS, start=1):
        print(f"Building node tiles for zoom {zoom}...")
        tiles = defaultdict(list)

        included_node_ids = included_node_ids_by_zoom[zoom]
        include_label = zoom >= LABEL_MIN_ZOOM

        for node in nodes.values():
            if node["page_title"] not in included_node_ids:
                continue

            tx, ty = get_tile_xy(node["x"], node["y"], bounds, zoom)
            tiles[(tx, ty)].append(
                make_node_payload(
                    node,
                    page_categories,
                    cluster_metadata,
                    include_label=include_label,
                )
            )

        zoom_dir = OUTPUT_DIR / f"z{zoom}"
        ensure_dir(zoom_dir)

        for (tx, ty), tile_nodes in tiles.items():
            filename = f"nodes_{tx}_{ty}.json"
            with (zoom_dir / filename).open("w", encoding="utf-8") as f:
                json.dump(tile_nodes, f, ensure_ascii=False)

        manifest["zoom_levels"][str(zoom)] = {
            "tiles_per_axis": 2 ** zoom,
            "node_tiles": len(tiles),
            "edge_tiles": 0,
            "interactive_enabled": zoom >= INTERACTIVE_MIN_ZOOM,
            "min_in_degree": MIN_IN_DEGREE_BY_ZOOM.get(zoom, 0),
            "included_nodes": len(included_node_ids),
        }

        elapsed = time.time() - node_zoom_started_at
        rate = zoom_index / elapsed if elapsed > 0 else 0
        remaining = total_node_zooms - zoom_index
        eta = remaining / rate if rate > 0 else 0
        print(
            f"[node_tiles] {zoom_index}/{total_node_zooms} zooms | "
            f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
        )

    edge_zoom_started_at = time.time()
    total_edge_zooms = len(ZOOM_LEVELS)

    for zoom_index, zoom in enumerate(ZOOM_LEVELS, start=1):
        print(f"Building edge tiles for zoom {zoom}...")
        tiles = defaultdict(list)

        included_node_ids = included_node_ids_by_zoom[zoom]

        for source, target in edges:
            s = nodes.get(source)
            t = nodes.get(target)

            if not s or not t:
                continue

            if source not in included_node_ids or target not in included_node_ids:
                continue

            tx, ty = get_tile_xy(s["x"], s["y"], bounds, zoom)
            tiles[(tx, ty)].append({
                "source": source,
                "target": target,
            })

        zoom_dir = OUTPUT_DIR / f"z{zoom}"
        ensure_dir(zoom_dir)

        for (tx, ty), tile_edges in tiles.items():
            filename = f"edges_{tx}_{ty}.json"
            with (zoom_dir / filename).open("w", encoding="utf-8") as f:
                json.dump(tile_edges, f, ensure_ascii=False)

        manifest["zoom_levels"][str(zoom)]["edge_tiles"] = len(tiles)

        elapsed = time.time() - edge_zoom_started_at
        rate = zoom_index / elapsed if elapsed > 0 else 0
        remaining = total_edge_zooms - zoom_index
        eta = remaining / rate if rate > 0 else 0
        print(
            f"[edge_tiles] {zoom_index}/{total_edge_zooms} zooms | "
            f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
        )

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to {OUTPUT_DIR.resolve()}")


def main():
    build_spatial_chunks()


if __name__ == "__main__":
    main()

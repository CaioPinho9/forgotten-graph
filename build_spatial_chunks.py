import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from db import (
    init_db,
    load_cluster_metadata,
    load_nodes_layout,
    load_page_categories,
)
from zoom_config import load_zoom_config

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "graph_chunks"
SEARCH_INDEX_FILE = OUTPUT_DIR / "search_index.json"

ZOOM_CONFIG = load_zoom_config()
INTERACTIVE_MIN_ZOOM = ZOOM_CONFIG["interactive_min_zoom"]
ZOOM_LEVELS = list(range(INTERACTIVE_MIN_ZOOM, ZOOM_CONFIG["chunk_max_zoom"] + 1))
LABEL_MIN_ZOOM = ZOOM_CONFIG["label_min_zoom"]


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


def get_tile_xy(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)

    tiles_per_axis = 2 ** zoom

    nx = (x - min_x) / width
    ny = 1.0 - ((y - min_y) / height)

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

    bounds = get_bounds(nodes)
    print(f"Bounds: {bounds}")
    included_node_ids = set(nodes.keys())

    manifest = {
        "bounds": {
            "min_x": bounds[0],
            "max_x": bounds[1],
            "min_y": bounds[2],
            "max_y": bounds[3],
        },
        "zoom_levels": {},
        "zoom_config": ZOOM_CONFIG,
        "label_min_zoom": LABEL_MIN_ZOOM,
        "interactive_min_zoom": INTERACTIVE_MIN_ZOOM,
        "adjacency": {
            "mode": "api",
            "endpoint": "/api/adjacency?node_id={node_id}",
        },
    }

    node_zoom_started_at = time.time()
    total_node_zooms = len(ZOOM_LEVELS)

    for zoom_index, zoom in enumerate(ZOOM_LEVELS, start=1):
        print(f"Building node tiles for zoom {zoom}...")
        tiles = defaultdict(list)

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
            "interactive_enabled": zoom >= INTERACTIVE_MIN_ZOOM,
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

    print("Writing search index...")
    search_index = [
        make_node_payload(
            node,
            page_categories,
            cluster_metadata,
            include_label=True,
        )
        for node in sorted(nodes.values(), key=lambda n: n["page_title"].lower())
    ]
    with SEARCH_INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(search_index, f, ensure_ascii=False)

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to {OUTPUT_DIR.resolve()}")


def main():
    build_spatial_chunks()


if __name__ == "__main__":
    main()

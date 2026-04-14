import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageDraw

from db import init_db, load_nodes_layout, load_edges_for_chunks
from zoom_config import load_zoom_config

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR_NODES = DATA_DIR / "tiles_nodes"
OUTPUT_DIR_EDGES = DATA_DIR / "tiles_edges"

ZOOM_CONFIG = load_zoom_config()
STATIC_TILE_MAX_ZOOM = ZOOM_CONFIG["static_tile_max_zoom"]
MAX_EDGE_TILE_ZOOM = ZOOM_CONFIG["edge_tile_max_zoom"]

TILE_SIZE = 1024

NODES_BACKGROUND = (0, 0, 0, 0)
EDGES_BACKGROUND = (0, 0, 0, 0)

EDGE_ALPHA = 40
EDGE_WIDTH_BASE = 1.1
EDGE_SEGMENTS = 12
EDGE_ARROW_SIZE_BASE = 7.0
EDGE_ARROW_SPREAD_BASE = 4.0

EDGE_SHARD_COUNT = 128


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    hex_color = (hex_color or "").lstrip("#")
    if len(hex_color) != 6:
        return 79, 70, 229, alpha
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b, alpha


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(
    c1: tuple[int, int, int, int],
    c2: tuple[int, int, int, int],
    t: float,
) -> tuple[int, int, int, int]:
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
        int(lerp(c1[3], c2[3], t)),
    )


def edge_alpha_for_node_sizes(
    source_size: float,
    target_size: float,
    base_alpha: int = EDGE_ALPHA,
) -> int:
    max_size = max(float(source_size or 0.0), float(target_size or 0.0), 0.0)
    alpha_multiplier = clamp(0.9 + math.log1p(max_size) * 0.45, 0.9, 2.6)
    return int(round(clamp(base_alpha * alpha_multiplier, base_alpha, 255)))


def get_bounds(nodes: list[dict]) -> tuple[float, float, float, float]:
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    return min(xs), max(xs), min(ys), max(ys)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def world_to_global_pixel(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)

    tiles_per_axis = 2 ** zoom

    nx = (x - min_x) / width
    ny = 1.0 - ((y - min_y) / height)

    global_px = nx * tiles_per_axis * TILE_SIZE
    global_py = ny * tiles_per_axis * TILE_SIZE
    return global_px, global_py


def clip_tile_range(min_px: float, max_px: float, tiles_per_axis: int) -> tuple[int, int]:
    min_t = max(0, min(tiles_per_axis - 1, int(math.floor(min_px / TILE_SIZE))))
    max_t = max(0, min(tiles_per_axis - 1, int(math.floor(max_px / TILE_SIZE))))
    return min_t, max_t


def draw_gradient_line_on_tile(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color1: tuple[int, int, int, int],
    color2: tuple[int, int, int, int],
    width: int,
    segments: int = EDGE_SEGMENTS,
) -> None:
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments

        sx = lerp(x1, x2, t0)
        sy = lerp(y1, y2, t0)
        ex = lerp(x1, x2, t1)
        ey = lerp(y1, y2, t1)

        color = lerp_color(color1, color2, (t0 + t1) * 0.5)
        draw.line((sx, sy, ex, ey), fill=color, width=width)


def draw_arrowhead_on_tile(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int, int],
    size: float,
    spread: float,
) -> None:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return

    ux = dx / length
    uy = dy / length

    base_x = x2 - ux * size
    base_y = y2 - uy * size

    px = -uy
    py = ux

    left = (base_x + px * spread, base_y + py * spread)
    right = (base_x - px * spread, base_y - py * spread)

    draw.polygon([(x2, y2), left, right], fill=color)


def build_node_tile_ops_at_zoom(
    nodes: list[dict],
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> tuple[dict[tuple[int, int], list[tuple]], int]:
    tiles_per_axis = 2 ** zoom
    tile_node_ops: dict[tuple[int, int], list[tuple]] = {}
    total_ops = 0

    total_nodes = len(nodes)
    started_at = time.time()

    for node_index, node in enumerate(nodes, start=1):
        global_px, global_py = world_to_global_pixel(node["x"], node["y"], bounds, zoom)

        color = hex_to_rgba(node["cluster_color"], 140)
        radius = max(1, int(max(1.0, node["node_size"] * (0.35 + 0.15 * zoom))))

        min_tx, max_tx = clip_tile_range(global_px - radius, global_px + radius, tiles_per_axis)
        min_ty, max_ty = clip_tile_range(global_py - radius, global_py + radius, tiles_per_axis)

        for tx in range(min_tx, max_tx + 1):
            for ty in range(min_ty, max_ty + 1):
                local_x = global_px - tx * TILE_SIZE
                local_y = global_py - ty * TILE_SIZE

                tile_node_ops.setdefault((tx, ty), []).append(
                    (local_x, local_y, radius, color)
                )
                total_ops += 1

        if node_index % 5000 == 0 or node_index == total_nodes:
            elapsed = time.time() - started_at
            rate = node_index / elapsed if elapsed > 0 else 0.0
            remaining = total_nodes - node_index
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"[node_precompute z{zoom}] {node_index}/{total_nodes} nodes | "
                f"tiles {len(tile_node_ops)} | ops {total_ops} | "
                f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
            )

    return tile_node_ops, total_ops


def merge_child_node_ops_to_parent(
    child_ops: dict[tuple[int, int], list[tuple]]
) -> dict[tuple[int, int], list[tuple]]:
    parent_ops: dict[tuple[int, int], list[tuple]] = {}

    for (child_tx, child_ty), ops in child_ops.items():
        parent_tx = child_tx // 2
        parent_ty = child_ty // 2

        offset_x = (child_tx % 2) * (TILE_SIZE / 2)
        offset_y = (child_ty % 2) * (TILE_SIZE / 2)

        parent_list = parent_ops.setdefault((parent_tx, parent_ty), [])

        for local_x, local_y, radius, color in ops:
            parent_local_x = offset_x + (local_x / 2.0)
            parent_local_y = offset_y + (local_y / 2.0)
            parent_radius = max(1, radius / 2.0)

            parent_list.append(
                (parent_local_x, parent_local_y, parent_radius, color)
            )

    return parent_ops


def render_sparse_node_tiles(
    output_dir: Path,
    zoom: int,
    tile_draw_ops: dict[tuple[int, int], list[tuple]],
    rendered_tiles_so_far: int,
    total_tiles_to_render: int,
    started_at: float,
) -> int:
    zoom_dir = output_dir / str(zoom)
    ensure_dir(zoom_dir)

    rendered_now = 0
    tiles_in_zoom = len(tile_draw_ops)

    for tile_index, ((tx, ty), ops) in enumerate(tile_draw_ops.items(), start=1):
        img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), NODES_BACKGROUND)
        draw = ImageDraw.Draw(img, "RGBA")

        for local_x, local_y, radius, color in ops:
            draw.ellipse(
                (local_x - radius, local_y - radius, local_x + radius, local_y + radius),
                fill=color,
            )

        tile_dir = zoom_dir / str(tx)
        ensure_dir(tile_dir)
        img.save(tile_dir / f"{ty}.png")

        rendered_now += 1
        total_done = rendered_tiles_so_far + rendered_now

        if tile_index % 100 == 0 or tile_index == tiles_in_zoom:
            elapsed = time.time() - started_at
            rate = total_done / elapsed if elapsed > 0 else 0.0
            remaining = total_tiles_to_render - total_done
            eta = remaining / rate if rate > 0 else 0.0

            print(
                f"[nodes z{zoom}] {tile_index}/{tiles_in_zoom} tiles in zoom | "
                f"total {total_done}/{total_tiles_to_render} | "
                f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
            )

    return rendered_now


def render_node_tiles(
    nodes: list[dict],
    bounds: tuple[float, float, float, float],
) -> None:
    reset_output_dir(OUTPUT_DIR_NODES)

    max_zoom = STATIC_TILE_MAX_ZOOM
    print(f"Precomputing NODE tiles only at max zoom z{max_zoom}...")
    max_zoom_ops, max_zoom_total_ops = build_node_tile_ops_at_zoom(nodes, bounds, max_zoom)

    zoom_to_ops: dict[int, dict[tuple[int, int], list[tuple]]] = {max_zoom: max_zoom_ops}
    current_ops = max_zoom_ops

    for zoom in range(max_zoom - 1, -1, -1):
        print(f"Merging NODE child tiles z{zoom + 1} -> z{zoom}...")
        current_ops = merge_child_node_ops_to_parent(current_ops)
        zoom_to_ops[zoom] = current_ops

    total_tiles_to_render = sum(len(zoom_to_ops[z]) for z in range(0, max_zoom + 1))
    print(f"Total NODE tiles to render: {total_tiles_to_render}")

    started_at = time.time()
    rendered_tiles_so_far = 0

    for zoom in range(0, max_zoom + 1):
        tile_node_ops = zoom_to_ops[zoom]
        rendered_tiles = render_sparse_node_tiles(
            output_dir=OUTPUT_DIR_NODES,
            zoom=zoom,
            tile_draw_ops=tile_node_ops,
            rendered_tiles_so_far=rendered_tiles_so_far,
            total_tiles_to_render=total_tiles_to_render,
            started_at=started_at,
        )

        rendered_tiles_so_far += rendered_tiles
        elapsed = time.time() - started_at
        rate = rendered_tiles_so_far / elapsed if elapsed > 0 else 0.0
        remaining = total_tiles_to_render - rendered_tiles_so_far
        eta = remaining / rate if rate > 0 else 0.0

        print(
            f"[node_tiles] zoom {zoom} done | "
            f"zoom tiles {len(tile_node_ops)} | "
            f"total {rendered_tiles_so_far}/{total_tiles_to_render} | "
            f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
        )

    print(f"Node max-zoom ops: {max_zoom_total_ops}")


def tile_shard_index(tx: int, ty: int, shard_count: int = EDGE_SHARD_COUNT) -> int:
    return ((tx * 73856093) ^ (ty * 19349663)) % shard_count


def iter_segment_tiles(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tiles_per_axis: int,
):
    tx0 = int(math.floor(x0 / TILE_SIZE))
    ty0 = int(math.floor(y0 / TILE_SIZE))
    tx1 = int(math.floor(x1 / TILE_SIZE))
    ty1 = int(math.floor(y1 / TILE_SIZE))

    tx0 = max(0, min(tiles_per_axis - 1, tx0))
    ty0 = max(0, min(tiles_per_axis - 1, ty0))
    tx1 = max(0, min(tiles_per_axis - 1, tx1))
    ty1 = max(0, min(tiles_per_axis - 1, ty1))

    yield (tx0, ty0)

    if tx0 == tx1 and ty0 == ty1:
        return

    dx = x1 - x0
    dy = y1 - y0

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)

    inv_dx = float("inf") if dx == 0 else abs(1.0 / dx)
    inv_dy = float("inf") if dy == 0 else abs(1.0 / dy)

    tx = tx0
    ty = ty0

    if step_x > 0:
        next_vert = (tx + 1) * TILE_SIZE
        t_max_x = (next_vert - x0) * inv_dx
    elif step_x < 0:
        next_vert = tx * TILE_SIZE
        t_max_x = (x0 - next_vert) * inv_dx
    else:
        t_max_x = float("inf")

    if step_y > 0:
        next_horiz = (ty + 1) * TILE_SIZE
        t_max_y = (next_horiz - y0) * inv_dy
    elif step_y < 0:
        next_horiz = ty * TILE_SIZE
        t_max_y = (y0 - next_horiz) * inv_dy
    else:
        t_max_y = float("inf")

    t_delta_x = float("inf") if step_x == 0 else TILE_SIZE * inv_dx
    t_delta_y = float("inf") if step_y == 0 else TILE_SIZE * inv_dy

    safety = 0
    max_steps = (abs(tx1 - tx0) + abs(ty1 - ty0) + 4) * 4

    while (tx != tx1 or ty != ty1) and safety < max_steps:
        safety += 1

        if t_max_x < t_max_y:
            tx += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            ty += step_y
            t_max_y += t_delta_y
        else:
            tx += step_x
            ty += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y

        if 0 <= tx < tiles_per_axis and 0 <= ty < tiles_per_axis:
            yield (tx, ty)
        else:
            break


def make_zoom_temp_dir(base_temp_dir: str, zoom: int) -> str:
    path = os.path.join(base_temp_dir, f"z{zoom}")
    os.makedirs(path, exist_ok=True)
    return path


def make_shard_paths(temp_zoom_dir: str, shard_count: int = EDGE_SHARD_COUNT) -> list[str]:
    return [os.path.join(temp_zoom_dir, f"shard_{i:03d}.txt") for i in range(shard_count)]


def build_edge_membership_temp_at_zoom(
    nodes: list[dict],
    edges: list[tuple[str, str]],
    bounds: tuple[float, float, float, float],
    zoom: int,
    temp_zoom_dir: str,
    shard_count: int = EDGE_SHARD_COUNT,
) -> tuple[int, int]:
    tiles_per_axis = 2 ** zoom
    node_by_id = {n["page_title"]: n for n in nodes}
    shard_paths = make_shard_paths(temp_zoom_dir, shard_count)

    total_edges = len(edges)
    total_memberships = 0
    started_at = time.time()
    touched_shards = set()

    with ExitStack() as stack:
        shard_files = [
            stack.enter_context(open(path, "a", encoding="utf-8"))
            for path in shard_paths
        ]

        for edge_index, (source, target) in enumerate(edges, start=1):
            s = node_by_id.get(source)
            t = node_by_id.get(target)
            if not s or not t:
                continue

            gx1, gy1 = world_to_global_pixel(s["x"], s["y"], bounds, zoom)
            gx2, gy2 = world_to_global_pixel(t["x"], t["y"], bounds, zoom)

            seen_tiles = set()
            for tx, ty in iter_segment_tiles(gx1, gy1, gx2, gy2, tiles_per_axis):
                if (tx, ty) in seen_tiles:
                    continue
                seen_tiles.add((tx, ty))

                shard_idx = tile_shard_index(tx, ty, shard_count)
                touched_shards.add(shard_idx)
                shard_files[shard_idx].write(f"{tx}\t{ty}\t{edge_index - 1}\n")
                total_memberships += 1

            if edge_index % 100000 == 0 or edge_index == total_edges:
                elapsed = time.time() - started_at
                rate = edge_index / elapsed if elapsed > 0 else 0.0
                remaining = total_edges - edge_index
                eta = remaining / rate if rate > 0 else 0.0

                print(
                    f"[edge_precompute z{zoom}] {edge_index}/{total_edges} edges | "
                    f"memberships {total_memberships} | "
                    f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
                )

    return total_memberships, len(touched_shards)


def render_edge_tile_from_ids(
    draw: ImageDraw.ImageDraw,
    edge_ids: list[int],
    tx: int,
    ty: int,
    zoom: int,
    edges: list[tuple[str, str]],
    node_by_id: dict[str, dict],
    bounds: tuple[float, float, float, float],
) -> None:
    for edge_id in edge_ids:
        source, target = edges[edge_id]

        s = node_by_id.get(source)
        t = node_by_id.get(target)
        if not s or not t:
            continue

        gx1, gy1 = world_to_global_pixel(s["x"], s["y"], bounds, zoom)
        gx2, gy2 = world_to_global_pixel(t["x"], t["y"], bounds, zoom)

        local_x1 = gx1 - tx * TILE_SIZE
        local_y1 = gy1 - ty * TILE_SIZE
        local_x2 = gx2 - tx * TILE_SIZE
        local_y2 = gy2 - ty * TILE_SIZE

        edge_alpha = edge_alpha_for_node_sizes(s["node_size"], t["node_size"])
        c1 = hex_to_rgba(s["cluster_color"], edge_alpha)
        c2 = hex_to_rgba(t["cluster_color"], edge_alpha)
        edge_width = max(1, int(round(EDGE_WIDTH_BASE + zoom * 0.12)))

        draw_gradient_line_on_tile(
            draw,
            local_x1,
            local_y1,
            local_x2,
            local_y2,
            c1,
            c2,
            width=edge_width,
        )

        arrow_size = max(EDGE_ARROW_SIZE_BASE, edge_width * 4.0)
        arrow_spread = max(EDGE_ARROW_SPREAD_BASE, edge_width * 2.2)

        draw_arrowhead_on_tile(
            draw,
            local_x1,
            local_y1,
            local_x2,
            local_y2,
            c2,
            size=arrow_size,
            spread=arrow_spread,
        )


def process_edge_zoom_from_temp(
    zoom: int,
    temp_zoom_dir: str,
    next_temp_zoom_dir: str | None,
    output_dir: Path,
    edges: list[tuple[str, str]],
    node_by_id: dict[str, dict],
    bounds: tuple[float, float, float, float],
    rendered_tiles_so_far: int,
    total_tiles_hint: int | None,
    started_at: float,
    shard_count: int = EDGE_SHARD_COUNT,
) -> tuple[int, int]:
    zoom_dir = output_dir / str(zoom)
    ensure_dir(zoom_dir)

    shard_paths = make_shard_paths(temp_zoom_dir, shard_count)
    parent_memberships_written = 0
    rendered_now = 0

    parent_files = None
    if next_temp_zoom_dir is not None:
        parent_paths = make_shard_paths(next_temp_zoom_dir, shard_count)
        parent_stack = ExitStack()
        parent_files = [
            parent_stack.enter_context(open(path, "a", encoding="utf-8"))
            for path in parent_paths
        ]
    else:
        parent_stack = None

    try:
        for shard_index, shard_path in enumerate(shard_paths, start=1):
            if not os.path.exists(shard_path) or os.path.getsize(shard_path) == 0:
                continue

            tile_to_edge_ids: dict[tuple[int, int], list[int]] = defaultdict(list)

            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    tx_s, ty_s, edge_id_s = line.rstrip("\n").split("\t")
                    tx = int(tx_s)
                    ty = int(ty_s)
                    edge_id = int(edge_id_s)
                    tile_to_edge_ids[(tx, ty)].append(edge_id)

            for (tx, ty), ids in tile_to_edge_ids.items():
                deduped_ids = list(dict.fromkeys(ids))

                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), EDGES_BACKGROUND)
                draw = ImageDraw.Draw(img, "RGBA")
                render_edge_tile_from_ids(
                    draw=draw,
                    edge_ids=deduped_ids,
                    tx=tx,
                    ty=ty,
                    zoom=zoom,
                    edges=edges,
                    node_by_id=node_by_id,
                    bounds=bounds,
                )

                tile_dir = zoom_dir / str(tx)
                ensure_dir(tile_dir)
                img.save(tile_dir / f"{ty}.png")

                rendered_now += 1
                total_done = rendered_tiles_so_far + rendered_now

                if parent_files is not None:
                    parent_tx = tx // 2
                    parent_ty = ty // 2
                    shard_idx = tile_shard_index(parent_tx, parent_ty, shard_count)
                    pf = parent_files[shard_idx]
                    for edge_id in deduped_ids:
                        pf.write(f"{parent_tx}\t{parent_ty}\t{edge_id}\n")
                        parent_memberships_written += 1

                if rendered_now % 100 == 0:
                    elapsed = time.time() - started_at
                    rate = total_done / elapsed if elapsed > 0 else 0.0

                    if total_tiles_hint:
                        remaining = total_tiles_hint - total_done
                        eta = remaining / rate if rate > 0 else 0.0
                        eta_text = format_eta(eta)
                        total_text = f"{total_done}/{total_tiles_hint}"
                        extra = f" | eta {eta_text}"
                    else:
                        total_text = str(total_done)
                        extra = ""

                    print(
                        f"[edge_tiles z{zoom}] rendered {rendered_now} tiles in zoom | "
                        f"total {total_text} | elapsed {format_eta(elapsed)} | "
                        f"tiles/sec {rate:.2f}{extra}"
                    )

            os.remove(shard_path)

            elapsed = time.time() - started_at
            print(
                f"[edge_tiles z{zoom}] shard {shard_index}/{shard_count} done | "
                f"zoom tiles so far {rendered_now} | elapsed {format_eta(elapsed)}"
            )
    finally:
        if parent_stack is not None:
            parent_stack.close()

    return rendered_now, parent_memberships_written


def render_edge_tiles(
    nodes: list[dict],
    edges: list[tuple[str, str]],
    bounds: tuple[float, float, float, float],
) -> None:
    reset_output_dir(OUTPUT_DIR_EDGES)

    node_by_id = {n["page_title"]: n for n in nodes}
    max_zoom = MAX_EDGE_TILE_ZOOM

    with tempfile.TemporaryDirectory(prefix="edge_tiles_") as temp_root:
        current_temp_dir = make_zoom_temp_dir(temp_root, max_zoom)

        print(f"Precomputing EDGE tile membership to temp files only at max zoom z{max_zoom}...")
        max_memberships, touched_shards = build_edge_membership_temp_at_zoom(
            nodes=nodes,
            edges=edges,
            bounds=bounds,
            zoom=max_zoom,
            temp_zoom_dir=current_temp_dir,
            shard_count=EDGE_SHARD_COUNT,
        )

        print(
            f"Max zoom membership written to temp files | "
            f"memberships {max_memberships} | touched shards {touched_shards}"
        )

        started_at = time.time()
        rendered_tiles_so_far = 0

        for zoom in range(max_zoom, -1, -1):
            next_temp_dir = make_zoom_temp_dir(temp_root, zoom - 1) if zoom > 0 else None

            print(f"Rendering EDGE zoom z{zoom} from temp shards...")
            rendered_now, parent_memberships_written = process_edge_zoom_from_temp(
                zoom=zoom,
                temp_zoom_dir=current_temp_dir,
                next_temp_zoom_dir=next_temp_dir,
                output_dir=OUTPUT_DIR_EDGES,
                edges=edges,
                node_by_id=node_by_id,
                bounds=bounds,
                rendered_tiles_so_far=rendered_tiles_so_far,
                total_tiles_hint=None,
                started_at=started_at,
                shard_count=EDGE_SHARD_COUNT,
            )

            rendered_tiles_so_far += rendered_now
            elapsed = time.time() - started_at
            rate = rendered_tiles_so_far / elapsed if elapsed > 0 else 0.0

            print(
                f"[edge_tiles] zoom {zoom} done | "
                f"zoom tiles {rendered_now} | "
                f"parent memberships written {parent_memberships_written} | "
                f"total rendered tiles {rendered_tiles_so_far} | "
                f"elapsed {format_eta(elapsed)} | "
                f"tiles/sec {rate:.2f}"
            )

            current_temp_dir = next_temp_dir if next_temp_dir is not None else current_temp_dir


def build_tiles():
    init_db()

    print("Loading nodes from DB...")
    nodes = load_nodes_layout()
    print(f"Loaded {len(nodes)} nodes")

    print("Loading edges from DB...")
    edges = load_edges_for_chunks(discovered_only=True)
    print(f"Loaded {len(edges)} edges")

    bounds = get_bounds(nodes)

    render_node_tiles(nodes, bounds)
    render_edge_tiles(nodes, edges, bounds)


if __name__ == "__main__":
    build_tiles()

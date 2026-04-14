import time
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from db import init_db, load_nodes_layout

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "tiles"

STATIC_TILE_MAX_ZOOM = 6
TILE_ZOOMS = list(range(STATIC_TILE_MAX_ZOOM + 1))
TILE_SIZE = 1024

BACKGROUND = (17, 24, 39, 255)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (79, 70, 229, alpha)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


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


def world_to_tile_pixel(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> tuple[int, int, int, int]:
    min_x, max_x, min_y, max_y = bounds
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)

    tiles_per_axis = 2 ** zoom

    nx = (x - min_x) / width
    ny = 1.0 - ((y - min_y) / height)

    global_px = nx * tiles_per_axis * TILE_SIZE
    global_py = ny * tiles_per_axis * TILE_SIZE

    tile_x = min(tiles_per_axis - 1, max(0, int(global_px // TILE_SIZE)))
    tile_y = min(tiles_per_axis - 1, max(0, int(global_py // TILE_SIZE)))

    local_x = int(global_px - tile_x * TILE_SIZE)
    local_y = int(global_py - tile_y * TILE_SIZE)

    return tile_x, tile_y, local_x, local_y


def build_tiles():
    init_db()
    reset_output_dir(OUTPUT_DIR)

    print("Loading nodes from DB...")
    nodes = load_nodes_layout()
    print(f"Loaded {len(nodes)} nodes")

    bounds = get_bounds(nodes)

    started_at = time.time()
    total_zooms = len(TILE_ZOOMS)

    for zoom_index, zoom in enumerate(TILE_ZOOMS, start=1):
        print(f"Rendering tiles for zoom {zoom}...")
        tiles_per_axis = 2 ** zoom
        zoom_dir = OUTPUT_DIR / str(zoom)
        ensure_dir(zoom_dir)

        tile_images = {}
        tile_draws = {}

        for tx in range(tiles_per_axis):
            for ty in range(tiles_per_axis):
                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), BACKGROUND)
                draw = ImageDraw.Draw(img, "RGBA")
                tile_images[(tx, ty)] = img
                tile_draws[(tx, ty)] = draw

        for node in nodes:
            tx, ty, px, py = world_to_tile_pixel(node["x"], node["y"], bounds, zoom)
            draw = tile_draws[(tx, ty)]

            color = hex_to_rgba(node["cluster_color"], 180)
            r = max(1, int(max(1.0, node["node_size"] * (0.35 + 0.15 * zoom))))
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

        for (tx, ty), img in tile_images.items():
            tile_dir = zoom_dir / str(tx)
            ensure_dir(tile_dir)
            img.save(tile_dir / f"{ty}.png")

        elapsed = time.time() - started_at
        rate = zoom_index / elapsed if elapsed > 0 else 0
        remaining = total_zooms - zoom_index
        eta = remaining / rate if rate > 0 else 0
        print(
            f"[render_tiles] {zoom_index}/{total_zooms} zooms | "
            f"elapsed {format_eta(elapsed)} | eta {format_eta(eta)}"
        )

    print(f"Saved static tiles to {OUTPUT_DIR.resolve()}")


def main():
    build_tiles()


if __name__ == "__main__":
    main()

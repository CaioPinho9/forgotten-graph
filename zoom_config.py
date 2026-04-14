import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ZOOM_CONFIG_PATH = BASE_DIR / "zoom_config.json"


def load_zoom_config() -> dict[str, int]:
    with ZOOM_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = {
        "static_tile_max_zoom",
        "interactive_min_zoom",
        "label_min_zoom",
        "chunk_max_zoom",
        "edge_tile_max_zoom",
    }
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"Missing zoom config keys: {sorted(missing)}")

    normalized = {key: int(value) for key, value in config.items()}

    if normalized["interactive_min_zoom"] != normalized["static_tile_max_zoom"] + 1:
        raise ValueError(
            "interactive_min_zoom must be exactly static_tile_max_zoom + 1"
        )
    if normalized["label_min_zoom"] < normalized["interactive_min_zoom"]:
        raise ValueError("label_min_zoom must be >= interactive_min_zoom")
    if normalized["chunk_max_zoom"] < normalized["label_min_zoom"]:
        raise ValueError("chunk_max_zoom must be >= label_min_zoom")

    return normalized

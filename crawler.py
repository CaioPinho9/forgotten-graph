import csv
import json
import time
import threading
import requests

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://forgottenrealms.fandom.com/api.php"

DATA_DIR = Path("data")

SEARCHED_FILE = DATA_DIR / "searched_pages.txt"
FAILED_FILE = DATA_DIR / "failed_pages.txt"
PAGES_FILE = DATA_DIR / "pages.jsonl"
EDGES_FILE = DATA_DIR / "edges.csv"
DISCOVERED_TITLES_FILE = DATA_DIR / "discovered_titles.txt"

# ---- Tuning ----
EXPECTED_DISCOVERED_TOTAL = 72666
DISCOVERY_SLEEP_SECONDS = 0.05
PAGE_SLEEP_SECONDS = 0.0
REQUEST_TIMEOUT = 30
MAX_WORKERS = 4
FLUSH_EVERY = 50
TITLE_BATCH_SIZE = 20

_thread_local = threading.local()


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "forgotten-graph/1.0 (personal research script)",
        "Accept": "application/json",
    })

    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = create_session()
    return _thread_local.session


def get_json(params: dict, timeout: int = REQUEST_TIMEOUT, extra_sleep: float = 0.0) -> dict:
    session = get_session()
    response = session.get(API_URL, params=params, timeout=timeout)

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 15
        print(f"Rate limited (429). Waiting {wait_seconds}s...")
        time.sleep(wait_seconds)
        response = session.get(API_URL, params=params, timeout=timeout)

    response.raise_for_status()

    if extra_sleep > 0:
        time.sleep(extra_sleep)

    return response.json()


def load_line_set(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_lines(path: Path, values: list[str]) -> None:
    if not values:
        return

    with path.open("a", encoding="utf-8") as f:
        for value in values:
            f.write(value + "\n")


def append_jsonl_many(path: Path, records: list[dict]) -> None:
    if not records:
        return

    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_edges_file_header() -> None:
    if not EDGES_FILE.exists() or EDGES_FILE.stat().st_size == 0:
        with EDGES_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "target"])
            writer.writeheader()


def append_edges_many(edges: list[dict]) -> None:
    if not edges:
        return

    with EDGES_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writerows(edges)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def chunked(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def should_refresh_discovered_titles(expected_total: int = EXPECTED_DISCOVERED_TOTAL) -> bool:
    if not DISCOVERED_TITLES_FILE.exists():
        return True

    discovered_titles = load_line_set(DISCOVERED_TITLES_FILE)
    discovered_count = len(discovered_titles)

    print(f"Discovered titles on disk: {discovered_count}")
    return discovered_count < expected_total


def get_all_pages() -> int:
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": "500",
    }

    seen_titles = load_line_set(DISCOVERED_TITLES_FILE)
    new_titles_count = 0

    while True:
        data = get_json(params, extra_sleep=DISCOVERY_SLEEP_SECONDS)

        batch = data.get("query", {}).get("allpages", [])
        new_titles_buffer = []

        for page in batch:
            title = page["title"]

            if title in seen_titles:
                continue

            seen_titles.add(title)
            new_titles_buffer.append(title)

        if new_titles_buffer:
            append_lines(DISCOVERED_TITLES_FILE, new_titles_buffer)
            new_titles_count += len(new_titles_buffer)

        print(
            f"New titles found this refresh: {new_titles_count} | "
            f"total discovered on disk: {len(seen_titles)}"
        )

        if "continue" not in data:
            break

        params.update(data["continue"])

    return len(seen_titles)


def get_pages_links_and_categories_batch(titles: list[str]) -> list[dict]:
    """
    Fetch multiple titles in one API request sequence.
    Returns one record per title.
    """
    results = {
        title: {
            "page_title": title,
            "links_out_count": 0,
            "links_out": [],
            "categories_count": 0,
            "categories": [],
            "error": None,
        }
        for title in titles
    }

    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "links|categories",
        "pllimit": "max",
        "cllimit": "max",
    }

    try:
        while True:
            data = get_json(params, extra_sleep=PAGE_SLEEP_SECONDS)
            pages = data.get("query", {}).get("pages", {})

            for page_data in pages.values():
                title = page_data.get("title")
                if not title or title not in results:
                    continue

                for link in page_data.get("links", []):
                    if link.get("ns") == 0:
                        results[title]["links_out"].append(link["title"])

                for cat in page_data.get("categories", []):
                    cat_title = cat["title"]
                    if cat_title.startswith("Category:"):
                        cat_title = cat_title[len("Category:"):]
                    results[title]["categories"].append(cat_title)

            if "continue" not in data:
                break

            params.update(data["continue"])

        final_records = []
        for title in titles:
            record = results[title]
            record["links_out"] = sorted(set(record["links_out"]))
            record["categories"] = sorted(set(record["categories"]))
            record["links_out_count"] = len(record["links_out"])
            record["categories_count"] = len(record["categories"])
            final_records.append(record)

        return final_records

    except Exception as e:
        return [
            {
                "page_title": title,
                "links_out_count": 0,
                "links_out": [],
                "categories_count": 0,
                "categories": [],
                "error": str(e),
            }
            for title in titles
        ]


def fetch_batch(batch_titles: list[str]) -> list[dict]:
    return get_pages_links_and_categories_batch(batch_titles)


def flush_buffers(
    page_buffer: list[dict],
    edge_buffer: list[dict],
    searched_buffer: list[str],
    failed_buffer: list[str],
) -> None:
    append_jsonl_many(PAGES_FILE, page_buffer)
    append_edges_many(edge_buffer)
    append_lines(SEARCHED_FILE, searched_buffer)
    append_lines(FAILED_FILE, failed_buffer)

    page_buffer.clear()
    edge_buffer.clear()
    searched_buffer.clear()
    failed_buffer.clear()


def build_dataset(max_pages: int | None = None) -> None:
    ensure_edges_file_header()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if should_refresh_discovered_titles():
        print("Refreshing discovered titles from API...")
        discovered_total = get_all_pages()
        print(f"Finished discovery refresh. Total discovered titles: {discovered_total}")
    else:
        discovered_total = len(load_line_set(DISCOVERED_TITLES_FILE))
        print(
            f"Skipping get_all_pages(): {DISCOVERED_TITLES_FILE} already has "
            f"{discovered_total} titles."
        )

    discovered_titles = sorted(load_line_set(DISCOVERED_TITLES_FILE))
    searched_set = load_line_set(SEARCHED_FILE)
    failed_set = load_line_set(FAILED_FILE)

    pending_titles = [title for title in discovered_titles if title not in searched_set]

    if max_pages is not None:
        pending_titles = pending_titles[:max_pages]

    total = len(pending_titles)
    pending_batches = list(chunked(pending_titles, TITLE_BATCH_SIZE))

    print(f"Total discovered: {len(discovered_titles)}")
    print(f"Already searched: {len(searched_set)}")
    print(f"Failed before: {len(failed_set)}")
    print(f"Pending pages now: {total}")
    print(f"Pending batches now: {len(pending_batches)}")

    if total == 0:
        print("Nothing left to process.")
        return

    started_at = time.time()
    done_pages = 0
    ok_count = 0
    error_count = 0

    page_buffer = []
    edge_buffer = []
    searched_buffer = []
    failed_buffer = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(fetch_batch, batch_titles): batch_titles
            for batch_titles in pending_batches
        }

        for future in as_completed(future_to_batch):
            batch_results = future.result()

            for result in batch_results:
                title = result["page_title"]
                page_buffer.append(result)

                if result["error"] is None:
                    ok_count += 1
                    searched_buffer.append(title)

                    for target in result["links_out"]:
                        edge_buffer.append({
                            "source": title,
                            "target": target,
                        })
                else:
                    error_count += 1
                    failed_buffer.append(title)

                done_pages += 1

            if done_pages % 200 == 0:
                elapsed = time.time() - started_at
                avg = elapsed / done_pages if done_pages else 0
                remaining = total - done_pages
                eta_seconds = avg * remaining

                print(
                    f"[{done_pages}/{total}] {(done_pages / total) * 100:.2f}% | "
                    f"OK={ok_count} ERROR={error_count} | "
                    f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta_seconds)}"
                )

            if len(page_buffer) >= FLUSH_EVERY:
                flush_buffers(page_buffer, edge_buffer, searched_buffer, failed_buffer)

    flush_buffers(page_buffer, edge_buffer, searched_buffer, failed_buffer)

    total_elapsed = time.time() - started_at
    print(
        f"Done. OK: {ok_count}, ERROR: {error_count}, "
        f"total elapsed: {format_seconds(total_elapsed)}"
    )


if __name__ == "__main__":
    build_dataset()

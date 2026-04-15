import csv
import os
import random
import time
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import db

CSV_DIR = Path(__file__).resolve().parent / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PATHLONGEST_RESULTS_CSV = CSV_DIR / "pathlongest_simple_results.csv"

# Restrict analysis to pages that were actually searched.
searched_titles = db.load_searched_title_set()
all_page_edges = db.read_filtered_edges(discovered_only=False)[1]
page_edges = [
    (source_title, target_title)
    for source_title, target_title in all_page_edges
    if source_title in searched_titles and target_title in searched_titles
]
page_titles = sorted(searched_titles)
in_degree = {title: 0 for title in page_titles}
out_degree = {title: 0 for title in page_titles}
adjacency = {title: [] for title in page_titles}

for source_title, target_title in page_edges:
    out_degree[source_title] += 1
    in_degree[target_title] += 1
    adjacency[source_title].append(target_title)

dead_ends = [title for title in page_titles if out_degree[title] == 0]
orphans = [title for title in page_titles if in_degree[title] == 0]
orphan_dead_ends = [title for title in page_titles if in_degree[title] == 0 and out_degree[title] == 0]

start_candidates = [title for title in page_titles if out_degree[title] > 0]  # non dead-ends
end_candidates = [title for title in page_titles if in_degree[title] > 0]  # non orphans


def format_percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def read_biggest_target_row(path: str) -> tuple[str, str, int] | None:
    if not os.path.exists(path):
        return None

    biggest: tuple[str, str, int] | None = None
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                path_length = int(row["path_length"])
            except (KeyError, TypeError, ValueError):
                continue
            start = row.get("start", "")
            end = row.get("end", "")
            if biggest is None or path_length > biggest[2]:
                biggest = (start, end, path_length)
    return biggest


def read_biggest_longest(path: str) -> int:
    if not os.path.exists(path):
        return -1

    best = -1
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                path_length = int(row["path_length"])
            except (KeyError, TypeError, ValueError):
                continue
            if path_length > best:
                best = path_length
    return best


total_pages = len(page_titles)
print(f"Total pages: {total_pages}")
print(f"Dead-ends: {len(dead_ends)} ({format_percentage(len(dead_ends), total_pages)})")
print(f"Orphans: {len(orphans)} ({format_percentage(len(orphans), total_pages)})")
print(
    "Orphan dead-ends: "
    f"{len(orphan_dead_ends)} ({format_percentage(len(orphan_dead_ends), total_pages)})"
)

if not start_candidates:
    raise RuntimeError("No valid start nodes found (all pages are dead-ends).")

if not end_candidates:
    raise RuntimeError("No valid end nodes found (all pages are orphans).")


class Node:
    def __init__(self, title, parent=None, depth=0):
        self.title = title
        self.parent = parent
        self.depth = depth


def build_path(node: Node) -> list[Node]:
    path = []
    current = node
    while current is not None:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def run_bfs(start_node_title: str, end_node_title: str) -> dict:
    run_started_at = time.time()
    search = deque()
    discovered = set()

    start_node = Node(start_node_title, depth=0)
    search.append(start_node)
    discovered.add(start_node_title)

    found_node = None
    farthest_node = start_node

    while search:
        current_node = search.popleft()

        if current_node.title == end_node_title and found_node is None:
            found_node = current_node

        if current_node.depth > farthest_node.depth:
            farthest_node = current_node

        neighbors = adjacency[current_node.title]
        for neighbor_title in neighbors:
            if neighbor_title not in discovered:
                discovered.add(neighbor_title)
                search.append(
                    Node(
                        title=neighbor_title,
                        parent=current_node,
                        depth=current_node.depth + 1,
                    )
                )

    result = {
        "start_node_title": start_node_title,
        "end_node_title": end_node_title,
        "found": found_node is not None,
        "run_seconds": time.time() - run_started_at,
    }

    if found_node is not None:
        path = build_path(found_node)
        result["target_path_length"] = found_node.depth
        result["target_path"] = " -> ".join(node.title for node in path)
    else:
        result["target_path_length"] = -1
        result["target_path"] = "NOT FOUND"

    longest_path = build_path(farthest_node)
    result["longest_path_length"] = farthest_node.depth
    result["farthest_title"] = farthest_node.title
    result["longest_path"] = " -> ".join(node.title for node in longest_path)
    return result


# 1. PRE-PROCESS: Create a reverse adjacency list
# This allows us to "walk backwards" from your end nodes.
print("Building reverse adjacency map...")
reverse_adjacency = {title: [] for title in page_titles}
for source, target in page_edges:
    reverse_adjacency[target].append(source)

TARGET_ENDS = ["1567 DR"]


def find_longest_incoming_path(end_node: str, iterations: int = 1000):
    """
    Greedily walks BACKWARDS from an end node to find long simple paths.
    """
    best_path = []

    for _ in range(iterations):
        current_path = [end_node]
        visited = {end_node}
        current = end_node

        while True:
            # Look at nodes that link TO our current node
            sources = [s for s in reverse_adjacency.get(current, []) if s not in visited]
            if not sources:
                break

            # Heuristic: Pick a source node at random to explore a new branch
            next_node = random.choice(sources)

            current_path.append(next_node)
            visited.add(next_node)
            current = next_node

        if len(current_path) > len(best_path):
            best_path = current_path

    # Reverse it back so it reads Start -> End
    final_path = list(reversed(best_path))
    return {
        "end_node": end_node,
        "length": len(final_path) - 1,
        "path": " -> ".join(final_path)
    }


if __name__ == "__main__":
    workers = len(TARGET_ENDS)  # One process per target node
    iterations = int(os.environ.get("PATHLONGEST_ITERATIONS", "100000"))

    if not os.path.exists(PATHLONGEST_RESULTS_CSV):
        with open(PATHLONGEST_RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["end_node", "path_length", "path"])

    print(f"Searching for longest paths ending at: {TARGET_ENDS}")
    print(f"Iterations per target: {iterations}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(find_longest_incoming_path, target, iterations) for target in TARGET_ENDS]

        for future in as_completed(futures):
            res = future.result()
            print(f"\nTarget: {res['end_node']}")
            print(f"Longest Simple Path found: {res['length']} steps")

            with open(PATHLONGEST_RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([res['end_node'], res['length'], res['path']])

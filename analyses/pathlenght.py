import csv
import os
import time
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import db

CSV_DIR = Path(__file__).resolve().parent / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PATHLENGHT_CSV = CSV_DIR / "pathlength_records.csv"
LONGEST_PATHLENGHT_CSV = CSV_DIR / "pathlength_longest_records.csv"

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


# Create CSVs if they do not exist
if not os.path.exists(PATHLENGHT_CSV):
    with open(PATHLENGHT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "end", "path_length", "path"])

if not os.path.exists(LONGEST_PATHLENGHT_CSV):
    with open(LONGEST_PATHLENGHT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "farthest_end", "path_length", "path"])


end_nodes_titles = ["1567 DR", "Logan's Walk", "1570 DR"]
jobs = [(start_node_title, end_node_title) for end_node_title in end_nodes_titles for start_node_title in start_candidates]
total_jobs = len(jobs)

best_target_checkpoint = read_biggest_target_row(PATHLENGHT_CSV)
resume_from_job_index = 0
if best_target_checkpoint is not None:
    checkpoint_start, checkpoint_end, checkpoint_length = best_target_checkpoint
    try:
        resume_from_job_index = jobs.index((checkpoint_start, checkpoint_end)) + 1
        print(
            f"Checkpoint found in {PATHLENGHT_CSV}: start={checkpoint_start}, "
            f"end={checkpoint_end}, path_length={checkpoint_length}. "
            f"Resuming from next job ({resume_from_job_index}/{total_jobs})."
        )
    except ValueError:
        print(
            f"Checkpoint pair ({checkpoint_start}, {checkpoint_end}) not found in current job list; "
            "starting from first job."
        )

best_target_path_length = best_target_checkpoint[2] if best_target_checkpoint is not None else -1
best_longest_path_length = read_biggest_longest(LONGEST_PATHLENGHT_CSV)

thread_count = max(1, int(os.environ.get("PATHLENGHT_THREADS", os.cpu_count() or 1)))
max_in_flight = max(thread_count * 4, 32)

completed_jobs = resume_from_job_index
all_runs_started_at = time.time()

print(f"Planned runs: {total_jobs}")
print(f"Thread workers: {thread_count} (max in-flight: {max_in_flight})")



# Files
CSV_OUTPUT = CSV_DIR / "pathlength_graph_distances.csv"


def run_bfs_analysis(start_node: str):
    """
    Performs one BFS to find the 'farthest' node.
    In a BFS, the distance to the farthest node IS the longest shortest path
    from that specific start point.
    """
    start_time = time.time()

    # Track distance to avoid cycles and find depths
    # Using a dict is faster than a Node class for millions of iterations
    distances = {start_node: 0}
    predecessors = {start_node: None}
    queue = deque([start_node])

    farthest_node = start_node
    max_depth = 0

    while queue:
        current = queue.popleft()
        current_depth = distances[current]

        if current_depth > max_depth:
            max_depth = current_depth
            farthest_node = current

        # Explore neighbors
        for neighbor in adjacency.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = current_depth + 1
                predecessors[neighbor] = current
                queue.append(neighbor)

    # Reconstruct the path to the farthest node
    path = []
    curr = farthest_node
    while curr is not None:
        path.append(curr)
        curr = predecessors[curr]
    path.reverse()

    return {
        "start": start_node,
        "farthest": farthest_node,
        "length": max_depth,
        "path": " -> ".join(path),
        "runtime": time.time() - start_time
    }


if __name__ == "__main__":
    # 1. Setup Data (Assume adjacency and start_candidates are loaded as before)
    # Ensure adjacency is available to child processes

    # 2. Prepare Output
    if not os.path.exists(CSV_OUTPUT):
        with open(CSV_OUTPUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_node", "farthest_node", "depth", "path"])

    # 3. Parallel Processing (The "Fix" for slowness)
    # ProcessPoolExecutor uses multiple CPU cores, bypassing the Python GIL bottleneck.
    workers = max(1, os.cpu_count() - 1)
    print(f"Running BFS on {len(start_candidates)} nodes using {workers} cores...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_bfs_analysis, node): node for node in start_candidates}

        completed = 0
        for future in as_completed(futures):
            result = future.result()

            # Save every result or only new 'records'?
            # Appending all to CSV for full analysis:
            with open(CSV_OUTPUT, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([result["start"], result["farthest"], result["length"], result["path"]])

            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(start_candidates)} nodes analyzed...")

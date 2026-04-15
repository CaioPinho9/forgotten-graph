import csv
import os
import random
from pathlib import Path
from collections import deque

import db

CSV_DIR = Path(__file__).resolve().parent / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PATHSIZE_CSV = CSV_DIR / "pathsize_random_records.csv"
LONGEST_PATHSIZE_CSV = CSV_DIR / "pathsize_random_longest_records.csv"

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


def get_random_node_title():
    return random.choice(start_candidates)


def get_random_end_node_title():
    return random.choice(end_candidates)


def build_path(node: Node) -> list[Node]:
    path = []
    current = node

    while current is not None:
        path.append(current)
        current = current.parent

    path.reverse()
    return path


# Create CSVs if they do not exist
if not os.path.exists(PATHSIZE_CSV):
    with open(PATHSIZE_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "end", "path_length", "path"])

while True:
    search = deque()
    discovered = set()

    start_node_title = get_random_node_title()
    end_node_title = get_random_end_node_title()

    start_node = Node(start_node_title, depth=0)
    search.append(start_node)
    discovered.add(start_node_title)

    print(f"Finding path from {start_node_title} to {end_node_title}...")

    farthest_node = start_node

    while search:
        current_node = search.popleft()

        # Save the first occurrence of the target.
        # In BFS, the first time we find it is the shortest path.
        if current_node.title == end_node_title:
            path = build_path(current_node)
            path_titles = [node.title for node in path]
            steps = current_node.depth

            print("Target path:")
            print(" -> ".join(path_titles))
            print(f"Found path from {start_node_title} to {end_node_title} in {steps} steps!")

            with open(PATHSIZE_CSV, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([start_node_title, end_node_title, steps, " -> ".join(path_titles)])
            break

        # Keep tracking the deepest node even after finding the target
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

    print(f"No path found from {start_node_title} to {end_node_title}")

    with open(PATHSIZE_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([start_node_title, end_node_title, -1, "NOT FOUND"])

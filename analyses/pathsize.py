import csv
import os
import random
from collections import deque

import db

page_titles = db.load_discovered_titles()


class Node:
    def __init__(self, title, parent=None, depth=0):
        self.title = title
        self.parent = parent
        self.depth = depth


def get_random_node_title():
    return random.choice(page_titles)


def build_path(node: Node) -> list[Node]:
    path = []
    current = node

    while current is not None:
        path.append(current)
        current = current.parent

    path.reverse()
    return path


# Create CSVs if they do not exist
if not os.path.exists("pathsize.csv"):
    with open("pathsize.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "end", "count", "path"])

if not os.path.exists("longest_pathsize.csv"):
    with open("longest_pathsize.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start", "farthest_end", "count", "path"])


while True:
    search = deque()
    discovered = set()

    start_node_title = get_random_node_title()
    end_node_title = get_random_node_title()

    start_node = Node(start_node_title, depth=0)
    search.append(start_node)
    discovered.add(start_node_title)

    print(f"Finding path from {start_node_title} to {end_node_title}...")

    found_node = None
    farthest_node = start_node

    while search:
        current_node = search.popleft()

        # Save the first occurrence of the target.
        # In BFS, the first time we find it is the shortest path.
        if current_node.title == end_node_title and found_node is None:
            found_node = current_node

        # Keep tracking the deepest node even after finding the target
        if current_node.depth > farthest_node.depth:
            farthest_node = current_node

        neighbors = db.load_nodes_edges_by_title(current_node.title)
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

    # Save path to requested target
    if found_node is not None:
        path = build_path(found_node)
        path_titles = [node.title for node in path]
        steps = found_node.depth

        print("Target path:")
        print(" -> ".join(path_titles))
        print(f"Found path from {start_node_title} to {end_node_title} in {steps} steps!")

        with open("pathsize.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([start_node_title, end_node_title, steps, " -> ".join(path_titles)])
    else:
        print(f"No path found from {start_node_title} to {end_node_title}")

        with open("pathsize.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([start_node_title, end_node_title, -1, "NOT FOUND"])

    # Save longest reachable path from this same start node
    longest_path = build_path(farthest_node)
    longest_path_titles = [node.title for node in longest_path]
    longest_steps = farthest_node.depth

    print("Longest reachable path from start:")
    print(" -> ".join(longest_path_titles))
    print(
        f"Longest reachable path from {start_node_title} ends at "
        f"{farthest_node.title} in {longest_steps} steps!"
    )

    with open("longest_pathsize.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            start_node_title,
            farthest_node.title,
            longest_steps,
            " -> ".join(longest_path_titles),
        ])

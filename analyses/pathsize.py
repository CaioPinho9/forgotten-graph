import csv
import os
import random

import db

page_titles = db.load_discovered_titles()

search = []
discovered = set()

def get_random_node_title():
    return random.choice(page_titles)

start_node = get_random_node_title()
end_node = get_random_node_title()
count = 0

search.append(start_node)

print(f"Finding path from {start_node} to {end_node}...")

# Start csv if not exists
if not os.path.exists('pathsize.csv'):
    with open('pathsize.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['start', 'end', 'count'])


while True:
    current_node = search.pop(0)
    if current_node == end_node:
        print(f"Found path from {start_node} to {end_node} in {count} steps!")
        # append in csv the count
        with open('pathsize.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([start_node,end_node,count])
        break

    if current_node in discovered:
        continue

    to_search = db.load_nodes_edges_by_title(current_node)
    for search_node in to_search:
        search.append(search_node)
    discovered.add(current_node)
    count += 1

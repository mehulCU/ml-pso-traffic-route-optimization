import json
import pandas as pd
from collections import defaultdict, deque
from math import radians, sin, cos, sqrt, atan2


# -----------------------------------
# Haversine distance
# -----------------------------------

def haversine(coord1, coord2):
    """
    Calculate distance between two
    longitude/latitude coordinates in meters.
    """

    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # Earth radius in meters

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    dlat = lat2 - lat1
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# -----------------------------------
# Configuration
# -----------------------------------

TOLERANCE_METERS = 5.0


# -----------------------------------
# Load selected 100 roads
# -----------------------------------

top_segments = pd.read_csv(
    "ml/top_100_segments_train_only.csv"
)

segment_ids = set(
    top_segments["segment_id"].astype(int)
)


print("-----------------------------------")
print("TOLERANCE-BASED ROAD CONNECTIVITY")
print("-----------------------------------")

print("Selected roads:", len(segment_ids))
print("Connection tolerance:", TOLERANCE_METERS, "meters")


# -----------------------------------
# Load GeoJSON
# -----------------------------------

geojson_path = (
    "probe_counts/geojson/"
    "new_delhi__2024-08-11_to_2024-08-11_.geojson"
)

with open(geojson_path, "r") as f:
    data = json.load(f)


# -----------------------------------
# Extract selected roads
# -----------------------------------

roads = {}

for feature in data["features"]:

    properties = feature.get("properties", {})

    segment_id = properties.get("segmentId")

    if segment_id not in segment_ids:
        continue

    geometry = feature.get("geometry")

    if not geometry:
        continue

    coordinates = geometry.get("coordinates")

    if not coordinates:
        continue

    start = tuple(coordinates[0])
    end = tuple(coordinates[-1])

    roads[segment_id] = {
        "start": start,
        "end": end,
        "street_name": properties.get("streetName"),
        "distance": properties.get("distance")
    }


print("Roads with geometry:", len(roads))


# -----------------------------------
# Build adjacency using tolerance
# -----------------------------------

segment_list = list(roads.keys())

adjacency = defaultdict(set)

connection_count = 0

for i in range(len(segment_list)):

    road_a_id = segment_list[i]

    road_a = roads[road_a_id]

    for j in range(i + 1, len(segment_list)):

        road_b_id = segment_list[j]

        road_b = roads[road_b_id]

        # Compare all endpoint combinations
        distances = [

            haversine(
                road_a["start"],
                road_b["start"]
            ),

            haversine(
                road_a["start"],
                road_b["end"]
            ),

            haversine(
                road_a["end"],
                road_b["start"]
            ),

            haversine(
                road_a["end"],
                road_b["end"]
            )

        ]

        minimum_distance = min(distances)

        if minimum_distance <= TOLERANCE_METERS:

            adjacency[road_a_id].add(road_b_id)
            adjacency[road_b_id].add(road_a_id)

            connection_count += 1


# -----------------------------------
# Find connected components
# -----------------------------------

visited = set()
components = []

for segment_id in roads:

    if segment_id in visited:
        continue

    component = []

    queue = deque([segment_id])

    visited.add(segment_id)

    while queue:

        current = queue.popleft()

        component.append(current)

        for neighbor in adjacency[current]:

            if neighbor not in visited:

                visited.add(neighbor)

                queue.append(neighbor)

    components.append(component)


# -----------------------------------
# Sort components
# -----------------------------------

components.sort(
    key=len,
    reverse=True
)


# -----------------------------------
# Results
# -----------------------------------

print("\n-----------------------------------")
print("RESULTS")
print("-----------------------------------")

print(
    "Number of road-to-road connections:",
    connection_count
)

print(
    "Number of connected components:",
    len(components)
)

print("\nComponent sizes:")

for i, component in enumerate(
    components,
    start=1
):

    print(
        f"Component {i}: {len(component)} roads"
    )


connected_count = sum(
    1
    for segment_id in roads
    if len(adjacency[segment_id]) > 0
)

isolated = [
    segment_id
    for segment_id in roads
    if len(adjacency[segment_id]) == 0
]


print("\nRoads with at least one connection:")
print(connected_count)

print("\nIsolated roads:")
print(len(isolated))

if isolated:

    print("\nIsolated roads:")

    for segment_id in isolated:

        print(
            segment_id,
            roads[segment_id]["street_name"]
        )


# -----------------------------------
# Largest component
# -----------------------------------

print("\nLargest component:")
print(len(components[0]), "roads")

print("\nFirst 10 roads in largest component:")

for segment_id in components[0][:10]:

    print(
        segment_id,
        "|",
        roads[segment_id]["street_name"]
    )
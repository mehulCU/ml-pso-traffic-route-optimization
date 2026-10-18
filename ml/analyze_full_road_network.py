import json
from collections import defaultdict, deque
import math


TOLERANCE_METERS = 5.0


def haversine(coord1, coord2):

    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# -----------------------------------
# Load full GeoJSON
# -----------------------------------

geojson_path = (
    "probe_counts/geojson/"
    "new_delhi__2024-08-11_to_2024-08-11_.geojson"
)

print("-----------------------------------")
print("FULL ROAD NETWORK ANALYSIS")
print("-----------------------------------")

print("Loading GeoJSON...")

with open(geojson_path, "r") as f:
    data = json.load(f)


# -----------------------------------
# Extract endpoints
# -----------------------------------

roads = {}

for feature in data["features"]:

    properties = feature.get("properties", {})
    geometry = feature.get("geometry")

    if not geometry:
        continue

    coordinates = geometry.get("coordinates")

    if not coordinates:
        continue

    segment_id = properties.get("segmentId")

    start = tuple(coordinates[0])
    end = tuple(coordinates[-1])

    roads[segment_id] = {
        "start": start,
        "end": end
    }


print("Road segments:", len(roads))


# -----------------------------------
# Exact endpoint mapping
# -----------------------------------

endpoint_to_roads = defaultdict(list)

for segment_id, road in roads.items():

    endpoint_to_roads[
        road["start"]
    ].append(segment_id)

    endpoint_to_roads[
        road["end"]
    ].append(segment_id)


# -----------------------------------
# Calculate connectivity
# -----------------------------------

adjacency = defaultdict(set)

connection_count = 0

for connected_roads in endpoint_to_roads.values():

    if len(connected_roads) < 2:
        continue

    for i in range(len(connected_roads)):

        for j in range(i + 1, len(connected_roads)):

            a = connected_roads[i]
            b = connected_roads[j]

            adjacency[a].add(b)
            adjacency[b].add(a)

            connection_count += 1


# -----------------------------------
# Connected components
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


components.sort(
    key=len,
    reverse=True
)


# -----------------------------------
# Degree statistics
# -----------------------------------

degrees = [
    len(adjacency[segment_id])
    for segment_id in roads
]


# -----------------------------------
# Results
# -----------------------------------

print("\n-----------------------------------")
print("RESULTS")
print("-----------------------------------")

print(
    "Road segments:",
    len(roads)
)

print(
    "Road-to-road connections:",
    connection_count
)

print(
    "Connected components:",
    len(components)
)

print(
    "Largest component:",
    len(components[0]),
    "roads"
)

print(
    "Smallest component:",
    len(components[-1]),
    "roads"
)

print("\nLargest 10 components:")

for i, component in enumerate(
    components[:10],
    start=1
):

    print(
        f"Component {i}: {len(component)} roads"
    )

print("\nDegree statistics:")

print(
    "Minimum:",
    min(degrees)
)

print(
    "Maximum:",
    max(degrees)
)

print(
    "Average:",
    sum(degrees) / len(degrees)
)

print(
    "Roads with at least one connection:",
    sum(d > 0 for d in degrees)
)

print(
    "Isolated roads:",
    sum(d == 0 for d in degrees)
)
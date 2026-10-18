import json
import pandas as pd
import math
from collections import defaultdict, deque


TOLERANCE_METERS = 1.0


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
# LOAD GEOJSON
# -----------------------------------

geojson_path = (
    "probe_counts/geojson/"
    "new_delhi__2024-08-11_to_2024-08-11_.geojson"
)

print("-----------------------------------")
print("BUILDING FULL ROAD GRAPH")
print("-----------------------------------")

print("Loading GeoJSON...")

with open(geojson_path, "r") as f:
    data = json.load(f)


# -----------------------------------
# EXTRACT ROAD SEGMENTS
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

    if segment_id is None:
        continue

    roads[segment_id] = {
        "coordinates": coordinates,
        "street_name": properties.get("streetName"),
        "distance": properties.get("distance"),
        "speed_limit": properties.get("speedLimit"),
        "frc": properties.get("frc")
    }


print("Road segments:", len(roads))


# -----------------------------------
# CREATE NODES
# -----------------------------------

print("Creating road nodes...")

nodes = []
endpoint_to_node = {}

for segment_id, road in roads.items():

    start = tuple(road["coordinates"][0])
    end = tuple(road["coordinates"][-1])

    for endpoint_type, coordinate in [
        ("start", start),
        ("end", end)
    ]:

        assigned_node = None

        for node in nodes:

            distance = haversine(
                coordinate,
                node["coordinate"]
            )

            if distance <= TOLERANCE_METERS:

                assigned_node = node["node_id"]
                break

        if assigned_node is None:

            assigned_node = len(nodes)

            nodes.append({
                "node_id": assigned_node,
                "coordinate": coordinate
            })

        endpoint_to_node[
            (segment_id, endpoint_type)
        ] = assigned_node


print("Graph nodes:", len(nodes))


# -----------------------------------
# CREATE EDGES
# -----------------------------------

edges = []

for segment_id, road in roads.items():

    start_node = endpoint_to_node[
        (segment_id, "start")
    ]

    end_node = endpoint_to_node[
        (segment_id, "end")
    ]

    edges.append({
        "segment_id": segment_id,
        "start_node": start_node,
        "end_node": end_node,
        "street_name": road["street_name"],
        "distance": road["distance"],
        "speed_limit": road["speed_limit"],
        "frc": road["frc"]
    })


print("Graph edges:", len(edges))


# -----------------------------------
# BUILD ADJACENCY
# -----------------------------------

adjacency = defaultdict(set)

for edge in edges:

    a = edge["start_node"]
    b = edge["end_node"]

    adjacency[a].add(b)
    adjacency[b].add(a)


# -----------------------------------
# FIND CONNECTED COMPONENTS
# -----------------------------------

print("Finding connected components...")

visited = set()
components = []

for node in nodes:

    node_id = node["node_id"]

    if node_id in visited:
        continue

    component = []

    queue = deque([node_id])
    visited.add(node_id)

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
# KEEP LARGEST COMPONENT
# -----------------------------------

largest_nodes = set(components[0])

largest_edges = [
    edge
    for edge in edges
    if (
        edge["start_node"] in largest_nodes
        and
        edge["end_node"] in largest_nodes
    )
]


# -----------------------------------
# NODE DEGREE
# -----------------------------------

degree_data = []

for node_id in largest_nodes:

    degree_data.append({
        "node_id": node_id,
        "degree": len(adjacency[node_id])
    })


# -----------------------------------
# SAVE NODES
# -----------------------------------

nodes_df = pd.DataFrame([
    {
        "node_id": node["node_id"],
        "longitude": node["coordinate"][0],
        "latitude": node["coordinate"][1]
    }
    for node in nodes
    if node["node_id"] in largest_nodes
])

nodes_df.to_csv(
    "ml/full_graph_nodes.csv",
    index=False
)


# -----------------------------------
# SAVE EDGES
# -----------------------------------

edges_df = pd.DataFrame(largest_edges)

edges_df.to_csv(
    "ml/full_graph_edges.csv",
    index=False
)


# -----------------------------------
# SAVE NODE DEGREES
# -----------------------------------

degree_df = pd.DataFrame(degree_data)

degree_df.to_csv(
    "ml/full_graph_node_degrees.csv",
    index=False
)


# -----------------------------------
# RESULTS
# -----------------------------------

print("\n-----------------------------------")
print("FULL GRAPH RESULTS")
print("-----------------------------------")

print(
    "Original road segments:",
    len(roads)
)

print(
    "Original graph nodes:",
    len(nodes)
)

print(
    "Connected components:",
    len(components)
)

print(
    "Largest component nodes:",
    len(largest_nodes)
)

print(
    "Largest component edges:",
    len(largest_edges)
)

print("\nComponent sizes:")

for i, component in enumerate(
    components,
    start=1
):

    print(
        f"Component {i}: {len(component)} nodes"
    )

print("\nDegree distribution:")

print(
    degree_df["degree"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSaved files:")

print("ml/full_graph_nodes.csv")
print("ml/full_graph_edges.csv")
print("ml/full_graph_node_degrees.csv")
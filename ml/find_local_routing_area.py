import pandas as pd
import math


def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0

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

    return R * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


nodes = pd.read_csv("ml/full_graph_nodes.csv")
junctions = pd.read_csv("ml/full_graph_junctions.csv")

# Highest-degree junction
source_node = 4479

source = nodes[
    nodes["node_id"] == source_node
].iloc[0]

print("-----------------------------------")
print("LOCAL ROUTING AREA")
print("-----------------------------------")

print(
    f"Reference node: {source_node}"
)

print(
    f"Coordinates: "
    f"{source['longitude']}, "
    f"{source['latitude']}"
)

# Calculate distance from reference junction
junctions["distance_km"] = junctions.apply(
    lambda row: haversine(
        source["longitude"],
        source["latitude"],
        row["longitude"],
        row["latitude"]
    ),
    axis=1
)

nearby = junctions[
    (junctions["node_id"] != source_node)
    & (junctions["distance_km"] <= 2.0)
].copy()

nearby = nearby.sort_values(
    ["distance_km", "degree"]
)

print("\nJunctions within 2 km:")
print(
    nearby[
        [
            "node_id",
            "degree",
            "longitude",
            "latitude",
            "distance_km"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

nearby.to_csv(
    "ml/local_junctions.csv",
    index=False
)

print("\nSaved:")
print("ml/local_junctions.csv")
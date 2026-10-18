import json
import pandas as pd

# -----------------------------------
# Load selected 100 road segments
# -----------------------------------

top_segments = pd.read_csv(
    "ml/top_100_segments_train_only.csv"
)

segment_ids = set(
    top_segments["segment_id"].astype(int)
)

print("-----------------------------------")
print("ROAD NETWORK ANALYSIS")
print("-----------------------------------")

print("Selected road segments:", len(segment_ids))


# -----------------------------------
# Load one traffic GeoJSON
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

selected = []

for feature in data["features"]:
    properties = feature.get("properties", {})

    segment_id = properties.get("segmentId")

    if segment_id in segment_ids:
        geometry = feature.get("geometry")

        selected.append({
            "segment_id": segment_id,
            "street_name": properties.get("streetName"),
            "distance": properties.get("distance"),
            "speed_limit": properties.get("speedLimit"),
            "frc": properties.get("frc"),
            "geometry_type": geometry.get("type") if geometry else None,
            "coordinates": geometry.get("coordinates") if geometry else None
        })


# -----------------------------------
# Basic checks
# -----------------------------------

print("Selected roads found:", len(selected))

missing_geometry = [
    x for x in selected
    if not x["coordinates"]
]

print("Missing geometry:", len(missing_geometry))

print("\nFirst 5 selected roads:")

for road in selected[:5]:

    coords = road["coordinates"]

    print("-----------------------------------")
    print("Segment:", road["segment_id"])
    print("Street:", road["street_name"])
    print("Distance:", road["distance"])
    print("FRC:", road["frc"])
    print("Geometry:", road["geometry_type"])
    print("Number of coordinates:", len(coords))

    print("Start:", coords[0])
    print("End:", coords[-1])
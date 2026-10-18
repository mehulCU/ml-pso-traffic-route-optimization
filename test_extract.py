import json

file = "probe_counts/geojson/new_delhi__2024-08-11_to_2024-08-11_.geojson"

with open(file, "r") as f:
    data = json.load(f)

features = data["features"]

# Take the first actual road segment
road = features[1]["properties"]

segment_id = road["segmentId"]
street_name = road.get("streetName")
speed_limit = road.get("speedLimit")
frc = road.get("frc")
distance = road.get("distance")

print("ROAD INFORMATION")
print("----------------")
print("Segment ID:", segment_id)
print("Street:", street_name)
print("Speed Limit:", speed_limit)
print("FRC:", frc)
print("Distance:", distance)

print("\nHOURLY TRAFFIC")
print("--------------")

for record in road["segmentProbeCounts"][:10]:
    time_set = record["timeSet"]
    probe_count = record["probeCount"]

    # timeSet 2 = 00:00, timeSet 3 = 01:00, etc.
    hour = time_set - 2

    print(
        f"Hour: {hour:02d}:00 | "
        f"Probe Count: {probe_count}"
    )

import json
import csv

input_file = (
    "probe_counts/geojson/"
    "new_delhi__2024-08-11_to_2024-08-11_.geojson"
)

output_file = "traffic_test.csv"

with open(input_file, "r") as f:
    data = json.load(f)

features = data["features"]

columns = [
    "date",
    "hour",
    "segment_id",
    "street_name",
    "speed_limit",
    "frc",
    "distance",
    "probe_count"
]

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)

    for feature in features[1:]:

        road = feature["properties"]

        for record in road.get("segmentProbeCounts", []):

            writer.writerow([
                "2024-08-11",
                record["timeSet"] - 2,
                road["segmentId"],
                road.get("streetName"),
                road.get("speedLimit"),
                road.get("frc"),
                road.get("distance"),
                record["probeCount"]
            ])

print("Extraction completed!")
print("File created:", output_file)

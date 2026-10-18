import csv
import glob
import os
import re
import ijson

INPUT_FOLDER = "probe_counts/geojson"
OUTPUT_FILE = "traffic_all.csv"


def extract_date(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_to_", filename)

    if match:
        return match.group(1)

    return None


files = sorted(
    glob.glob(
        os.path.join(
            INPUT_FOLDER,
            "new_delhi__*_to_*_.geojson"
        )
    )
)

print(f"Found {len(files)} files.")

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

total_rows = 0

with open(OUTPUT_FILE, "w", newline="") as output:

    writer = csv.writer(output)
    writer.writerow(columns)

    for file_number, file in enumerate(files, start=1):

        date = extract_date(os.path.basename(file))

        print(
            f"[{file_number}/{len(files)}] "
            f"Processing {date}..."
        )

        file_rows = 0

        with open(file, "rb") as f:

            # Read each GeoJSON feature one at a time
            features = ijson.items(f, "features.item")

            for feature in features:

                properties = feature.get("properties", {})

                # Skip the first metadata feature
                if "segmentProbeCounts" not in properties:
                    continue

                segment_id = properties.get("segmentId")
                street_name = properties.get("streetName")
                speed_limit = properties.get("speedLimit")
                frc = properties.get("frc")
                distance = properties.get("distance")

                for record in properties["segmentProbeCounts"]:

                    time_set = record["timeSet"]

                    # timeSet 2 = 00:00
                    # timeSet 3 = 01:00
                    # ...
                    # timeSet 25 = 23:00
                    hour = time_set - 2

                    probe_count = record["probeCount"]

                    writer.writerow([
                        date,
                        hour,
                        segment_id,
                        street_name,
                        speed_limit,
                        frc,
                        distance,
                        probe_count
                    ])

                    file_rows += 1

        total_rows += file_rows

        print(
            f"    Rows written: {file_rows:,}"
        )

print("\nExtraction completed!")
print(f"Total rows: {total_rows:,}")
print(f"Output file: {OUTPUT_FILE}")

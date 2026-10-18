import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "ml/regional_data/regional_traffic.csv"
BATCH_SIZE = 10_000

INSERT_SQL = """
    INSERT IGNORE INTO traffic_observations (
        segment_id, timestamp, date, hour, street_name,
        speed_limit, frc, distance, probe_count,
        hour_sin, hour_cos, day_of_week
    ) VALUES (
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s
    )
"""

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    required_columns = [
        "segment_id", "timestamp", "date", "hour", "street_name",
        "speed_limit", "frc", "distance", "probe_count",
        "hour_sin", "hour_cos", "day_of_week"
    ]

    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

    total_processed = 0

    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(CSV_PATH, chunksize=BATCH_SIZE), start=1
        ):
            missing_columns = set(required_columns) - set(chunk.columns)
            if missing_columns:
                raise ValueError(
                    f"CSV is missing required columns: {sorted(missing_columns)}"
                )

            chunk = chunk[required_columns].copy()

            # Convert dates and timestamps into MySQL-compatible values.
            chunk["timestamp"] = pd.to_datetime(
                chunk["timestamp"], errors="raise"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            chunk["date"] = pd.to_datetime(
                chunk["date"], errors="raise"
            ).dt.strftime("%Y-%m-%d")

            # MySQL expects None rather than pandas NaN for nullable text.
            chunk["street_name"] = chunk["street_name"].where(
                chunk["street_name"].notna(), None
            )

            # Convert NumPy/pandas values to ordinary Python values.
            rows = [
                tuple(None if pd.isna(value) else value.item()
                      if hasattr(value, "item") else value
                      for value in row)
                for row in chunk.itertuples(index=False, name=None)
            ]

            with connection.cursor() as cursor:
                cursor.executemany(INSERT_SQL, rows)
            connection.commit()

            total_processed += len(rows)
            print(
                f"Batch {chunk_number}: processed {len(rows):,} rows "
                f"(total processed: {total_processed:,})"
            )

        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM traffic_observations")
            final_count = cursor.fetchone()[0]

        print("\nImport finished.")
        print(f"Rows processed from CSV: {total_processed:,}")
        print(f"Rows currently in MySQL: {final_count:,}")

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

if __name__ == "__main__":
    main()

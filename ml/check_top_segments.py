import pandas as pd


# --------------------------------------------------
# 1. Load traffic data
# --------------------------------------------------

df = pd.read_csv("traffic_all.csv")

df["date"] = pd.to_datetime(df["date"])


# --------------------------------------------------
# 2. Load selected top 100 segments
# --------------------------------------------------

top_segments = pd.read_csv(
    "ml/top_100_segments.csv"
)

segment_ids = top_segments["segment_id"].tolist()


# --------------------------------------------------
# 3. Keep only selected segments
# --------------------------------------------------

df = df[df["segment_id"].isin(segment_ids)].copy()


# --------------------------------------------------
# 4. Count observations per segment
# --------------------------------------------------

counts = (
    df.groupby("segment_id")
    .size()
    .sort_values()
)


# --------------------------------------------------
# 5. Display summary
# --------------------------------------------------

print("-----------------------------------")
print("TOP 100 SEGMENT COMPLETENESS")
print("-----------------------------------")

print("\nExpected observations per segment: 480")

print("\nMinimum observations:")
print(counts.min())

print("\nMaximum observations:")
print(counts.max())

print("\nAverage observations:")
print(counts.mean())


# --------------------------------------------------
# 6. Check incomplete segments
# --------------------------------------------------

incomplete = counts[counts < 480]

print("\nNumber of incomplete segments:")
print(len(incomplete))

if len(incomplete) > 0:

    print("\nIncomplete segments:")
    print(incomplete)

else:

    print("\nAll 100 segments have complete data.")


# --------------------------------------------------
# 7. Check duplicate segment/date/hour combinations
# --------------------------------------------------

duplicates = df.duplicated(
    subset=["segment_id", "date", "hour"]
).sum()

print("\nDuplicate segment/date/hour records:")
print(duplicates)
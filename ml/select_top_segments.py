import pandas as pd


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("traffic_all.csv")


# --------------------------------------------------
# 2. Calculate average probe count for each segment
# --------------------------------------------------

segment_stats = (
    df.groupby("segment_id")["probe_count"]
    .mean()
    .sort_values(ascending=False)
)


# --------------------------------------------------
# 3. Select top 100 busiest segments
# --------------------------------------------------

top_100 = segment_stats.head(100)


# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("-----------------------------------")
print("TOP 100 BUSIEST ROAD SEGMENTS")
print("-----------------------------------")

print(top_100.head(20))


# --------------------------------------------------
# 5. Basic information
# --------------------------------------------------

print("\nNumber of selected segments:", len(top_100))

print(
    "\nHighest average probe count:",
    top_100.max()
)

print(
    "Lowest average probe count:",
    top_100.min()
)


# --------------------------------------------------
# 6. Save segment IDs
# --------------------------------------------------

top_100.reset_index().to_csv(
    "ml/top_100_segments.csv",
    index=False
)

print("\nSaved:")
print("ml/top_100_segments.csv")
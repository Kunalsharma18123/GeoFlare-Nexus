from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"

INPUT_FILE = DATA_DIR / "firms_historical_india.csv"
OUTPUT_FILE = DATA_DIR / "firms_historical_features.csv"


print("GeoFlare - Historical Feature Engineering")
print()

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Input records: {len(df):,}")
print(f"Input columns: {len(df.columns)}")
print()


print("Creating timestamp...")

df["acq_date"] = pd.to_datetime(df["acq_date"])

df["acq_time"] = (
    df["acq_time"]
    .astype(str)
    .str.zfill(4)
)

df["hour"] = df["acq_time"].str[:2].astype(int)
df["minute"] = df["acq_time"].str[2:].astype(int)

df["datetime"] = (
    df["acq_date"]
    + pd.to_timedelta(df["hour"], unit="h")
    + pd.to_timedelta(df["minute"], unit="m")
)


print("Creating temporal features...")

df["day_of_year"] = df["datetime"].dt.dayofyear
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["is_day"] = (
    df["daynight"].astype(str).str.upper() == "D"
).astype(int)


print("Creating confidence feature...")

confidence_map = {
    "l": 0.33,
    "n": 0.66,
    "h": 1.00,
}

df["confidence_score"] = (
    df["confidence"]
    .astype(str)
    .str.lower()
    .map(confidence_map)
    .fillna(0.0)
)


print("Creating FRP features...")

df["frp"] = pd.to_numeric(
    df["frp"],
    errors="coerce",
)

df["bright_ti4"] = pd.to_numeric(
    df["bright_ti4"],
    errors="coerce",
)

df["bright_ti5"] = pd.to_numeric(
    df["bright_ti5"],
    errors="coerce",
)

df["scan"] = pd.to_numeric(
    df["scan"],
    errors="coerce",
)

df["track"] = pd.to_numeric(
    df["track"],
    errors="coerce",
)


print("Creating spatial grid...")

GRID_SIZE = 0.01

df["grid_lat"] = (
    np.floor(df["latitude"] / GRID_SIZE)
    * GRID_SIZE
)

df["grid_lon"] = (
    np.floor(df["longitude"] / GRID_SIZE)
    * GRID_SIZE
)

df["location_id"] = (
    df["grid_lat"].round(2).astype(str)
    + "_"
    + df["grid_lon"].round(2).astype(str)
)


print("Calculating temporal activity features...")

df = df.sort_values(
    ["location_id", "datetime"]
).reset_index(drop=True)


group = df.groupby("location_id", sort=False)

df["previous_detection"] = group["datetime"].shift(1)

df["days_since_previous_detection"] = (
    (
        df["datetime"]
        - df["previous_detection"]
    ).dt.total_seconds()
    / 86400
)

df["days_since_previous_detection"] = (
    df["days_since_previous_detection"]
    .fillna(-1)
)


print("Calculating persistence features...")

df["detection_date"] = df["datetime"].dt.date

first_date = group["datetime"].transform("min")
last_date = group["datetime"].transform("max")

df["location_first_seen"] = first_date
df["location_last_seen"] = last_date

df["persistence_days"] = (
    (
        df["location_last_seen"]
        - df["location_first_seen"]
    ).dt.total_seconds()
    / 86400
    + 1
)

df["detections_at_location"] = (
    group["location_id"].transform("size")
)


print("Calculating FRP statistics...")

df["mean_frp_location"] = (
    group["frp"].transform("mean")
)

df["max_frp_location"] = (
    group["frp"].transform("max")
)

df["min_frp_location"] = (
    group["frp"].transform("min")
)

df["frp_std_location"] = (
    group["frp"].transform("std")
    .fillna(0)
)


print("Calculating persistence indicator...")

df["persistent_source_candidate"] = (
    (
        (df["persistence_days"] >= 3)
        & (df["detections_at_location"] >= 3)
    )
).astype(int)


print("Calculating intensity indicators...")

df["high_frp"] = (
    df["frp"] >= df["frp"].quantile(0.90)
).astype(int)

df["high_confidence"] = (
    df["confidence_score"] >= 0.66
).astype(int)


print("Cleaning feature columns...")

numeric_columns = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp",
    "hour",
    "minute",
    "day_of_year",
    "day_of_week",
    "month",
    "is_weekend",
    "is_day",
    "confidence_score",
    "grid_lat",
    "grid_lon",
    "days_since_previous_detection",
    "persistence_days",
    "detections_at_location",
    "mean_frp_location",
    "max_frp_location",
    "min_frp_location",
    "frp_std_location",
    "persistent_source_candidate",
    "high_frp",
    "high_confidence",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


print("Checking missing values...")

missing = df[numeric_columns].isna().sum()

missing = missing[missing > 0]

if len(missing) > 0:
    print(missing)
else:
    print("No missing values in feature columns.")


print()
print("Feature summary:")
print()

summary_columns = [
    "frp",
    "confidence_score",
    "persistence_days",
    "detections_at_location",
    "mean_frp_location",
    "max_frp_location",
    "days_since_previous_detection",
]

print(df[summary_columns].describe())


print()
print("Persistent source candidates:")

candidate_count = (
    df["persistent_source_candidate"]
    .sum()
)

print(f"{candidate_count:,}")


print()
print("Saving feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"Saved: {OUTPUT_FILE}")
print(f"Final records: {len(df):,}")
print(f"Final columns: {len(df.columns)}")
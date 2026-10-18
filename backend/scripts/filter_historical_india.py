from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"

INPUT_FILE = DATA_DIR / "firms_historical_raw.csv"
OUTPUT_FILE = DATA_DIR / "firms_historical_india.csv"

WORLD_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)


print("Loading historical FIRMS data...")

firms = pd.read_csv(INPUT_FILE)

print(f"Raw records: {len(firms):,}")

print("Loading country boundaries...")

world = gpd.read_file(WORLD_URL)

india = world[world["ADMIN"] == "India"].copy()

if india.empty:
    raise RuntimeError("India boundary was not found.")

print("Filtering detections to India...")

points = gpd.GeoDataFrame(
    firms,
    geometry=gpd.points_from_xy(
        firms["longitude"],
        firms["latitude"],
    ),
    crs="EPSG:4326",
)

india_points = gpd.sjoin(
    points,
    india[["geometry"]],
    how="inner",
    predicate="within",
)

india_points = india_points.drop(columns=["geometry", "index_right"])

india_points.to_csv(OUTPUT_FILE, index=False)

print()
print(f"India records: {len(india_points):,}")
print(f"Outside India: {len(firms) - len(india_points):,}")
print(f"Saved: {OUTPUT_FILE}")
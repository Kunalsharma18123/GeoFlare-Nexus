from pathlib import Path

import geopandas as gpd
import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"

RAW_FILE = DATA_DIR / "firms_raw.csv"
OUTPUT_FILE = DATA_DIR / "firms_india.csv"


# -------------------------------------------------------------------
# Load FIRMS data
# -------------------------------------------------------------------

print("🇮🇳 GeoFlare — India spatial filtering")
print()

if not RAW_FILE.exists():
    raise FileNotFoundError(f"FIRMS file not found: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

print(f"Raw detections: {len(df):,}")


# -------------------------------------------------------------------
# Convert FIRMS points into GeoDataFrame
# -------------------------------------------------------------------

fires = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"],
    ),
    crs="EPSG:4326",
)


# -------------------------------------------------------------------
# Load Natural Earth country boundaries
# -------------------------------------------------------------------

world_url = (
    "https://naturalearth.s3.amazonaws.com/"
    "110m_cultural/ne_110m_admin_0_countries.zip"
)

print("Downloading Natural Earth country boundaries...")

world = gpd.read_file(world_url)

india = world[world["ADMIN"] == "India"]

if india.empty:
    raise RuntimeError("India boundary was not found.")


# -------------------------------------------------------------------
# Spatial filter
# -------------------------------------------------------------------

india = india.to_crs(fires.crs)

india_fires = gpd.sjoin(
    fires,
    india[["geometry"]],
    predicate="within",
    how="inner",
)

# Remove spatial-join helper columns
india_fires = india_fires.drop(
    columns=["index_right"],
    errors="ignore",
)

# Remove geometry before saving CSV
india_fires = pd.DataFrame(india_fires.drop(columns="geometry"))


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

india_fires.to_csv(OUTPUT_FILE, index=False)

print()
print(f" India detections: {len(india_fires):,}")
print(f"❌ Removed outside India: {len(df) - len(india_fires):,}")
print()
print(f" Saved: {OUTPUT_FILE}")
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise RuntimeError("FIRMS_MAP_KEY is not set in .env")


# GeoFlare data directory
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# NASA FIRMS configuration
SOURCE = "VIIRS_NOAA20_NRT"
AREA = "68.1,8.0,97.4,37.1"  # India bounding box
DAYS = 1

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{SOURCE}/{AREA}/{DAYS}"
)

print("GeoFlare — FIRMS ingestion")
print(f"Source : {SOURCE}")
print(f"Area   : India")
print(f"Days   : {DAYS}")
print()
print("Fetching FIRMS data...")

response = requests.get(url, timeout=60)

if not response.ok:
    print("\n❌ FIRMS API request failed")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    raise SystemExit(1)

output_file = DATA_DIR / "firms_raw.csv"
output_file.write_bytes(response.content)

df = pd.read_csv(output_file)

print()
print(f"Downloaded {len(df):,} detections")
print(f"Saved to: {output_file}")

print()
print("Columns:")
for column in df.columns:
    print(f"  • {column}")

print()
print("First 5 detections:")
print(df.head().to_string(index=False))

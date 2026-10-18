from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"
OSM_DIR = DATA_DIR / "osm"

OSM_DIR.mkdir(parents=True, exist_ok=True)


INPUT_FILE = DATA_DIR / "firms_historical_features.csv"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# Test region around northern India.
# We will expand this to tiled India-wide collection after
# confirming the Overpass request works.
SOUTH = 28.0
WEST = 76.0
NORTH = 29.0
EAST = 78.0


QUERY = f"""
[out:json][timeout:120];

(
  nwr["landuse"="industrial"]({SOUTH},{WEST},{NORTH},{EAST});

  nwr["power"="plant"]({SOUTH},{WEST},{NORTH},{EAST});
  nwr["power"="generator"]({SOUTH},{WEST},{NORTH},{EAST});

  nwr["man_made"="works"]({SOUTH},{WEST},{NORTH},{EAST});

  nwr["industrial"="oil"]({SOUTH},{WEST},{NORTH},{EAST});
  nwr["industrial"="gas"]({SOUTH},{WEST},{NORTH},{EAST});

  nwr["man_made"="petroleum_well"]({SOUTH},{WEST},{NORTH},{EAST});

  nwr["highway"~"motorway|trunk|primary|secondary"]({SOUTH},{WEST},{NORTH},{EAST});

  nwr["place"~"city|town|village"]({SOUTH},{WEST},{NORTH},{EAST});
);

out center tags;
"""


print("GeoFlare - OpenStreetMap downloader")
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

print(f"Test bounding box:")
print(f"South: {SOUTH}")
print(f"West:  {WEST}")
print(f"North: {NORTH}")
print(f"East: {EAST}")
print()

print("Sending query to Overpass API...")


response = requests.post(
    OVERPASS_URL,
    data={"data": QUERY},
    headers={
        "User-Agent": "GeoFlare/0.1",
        "Accept": "application/json",
    },
    timeout=180,
)


if not response.ok:
    print(f"Request failed: HTTP {response.status_code}")
    print(response.text[:2000])
    raise SystemExit(1)


data = response.json()

elements = data.get("elements", [])

print()
print(f"OSM elements received: {len(elements):,}")


if not elements:
    print("No OSM elements were returned for this test region.")
    raise SystemExit(0)


import json

raw_file = OSM_DIR / "osm_test.json"

with open(raw_file, "w", encoding="utf-8") as file:
    json.dump(data, file)


print(f"Raw OSM data saved: {raw_file}")
print()
print("OSM test query completed successfully.")
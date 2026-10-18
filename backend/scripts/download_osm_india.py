from pathlib import Path
import json
import time

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"
OSM_DIR = DATA_DIR / "osm" / "india_tiles"

OSM_DIR.mkdir(parents=True, exist_ok=True)


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# India coverage
SOUTH = 8.0
WEST = 68.0
NORTH = 37.5
EAST = 97.5

# Tile size in degrees
TILE_HEIGHT = 2.0
TILE_WIDTH = 2.0


QUERY_TEMPLATE = """
[out:json][timeout:180];

(
  nwr["landuse"="industrial"]({south},{west},{north},{east});

  nwr["power"="plant"]({south},{west},{north},{east});
  nwr["power"="generator"]({south},{west},{north},{east});

  nwr["man_made"="works"]({south},{west},{north},{east});

  nwr["industrial"="oil"]({south},{west},{north},{east});
  nwr["industrial"="gas"]({south},{west},{north},{east});

  nwr["man_made"="petroleum_well"]({south},{west},{north},{east});

  nwr["highway"~"motorway|trunk|primary|secondary"]({south},{west},{north},{east});

  nwr["place"~"city|town|village"]({south},{west},{north},{east});
);

out center tags;
"""


session = requests.Session()

session.headers.update(
    {
        "User-Agent": "GeoFlare/0.1 (geospatial fire monitoring prototype)",
        "Accept": "application/json",
    }
)


def fetch_tile(south, west, north, east, output_file):
    query = QUERY_TEMPLATE.format(
        south=south,
        west=west,
        north=north,
        east=east,
    )

    for attempt in range(3):
        try:
            response = session.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=240,
            )

            if response.ok:
                data = response.json()

                with open(output_file, "w", encoding="utf-8") as file:
                    json.dump(data, file)

                return data

            print(
                f"HTTP {response.status_code} "
                f"on attempt {attempt + 1}"
            )

            print(response.text[:500])

        except requests.RequestException as exc:
            print(
                f"Request error on attempt {attempt + 1}: {exc}"
            )

        if attempt < 2:
            print("Waiting before retry...")
            time.sleep(10)

    return None


print("GeoFlare - India OSM tiled downloader")
print()

print(
    f"Coverage: "
    f"{SOUTH},{WEST} to {NORTH},{EAST}"
)
print(
    f"Tile size: "
    f"{TILE_HEIGHT} x {TILE_WIDTH} degrees"
)
print()


tile_number = 0
successful_tiles = 0
skipped_tiles = 0
failed_tiles = 0


south = SOUTH

while south < NORTH:
    north = min(
        south + TILE_HEIGHT,
        NORTH,
    )

    west = WEST

    while west < EAST:
        east = min(
            west + TILE_WIDTH,
            EAST,
        )

        tile_number += 1

        tile_name = (
            f"tile_{south:.1f}_{west:.1f}_"
            f"{north:.1f}_{east:.1f}.json"
        )

        output_file = OSM_DIR / tile_name

        print(
            f"[{tile_number}] "
            f"{south:.1f},{west:.1f} -> "
            f"{north:.1f},{east:.1f}"
        )

        if output_file.exists():
            print("  Already downloaded. Skipping.")
            skipped_tiles += 1

            west += TILE_WIDTH
            continue

        print("  Downloading...")

        data = fetch_tile(
            south,
            west,
            north,
            east,
            output_file,
        )

        if data is None:
            print("  Failed.")
            failed_tiles += 1
        else:
            count = len(data.get("elements", []))

            print(
                f"  Received {count:,} OSM elements."
            )

            successful_tiles += 1

        print()

        time.sleep(3)

        west += TILE_WIDTH

    south += TILE_HEIGHT


print("Download completed.")
print()
print(f"Total tiles: {tile_number}")
print(f"Downloaded: {successful_tiles}")
print(f"Skipped: {skipped_tiles}")
print(f"Failed: {failed_tiles}")
print(f"Tile directory: {OSM_DIR}")
from pathlib import Path
import math

import pandas as pd
import requests
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIRMS_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "firms_historical_osm_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "firms_historical_osm_landcover_features.csv"
)

RASTER_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "landcover"
)

BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map"
)

LANDCOVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare or sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def tile_coordinate(value, is_latitude):
    if is_latitude:
        base = math.floor(value / 3.0) * 3
        return base
    else:
        base = math.floor(value / 3.0) * 3
        return base


def format_tile(lat, lon):
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"

    return (
        f"{lat_prefix}{abs(lat):02d}"
        f"{lon_prefix}{abs(lon):03d}"
    )


def get_tile_name(latitude, longitude):
    tile_lat = tile_coordinate(latitude, True)
    tile_lon = tile_coordinate(longitude, False)

    return format_tile(tile_lat, tile_lon)


def download_tile(tile_name):
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"ESA_WorldCover_10m_2021_v200_"
        f"{tile_name}_Map.tif"
    )

    output_path = RASTER_DIR / filename

    if output_path.exists():
        print(f"  Already exists: {filename}")
        return output_path

    url = f"{BASE_URL}/{filename}"

    print(f"  Downloading: {tile_name}")

    response = requests.get(
        url,
        stream=True,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {tile_name}: "
            f"HTTP {response.status_code}"
        )

    total_size = int(
        response.headers.get("content-length", 0)
    )

    downloaded = 0

    with output_path.open("wb") as file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percent = (
                        downloaded / total_size
                    ) * 100

                    print(
                        f"\r    {percent:6.1f}%",
                        end="",
                    )

    print()

    return output_path


def sample_tile(dataset, points):
    coordinates = list(
        zip(
            points["longitude"],
            points["latitude"],
        )
    )

    values = []

    for value in dataset.sample(coordinates):
        values.append(int(value[0]))

    return values


def main():
    print("Loading FIRMS + OSM dataset...")

    firms = pd.read_csv(FIRMS_FILE)

    print(f"Records: {len(firms):,}")

    required_columns = {
        "latitude",
        "longitude",
    }

    missing = required_columns - set(firms.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    firms["worldcover_tile"] = [
        get_tile_name(lat, lon)
        for lat, lon in zip(
            firms["latitude"],
            firms["longitude"],
        )
    ]

    tiles = sorted(
        firms["worldcover_tile"].unique()
    )

    print(f"Unique WorldCover tiles needed: {len(tiles)}")
    print()

    tile_paths = {}

    for index, tile in enumerate(tiles, start=1):
        print(
            f"[{index}/{len(tiles)}] {tile}"
        )

        tile_paths[tile] = download_tile(tile)

    print()
    print("Sampling land-cover classes...")

    firms["landcover_class"] = 0

    for index, tile in enumerate(tiles, start=1):
        mask = firms["worldcover_tile"] == tile

        points = firms.loc[
            mask,
            ["latitude", "longitude"],
        ]

        print(
            f"[{index}/{len(tiles)}] "
            f"{tile}: {len(points):,} points"
        )

        with rasterio.open(tile_paths[tile]) as dataset:
            values = sample_tile(
                dataset,
                points,
            )

        firms.loc[
            mask,
            "landcover_class",
        ] = values

    firms["landcover_name"] = (
        firms["landcover_class"]
        .map(LANDCOVER_NAMES)
        .fillna("Unknown")
    )

    firms.drop(
        columns=["worldcover_tile"],
        inplace=True,
    )

    firms["vegetation_context"] = (
        firms["landcover_class"].isin(
            [10, 20, 30]
        )
    ).astype("int8")

    firms["agriculture_context"] = (
        firms["landcover_class"] == 40
    ).astype("int8")

    firms["builtup_context"] = (
        firms["landcover_class"] == 50
    ).astype("int8")

    firms["water_context"] = (
        firms["landcover_class"] == 80
    ).astype("int8")

    firms["wetland_context"] = (
        firms["landcover_class"] == 90
    ).astype("int8")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    firms.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Land-cover enrichment completed.")
    print(f"Records: {len(firms):,}")
    print(f"Columns: {len(firms.columns)}")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("Land-cover distribution:")

    distribution = (
        firms["landcover_name"]
        .value_counts()
    )

    for name, count in distribution.items():
        percentage = (
            count / len(firms)
        ) * 100

        print(
            f"  {name}: "
            f"{count:,} ({percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()
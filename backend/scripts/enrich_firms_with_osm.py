from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIRMS_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "firms_historical_features.csv"
)

OSM_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "osm"
    / "osm_context_points.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "firms_historical_osm_features.csv"
)

EARTH_RADIUS_KM = 6371.0088


def latlon_to_unit_sphere(latitudes, longitudes):
    lat = np.radians(latitudes)
    lon = np.radians(longitudes)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.column_stack((x, y, z))


def chord_to_km(chord_distance):
    chord_distance = np.clip(chord_distance, 0, 2)

    angle = 2 * np.arcsin(chord_distance / 2)

    return EARTH_RADIUS_KM * angle


def calculate_nearest_distance(firms_coords, osm_coords):
    if len(osm_coords) == 0:
        return np.full(len(firms_coords), np.nan)

    tree = cKDTree(osm_coords)

    distances, _ = tree.query(firms_coords, k=1)

    return chord_to_km(distances)


def calculate_nearby_count(
    firms_coords,
    osm_coords,
    radius_km,
):
    if len(osm_coords) == 0:
        return np.zeros(len(firms_coords), dtype=np.int32)

    tree = cKDTree(osm_coords)

    radius_chord = 2 * np.sin(
        radius_km / (2 * EARTH_RADIUS_KM)
    )

    counts = tree.query_ball_point(
        firms_coords,
        r=radius_chord,
        return_length=True,
    )

    return counts.astype(np.int32)


def main():
    print("Loading FIRMS dataset...")
    firms = pd.read_csv(FIRMS_FILE)

    print(f"FIRMS records: {len(firms):,}")

    print("Loading OSM contextual features...")
    osm = pd.read_csv(OSM_FILE)

    print(f"OSM features: {len(osm):,}")

    required_firms = {"latitude", "longitude"}

    if not required_firms.issubset(firms.columns):
        missing = required_firms - set(firms.columns)
        raise ValueError(
            f"Missing FIRMS columns: {sorted(missing)}"
        )

    required_osm = {
        "category",
        "latitude",
        "longitude",
    }

    if not required_osm.issubset(osm.columns):
        missing = required_osm - set(osm.columns)
        raise ValueError(
            f"Missing OSM columns: {sorted(missing)}"
        )

    firms_coords = latlon_to_unit_sphere(
        firms["latitude"].to_numpy(),
        firms["longitude"].to_numpy(),
    )

    categories = {
        "industrial": {
            "distance_column": "distance_to_industry_km",
            "near_column": "near_industry_5km",
            "count_column": "industrial_features_5km",
            "radius_km": 5,
        },
        "power": {
            "distance_column": "distance_to_power_km",
            "near_column": "near_power_10km",
            "count_column": "power_features_10km",
            "radius_km": 10,
        },
        "oil_gas": {
            "distance_column": "distance_to_oil_gas_km",
            "near_column": "near_oil_gas_10km",
            "count_column": "oil_gas_features_10km",
            "radius_km": 10,
        },
        "road": {
            "distance_column": "distance_to_road_km",
            "near_column": "near_road_1km",
            "count_column": "road_features_1km",
            "radius_km": 1,
        },
        "settlement": {
            "distance_column": "distance_to_settlement_km",
            "near_column": "near_settlement_5km",
            "count_column": "settlement_features_5km",
            "radius_km": 5,
        },
    }

    for category, config in categories.items():
        print()
        print(f"Processing OSM category: {category}")

        category_osm = osm[
            osm["category"] == category
        ].copy()

        print(
            f"  OSM features: {len(category_osm):,}"
        )

        if category_osm.empty:
            firms[config["distance_column"]] = np.nan
            firms[config["near_column"]] = 0
            firms[config["count_column"]] = 0
            continue

        osm_coords = latlon_to_unit_sphere(
            category_osm["latitude"].to_numpy(),
            category_osm["longitude"].to_numpy(),
        )

        print("  Calculating nearest distance...")

        distances = calculate_nearest_distance(
            firms_coords,
            osm_coords,
        )

        firms[config["distance_column"]] = distances

        print("  Calculating nearby feature counts...")

        counts = calculate_nearby_count(
            firms_coords,
            osm_coords,
            config["radius_km"],
        )

        firms[config["count_column"]] = counts

        firms[config["near_column"]] = (
            counts > 0
        ).astype(np.int8)

        print(
            f"  Median distance: "
            f"{np.nanmedian(distances):.3f} km"
        )

        print(
            f"  Within radius: "
            f"{(counts > 0).sum():,}"
        )

    print()
    print("Creating combined geospatial indicators...")

    firms["industrial_context"] = (
        (firms["near_industry_5km"] == 1)
        | (firms["near_power_10km"] == 1)
        | (firms["near_oil_gas_10km"] == 1)
    ).astype(np.int8)

    firms["settlement_context"] = (
        firms["near_settlement_5km"] == 1
    ).astype(np.int8)

    firms["road_context"] = (
        firms["near_road_1km"] == 1
    ).astype(np.int8)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    firms.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("OSM enrichment completed.")
    print(f"Input FIRMS records: {len(firms):,}")
    print(f"Output columns: {len(firms.columns)}")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("New OSM features:")

    osm_columns = [
        "distance_to_industry_km",
        "distance_to_power_km",
        "distance_to_oil_gas_km",
        "distance_to_road_km",
        "distance_to_settlement_km",
        "near_industry_5km",
        "near_power_10km",
        "near_oil_gas_10km",
        "near_road_1km",
        "near_settlement_5km",
        "industrial_context",
        "settlement_context",
        "road_context",
    ]

    for column in osm_columns:
        print(f"  {column}")


if __name__ == "__main__":
    main()
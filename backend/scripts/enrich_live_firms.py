from pathlib import Path
import argparse
import math
import requests

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]

FIRMS_PATH = ROOT / "backend" / "data" / "firms_india.csv"

HISTORICAL_PATH = (
    ROOT
    / "backend"
    / "data"
    / "firms_historical_osm_landcover_features.csv"
)

OSM_PATH = (
    ROOT
    / "backend"
    / "data"
    / "osm"
    / "osm_context_points.csv"
)

WORLDCOVER_DIR = (
    ROOT
    / "backend"
    / "data"
    / "landcover"
)

DEFAULT_OUTPUT = (
    ROOT
    / "backend"
    / "data"
    / "firms_live_features.csv"
)

EARTH_RADIUS_KM = 6371.0088

HISTORY_WINDOW_DAYS = 30

WORLDCOVER_BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map"
)

WORLDCOVER_CLASSES = {
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


def latlon_to_xyz(lat, lon):
    """Convert latitude/longitude to 3D unit-sphere coordinates."""

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack((x, y, z))


def angular_distance_to_km(distance):
    """Convert unit-sphere chord distance to approximate kilometers."""

    return distance * EARTH_RADIUS_KM


def build_datetime(df):
    """Build acquisition datetime from FIRMS date and HHMM time."""

    date = (
        df["acq_date"]
        .astype(str)
        .str.strip()
    )

    time = (
        df["acq_time"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(4)
    )

    return pd.to_datetime(
        date + " " + time,
        format="%Y-%m-%d %H%M",
        errors="coerce",
    )


def normalize_confidence(live):
    """Convert FIRMS confidence into GeoFlare confidence_score."""

    if "confidence_score" in live.columns:
        live["confidence_score"] = pd.to_numeric(
            live["confidence_score"],
            errors="coerce",
        )
        return live

    if "confidence" not in live.columns:
        raise ValueError(
            "FIRMS input does not contain either "
            "'confidence' or 'confidence_score'."
        )

    confidence_numeric = pd.to_numeric(
        live["confidence"],
        errors="coerce",
    )

    confidence_map = {
        "l": 0.33,
        "n": 0.66,
        "h": 1.00,
    }

    confidence_categorical = (
        live["confidence"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(confidence_map)
    )

    live["confidence_score"] = (
        confidence_numeric
        .fillna(confidence_categorical)
    )

    return live


def add_basic_features(live):
    """Create basic FIRMS and temporal features."""

    print("Building basic FIRMS features...")

    live = normalize_confidence(live)

    live["acq_datetime"] = build_datetime(live)

    live["hour"] = (
        live["acq_datetime"].dt.hour
    )

    live["minute"] = (
        live["acq_datetime"].dt.minute
    )

    live["day_of_year"] = (
        live["acq_datetime"].dt.dayofyear
    )

    live["day_of_week"] = (
        live["acq_datetime"].dt.dayofweek
    )

    live["month"] = (
        live["acq_datetime"].dt.month
    )

    live["is_weekend"] = (
        live["day_of_week"] >= 5
    ).astype(int)

    live["is_day"] = (
        live["daynight"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "D"
    ).astype(int)

    return live


def add_location_features(live):
    """Create the same approximate 0.01-degree location key."""

    print("Building spatial location features...")

    live["grid_lat"] = (
        np.floor(live["latitude"] / 0.01)
        * 0.01
    )

    live["grid_lon"] = (
        np.floor(live["longitude"] / 0.01)
        * 0.01
    )

    live["location_id"] = (
        live["grid_lat"]
        .round(2)
        .map(lambda value: f"{value:.2f}")
        + "_"
        + live["grid_lon"]
        .round(2)
        .map(lambda value: f"{value:.2f}")
    )

    return live


def add_temporal_features(live, historical):
    """
    Build historical context using only observations within the
    previous HISTORY_WINDOW_DAYS before each live event.
    """

    print(
        f"Building temporal features "
        f"(previous {HISTORY_WINDOW_DAYS} days only)..."
    )

    historical = historical.copy()

    if "acq_datetime" not in historical.columns:
        historical["acq_datetime"] = build_datetime(
            historical
        )

    if "location_id" not in historical.columns:

        historical["grid_lat"] = (
            np.floor(
                historical["latitude"] / 0.01
            )
            * 0.01
        )

        historical["grid_lon"] = (
            np.floor(
                historical["longitude"] / 0.01
            )
            * 0.01
        )

        historical["location_id"] = (
            historical["grid_lat"]
            .round(2)
            .map(lambda value: f"{value:.2f}")
            + "_"
            + historical["grid_lon"]
            .round(2)
            .map(lambda value: f"{value:.2f}")
        )

    history = historical[
        [
            "location_id",
            "acq_datetime",
            "frp",
        ]
    ].dropna(
        subset=[
            "location_id",
            "acq_datetime",
            "frp",
        ]
    ).copy()

    history = history.sort_values(
        [
            "location_id",
            "acq_datetime",
        ]
    )

    history_groups = {
        location_id: group
        for location_id, group
        in history.groupby("location_id")
    }

    persistence_days = []
    detections = []
    mean_frp = []
    max_frp = []
    min_frp = []
    std_frp = []
    previous_detection = []
    days_since_previous = []

    for _, row in live.iterrows():

        location_id = row["location_id"]
        event_time = row["acq_datetime"]
        current_frp = float(row["frp"])

        group = history_groups.get(
            location_id
        )

        if (
            group is None
            or pd.isna(event_time)
        ):
            persistence_days.append(1.0)
            detections.append(1)
            mean_frp.append(current_frp)
            max_frp.append(current_frp)
            min_frp.append(current_frp)
            std_frp.append(0.0)
            previous_detection.append(pd.NaT)
            days_since_previous.append(-1.0)
            continue

        window_start = (
            event_time
            - pd.Timedelta(
                days=HISTORY_WINDOW_DAYS
            )
        )

        previous = group[
            (
                group["acq_datetime"] >= window_start
            )
            & (
                group["acq_datetime"] < event_time
            )
        ]

        if previous.empty:
            persistence_days.append(1.0)
            detections.append(1)
            mean_frp.append(current_frp)
            max_frp.append(current_frp)
            min_frp.append(current_frp)
            std_frp.append(0.0)
            previous_detection.append(pd.NaT)
            days_since_previous.append(-1.0)
            continue

        previous_times = (
            previous["acq_datetime"]
        )

        first_seen = previous_times.min()
        last_seen = previous_times.max()

        persistence = (
            event_time - first_seen
        ).total_seconds() / 86400.0

        persistence_days.append(
            max(1.0, persistence)
        )

        previous_frp = (
            previous["frp"]
            .astype(float)
        )

        combined_frp = np.append(
            previous_frp.to_numpy(),
            current_frp,
        )

        detections.append(
            len(combined_frp)
        )

        mean_frp.append(
            float(np.mean(combined_frp))
        )

        max_frp.append(
            float(np.max(combined_frp))
        )

        min_frp.append(
            float(np.min(combined_frp))
        )

        std_frp.append(
            float(np.std(combined_frp))
        )

        previous_detection.append(
            last_seen
        )

        delta = (
            event_time - last_seen
        ).total_seconds() / 86400.0

        days_since_previous.append(
            max(0.0, delta)
        )

    live["previous_detection"] = (
        previous_detection
    )

    live["days_since_previous_detection"] = (
        days_since_previous
    )

    live["persistence_days"] = (
        persistence_days
    )

    live["detections_at_location"] = (
        detections
    )

    live["mean_frp_location"] = (
        mean_frp
    )

    live["max_frp_location"] = (
        max_frp
    )

    live["min_frp_location"] = (
        min_frp
    )

    live["frp_std_location"] = (
        std_frp
    )

    live["persistent_source_candidate"] = (
        (
            live["persistence_days"] >= 7
        )
        & (
            live["detections_at_location"] >= 7
        )
    ).astype(int)

    historical_frp_90 = (
        historical["frp"]
        .astype(float)
        .quantile(0.90)
    )

    live["high_frp"] = (
        live["frp"] >= historical_frp_90
    ).astype(int)

    live["high_confidence"] = (
        live["confidence_score"] >= 0.66
    ).astype(int)

    return live


def build_osm_trees(osm):
    """Build KD-trees for OSM categories."""

    trees = {}

    for category in (
        osm["category"]
        .dropna()
        .unique()
    ):

        subset = osm[
            osm["category"] == category
        ].copy()

        if subset.empty:
            continue

        coordinates = latlon_to_xyz(
            subset["latitude"].to_numpy(),
            subset["longitude"].to_numpy(),
        )

        trees[category] = {
            "tree": cKDTree(coordinates),
        }

    return trees


def add_osm_features(live, osm):
    """Add nearest OSM infrastructure distances and context."""

    print("Building OSM spatial indexes...")

    trees = build_osm_trees(osm)

    mappings = {
        "industrial": "distance_to_industry_km",
        "power": "distance_to_power_km",
        "oil_gas": "distance_to_oil_gas_km",
        "road": "distance_to_road_km",
        "settlement": "distance_to_settlement_km",
    }

    points = latlon_to_xyz(
        live["latitude"].to_numpy(),
        live["longitude"].to_numpy(),
    )

    for category, output_column in mappings.items():

        print(
            f"Calculating nearest "
            f"{category} distance..."
        )

        if category not in trees:
            live[output_column] = np.nan
            continue

        distances, _ = (
            trees[category]["tree"]
            .query(points)
        )

        live[output_column] = (
            angular_distance_to_km(
                distances
            )
        )

    live["near_industry_5km"] = (
        live["distance_to_industry_km"] <= 5
    ).astype(int)

    live["near_power_10km"] = (
        live["distance_to_power_km"] <= 10
    ).astype(int)

    live["near_oil_gas_10km"] = (
        live["distance_to_oil_gas_km"] <= 10
    ).astype(int)

    live["near_road_1km"] = (
        live["distance_to_road_km"] <= 1
    ).astype(int)

    live["near_settlement_5km"] = (
        live["distance_to_settlement_km"] <= 5
    ).astype(int)

    live["industrial_context"] = (
        (
            live["near_industry_5km"] == 1
        )
        | (
            live["near_power_10km"] == 1
        )
        | (
            live["near_oil_gas_10km"] == 1
        )
    ).astype(int)

    live["settlement_context"] = (
        live["near_settlement_5km"]
    ).astype(int)

    live["road_context"] = (
        live["near_road_1km"]
    ).astype(int)

    return live


def worldcover_tile_name(latitude, longitude):
    """
    Return ESA WorldCover 3x3-degree tile name.

    Tiles use the southwest corner:
        N30E075
        N00E078
        S03E...
    """

    tile_lat = math.floor(
        latitude / 3
    ) * 3

    tile_lon = math.floor(
        longitude / 3
    ) * 3

    if tile_lat >= 0:
        lat_prefix = "N"
        lat_value = tile_lat
    else:
        lat_prefix = "S"
        lat_value = abs(tile_lat)

    if tile_lon >= 0:
        lon_prefix = "E"
        lon_value = tile_lon
    else:
        lon_prefix = "W"
        lon_value = abs(tile_lon)

    return (
        f"{lat_prefix}{lat_value:02d}"
        f"{lon_prefix}{lon_value:03d}"
    )


def worldcover_path(tile_name):
    """Return local WorldCover tile path."""

    return (
        WORLDCOVER_DIR
        / (
            "ESA_WorldCover_10m_2021_v200_"
            f"{tile_name}_Map.tif"
        )
    )


def download_worldcover_tile(tile_name):
    """Download a WorldCover tile if it is not already cached."""

    WORLDCOVER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = worldcover_path(
        tile_name
    )

    if output_path.exists():
        return output_path

    url = (
        f"{WORLDCOVER_BASE_URL}/"
        f"{output_path.name}"
    )

    print(
        f"Downloading WorldCover tile "
        f"{tile_name}..."
    )

    response = requests.get(
        url,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    temporary_path = (
        output_path.with_suffix(".part")
    )

    with open(
        temporary_path,
        "wb",
    ) as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)

    temporary_path.replace(
        output_path
    )

    return output_path


def sample_worldcover(latitude, longitude):
    """Sample one WorldCover class at a coordinate."""

    tile_name = worldcover_tile_name(
        latitude,
        longitude,
    )

    path = download_worldcover_tile(
        tile_name
    )

    with rasterio.open(path) as src:

        try:
            values = list(
                src.sample(
                    [
                        (
                            longitude,
                            latitude,
                        )
                    ]
                )
            )

            if not values:
                return None

            value = values[0][0]

            if value == src.nodata:
                return None

            return int(value)

        except Exception:
            return None


def add_landcover_features(live):
    """
    Sample ESA WorldCover directly at every live FIRMS coordinate.
    """

    print(
        "Sampling WorldCover directly "
        "at live coordinates..."
    )

    classes = []

    unique_tiles = set()

    for _, row in live.iterrows():

        unique_tiles.add(
            worldcover_tile_name(
                row["latitude"],
                row["longitude"],
            )
        )

    print(
        f"WorldCover tiles required: "
        f"{len(unique_tiles)}"
    )

    for _, row in live.iterrows():

        value = sample_worldcover(
            row["latitude"],
            row["longitude"],
        )

        classes.append(value)

    live["landcover_class"] = classes

    live["landcover_name"] = (
        live["landcover_class"]
        .map(WORLDCOVER_CLASSES)
    )

    live["vegetation_context"] = (
        live["landcover_class"]
        .isin(
            [
                10,
                20,
                30,
            ]
        )
    ).astype(int)

    live["agriculture_context"] = (
        live["landcover_class"] == 40
    ).astype(int)

    live["builtup_context"] = (
        live["landcover_class"] == 50
    ).astype(int)

    live["water_context"] = (
        live["landcover_class"] == 80
    ).astype(int)

    live["wetland_context"] = (
        live["landcover_class"]
        .isin(
            [
                90,
                95,
            ]
        )
    ).astype(int)

    return live


def validate_live_features(live):
    """Validate the model feature schema."""

    required_features = [
        "frp",
        "confidence_score",
        "latitude",
        "longitude",
        "hour",
        "minute",
        "day_of_year",
        "day_of_week",
        "month",
        "is_weekend",
        "is_day",
        "persistence_days",
        "detections_at_location",
        "mean_frp_location",
        "max_frp_location",
        "min_frp_location",
        "frp_std_location",
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
        "vegetation_context",
        "agriculture_context",
        "builtup_context",
        "water_context",
        "wetland_context",
    ]

    missing = [
        column
        for column in required_features
        if column not in live.columns
    ]

    if missing:
        raise ValueError(
            "Live feature generation is missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    for column in required_features:

        live[column] = pd.to_numeric(
            live[column],
            errors="coerce",
        )

    missing_values = (
        live[required_features]
        .isna()
        .sum()
    )

    problematic = missing_values[
        missing_values > 0
    ]

    if not problematic.empty:

        print(
            "\nWarning: missing model features:"
        )

        print(
            problematic
        )

    return live


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Enrich live NASA FIRMS detections "
            "for GeoFlare inference."
        )
    )

    parser.add_argument(
        "--input",
        default=str(FIRMS_PATH),
        help="Live FIRMS India CSV.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output enriched CSV.",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"FIRMS input not found: "
            f"{input_path}"
        )

    if not HISTORICAL_PATH.exists():
        raise FileNotFoundError(
            "Historical enriched dataset not found: "
            f"{HISTORICAL_PATH}"
        )

    if not OSM_PATH.exists():
        raise FileNotFoundError(
            f"OSM context file not found: "
            f"{OSM_PATH}"
        )

    print("Loading live FIRMS data...")

    live = pd.read_csv(
        input_path
    )

    print("Loading historical data...")

    historical = pd.read_csv(
        HISTORICAL_PATH
    )

    print("Loading OSM context...")

    osm = pd.read_csv(
        OSM_PATH
    )

    print(
        f"Live detections: "
        f"{len(live):,}"
    )

    print(
        f"Historical detections: "
        f"{len(historical):,}"
    )

    print(
        f"OSM features: "
        f"{len(osm):,}"
    )

    live = add_basic_features(
        live
    )

    live = add_location_features(
        live
    )

    live = add_temporal_features(
        live,
        historical,
    )

    live = add_osm_features(
        live,
        osm,
    )

    live = add_landcover_features(
        live
    )

    live = validate_live_features(
        live
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    live.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nLive enrichment completed."
    )

    print(
        f"Output rows: "
        f"{len(live):,}"
    )

    print(
        f"Output columns: "
        f"{len(live.columns)}"
    )

    print(
        f"Saved to:\n"
        f"{output_path}"
    )

    print(
        "\nLand-cover distribution:"
    )

    print(
        live["landcover_name"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nSample:"
    )

    display_columns = [
        "latitude",
        "longitude",
        "frp",
        "confidence_score",
        "persistence_days",
        "detections_at_location",
        "distance_to_industry_km",
        "distance_to_power_km",
        "distance_to_oil_gas_km",
        "distance_to_road_km",
        "distance_to_settlement_km",
        "landcover_name",
        "agriculture_context",
        "vegetation_context",
        "industrial_context",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in live.columns
    ]

    print(
        live[
            available_columns
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
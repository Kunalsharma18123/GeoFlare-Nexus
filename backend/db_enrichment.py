import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from scipy.spatial import cKDTree
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.models import FireEvent, ThermalObservation
from backend.scripts.predict import predict_dataframe
from backend.scripts.score_risk import calculate_risk


# ============================================================================
# Paths
# ============================================================================

# This file lives at:
#
#   GeoFlare/backend/db_enrichment.py
#
# Therefore .parent is:
#
#   GeoFlare/backend
#
BACKEND_DIR = Path(__file__).resolve().parent

OSM_PATH = (
    BACKEND_DIR
    / "data"
    / "osm"
    / "osm_context_points.csv"
)

# Existing GeoFlare WorldCover data is expected here.
#
# We also support the older "landcover" directory as a fallback so that
# previously downloaded tiles do not become unusable.
WORLDCOVER_DIR = (
    BACKEND_DIR
    / "data"
    / "worldcover"
)

LEGACY_WORLDCOVER_DIR = (
    BACKEND_DIR
    / "data"
    / "landcover"
)

WORLDCOVER_BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map"
)


# ============================================================================
# Constants
# ============================================================================

EARTH_RADIUS_KM = 6371.0088

OSM_CATEGORIES = (
    "industrial",
    "power",
    "oil_gas",
    "road",
    "settlement",
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


# ============================================================================
# Coordinate helpers
# ============================================================================

def latlon_to_xyz(lat, lon):
    """
    Convert latitude/longitude coordinates to 3D unit-sphere coordinates.

    This allows cKDTree to perform efficient nearest-neighbour searches
    while accounting for the Earth's spherical geometry.
    """

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack(
        (
            x,
            y,
            z,
        )
    )


def xyz_distance_to_km(distance):
    """
    Convert unit-sphere chord distance to great-circle distance in km.

    cKDTree returns the straight-line chord distance between two points on
    the unit sphere. Convert that chord distance into the corresponding
    angular/geodesic distance.
    """

    distance = np.asarray(
        distance,
        dtype=float,
    )

    distance = np.clip(
        distance,
        0.0,
        2.0,
    )

    angular_distance = (
        2.0
        * np.arcsin(
            distance / 2.0
        )
    )

    return angular_distance * EARTH_RADIUS_KM


# ============================================================================
# OSM context
# ============================================================================

@lru_cache(maxsize=1)
def load_osm_trees():
    """
    Load the OSM contextual feature CSV and build one KD-tree per category.

    The CSV is loaded only once per Python process.
    """

    if not OSM_PATH.exists():
        raise FileNotFoundError(
            "OSM context file not found:\n"
            f"  {OSM_PATH}\n\n"
            "Expected file:\n"
            "  backend/data/osm/osm_context_points.csv"
        )

    osm = pd.read_csv(
        OSM_PATH
    )

    required_columns = {
        "category",
        "latitude",
        "longitude",
    }

    missing = (
        required_columns
        - set(osm.columns)
    )

    if missing:
        raise ValueError(
            "OSM context file is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    trees = {}

    for category in OSM_CATEGORIES:

        subset = (
            osm[
                osm["category"] == category
            ]
            .dropna(
                subset=[
                    "latitude",
                    "longitude",
                ]
            )
        )

        if subset.empty:
            continue

        coordinates = latlon_to_xyz(
            subset["latitude"].to_numpy(
                dtype=float
            ),
            subset["longitude"].to_numpy(
                dtype=float
            ),
        )

        trees[category] = cKDTree(
            coordinates
        )

    return trees


def get_osm_features(
    latitude,
    longitude,
):
    """
    Calculate nearest OSM contextual distances and proximity flags.
    """

    trees = load_osm_trees()

    point = latlon_to_xyz(
        np.array(
            [float(latitude)]
        ),
        np.array(
            [float(longitude)]
        ),
    )

    distances = {}

    column_map = {
        "industrial": "distance_to_industry_km",
        "power": "distance_to_power_km",
        "oil_gas": "distance_to_oil_gas_km",
        "road": "distance_to_road_km",
        "settlement": "distance_to_settlement_km",
    }

    for category, output_column in column_map.items():

        tree = trees.get(
            category
        )

        if tree is None:
            distances[
                output_column
            ] = 9999.0

            continue

        distance, _ = tree.query(
            point
        )

        distances[
            output_column
        ] = float(
            xyz_distance_to_km(
                distance[0]
            )
        )

    distances[
        "near_industry_5km"
    ] = int(
        distances[
            "distance_to_industry_km"
        ] <= 5.0
    )

    distances[
        "near_power_10km"
    ] = int(
        distances[
            "distance_to_power_km"
        ] <= 10.0
    )

    distances[
        "near_oil_gas_10km"
    ] = int(
        distances[
            "distance_to_oil_gas_km"
        ] <= 10.0
    )

    distances[
        "near_road_1km"
    ] = int(
        distances[
            "distance_to_road_km"
        ] <= 1.0
    )

    distances[
        "near_settlement_5km"
    ] = int(
        distances[
            "distance_to_settlement_km"
        ] <= 5.0
    )

    distances[
        "industrial_context"
    ] = int(
        distances[
            "near_industry_5km"
        ]
        or distances[
            "near_power_10km"
        ]
        or distances[
            "near_oil_gas_10km"
        ]
    )

    distances[
        "settlement_context"
    ] = int(
        distances[
            "near_settlement_5km"
        ]
    )

    distances[
        "road_context"
    ] = int(
        distances[
            "near_road_1km"
        ]
    )

    return distances


# ============================================================================
# ESA WorldCover
# ============================================================================

def worldcover_tile_name(
    latitude,
    longitude,
):
    """
    Determine the ESA WorldCover tile containing a coordinate.

    GeoFlare uses the 3-degree tile naming convention.
    """

    tile_lat = (
        math.floor(
            float(latitude) / 3
        )
        * 3
    )

    tile_lon = (
        math.floor(
            float(longitude) / 3
        )
        * 3
    )

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


def worldcover_filename(
    tile_name,
):
    """
    Return the standard ESA WorldCover filename.
    """

    return (
        "ESA_WorldCover_10m_2021_v200_"
        f"{tile_name}_Map.tif"
    )


def worldcover_path(
    tile_name,
):
    """
    Return the local WorldCover path.

    Prefer the current worldcover directory. If the tile already exists in
    the legacy landcover directory, reuse it.
    """

    filename = worldcover_filename(
        tile_name
    )

    current_path = (
        WORLDCOVER_DIR
        / filename
    )

    legacy_path = (
        LEGACY_WORLDCOVER_DIR
        / filename
    )

    if current_path.exists():
        return current_path

    if legacy_path.exists():
        return legacy_path

    return current_path


def download_worldcover_tile(
    tile_name,
):
    """
    Download a WorldCover tile if it is not already cached.
    """

    output_path = worldcover_path(
        tile_name
    )

    if output_path.exists():
        return output_path

    WORLDCOVER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = worldcover_filename(
        tile_name
    )

    url = (
        f"{WORLDCOVER_BASE_URL}/"
        f"{filename}"
    )

    print(
        f"Downloading WorldCover tile: "
        f"{tile_name}"
    )

    response = requests.get(
        url,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    temporary_path = (
        output_path.with_suffix(
            ".part"
        )
    )

    try:

        with open(
            temporary_path,
            "wb",
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(
                        chunk
                    )

        temporary_path.replace(
            output_path
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    return output_path


@lru_cache(maxsize=256)
def sample_worldcover(
    latitude,
    longitude,
):
    """
    Sample ESA WorldCover at one coordinate.

    Returns the numeric WorldCover class.

    Raster errors are deliberately allowed to propagate so that a broken
    or invalid tile does not silently become an artificial land-cover class.
    """

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    tile_name = worldcover_tile_name(
        latitude,
        longitude,
    )

    path = download_worldcover_tile(
        tile_name
    )

    try:

        with rasterio.open(
            path
        ) as src:

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

            if src.nodata is not None:
                if value == src.nodata:
                    return None

            if value is None:
                return None

            if np.isnan(value):
                return None

            return int(value)

    except Exception as exc:

        raise RuntimeError(
            "WorldCover sampling failed for "
            f"({latitude}, {longitude}) "
            f"using tile {tile_name}: {exc}"
        ) from exc


def get_landcover_features(
    latitude,
    longitude,
):
    """
    Convert WorldCover class into contextual features expected by GeoFlare.
    """

    landcover_class = sample_worldcover(
        float(latitude),
        float(longitude),
    )

    # If raster sampling returns no valid pixel, use the same neutral
    # fallback used by the enrichment pipeline.
    if landcover_class is None:
        landcover_class = 60

    return {
        "landcover_class": int(
            landcover_class
        ),

        "landcover_name": (
            WORLDCOVER_CLASSES.get(
                landcover_class,
                "Unknown",
            )
        ),

        "vegetation_context": int(
            landcover_class
            in {
                10,
                20,
                30,
            }
        ),

        "agriculture_context": int(
            landcover_class == 40
        ),

        "builtup_context": int(
            landcover_class == 50
        ),

        "water_context": int(
            landcover_class == 80
        ),

        "wetland_context": int(
            landcover_class
            in {
                90,
                95,
            }
        ),
    }


# ============================================================================
# PostgreSQL / PostGIS historical context
# ============================================================================

def get_location_history_features(
    session: Session,
    event: FireEvent,
    observation: ThermalObservation,
):
    """
    Build historical thermal-location features from PostgreSQL.

    Feature semantics:

    persistence_days
        Persistence of the CURRENT fire event.

    detections_at_location
        Number of observations historically associated with the same
        location_history record up to the current observation.

    FRP statistics
        Historical FRP statistics for that same location up to the current
        observation.

    Future observations are excluded.
    """

    query = text(
        """
        SELECT
            o.acq_datetime,
            o.frp
        FROM thermal_observations o
        JOIN event_observations eo
            ON eo.observation_id = o.id
        JOIN fire_events e
            ON e.id = eo.event_id
        WHERE e.location_history_id = :location_history_id
          AND o.acq_datetime <= :observation_time
        ORDER BY
            o.acq_datetime,
            o.id
        """
    )

    rows = session.execute(
        query,
        {
            "location_history_id": (
                event.location_history_id
            ),
            "observation_time": (
                observation.acq_datetime
            ),
        },
    ).mappings().all()

    # ------------------------------------------------------------------------
    # Current event persistence
    # ------------------------------------------------------------------------

    event_persistence_days = float(
        event.persistence_days
        if event.persistence_days is not None
        else 0.0
    )

    persistence_days = max(
        1.0,
        event_persistence_days,
    )

    # ------------------------------------------------------------------------
    # Historical location statistics
    # ------------------------------------------------------------------------

    if not rows:

        current_frp = (
            float(observation.frp)
            if observation.frp is not None
            else 0.0
        )

        return {
            "persistence_days": (
                persistence_days
            ),

            "detections_at_location": 1,

            "mean_frp_location": (
                current_frp
            ),

            "max_frp_location": (
                current_frp
            ),

            "min_frp_location": (
                current_frp
            ),

            "frp_std_location": 0.0,
        }

    frps = np.array(
        [
            float(row["frp"])
            for row in rows
            if row["frp"] is not None
        ],
        dtype=float,
    )

    if frps.size == 0:

        current_frp = (
            float(observation.frp)
            if observation.frp is not None
            else 0.0
        )

        frps = np.array(
            [
                current_frp
            ],
            dtype=float,
        )

    return {
        "persistence_days": (
            persistence_days
        ),

        "detections_at_location": int(
            len(rows)
        ),

        "mean_frp_location": float(
            np.mean(frps)
        ),

        "max_frp_location": float(
            np.max(frps)
        ),

        "min_frp_location": float(
            np.min(frps)
        ),

        "frp_std_location": float(
            np.std(frps)
        ),
    }


# ============================================================================
# Feature construction
# ============================================================================

def build_feature_row(
    session: Session,
    event: FireEvent,
    observation: ThermalObservation,
):
    """
    Build one complete ML inference row.

    The feature names remain compatible with the existing trained
    GeoFlare Random Forest.
    """

    if observation.acq_datetime is None:
        raise ValueError(
            "Observation has no acquisition datetime."
        )

    acq_datetime = (
        observation.acq_datetime
    )

    frp = (
        float(observation.frp)
        if observation.frp is not None
        else 0.0
    )

    confidence = (
        float(
            observation.confidence_score
        )
        if observation.confidence_score is not None
        else 0.0
    )

    # ------------------------------------------------------------------------
    # Database-backed temporal/location intelligence
    # ------------------------------------------------------------------------

    temporal = (
        get_location_history_features(
            session,
            event,
            observation,
        )
    )

    # ------------------------------------------------------------------------
    # OSM contextual intelligence
    # ------------------------------------------------------------------------

    osm = get_osm_features(
        observation.latitude,
        observation.longitude,
    )

    # ------------------------------------------------------------------------
    # ESA WorldCover contextual intelligence
    # ------------------------------------------------------------------------

    landcover = get_landcover_features(
        observation.latitude,
        observation.longitude,
    )

    # FIRMS acquisition time is UTC.
    is_day = int(
        6 <= acq_datetime.hour < 18
    )

    # ------------------------------------------------------------------------
    # Complete feature row
    # ------------------------------------------------------------------------

    row = {
        # FIRMS / thermal
        "frp": frp,
        "confidence_score": confidence,

        # Spatial
        "latitude": float(
            observation.latitude
        ),
        "longitude": float(
            observation.longitude
        ),

        # Temporal
        "hour": int(
            acq_datetime.hour
        ),
        "minute": int(
            acq_datetime.minute
        ),
        "day_of_year": int(
            acq_datetime
            .timetuple()
            .tm_yday
        ),
        "day_of_week": int(
            acq_datetime.weekday()
        ),
        "month": int(
            acq_datetime.month
        ),
        "is_weekend": int(
            acq_datetime.weekday() >= 5
        ),
        "is_day": is_day,

        # DB-backed historical features
        **temporal,

        # OSM features
        **osm,

        # WorldCover features
        **landcover,
    }

    return pd.DataFrame(
        [row]
    )


# ============================================================================
# Event observation lookup
# ============================================================================

def get_latest_event_observation(
    session: Session,
    event_id: int,
):
    """
    Return the latest thermal observation associated with an event.
    """

    query = text(
        """
        SELECT
            o.id
        FROM thermal_observations o
        JOIN event_observations eo
            ON eo.observation_id = o.id
        WHERE eo.event_id = :event_id
        ORDER BY
            o.acq_datetime DESC,
            o.id DESC
        LIMIT 1
        """
    )

    row = session.execute(
        query,
        {
            "event_id": event_id,
        },
    ).mappings().first()

    if row is None:
        return None

    return session.get(
        ThermalObservation,
        row["id"],
    )


# ============================================================================
# Intelligence pipeline
# ============================================================================

def update_event_intelligence(
    session: Session,
    event_id: int,
):
    """
    Run the complete GeoFlare intelligence pipeline for one event.

    Pipeline:

        PostgreSQL event
            |
            v
        DB-backed enrichment
            |
            +-- FIRMS features
            +-- event persistence
            +-- location history
            +-- OSM context
            +-- WorldCover
            |
            v
        Random Forest classification
            |
            v
        Risk engine
            |
            v
        fire_events update
    """

    # ------------------------------------------------------------------------
    # 1. Load event
    # ------------------------------------------------------------------------

    event = session.get(
        FireEvent,
        event_id,
    )

    if event is None:
        raise ValueError(
            f"Fire event {event_id} not found."
        )

    if event.location_history_id is None:
        raise ValueError(
            f"Fire event {event_id} has no "
            "location_history_id."
        )

    # ------------------------------------------------------------------------
    # 2. Find latest observation
    # ------------------------------------------------------------------------

    observation = (
        get_latest_event_observation(
            session,
            event_id,
        )
    )

    if observation is None:
        raise ValueError(
            f"Fire event {event_id} "
            "has no observations."
        )

    # ------------------------------------------------------------------------
    # 3. DB-backed enrichment
    # ------------------------------------------------------------------------

    features = build_feature_row(
        session,
        event,
        observation,
    )

    # ------------------------------------------------------------------------
    # 4. ML classification
    # ------------------------------------------------------------------------

    prediction = predict_dataframe(
        features
    )

    if prediction.empty:
        raise ValueError(
            f"Prediction returned no result "
            f"for event {event_id}."
        )

    prediction_row = (
        prediction.iloc[0]
    )

    # ------------------------------------------------------------------------
    # 5. Risk scoring
    # ------------------------------------------------------------------------

    risk = calculate_risk(
        prediction_row
    )

    # ------------------------------------------------------------------------
    # 6. Persist classification
    # ------------------------------------------------------------------------

    event.predicted_class = str(
        prediction_row[
            "predicted_class"
        ]
    )

    event.prediction_confidence = float(
        prediction_row[
            "prediction_confidence"
        ]
    )

    event.classification_status = str(
        prediction_row[
            "classification_status"
        ]
    )

    # ------------------------------------------------------------------------
    # 7. Persist risk
    # ------------------------------------------------------------------------

    event.risk_score = float(
        risk["risk_score"]
    )

    event.risk_level = str(
        risk["risk_level"]
    )

    event.decision_status = str(
        risk["decision_status"]
    )

    event.risk_reasons = str(
        risk["risk_reasons"]
    )

    # ------------------------------------------------------------------------
    # 8. Persist intelligence feature snapshot
    # ------------------------------------------------------------------------

    event.intelligence_features = {
        str(key): (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in prediction_row.items()
    }

    # ------------------------------------------------------------------------
    # 9. Commit
    # ------------------------------------------------------------------------

    session.commit()

    # ------------------------------------------------------------------------
    # 10. Return compact result
    # ------------------------------------------------------------------------

    return {
        "event_id": event.id,

        "predicted_class": (
            event.predicted_class
        ),

        "prediction_confidence": (
            event.prediction_confidence
        ),

        "classification_status": (
            event.classification_status
        ),

        "risk_score": (
            event.risk_score
        ),

        "risk_level": (
            event.risk_level
        ),

        "decision_status": (
            event.decision_status
        ),

        "risk_reasons": (
            event.risk_reasons
        ),
    }

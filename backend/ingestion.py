import hashlib
import logging
import os
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from backend.db import engine
from backend.db_enrichment import update_event_intelligence
from backend.event_engine import process_observation
from backend.ingestion_status import (
    mark_ingestion_failed,
    mark_ingestion_started,
    mark_ingestion_succeeded,
)
from backend.repository import insert_observation


load_dotenv()
logger = logging.getLogger(__name__)


FIRMS_SOURCE = "VIIRS_NOAA20_NRT"

INDIA_BBOX = "68.1,8.0,97.4,37.1"

FIRMS_DAY_RANGE = 5


def fetch_firms() -> pd.DataFrame:
    """
    Fetch recent VIIRS NOAA-20 NRT observations
    for the India bounding box.
    """

    firms_map_key = os.getenv("FIRMS_MAP_KEY")
    if not firms_map_key:
        raise RuntimeError("FIRMS_MAP_KEY is not set")

    firms_url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{firms_map_key}/"
        f"{FIRMS_SOURCE}/"
        f"{INDIA_BBOX}/"
        f"{FIRMS_DAY_RANGE}"
    )

    try:
        response = requests.get(
            firms_url,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"NASA FIRMS request failed ({type(exc).__name__})."
        ) from None

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )
        raise RuntimeError(
            f"NASA FIRMS returned HTTP {status_code}."
        ) from None

    dataframe = pd.read_csv(
        StringIO(response.text)
    )

    if dataframe.empty:
        return dataframe

    return dataframe


def parse_acquisition_datetime(row) -> datetime:
    """
    FIRMS acq_date + acq_time represent acquisition
    time in UTC.
    """

    date_text = str(
        row["acq_date"]
    )

    time_value = row["acq_time"]

    if pd.isna(time_value):
        raise ValueError(
            "Missing acq_time"
        )

    time_text = str(
        int(float(time_value))
    ).zfill(4)

    value = datetime.strptime(
        f"{date_text} {time_text}",
        "%Y-%m-%d %H%M",
    )

    return value.replace(
        tzinfo=timezone.utc
    )


def normalize_confidence(value) -> float | None:
    """
    Normalize FIRMS confidence into a numeric score.

    Supports both numeric and categorical FIRMS values.
    """

    if pd.isna(value):
        return None

    if isinstance(value, str):
        normalized = (
            value
            .strip()
            .lower()
        )

        mapping = {
            "l": 0.30,
            "n": 0.60,
            "h": 0.90,
            "low": 0.30,
            "nominal": 0.60,
            "high": 0.90,
        }

        if normalized in mapping:
            return mapping[normalized]

    try:
        numeric = float(value)

        if numeric > 1:
            return numeric / 100.0

        return numeric

    except (TypeError, ValueError):
        return None


def make_observation_key(
    row,
    acq_datetime: datetime,
) -> str:
    """
    Generate a deterministic identity for one FIRMS
    satellite observation.

    The key allows repeated ingestion runs to be safely
    deduplicated.
    """

    raw = "|".join(
        [
            str(
                row.get(
                    "satellite",
                    "",
                )
            ),
            str(
                row.get(
                    "instrument",
                    "",
                )
            ),
            acq_datetime.isoformat(),
            f"{float(row['latitude']):.6f}",
            f"{float(row['longitude']):.6f}",
        ]
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def ingest_dataframe(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Ingest FIRMS observations into PostgreSQL,
    associate them with thermal events, and run
    DB-backed intelligence for affected events.
    """

    if dataframe.empty:
        return {
            "fetched": 0,
            "inserted": 0,
            "duplicates": 0,
            "events_created": 0,
            "events_updated": 0,
            "intelligence_processed": 0,
            "intelligence_failed": 0,
        }

    required_columns = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "frp",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "FIRMS response is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    inserted = 0
    duplicates = 0
    events_created = 0
    events_updated = 0

    intelligence_processed = 0
    intelligence_failed = 0

    touched_event_ids = set()

    # Process observations chronologically so event
    # lifecycle decisions are deterministic.
    dataframe = (
        dataframe
        .sort_values(
            [
                "acq_date",
                "acq_time",
            ]
        )
        .reset_index(drop=True)
    )

    with Session(engine) as session:

        for _, row in dataframe.iterrows():

            latitude = float(
                row["latitude"]
            )

            longitude = float(
                row["longitude"]
            )

            acq_datetime = (
                parse_acquisition_datetime(
                    row
                )
            )

            observation_key = (
                make_observation_key(
                    row,
                    acq_datetime,
                )
            )

            confidence_score = (
                normalize_confidence(
                    row.get(
                        "confidence"
                    )
                )
            )

            observation, was_inserted = (
                insert_observation(
                    session=session,
                    observation_key=observation_key,
                    latitude=latitude,
                    longitude=longitude,
                    acq_datetime=acq_datetime,
                    satellite=str(
                        row.get(
                            "satellite",
                            "",
                        )
                    ),
                    instrument=str(
                        row.get(
                            "instrument",
                            "",
                        )
                    ),
                    frp=(
                        float(row["frp"])
                        if pd.notna(
                            row["frp"]
                        )
                        else None
                    ),
                    confidence_score=(
                        confidence_score
                    ),
                )
            )

            if not was_inserted:
                duplicates += 1
                continue

            inserted += 1

            event, created = (
                process_observation(
                    session,
                    observation,
                )
            )

            touched_event_ids.add(
                event.id
            )

            if created:
                events_created += 1
            else:
                events_updated += 1

        # ---------------------------------------------------------
        # Run DB-backed intelligence only for events touched by
        # newly inserted observations.
        # ---------------------------------------------------------

        print()
        print(
            "Running GeoFlare intelligence..."
        )

        print(
            f"Events requiring intelligence: "
            f"{len(touched_event_ids)}"
        )

        for event_id in sorted(
            touched_event_ids
        ):
            try:
                result = (
                    update_event_intelligence(
                        session,
                        event_id,
                    )
                )

                intelligence_processed += 1

                print(
                    f"  Event {event_id}: "
                    f"{result['predicted_class']} | "
                    f"{result['prediction_confidence']:.3f} | "
                    f"{result['risk_level']} "
                    f"({result['risk_score']:.2f})"
                )

            except Exception as exc:
                intelligence_failed += 1

                print(
                    f"  Event {event_id}: "
                    f"INTELLIGENCE FAILED - {exc}"
                )

    return {
        "fetched": len(dataframe),
        "inserted": inserted,
        "duplicates": duplicates,
        "events_created": events_created,
        "events_updated": events_updated,
        "intelligence_processed": (
            intelligence_processed
        ),
        "intelligence_failed": (
            intelligence_failed
        ),
    }


def run_ingestion() -> dict:
    """
    Execute one complete FIRMS ingestion cycle.
    """

    mark_ingestion_started()

    try:
        dataframe = fetch_firms()
        result = ingest_dataframe(dataframe)
    except Exception:
        mark_ingestion_failed()
        logger.exception("FIRMS ingestion cycle failed")
        raise

    mark_ingestion_succeeded(result)
    return result


if __name__ == "__main__":

    result = run_ingestion()

    print()
    print(
        "GeoFlare FIRMS ingestion"
    )
    print(
        "-------------------------"
    )

    print(
        f"Fetched:              "
        f"{result['fetched']}"
    )

    print(
        f"Inserted:             "
        f"{result['inserted']}"
    )

    print(
        f"Duplicates:           "
        f"{result['duplicates']}"
    )

    print(
        f"Events created:       "
        f"{result['events_created']}"
    )

    print(
        f"Events updated:       "
        f"{result['events_updated']}"
    )

    print(
        f"Intelligence:         "
        f"{result['intelligence_processed']}"
    )

    print(
        f"Intelligence failed:  "
        f"{result['intelligence_failed']}"
    )
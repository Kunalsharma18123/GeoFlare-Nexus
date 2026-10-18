from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.models import LocationHistory


LOCATION_MATCH_RADIUS_KM = 5.0


def find_location_history(
    session: Session,
    latitude: float,
    longitude: float,
) -> LocationHistory | None:

    radius_meters = LOCATION_MATCH_RADIUS_KM * 1000

    statement = text("""
        SELECT
            id,
            ST_Distance(
                geom,
                ST_SetSRID(
                    ST_MakePoint(:longitude, :latitude),
                    4326
                )::geography
            ) AS distance_m
        FROM location_history
        WHERE ST_DWithin(
            geom,
            ST_SetSRID(
                ST_MakePoint(:longitude, :latitude),
                4326
            )::geography,
            :radius
        )
        ORDER BY distance_m ASC
        LIMIT 1
    """)

    result = session.execute(
        statement,
        {
            "longitude": longitude,
            "latitude": latitude,
            "radius": radius_meters,
        },
    ).first()

    if result is None:
        return None

    return session.get(
        LocationHistory,
        result.id,
    )


def create_location_history(
    session: Session,
    latitude: float,
    longitude: float,
    activity_time,
) -> LocationHistory:

    history = LocationHistory(
        geom=(
            f"SRID=4326;POINT("
            f"{longitude} {latitude}"
            f")"
        ),
        first_activity_at=activity_time,
        last_activity_at=activity_time,
        event_count=1,
        total_detection_count=1,
    )

    session.add(history)
    session.flush()

    return history


def record_new_event(
    session: Session,
    latitude: float,
    longitude: float,
    activity_time,
) -> LocationHistory:
    """
    Record a new thermal activity episode.

    A new event increments:
    - event_count
    - total_detection_count
    """

    history = find_location_history(
        session,
        latitude,
        longitude,
    )

    if history is None:
        return create_location_history(
            session,
            latitude,
            longitude,
            activity_time,
        )

    history.last_activity_at = max(
        history.last_activity_at,
        activity_time,
    )

    history.event_count += 1
    history.total_detection_count += 1

    session.flush()

    return history


def record_event_detection(
    session: Session,
    history_id: int,
    activity_time,
) -> LocationHistory:
    """
    Record an observation belonging to an existing event.

    The event already knows its location_history_id, so we
    do not perform another spatial lookup.
    """

    history = session.get(
        LocationHistory,
        history_id,
    )

    if history is None:
        raise ValueError(
            f"Location history {history_id} does not exist"
        )

    history.last_activity_at = max(
        history.last_activity_at,
        activity_time,
    )

    history.total_detection_count += 1

    session.flush()

    return history
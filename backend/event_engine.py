from datetime import timedelta

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from backend.location_history import (
    record_event_detection,
    record_new_event,
)
from backend.models import (
    EventObservation,
    FireEvent,
    ThermalObservation,
)


EVENT_MATCH_RADIUS_KM = 5.0
EVENT_GAP_HOURS = 48.0


def close_stale_events(
    session: Session,
    as_of,
) -> int:
    """
    Close ACTIVE events that have not received a detection
    within the configured event gap.
    """

    cutoff = as_of - timedelta(
        hours=EVENT_GAP_HOURS
    )

    statement = (
        update(FireEvent)
        .where(
            FireEvent.status == "ACTIVE",
            FireEvent.last_detected_at < cutoff,
        )
        .values(
            status="CLOSED",
            updated_at=func.now(),
        )
    )

    result = session.execute(statement)
    session.flush()

    return result.rowcount


def get_existing_event_for_observation(
    session: Session,
    observation: ThermalObservation,
) -> FireEvent | None:
    """
    Check whether an observation has already been assigned
    to an event.
    """

    statement = (
        select(FireEvent)
        .join(
            EventObservation,
            EventObservation.event_id == FireEvent.id,
        )
        .where(
            EventObservation.observation_id
            == observation.id
        )
        .limit(1)
    )

    return session.scalar(statement)


def find_active_event(
    session: Session,
    observation: ThermalObservation,
) -> FireEvent | None:
    """
    Find the nearest ACTIVE event within the configured
    spatial and temporal continuity window.
    """

    cutoff = observation.acq_datetime - timedelta(
        hours=EVENT_GAP_HOURS
    )

    radius_meters = EVENT_MATCH_RADIUS_KM * 1000

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
        FROM fire_events
        WHERE status = 'ACTIVE'
          AND last_detected_at >= :cutoff
          AND ST_DWithin(
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
            "longitude": observation.longitude,
            "latitude": observation.latitude,
            "cutoff": cutoff,
            "radius": radius_meters,
        },
    ).first()

    if result is None:
        return None

    return session.get(
        FireEvent,
        result.id,
    )


def create_event(
    session: Session,
    observation: ThermalObservation,
) -> FireEvent:
    """
    Create a new thermal activity episode and associate it
    with a historical location.
    """

    history = record_new_event(
        session,
        latitude=observation.latitude,
        longitude=observation.longitude,
        activity_time=observation.acq_datetime,
    )

    event = FireEvent(
        location_history_id=history.id,
        geom=(
            f"SRID=4326;POINT("
            f"{observation.longitude} "
            f"{observation.latitude}"
            f")"
        ),
        first_detected_at=observation.acq_datetime,
        last_detected_at=observation.acq_datetime,
        detection_count=1,
        persistence_days=0.0,
        status="ACTIVE",
    )

    session.add(event)
    session.flush()

    link = EventObservation(
        event_id=event.id,
        observation_id=observation.id,
    )

    session.add(link)

    session.commit()
    session.refresh(event)

    return event


def update_event(
    session: Session,
    event: FireEvent,
    observation: ThermalObservation,
) -> FireEvent:
    """
    Add a new observation to an existing event.

    Duplicate observations are ignored.
    """

    existing_link = session.get(
        EventObservation,
        {
            "event_id": event.id,
            "observation_id": observation.id,
        },
    )

    if existing_link is not None:
        return event

    event.last_detected_at = max(
        event.last_detected_at,
        observation.acq_datetime,
    )

    event.detection_count += 1

    duration = (
        event.last_detected_at
        - event.first_detected_at
    )

    event.persistence_days = max(
        duration.total_seconds() / 86400,
        0.0,
    )

    link = EventObservation(
        event_id=event.id,
        observation_id=observation.id,
    )

    session.add(link)

    if event.location_history_id is None:
        raise ValueError(
            f"Event {event.id} has no location history"
        )

    record_event_detection(
        session,
        history_id=event.location_history_id,
        activity_time=observation.acq_datetime,
    )

    session.commit()
    session.refresh(event)

    return event


def process_observation(
    session: Session,
    observation: ThermalObservation,
) -> tuple[FireEvent, bool]:
    """
    Process one FIRMS observation.

    Returns:
        (event, created)

    created=True:
        A new event was created.

    created=False:
        The observation was associated with an existing
        event or was already processed.
    """

    # Idempotency must happen before stale-event processing.
    existing_event = get_existing_event_for_observation(
        session,
        observation,
    )

    if existing_event is not None:
        return existing_event, False

    # Close events that have gone stale.
    close_stale_events(
        session,
        observation.acq_datetime,
    )

    # Find an existing active event.
    event = find_active_event(
        session,
        observation,
    )

    # No active event -> create a new episode.
    if event is None:
        event = create_event(
            session,
            observation,
        )

        return event, True

    # Existing event -> extend the episode.
    event = update_event(
        session,
        event,
        observation,
    )

    return event, False

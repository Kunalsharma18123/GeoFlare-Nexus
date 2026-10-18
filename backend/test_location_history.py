from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.db import engine
from backend.event_engine import process_observation
from backend.models import (
    EventObservation,
    FireEvent,
    LocationHistory,
    ThermalObservation,
)


def make_observation(
    session,
    key,
    latitude,
    longitude,
    when,
):
    observation = ThermalObservation(
        observation_key=key,
        latitude=latitude,
        longitude=longitude,
        geom=f"SRID=4326;POINT({longitude} {latitude})",
        acq_datetime=when,
        satellite="TEST",
        instrument="TEST",
        frp=20.0,
        confidence_score=0.9,
        source="TEST",
    )

    session.add(observation)
    session.flush()

    return observation


def main():
    with Session(engine) as session:

        # Clear previous lifecycle test data.
        test_observations = session.scalars(
            select(ThermalObservation).where(
                ThermalObservation.source == "TEST"
            )
        ).all()

        test_ids = [
            observation.id
            for observation in test_observations
        ]

        if test_ids:
            session.execute(
                delete(EventObservation).where(
                    EventObservation.observation_id.in_(test_ids)
                )
            )

            for observation in test_observations:
                session.delete(observation)

        test_events = session.scalars(
            select(FireEvent).where(
                FireEvent.predicted_class == "LIFECYCLE_TEST"
            )
        ).all()

        for event in test_events:
            session.delete(event)

        session.execute(
            delete(LocationHistory)
        )

        session.commit()

        base = datetime(
            2026,
            9,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        )

        observations = [
            make_observation(
                session,
                "HISTORY-001",
                23.6600,
                87.1500,
                base,
            ),
            make_observation(
                session,
                "HISTORY-002",
                23.6620,
                87.1520,
                base + timedelta(days=1),
            ),
            make_observation(
                session,
                "HISTORY-003",
                23.6640,
                87.1540,
                base + timedelta(days=2),
            ),

            # New episode at the same location.
            make_observation(
                session,
                "HISTORY-004",
                23.6630,
                87.1530,
                base + timedelta(days=5),
            ),
        ]

        session.commit()

        print("Testing location history")
        print()

        for observation in observations:
            event, created = process_observation(
                session,
                observation,
            )

            if event.predicted_class is None:
                event.predicted_class = "LIFECYCLE_TEST"
                session.commit()

            action = "CREATED" if created else "UPDATED"

            print(
                f"{action} | "
                f"{observation.observation_key} -> "
                f"Event {event.id} | "
                f"Detections: {event.detection_count} | "
                f"Persistence: "
                f"{event.persistence_days:.2f} days"
            )

        print()
        print("Location history")
        print("----------------")

        histories = session.scalars(
            select(LocationHistory)
        ).all()

        for history in histories:
            print(
                f"Location {history.id} | "
                f"Events: {history.event_count} | "
                f"Total detections: "
                f"{history.total_detection_count} | "
                f"First activity: "
                f"{history.first_activity_at} | "
                f"Last activity: "
                f"{history.last_activity_at}"
            )


if __name__ == "__main__":
    main()
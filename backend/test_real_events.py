from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import engine
from backend.event_engine import process_observation
from backend.models import FireEvent, ThermalObservation


def main():
    with Session(engine) as session:
        observations = session.scalars(
            select(ThermalObservation)
            .where(ThermalObservation.source == "NASA_FIRMS")
            .order_by(
                ThermalObservation.acq_datetime,
                ThermalObservation.id,
            )
        ).all()

        print(f"Loaded {len(observations)} FIRMS observations")
        print()

        for observation in observations:
            event, created = process_observation(
                session,
                observation,
            )

            action = "CREATED" if created else "UPDATED"

            print(
                f"{action} | "
                f"Observation {observation.id} -> "
                f"Event {event.id} | "
                f"Detections: {event.detection_count} | "
                f"Persistence: "
                f"{event.persistence_days:.3f} days | "
                f"Status: {event.status}"
            )

        print()

        events = session.scalars(
            select(FireEvent)
            .order_by(FireEvent.id)
        ).all()

        print(f"Total events: {len(events)}")
        print()

        for event in events:
            print(
                f"Event {event.id} | "
                f"Detections: {event.detection_count} | "
                f"Persistence: "
                f"{event.persistence_days:.3f} days | "
                f"Status: {event.status}"
            )


if __name__ == "__main__":
    main()
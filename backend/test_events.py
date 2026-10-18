from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.db import engine
from backend.event_engine import process_observation
from backend.models import (
    EventObservation,
    FireEvent,
    ThermalObservation,
)


TEST_SOURCE = "GEOfLARE_LIFECYCLE_TEST"
TEST_KEYS = [
    "LIFECYCLE-001",
    "LIFECYCLE-002",
    "LIFECYCLE-003",
    "LIFECYCLE-004",
]

# Deliberately isolated from the real FIRMS events.
TEST_LATITUDE = 0.1234
TEST_LONGITUDE = 0.1234


def make_observation(
    session,
    key,
    when,
):
    observation = ThermalObservation(
        observation_key=key,
        latitude=TEST_LATITUDE,
        longitude=TEST_LONGITUDE,
        geom=(
            f"SRID=4326;POINT("
            f"{TEST_LONGITUDE} {TEST_LATITUDE}"
            f")"
        ),
        acq_datetime=when,
        satellite="TEST",
        instrument="TEST",
        frp=20.0,
        confidence_score=0.9,
        source=TEST_SOURCE,
    )

    session.add(observation)
    session.flush()

    return observation


def cleanup(session):
    observations = session.scalars(
        select(ThermalObservation).where(
            ThermalObservation.source == TEST_SOURCE
        )
    ).all()

    observation_ids = [
        observation.id
        for observation in observations
    ]

    if observation_ids:
        session.execute(
            delete(EventObservation).where(
                EventObservation.observation_id.in_(
                    observation_ids
                )
            )
        )

        for observation in observations:
            session.delete(observation)

    session.flush()

    test_events = session.scalars(
        select(FireEvent).where(
            FireEvent.geom
            == (
                f"SRID=4326;POINT("
                f"{TEST_LONGITUDE} {TEST_LATITUDE}"
                f")"
            )
        )
    ).all()

    for event in test_events:
        session.delete(event)

    session.commit()


def main():
    with Session(engine) as session:
        # Remove leftovers from an interrupted previous test.
        cleanup(session)

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
                "LIFECYCLE-001",
                base,
            ),
            make_observation(
                session,
                "LIFECYCLE-002",
                base + timedelta(days=1),
            ),
            make_observation(
                session,
                "LIFECYCLE-003",
                base + timedelta(days=2),
            ),
            make_observation(
                session,
                "LIFECYCLE-004",
                base + timedelta(days=5),
            ),
        ]

        session.commit()

        print("Testing event lifecycle")
        print("----------------------")
        print()

        processed_events = []

        for observation in observations:
            event, created = process_observation(
                session,
                observation,
            )

            processed_events.append(event)

            action = (
                "CREATED"
                if created
                else "UPDATED"
            )

            print(
                f"{action} | "
                f"{observation.observation_key} -> "
                f"Event {event.id} | "
                f"Detections: {event.detection_count} | "
                f"Persistence: "
                f"{event.persistence_days:.2f} days"
            )

        unique_events = {
            event.id: event
            for event in processed_events
        }

        print()
        print("Validating")
        print("----------")

        assert len(unique_events) == 2, (
            f"Expected 2 events, got "
            f"{len(unique_events)}"
        )

        first_event = processed_events[0]
        second_event = processed_events[3]

        assert (
            first_event.id
            == processed_events[1].id
            == processed_events[2].id
        ), "First three observations should share one event"

        assert (
            second_event.id
            != first_event.id
        ), "Fourth observation should create a new event"

        assert (
            first_event.detection_count == 3
        ), (
            "First event should contain "
            f"3 detections, got "
            f"{first_event.detection_count}"
        )

        assert (
            abs(first_event.persistence_days - 2.0)
            < 0.001
        ), (
            "First event should have "
            f"2 days persistence, got "
            f"{first_event.persistence_days}"
        )

        assert (
            second_event.detection_count == 1
        ), (
            "Second event should contain "
            f"1 detection, got "
            f"{second_event.detection_count}"
        )

        assert (
            second_event.persistence_days == 0.0
        ), (
            "Second event should have "
            "0 days persistence"
        )

        print("PASS")
        print()
        print(
            f"Event A: {first_event.detection_count} "
            f"detections, "
            f"{first_event.persistence_days:.2f} days"
        )
        print(
            f"Event B: {second_event.detection_count} "
            f"detection, "
            f"{second_event.persistence_days:.2f} days"
        )

        # Always remove synthetic data after successful validation.
        cleanup(session)

        print()
        print("Synthetic test data cleaned up.")


if __name__ == "__main__":
    main()
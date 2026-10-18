from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.models import ThermalObservation


def insert_observation(
    session: Session,
    observation_key: str,
    latitude: float,
    longitude: float,
    acq_datetime: datetime,
    satellite: str | None,
    instrument: str | None,
    frp: float | None,
    confidence_score: float | None,
) -> tuple[ThermalObservation | None, bool]:

    statement = (
        insert(ThermalObservation)
        .values(
            observation_key=observation_key,
            latitude=latitude,
            longitude=longitude,
            geom=f"SRID=4326;POINT({longitude} {latitude})",
            acq_datetime=acq_datetime,
            satellite=satellite,
            instrument=instrument,
            frp=frp,
            confidence_score=confidence_score,
            source="NASA_FIRMS",
        )
        .on_conflict_do_nothing(
            index_elements=["observation_key"]
        )
        .returning(ThermalObservation.id)
    )

    result = session.execute(statement)
    observation_id = result.scalar_one_or_none()

    if observation_id is None:
        session.rollback()
        return None, False

    session.commit()

    observation = session.scalar(
        select(ThermalObservation).where(
            ThermalObservation.id == observation_id
        )
    )

    return observation, True
from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ThermalObservation(Base):
    __tablename__ = "thermal_observations"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    observation_key: Mapped[str] = mapped_column(
        Text,
        unique=True,
        nullable=False,
    )

    latitude: Mapped[float] = mapped_column(
        Double,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Double,
        nullable=False,
    )

    geom: Mapped[object] = mapped_column(
        Geography(
            geometry_type="POINT",
            srid=4326,
        ),
        nullable=False,
    )

    acq_datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    satellite: Mapped[str | None] = mapped_column(
        Text,
    )

    instrument: Mapped[str | None] = mapped_column(
        Text,
    )

    frp: Mapped[float | None] = mapped_column(
        Double,
    )

    confidence_score: Mapped[float | None] = mapped_column(
        Double,
    )

    source: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="NASA_FIRMS",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class FireEvent(Base):
    __tablename__ = "fire_events"

    location_history_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "location_history.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    geom: Mapped[object] = mapped_column(
        Geography(
            geometry_type="POINT",
            srid=4326,
        ),
        nullable=False,
    )

    first_detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    last_detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    detection_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    persistence_days: Mapped[float] = mapped_column(
        Double,
        nullable=False,
        default=0.0,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="ACTIVE",
    )

    predicted_class: Mapped[str | None] = mapped_column(
        Text,
    )

    prediction_confidence: Mapped[float | None] = mapped_column(
        Double,
    )

    risk_score: Mapped[float | None] = mapped_column(
        Double,
    )

    risk_level: Mapped[str | None] = mapped_column(
        Text,
    )

    classification_status: Mapped[str | None] = mapped_column(
        Text,
    )

    decision_status: Mapped[str | None] = mapped_column(
        Text,
    )

    risk_reasons: Mapped[str | None] = mapped_column(
        Text,
    )

    intelligence_features: Mapped[dict | None] = mapped_column(
        JSONB,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EventObservation(Base):
    __tablename__ = "event_observations"

    event_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "fire_events.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    observation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "thermal_observations.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class LocationHistory(Base):
    __tablename__ = "location_history"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    geom: Mapped[object] = mapped_column(
        Geography(
            geometry_type="POINT",
            srid=4326,
        ),
        nullable=False,
    )

    first_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    event_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    total_detection_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

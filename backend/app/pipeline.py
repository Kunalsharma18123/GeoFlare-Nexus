import logging

from sqlalchemy import text

from backend.db import engine


logger = logging.getLogger(__name__)


class GeoFlarePipeline:
    def health(self) -> dict:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            return {
                "status": "ok",
                "service": "GeoFlare API",
                "database": "connected",
                "pipeline": "operational",
            }

        except Exception:
            logger.exception("GeoFlare health check failed")
            return {
                "status": "degraded",
                "service": "GeoFlare API",
                "database": "error",
                "pipeline": "degraded",
            }

    def get_events(self) -> list[dict]:
        query = text(
            """
            SELECT
                e.id AS event_id,
                e.status AS event_status,
                e.first_detected_at,
                e.last_detected_at,
                e.detection_count,
                e.persistence_days,

                ST_Y(e.geom::geometry) AS latitude,
                ST_X(e.geom::geometry) AS longitude,

                o.acq_date,
                o.acq_time,
                o.acq_datetime,
                o.satellite,
                o.instrument,
                o.frp,
                o.confidence_score,

                COALESCE(e.predicted_class, 'uncertain') AS predicted_class,
                COALESCE(e.predicted_class, 'uncertain') AS model_predicted_class,
                COALESCE(e.prediction_confidence, 0) AS prediction_confidence,
                COALESCE(e.classification_status, 'PENDING') AS classification_status,

                COALESCE(e.risk_score, 0) AS risk_score,
                COALESCE(e.risk_level, 'LOW') AS risk_level,
                COALESCE(e.decision_status, 'PENDING') AS decision_status,
                COALESCE(e.risk_reasons, 'Awaiting event intelligence processing.') AS risk_reasons,

                COALESCE(
                    e.intelligence_features ->> 'landcover_name',
                    'Unknown'
                ) AS landcover_name,

                COALESCE(
                    (e.intelligence_features ->> 'distance_to_industry_km')::double precision,
                    0
                ) AS distance_to_industry_km,

                COALESCE(
                    (e.intelligence_features ->> 'distance_to_power_km')::double precision,
                    0
                ) AS distance_to_power_km,

                COALESCE(
                    (e.intelligence_features ->> 'distance_to_oil_gas_km')::double precision,
                    0
                ) AS distance_to_oil_gas_km,

                COALESCE(
                    (e.intelligence_features ->> 'distance_to_road_km')::double precision,
                    0
                ) AS distance_to_road_km,

                COALESCE(
                    (e.intelligence_features ->> 'distance_to_settlement_km')::double precision,
                    0
                ) AS distance_to_settlement_km,

                COALESCE(
                    (e.intelligence_features ->> 'near_industry_5km')::integer,
                    0
                ) AS near_industry_5km,

                COALESCE(
                    (e.intelligence_features ->> 'near_power_10km')::integer,
                    0
                ) AS near_power_10km,

                COALESCE(
                    (e.intelligence_features ->> 'near_oil_gas_10km')::integer,
                    0
                ) AS near_oil_gas_10km,

                COALESCE(
                    (e.intelligence_features ->> 'near_road_1km')::integer,
                    0
                ) AS near_road_1km,

                COALESCE(
                    (e.intelligence_features ->> 'near_settlement_5km')::integer,
                    0
                ) AS near_settlement_5km,

                COALESCE(
                    (e.intelligence_features ->> 'industrial_context')::integer,
                    0
                ) AS industrial_context,

                COALESCE(
                    (e.intelligence_features ->> 'settlement_context')::integer,
                    0
                ) AS settlement_context,

                COALESCE(
                    (e.intelligence_features ->> 'road_context')::integer,
                    0
                ) AS road_context,

                COALESCE(
                    (e.intelligence_features ->> 'vegetation_context')::integer,
                    0
                ) AS vegetation_context,

                COALESCE(
                    (e.intelligence_features ->> 'agriculture_context')::integer,
                    0
                ) AS agriculture_context,

                COALESCE(
                    (e.intelligence_features ->> 'builtup_context')::integer,
                    0
                ) AS builtup_context,

                COALESCE(
                    (e.intelligence_features ->> 'water_context')::integer,
                    0
                ) AS water_context,

                COALESCE(
                    (e.intelligence_features ->> 'wetland_context')::integer,
                    0
                ) AS wetland_context

            FROM fire_events e

            JOIN LATERAL (
                SELECT
                    o.id,
                    TO_CHAR(
                        o.acq_datetime AT TIME ZONE 'UTC',
                        'YYYY-MM-DD'
                    ) AS acq_date,
                    EXTRACT(
                        HOUR FROM o.acq_datetime AT TIME ZONE 'UTC'
                    ) * 100
                    +
                    EXTRACT(
                        MINUTE FROM o.acq_datetime AT TIME ZONE 'UTC'
                    ) AS acq_time,
                    o.acq_datetime,
                    o.satellite,
                    o.instrument,
                    o.frp,
                    o.confidence_score
                FROM thermal_observations o
                JOIN event_observations eo
                    ON eo.observation_id = o.id
                WHERE eo.event_id = e.id
                ORDER BY
                    o.acq_datetime DESC,
                    o.id DESC
                LIMIT 1
            ) o ON TRUE

            WHERE e.status = 'ACTIVE'

            ORDER BY
                e.risk_score DESC NULLS LAST,
                e.last_detected_at DESC
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(query).mappings().all()

        events = []

        for row in rows:
            event = dict(row)

            if event["acq_time"] is not None:
                event["acq_time"] = int(event["acq_time"])

            events.append(event)

        return events


pipeline = GeoFlarePipeline()

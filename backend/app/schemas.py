from datetime import datetime

from pydantic import BaseModel


class GeoFlareEvent(BaseModel):
    event_id: int
    event_status: str

    latitude: float
    longitude: float

    first_detected_at: datetime
    last_detected_at: datetime

    detection_count: int
    persistence_days: float

    acq_date: str
    acq_time: int
    acq_datetime: datetime

    satellite: str
    instrument: str

    frp: float
    confidence_score: float

    predicted_class: str
    model_predicted_class: str
    prediction_confidence: float
    classification_status: str

    risk_score: float
    risk_level: str
    decision_status: str
    risk_reasons: str

    landcover_name: str

    distance_to_industry_km: float
    distance_to_power_km: float
    distance_to_oil_gas_km: float
    distance_to_road_km: float
    distance_to_settlement_km: float

    near_industry_5km: int
    near_power_10km: int
    near_oil_gas_10km: int
    near_road_1km: int
    near_settlement_5km: int

    industrial_context: int
    settlement_context: int
    road_context: int

    vegetation_context: int
    agriculture_context: int
    builtup_context: int
    water_context: int
    wetland_context: int
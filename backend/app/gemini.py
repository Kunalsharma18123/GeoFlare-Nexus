import json
import os

import requests
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)


def build_event_prompt(event: dict) -> str:
    evidence = {
        "event_id": event["event_id"],
        "status": event["event_status"],
        "coordinates": {
            "latitude": event["latitude"],
            "longitude": event["longitude"],
        },
        "acquisition": {
            "datetime": event["acq_datetime"],
            "satellite": event["satellite"],
            "instrument": event["instrument"],
        },
        "thermal_observation": {
            "frp": event["frp"],
            "confidence_score": event["confidence_score"],
            "detection_count": event["detection_count"],
            "persistence_days": event["persistence_days"],
        },
        "random_forest_assessment": {
            "predicted_class": event["predicted_class"],
            "model_predicted_class": event["model_predicted_class"],
            "prediction_confidence": event["prediction_confidence"],
            "classification_status": event["classification_status"],
        },
        "risk_assessment": {
            "risk_score": event["risk_score"],
            "risk_level": event["risk_level"],
            "decision_status": event["decision_status"],
            "risk_reasons": event["risk_reasons"],
        },
        "context": {
            "landcover": event["landcover_name"],
            "distance_to_industry_km": event["distance_to_industry_km"],
            "distance_to_power_km": event["distance_to_power_km"],
            "distance_to_oil_gas_km": event["distance_to_oil_gas_km"],
            "distance_to_road_km": event["distance_to_road_km"],
            "distance_to_settlement_km": event["distance_to_settlement_km"],
            "near_industry_5km": event["near_industry_5km"],
            "near_power_10km": event["near_power_10km"],
            "near_oil_gas_10km": event["near_oil_gas_10km"],
            "near_road_1km": event["near_road_1km"],
            "near_settlement_5km": event["near_settlement_5km"],
            "industrial_context": event["industrial_context"],
            "settlement_context": event["settlement_context"],
            "road_context": event["road_context"],
            "vegetation_context": event["vegetation_context"],
            "agriculture_context": event["agriculture_context"],
            "builtup_context": event["builtup_context"],
            "water_context": event["water_context"],
            "wetland_context": event["wetland_context"],
        },
    }

    return (
        "You are GeoFlare's field operations and damage-prevention advisor. "
        "Analyze only the supplied event evidence. Do not invent missing facts, "
        "causes, locations, people, or infrastructure. The Random Forest output "
        "and risk score are model evidence, not proof. Treat an UNCERTAIN or "
        "low-confidence classification as unconfirmed.\n\n"
        "Your main goal is to reduce risk to people, nearby communities, "
        "property, industrial assets, and the environment. Recommend actions "
        "that are practical for an operations team and ordered by urgency. "
        "Never advise staff to approach, touch, sample, fight, or investigate "
        "the source in person without trained emergency personnel and the "
        "required safety controls. Include exclusion zones, responsible "
        "authorities, remote verification, and protective shutdown or isolation "
        "only when supported by the evidence and approved procedures.\n\n"
        "Return detailed but readable Markdown with exactly these sections:\n"
        "## What is happening\n"
        "Explain the likely situation in plain language. Interpret the thermal "
        "signal, FRP, detection count, persistence, satellite confidence, and "
        "Random Forest class/confidence. Clearly distinguish observed facts, "
        "model interpretation, and what cannot be confirmed remotely.\n"
        "## Why it matters\n"
        "Explain the plausible damage pathways supported by the evidence, such "
        "as spread, heat exposure, smoke, equipment damage, ignition of nearby "
        "material, or impact to people and the environment. Rank the concern "
        "using the risk score and nearby infrastructure/context, without claiming "
        "that damage has already occurred.\n"
        "## Immediate protection\n"
        "Give 4 to 6 numbered actions for the next 0 to 30 minutes. For every "
        "action include an owner (site security, incident commander, fire "
        "service, utility operator, or monitoring team), the purpose, and a "
        "verification step. Cover notification, access control, remote checks, "
        "protection of people/assets, and qualified isolation or shutdown only "
        "under an approved emergency procedure.\n"
        "## Next 1 to 24 hours\n"
        "Give a time-ordered plan for 1 to 24 hours: confirm with imagery or "
        "trained responders, inspect relevant nearby assets safely, repeat the "
        "thermal observation, document changes, and update the risk decision. "
        "State what result would change the plan.\n"
        "## Escalation triggers\n"
        "List explicit triggers for emergency services and incident command, "
        "including increasing FRP, a larger or repeated hotspot, smoke/flames, "
        "rapid spread, proximity to people or critical infrastructure, hazardous "
        "materials, or loss of control. For each trigger state the action.\n"
        "## Uncertainty and next evidence\n"
        "List the important unknowns and identify the single most valuable next "
        "piece of evidence. Explain how its result would change the response.\n"
        "Do not give medical, legal, or life-safety guarantees. If there is any "
        "credible immediate danger, tell the team to contact local emergency "
        "services and follow its emergency response plan.\n\n"
        "Event evidence:\n"
        f"{json.dumps(evidence, default=str, indent=2)}"
    )


def request_event_advice(event: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = requests.post(
        GEMINI_API_URL.format(model=model),
        params={"key": api_key},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": build_event_prompt(event)},
                    ],
                },
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1400,
            },
        },
        timeout=45,
    )
    response.raise_for_status()

    payload = response.json()
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini returned an empty or invalid response") from exc
import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pipeline import pipeline
from .schemas import GeoFlareEvent

from backend.ingestion import run_ingestion
from backend.ingestion_status import get_ingestion_status


logger = logging.getLogger(__name__)

INGEST_INTERVAL_SECONDS = 15 * 60
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
CORS_ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        ",".join(DEFAULT_CORS_ORIGINS),
    ).split(",")
    if origin.strip()
)


app = FastAPI(
    title="GeoFlare API",
    description="AI-Powered Geospatial Thermal Intelligence API",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ingestion_lock = asyncio.Lock()


async def automatic_ingestion_loop():
    await asyncio.sleep(5)

    while True:
        try:
            if not ingestion_lock.locked():
                async with ingestion_lock:
                    await asyncio.to_thread(run_ingestion)

        except Exception:
            logger.error(
                "Automatic ingestion failed; see ingestion status and backend logs."
            )

        await asyncio.sleep(INGEST_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup():
    asyncio.create_task(automatic_ingestion_loop())
    print("[GeoFlare] Automatic FIRMS ingestion enabled.")
    print(
        f"[GeoFlare] Ingestion interval: "
        f"{INGEST_INTERVAL_SECONDS // 60} minutes."
    )


@app.get("/health")
def health():
    health_status = pipeline.health()
    status_code = 200 if health_status["status"] == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content=health_status,
    )


@app.get("/ingestion/status")
def ingestion_status():
    return {
        "interval_seconds": INGEST_INTERVAL_SECONDS,
        **get_ingestion_status(),
    }


@app.get(
    "/events",
    response_model=list[GeoFlareEvent],
)
def get_events():
    return pipeline.get_events()


@app.get("/events/geojson")
def get_events_geojson():
    events = pipeline.get_events()

    features = []

    for event in events:
        properties = {
            key: value
            for key, value in event.items()
            if key not in {"latitude", "longitude"}
        }

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        event["longitude"],
                        event["latitude"],
                    ],
                },
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


@app.post("/ingest")
async def ingest():
    if ingestion_lock.locked():
        return {
            "status": "busy",
            "message": "An ingestion cycle is already running.",
        }

    async with ingestion_lock:
        result = await asyncio.to_thread(run_ingestion)

    return {
        "status": "ok",
        "message": "FIRMS ingestion completed.",
        **result,
    }
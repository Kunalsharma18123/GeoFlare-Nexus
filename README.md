
# GeoFlare

GeoFlare is a geospatial thermal intelligence dashboard for monitoring global and regional fire risk using satellite thermal data, AI risk scoring, and interactive visual analytics.

The project combines:
- a FastAPI backend for data ingestion and event APIs
- a PostgreSQL/Supabase data layer for persisted event intelligence
- a React + Vite frontend with a globe-driven risk dashboard
- ML-assisted classification and risk modeling for thermal anomaly events

## Tech stack

- Backend: Python, FastAPI, SQLAlchemy, GeoAlchemy2, Pandas, Rasterio
- Database: PostgreSQL via Supabase
- Frontend: React, Vite, Leaflet, react-globe.gl
- ML: scikit-learn, joblib
- Package management: uv for Python, npm for frontend

## Project structure

```text
GeoFlare/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   └── schemas.py
│   ├── db.py
│   ├── ingestion.py
│   ├── event_engine.py
│   ├── repository.py
│   ├── models.py
│   ├── location_history.py
│   ├── db_enrichment.py
│   ├── test_db.py
│   ├── test_events.py
│   ├── test_ingest.py
│   ├── test_location_history.py
│   └── test_real_events.py
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── data/
│   └── landcover/
├── .env.example
├── .env
├── pyproject.toml
├── README.md
└── src/
    └── geoflare/
```

## Features

- Global thermal event monitoring using an interactive globe
- Risk-tier classification for high, medium, and low severity events
- India-focused event map overlays for region-level analysis
- Event detail panels with persistence, confidence, and contextual metrics
- Satellite/imagery-inspired investigation cards
- FIRMS ingestion pipeline with enrichment and historical event processing
- PostgreSQL storage and centralized event retrieval through FastAPI

## Prerequisites

Before running the project, make sure you have:

- Python 3.12+
- uv installed
- Node.js 18+ and npm
- A Supabase Postgres project
- A NASA FIRMS API key

## Environment setup

1. Copy the example environment file:

```bash
copy .env.example .env
```

2. Update the values in `.env`:

```env
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_REF:YOUR_DB_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres
FIRMS_MAP_KEY=YOUR_NASA_FIRMS_MAP_KEY
```

### Supabase connection notes

- Use the Session pooler URI from Supabase Dashboard → Project Settings → Database → Connect.
- The driver format must include `postgresql+psycopg://`.
- If your password contains special characters such as `@`, encode them in the connection string (for example `%40`).
- Set `CORS_ALLOWED_ORIGINS` to the exact frontend origin(s), separated by commas.
- The frontend API origin is configured with `VITE_API_BASE_URL` in `frontend/.env`.

## Install dependencies

### Python backend

```bash
uv sync
```

### Frontend

```bash
cd frontend
npm install
```

## Run the application

### 1) Start the backend API

From the repository root:

```bash
uv run uvicorn backend.app.main:app --reload
```

Verify the health endpoint:

```bash
http://127.0.0.1:8000/health
```

The health endpoint returns HTTP `503` when the database check fails. Details
are recorded in backend logs; database connection details are not returned to
the caller.

### 2) Start the frontend

Copy `frontend/.env.example` to `frontend/.env` and configure
`VITE_API_BASE_URL` if the API is not at its local default. Vite embeds this
value into the frontend build, so set it in the build environment for each
deployment. Do not put credentials or other secrets in `VITE_*` variables;
they are exposed to browsers.

From the frontend folder:

```bash
cd frontend
npm run dev
```

Then open the local Vite URL, usually:

```text
http://127.0.0.1:5173
```

## Database validation

Test the database connection from the repository root:

```bash
uv run python -m backend.test_db
```

This validates the SQLAlchemy connection to Supabase and confirms required tables are reachable.

## API endpoints

The backend exposes the following main routes:

- `GET /health` – backend health and pipeline status
- `GET /ingestion/status` – latest ingestion attempt, last successful run, and result counts
- `GET /events` – latest event list for the dashboard
- `GET /events/geojson` – GeoJSON export of event points
- `POST /ingest` – trigger ingestion manually

## Common workflow

1. Set up `.env` with your database and FIRMS key.
2. Validate the database connection.
3. Start the backend server.
4. Start the frontend dev server.
5. Inspect the global risk dashboard and click event markers for investigation details.

## Testing

Run backend tests with:

```bash
uv run pytest
```

You can also run a more targeted check:

```bash
uv run python -m backend.test_real_events
```

## Troubleshooting

### Backend not starting

- Check whether `.env` exists and contains valid `DATABASE_URL`.
- Ensure the connection string uses `postgresql+psycopg://`.
- Check that your Supabase database allows connections from your machine.

### Frontend cannot reach backend

- Confirm the backend is running on port `8000`.
- Make sure `CORS_ALLOWED_ORIGINS` includes the exact frontend origin.
- Confirm `VITE_API_BASE_URL` in the frontend environment matches the API origin.

### FIRMS ingestion issues

- Verify `FIRMS_MAP_KEY` is present and valid.
- Check ingestion output in backend logs after running the API.
- Check `/ingestion/status` for the latest attempt and last successful run.

### Ingestion status limitations

Ingestion status is currently held in backend process memory. It resets when
the backend restarts and is not shared across multiple backend workers. Use a
single backend worker until status and ingestion scheduling are moved to
persistent/shared infrastructure.

## Notes

This project is designed for a live thermal-risk monitoring workflow and is best run with a valid Supabase environment and NASA FIRMS access key.

## License

This project is for internal/demo use unless otherwise specified by the repository owner.

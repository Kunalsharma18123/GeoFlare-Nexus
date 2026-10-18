import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.db import engine
from backend.repository import insert_observation


BASE_DIR = Path(__file__).resolve().parents[1]
LIVE_DATA_PATH = BASE_DIR / "backend" / "data" / "firms_live_features.csv"


def make_observation_key(row) -> str:
    raw = "|".join(
        [
            str(row["satellite"]),
            str(row["instrument"]),
            str(row["acq_datetime"]),
            f"{float(row['latitude']):.6f}",
            f"{float(row['longitude']):.6f}",
        ]
    )

    digest = hashlib.sha256(raw.encode()).hexdigest()

    return f"FIRMS:{digest}"


def parse_datetime(value) -> datetime:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)

    return timestamp.to_pydatetime()


def main():
    df = pd.read_csv(LIVE_DATA_PATH)

    print(f"Loaded {len(df)} FIRMS observations")

    inserted = 0
    duplicates = 0

    with Session(engine) as session:
        for _, row in df.iterrows():
            observation_key = make_observation_key(row)

            observation, was_inserted = insert_observation(
                session=session,
                observation_key=observation_key,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                acq_datetime=parse_datetime(row["acq_datetime"]),
                satellite=row.get("satellite"),
                instrument=row.get("instrument"),
                frp=float(row["frp"]) if pd.notna(row.get("frp")) else None,
                confidence_score=(
                    float(row["confidence_score"])
                    if pd.notna(row.get("confidence_score"))
                    else None
                ),
            )

            if was_inserted:
                inserted += 1
            else:
                duplicates += 1

    print(f"Inserted: {inserted}")
    print(f"Duplicates: {duplicates}")


if __name__ == "__main__":
    main()
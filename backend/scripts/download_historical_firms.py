import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise RuntimeError("FIRMS_MAP_KEY is not set in .env")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"
HISTORICAL_DIR = DATA_DIR / "historical"

HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)


SOURCE = "VIIRS_NOAA20_SP"

AREA = "68.1,6.5,97.4,37.1"

START_DATE = date(2026, 5, 1)
END_DATE = date(2026, 5, 31)

current_date = START_DATE
files = []


print("GeoFlare - Historical FIRMS downloader")
print(f"Source: {SOURCE}")
print(f"Period: {START_DATE} to {END_DATE}")
print()


while current_date <= END_DATE:
    remaining_days = (END_DATE - current_date).days + 1
    day_range = min(5, remaining_days)

    date_string = current_date.isoformat()

    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        f"api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{day_range}/{date_string}"
    )

    print(
        f"Fetching {date_string} "
        f"through {(current_date + timedelta(days=day_range - 1)).isoformat()}..."
    )

    response = requests.get(url, timeout=120)

    if not response.ok:
        print(f"Request failed with status {response.status_code}")
        print(response.text)
        raise SystemExit(1)

    output_file = HISTORICAL_DIR / f"firms_{date_string}.csv"
    output_file.write_bytes(response.content)

    df = pd.read_csv(output_file)

    print(f"  Records: {len(df):,}")

    files.append(output_file)

    current_date += timedelta(days=day_range)


print()
print("Combining historical files...")

frames = []

for file in files:
    frames.append(pd.read_csv(file))

historical = pd.concat(frames, ignore_index=True)

before_dedup = len(historical)

historical = historical.drop_duplicates()

after_dedup = len(historical)

output_file = DATA_DIR / "firms_historical_raw.csv"

historical.to_csv(output_file, index=False)

print()
print(f"Total records: {after_dedup:,}")
print(f"Duplicates removed: {before_dedup - after_dedup:,}")
print(f"Saved: {output_file}")
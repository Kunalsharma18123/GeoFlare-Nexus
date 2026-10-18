import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise RuntimeError("FIRMS_MAP_KEY is not set in .env")


url = (
    "https://firms.modaps.eosdis.nasa.gov/"
    f"api/data_availability/csv/{MAP_KEY}/VIIRS_NOAA20_SP"
)

print("Checking FIRMS NOAA-20 standard data availability...")

response = requests.get(url, timeout=60)

if not response.ok:
    print(f"Request failed with status {response.status_code}")
    print(response.text)
    raise SystemExit(1)

df = pd.read_csv(response.url)

print()
print(df.to_string(index=False))
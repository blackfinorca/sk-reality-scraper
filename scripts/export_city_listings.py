#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) < 3:
    print(json.dumps({"error": "usage: export_city_listings.py <city> <parquet_path>"}))
    sys.exit(1)

city = sys.argv[1].strip().lower()
parquet_path = Path(sys.argv[2]).expanduser().resolve()

if not parquet_path.exists():
    print(json.dumps({"error": f"Parquet file not found: {parquet_path}"}))
    sys.exit(1)

try:
    df = pd.read_parquet(parquet_path)
except Exception as exc:  # pragma: no cover
    print(json.dumps({"error": f"Failed reading parquet: {exc}"}))
    sys.exit(1)

if city:
    df = df[df["address_town"].fillna("").str.lower() == city]

records = df.to_dict(orient="records")
print(json.dumps(records, ensure_ascii=False))

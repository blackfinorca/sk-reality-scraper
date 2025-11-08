#!/usr/bin/env python3
"""
Export listings from the latest gold parquet snapshot as JSON.

This helper keeps the dev frontend decoupled from parquet internals by exposing
just the raw rows (no transformation).  Consumers can optionally restrict the
result set by city or limit the number of returned records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES: Iterable[Path] = (
    Path("dedupe_runs/latest/gold_listings.parquet"),
    Path("b2/realestate/gold/gold_listings_latest.parquet"),
    Path("parquet_runs/latest/gold_listings.parquet"),
)


def resolve_parquet_path(explicit: Optional[str]) -> Path:
    """Return the parquet path to use, preferring an explicit input."""

    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Parquet file not found: {candidate}")
        return candidate

    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No gold parquet snapshot found. Looked for: "
        + ", ".join(str(path) for path in DEFAULT_CANDIDATES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gold listings parquet to JSON.")
    parser.add_argument("--parquet-path", help="Custom parquet path (defaults to latest snapshot).")
    parser.add_argument("--limit", type=int, help="Limit number of returned rows.")
    parser.add_argument(
        "--city",
        help="Optional city filter (case-insensitive). "
        "Use 'all' or omit to export every record.",
    )
    args = parser.parse_args()

    parquet_path = resolve_parquet_path(args.parquet_path)
    df = pd.read_parquet(parquet_path)

    city = (args.city or "").strip().lower()
    if city and city not in {"all", "*"}:
        df = df[df["address_town"].fillna("").str.lower() == city]

    total_rows = int(df.shape[0])
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    def _to_serialisable(value):
        if isinstance(value, float) and (value != value):  # NaN check
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_to_serialisable(elem) for elem in value.tolist()]
        if isinstance(value, list):
            return [_to_serialisable(elem) for elem in value]
        if isinstance(value, tuple):
            return [_to_serialisable(elem) for elem in value]
        if isinstance(value, dict):
            return {key: _to_serialisable(val) for key, val in value.items()}
        return value

    records = []
    for row in df.to_dict(orient="records"):
        records.append({key: _to_serialisable(value) for key, value in row.items()})

    payload = {
        "data": records,
        "meta": {
            "total_rows": total_rows,
            "returned_rows": int(df.shape[0]),
            "parquet_path": str(parquet_path),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

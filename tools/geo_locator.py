"""Geocode missing coordinates for golden listings using Nominatim."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from requests import Response
from tqdm import tqdm


class RateLimiter:
    """Simple rate limiter ensuring a minimum interval between events."""

    def __init__(self, min_interval_sec: float = 1.0) -> None:
        self.min_interval = max(0.0, float(min_interval_sec))
        self._last_time: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_time is not None and self.min_interval > 0:
            elapsed = now - self._last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last_time = time.monotonic()


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def ensure_lat_lng_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lat/lng columns exist and are float dtype."""
    if "lat" not in df.columns:
        df["lat"] = pd.NA
    if "lng" not in df.columns:
        df["lng"] = pd.NA
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    return df


def build_query(address_street: Optional[str], address_town: Optional[str], country: str) -> Optional[str]:
    """Construct the geocoding query string."""
    parts = [
        part.strip()
        for part in [address_street or "", address_town or "", country or ""]
        if part and part.strip()
    ]
    if not parts:
        return None
    return ", ".join(parts)


def cache_key(address_street: Optional[str], address_town: Optional[str], country: str) -> str:
    """Return the normalized cache key."""
    street = (address_street or "").strip().lower()
    town = (address_town or "").strip().lower()
    cntry = (country or "").strip().lower()
    return f"{street}|{town}|{cntry}"


def cache_load(path: str) -> Dict[str, Dict[str, object]]:
    """Load cache data from JSON file."""
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        logging.warning("Failed to load cache from %s; starting with empty cache.", path)
    return {}


def cache_save(path: str, cache: Dict[str, Dict[str, object]]) -> None:
    """Persist cache to disk."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)


def _parse_response(resp: Response) -> Optional[Tuple[float, float]]:
    try:
        resp.raise_for_status()
        data = resp.json()
    except (ValueError, requests.HTTPError):
        return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    try:
        lat = float(first["lat"])
        lon = float(first["lon"])
    except (KeyError, ValueError, TypeError):
        return None
    return lat, lon


def geocode_nominatim(query: str, user_agent: str, rl: RateLimiter) -> Optional[Tuple[float, float]]:
    """Call Nominatim to geocode a query."""
    rl.wait()
    params = {"format": "json", "limit": 1, "q": query}
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=20)
    except requests.RequestException:
        return None
    return _parse_response(response)


def fill_coordinates(
    df: pd.DataFrame,
    country: str,
    user_agent: str,
    cache_path: str,
    max_new: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Fills missing lat/lng using cache + Nominatim, respecting max_new.
    Returns (df_with_coords, stats dict).
    Stats keys: total, with_coords, missing_coords, cache_hits, geocoded_now, failures, capped.
    """

    working = ensure_lat_lng_columns(df.copy())
    cache = cache_load(cache_path)
    rl = RateLimiter(1.0)

    stats = {
        "total": len(working),
        "with_coords": int((working["lat"].notna() & working["lng"].notna()).sum()),
        "missing_coords": 0,
        "cache_hits": 0,
        "geocoded_now": 0,
        "failures": 0,
        "capped": False,
    }

    new_calls = 0
    rows_to_process = working[working["lat"].isna() | working["lng"].isna()]

    for idx, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Geocoding", unit="listing"):
        street = row.get("address_street")
        town = row.get("address_town")

        query = build_query(street, town, country)
        if not query:
            stats["failures"] += 1
            continue

        key = cache_key(street, town, country)
        cached = cache.get(key)
        lat = None
        lng = None
        if cached is not None:
            try:
                lat = float(cached["lat"])
                lng = float(cached["lng"])
                stats["cache_hits"] += 1
            except (KeyError, ValueError, TypeError):
                lat = lng = None

        if lat is None or lng is None:
            if max_new is not None and new_calls >= max_new:
                stats["capped"] = True
                stats["failures"] += 1
                continue
            coords = geocode_nominatim(query, user_agent, rl)
            if coords is None:
                stats["failures"] += 1
                continue
            lat, lng = coords
            cache[key] = {"lat": lat, "lng": lng, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            stats["geocoded_now"] += 1
            new_calls += 1

        working.at[idx, "lat"] = lat
        working.at[idx, "lng"] = lng

    stats["with_coords"] = int((working["lat"].notna() & working["lng"].notna()).sum())
    stats["missing_coords"] = stats["total"] - stats["with_coords"]

    cache_save(cache_path, cache)
    return working, stats


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to Parquet with snappy compression."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode missing coordinates in golden listings.")
    parser.add_argument("--parquet", required=True, help="Input Parquet path.")
    parser.add_argument("--out-parquet", help="Output Parquet path (required unless --in-place).")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input Parquet.")
    parser.add_argument("--cache", default="data/geocode_cache.json", help="Cache JSON path.")
    parser.add_argument("--country", default="Slovakia", help="Country to include in queries.")
    parser.add_argument("--user-agent", required=True, help="User-Agent header for Nominatim requests.")
    parser.add_argument("--max-new", type=int, default=None, help="Maximum number of new geocode API calls.")
    parser.add_argument("--dry-run", default="false", help="If true, do not write output.")
    args = parser.parse_args()

    dry_run = _parse_bool(str(args.dry_run))
    if not args.in_place and not args.out_parquet:
        parser.error("--out-parquet is required unless --in-place is specified.")

    input_path = args.parquet
    output_path = args.parquet if args.in_place else args.out_parquet

    df = load_parquet(input_path)
    df, stats = fill_coordinates(
        df,
        country=args.country,
        user_agent=args.user_agent,
        cache_path=args.cache,
        max_new=args.max_new,
    )

    report = (
        f"Rows: {stats['total']}  | With coords: {stats['with_coords']}  | Missing: {stats['missing_coords']}\n"
        f"Cache hits: {stats['cache_hits']}  | Geocoded now: {stats['geocoded_now']}  | Failures: {stats['failures']}  | Capped: {str(stats['capped']).lower()}"
    )
    print(report)

    if dry_run:
        print("Dry-run mode enabled; not writing output Parquet.")
        return

    if output_path:
        write_parquet(df, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

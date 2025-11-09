"""Aggregate rent listings (prenájom) stats by city, room type, and geo clusters."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_PATTERNS = [
    "output/*prenajom*_output.csv",
    "output/*_prenajom_output.csv",
    "output/*prenajom*.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize rental listings.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_PATTERNS,
        help="CSV glob pattern(s) to scan for prenajom listings (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="output/rental_summary.json",
        help="Path to write JSON summary (default: %(default)s).",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.0,
        help="Ignore listings with price below this threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print summary to stdout as well as writing to file.",
    )
    return parser.parse_args()


def resolve_paths(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        expanded = glob(pattern)
        if not expanded:
            logging.debug("Pattern %s matched no files.", pattern)
        for path_str in expanded:
            candidate = Path(path_str)
            if candidate.exists():
                files.append(candidate)
    unique_files = sorted(set(files))
    return unique_files


def normalize_transaction(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def derive_city(row: pd.Series) -> str:
    for col in ("address_town", "address_norm", "address_ascii", "city"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown"


def derive_room_type(value: object) -> str:
    number = None
    if value is not None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
    if number is None or pd.isna(number):
        return "Unknown"
    bedrooms = max(1, int(round(number)))
    if bedrooms == 1:
        label = "1-izbový"
    else:
        label = f"{bedrooms}-izbový"
    return label


def derive_cluster_key(row: pd.Series) -> str:
    lat = row.get("lat") or row.get("latitude")
    lng = row.get("lng") or row.get("longitude")
    try:
        lat_val = float(lat)
        lng_val = float(lng)
        if not (pd.isna(lat_val) or pd.isna(lng_val)):
            return f"geo:{lat_val:.3f}|{lng_val:.3f}"
    except (TypeError, ValueError):
        pass

    street = ""
    for col in ("address_norm", "address_ascii", "address_street"):
        candidate = row.get(col)
        if isinstance(candidate, str) and candidate.strip():
            street = candidate.strip().lower()
            break
    city = derive_city(row).lower()
    if street:
        return f"text:{city}|{street}"
    return f"city:{city}"


def aggregate_stats(df: pd.DataFrame) -> Dict[str, object]:
    df = df.copy()
    df["city"] = df.apply(derive_city, axis=1)

    if "room_number" in df.columns:
        base_rooms = df["room_number"]
    elif "rooms" in df.columns:
        base_rooms = df["rooms"]
    else:
        base_rooms = None

    if base_rooms is not None:
        numeric_rooms = pd.to_numeric(base_rooms, errors="coerce")
        df["room_number_label"] = numeric_rooms.apply(derive_room_type)
    else:
        df["room_number_label"] = "Unknown"

    df["cluster_key"] = df.apply(derive_cluster_key, axis=1)
    lat_series = coerce_numeric(df.get("lat", pd.Series([pd.NA] * len(df))))
    lng_series = coerce_numeric(df.get("lng", pd.Series([pd.NA] * len(df))))
    df["lat_num"] = lat_series
    df["lng_num"] = lng_series

    def _fmt(value: object) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 2)

    city_stats: List[Dict[str, object]] = []
    for city, subset in df.groupby("city"):
        prices = subset["price"].dropna()
        if prices.empty:
            continue
        clean_city = city if (isinstance(city, str) and city) else "Unknown"
        city_stats.append(
            {
                "city": clean_city,
                "avg_price": _fmt(prices.mean()),
                "median_price": _fmt(prices.median()),
                "min_price": _fmt(prices.min()),
                "max_price": _fmt(prices.max()),
                "listings": int(len(subset)),
            }
        )
    city_stats.sort(key=lambda item: (item["avg_price"] is None, item["avg_price"]))

    def summarize_group(group: pd.DataFrame, label_field: str) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        grouped = group.groupby(label_field)
        for key, subset in grouped:
            clean_key = key if (isinstance(key, str) and key) else "Unknown"
            avg_price = subset["price"].mean()
            results.append(
                {
                    label_field: clean_key,
                    "avg_price": _fmt(avg_price),
                    "listings": int(len(subset)),
                }
            )
        results.sort(key=lambda item: (item["avg_price"] is None, item["avg_price"]))
        return results

    room_stats = summarize_group(df, "room_number_label")

    cluster_records: List[Dict[str, object]] = []
    for key, subset in df.groupby("cluster_key"):
        if len(subset) < 3:
            continue
        avg_price = subset["price"].mean()
        cluster_records.append(
            {
                "cluster": key,
                "city": subset["city"].mode().iloc[0] if not subset["city"].mode().empty else "Unknown",
                "avg_price": _fmt(avg_price),
                "listings": int(len(subset)),
                "center_lat": round(float(subset["lat_num"].mean()), 6)
                if subset["lat_num"].notna().any()
                else None,
                "center_lng": round(float(subset["lng_num"].mean()), 6)
                if subset["lng_num"].notna().any()
                else None,
            }
        )
    cluster_records.sort(key=lambda item: (item["avg_price"] is None, item["avg_price"]))

    return {
        "city_averages": city_stats,
        "room_number_averages": room_stats,
        "cluster_averages": cluster_records,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    files = resolve_paths(args.inputs)
    if not files:
        raise FileNotFoundError(f"No CSV files found for patterns: {args.inputs}")

    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Failed to read %s: %s", path, exc)
            continue
        frame["__source_path__"] = str(path)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No CSV data could be loaded.")

    df = pd.concat(frames, ignore_index=True)
    if "transaction" in df.columns:
        transaction_mask = df["transaction"].apply(normalize_transaction) == "prenajom"
        df = df[transaction_mask]
    if df.empty:
        raise RuntimeError("No prenajom listings found in provided CSV files.")

    df["price"] = coerce_numeric(df.get("price"))
    df = df[df["price"].notna()]
    if args.min_price > 0:
        df = df[df["price"] >= args.min_price]

    stats = aggregate_stats(df)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path) for path in files],
        "total_listings": int(len(df)),
        "filters": {"min_price": args.min_price, "transaction": "prenajom"},
        "averages": stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logging.info("Rental summary written to %s (listings: %s)", output_path, len(df))

    if args.do_print:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


SILVER_COLUMNS = [
    "listing_uid",
    "site",
    "transaction",
    "url",
    "title",
    "address_street",
    "city",
    "region",
    "rooms",
    "floor_area_m2",
    "status",
    "year_built",
    "top_year",
    "floor_raw",
    "energy_cert",
    "price_eur",
    "price_psm_eur",
    "currency",
    "valid_from",
]

LATEST_EXTRA_COLUMNS = ["hash", "first_seen", "last_seen", "is_current"]
HISTORY_EXTRA_COLUMNS = ["hash", "valid_to", "is_current"]
HASH_COLUMNS = [
    "transaction",
    "title",
    "address_street",
    "city",
    "rooms",
    "floor_area_m2",
    "status",
    "year_built",
    "top_year",
    "energy_cert",
    "price_eur",
    "price_psm_eur",
]


def make_city_slug(value: Any) -> str:
    cleaned = clean_string(value)
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^\w]+", "_", cleaned.lower())
    return cleaned.strip("_")


def is_city_close_match(city_value: Any, target_slug: str) -> bool:
    cleaned_slug = make_city_slug(city_value)
    if not cleaned_slug:
        return False
    if cleaned_slug == target_slug:
        return True
    if target_slug in cleaned_slug or cleaned_slug in target_slug:
        return True
    ratio = SequenceMatcher(None, cleaned_slug, target_slug).ratio()
    return ratio >= 0.8


def normalise_bucket_path(bucket: str) -> Path:
    """Translate a bucket URI into a local filesystem path."""
    bucket_sanitised = bucket.replace("://", "/")
    return Path(bucket_sanitised)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalise_numeric_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    cleaned = series.astype(str).str.replace("\xa0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace("\u202f", "", regex=False)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    cleaned = cleaned.replace({"": np.nan})
    return cleaned


def to_int(series: pd.Series) -> pd.Series:
    normalised = normalise_numeric_series(series)
    coerced = pd.to_numeric(normalised, errors="coerce").round()
    return coerced.astype("Int64")


def to_float(series: pd.Series) -> pd.Series:
    normalised = normalise_numeric_series(series)
    coerced = pd.to_numeric(normalised, errors="coerce")
    return coerced.astype(float)


def compute_listing_uid(row: Dict[str, Any]) -> str:
    property_id = clean_string(row.get("property_id"))
    if property_id:
        return property_id
    fallback_parts = [
        clean_string(row.get("link")).lower(),
        clean_string(row.get("property_name")).lower(),
        clean_string(row.get("address_street")).lower(),
        clean_string(row.get("address_town")).lower(),
    ]
    fallback_str = "|".join(fallback_parts).strip("|")
    if not fallback_str:
        fallback_str = hashlib.sha256(f"fallback|{row.get('link')}|{row.get('property_name')}".encode("utf-8")).hexdigest()
    return hashlib.sha256(fallback_str.encode("utf-8")).hexdigest()


def compute_hash(row: Dict[str, Any]) -> str:
    normalised_parts: List[str] = []
    for column in HASH_COLUMNS:
        value = row.get(column)
        if pd.isna(value):
            normalised_parts.append("")
            continue
        if isinstance(value, float):
            normalised_parts.append(f"{value:.6f}")
        else:
            normalised_parts.append(clean_string(value).lower())
    digest_input = "|".join(normalised_parts)
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def to_python(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        if np.isnan(value):
            return None
        return float(value)
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def gzip_copy(src: Path, dst: Path) -> None:
    with open(src, "rb") as source_file, gzip.open(dst, "wb") as gz_file:
        shutil.copyfileobj(source_file, gz_file)


def prepare_silver(df: pd.DataFrame, site: str, currency: str, date_str: str) -> pd.DataFrame:
    working = df.copy()
    for column in ["transaction", "property_id", "link", "property_name", "address_street", "address_town", "address_zrea", "building_status", "floor", "energ_cert"]:
        if column in working.columns:
            working[column] = working[column].map(clean_string)

    working["listing_uid"] = working.apply(compute_listing_uid, axis=1)
    working["site"] = site
    if "transaction" not in working.columns:
        working["transaction"] = ""
    working["transaction"] = working["transaction"].astype(str)
    working["url"] = working["link"]
    working["title"] = working["property_name"]
    working["city"] = working["address_town"]
    working["region"] = working["address_zrea"]
    working["rooms"] = to_int(working.get("room_number", pd.Series(dtype="Int64")))
    working["floor_area_m2"] = to_float(working.get("floor_area", pd.Series(dtype=float)))
    working.loc[working["floor_area_m2"] <= 0, "floor_area_m2"] = np.nan
    working["status"] = working["building_status"]
    working["year_built"] = to_int(working.get("year_of_construction", pd.Series(dtype="Int64")))
    working["top_year"] = to_int(working.get("year_of_top", pd.Series(dtype="Int64")))
    working["floor_raw"] = working["floor"]
    working["energy_cert"] = working["energ_cert"]
    working["price_eur"] = to_float(working.get("price", pd.Series(dtype=float)))
    working.loc[working["price_eur"] <= 0, "price_eur"] = np.nan
    price_area = to_float(working.get("price_area", pd.Series(dtype=float)))
    price_area = price_area.replace({0.0: np.nan})
    working["price_psm_eur"] = price_area
    need_calc = working["price_psm_eur"].isna() & working["price_eur"].notna() & working["floor_area_m2"].notna() & (working["floor_area_m2"] > 0)
    working.loc[need_calc, "price_psm_eur"] = working.loc[need_calc, "price_eur"] / working.loc[need_calc, "floor_area_m2"]
    working["currency"] = currency
    working["valid_from"] = date_str

    silver = working[SILVER_COLUMNS].copy()
    silver = silver.drop_duplicates(subset=["listing_uid"], keep="first").reset_index(drop=True)
    silver["rooms"] = silver["rooms"].astype("Int64")
    silver["year_built"] = silver["year_built"].astype("Int64")
    silver["top_year"] = silver["top_year"].astype("Int64")
    return silver


def load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return pd.DataFrame()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False, compression="zstd")
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Failed to write parquet file {path}: {exc}\n"
            "Ensure that a compatible pyarrow or fastparquet installation is available."
        ) from exc


def upsert_gold(
    silver: pd.DataFrame,
    gold_latest_path: Path,
    gold_history_path: Path,
    date_str: str,
) -> Dict[str, int]:
    latest_columns = SILVER_COLUMNS + LATEST_EXTRA_COLUMNS
    history_columns = SILVER_COLUMNS + HISTORY_EXTRA_COLUMNS

    if gold_latest_path.exists():
        latest_df = pd.read_parquet(gold_latest_path)
    else:
        latest_df = pd.DataFrame(columns=latest_columns)

    if gold_history_path.exists():
        history_df = pd.read_parquet(gold_history_path)
    else:
        history_df = pd.DataFrame(columns=history_columns)

    if "hash" not in silver.columns:
        silver_with_hash = silver.copy()
        silver_with_hash["hash"] = silver.apply(lambda row: compute_hash(row.to_dict()), axis=1)
    else:
        silver_with_hash = silver.copy()

    silver_with_hash["first_seen"] = date_str
    silver_with_hash["last_seen"] = date_str
    silver_with_hash["is_current"] = True

    latest_df = latest_df.reindex(columns=latest_columns)
    history_df = history_df.reindex(columns=history_columns)

    latest_df = latest_df.set_index("listing_uid", drop=False)
    history_df = history_df.set_index("listing_uid", drop=False)

    upserts_latest = 0
    new_history_rows = 0
    history_appends: List[Dict[str, Any]] = []

    for row in silver_with_hash.to_dict(orient="records"):
        listing_uid = row["listing_uid"]
        row = {key: to_python(value) for key, value in row.items()}
        if listing_uid in latest_df.index:
            previous = latest_df.loc[listing_uid].to_dict()
            if previous.get("hash") == row["hash"]:
                latest_df.at[listing_uid, "last_seen"] = date_str
                continue

            # close existing history row if present
            if listing_uid in history_df.index:
                current_active = history_df.loc[[listing_uid]]
                active_mask = current_active["is_current"] == True  # noqa: E712
                indices = current_active[active_mask].index
                for idx in indices:
                    history_df.at[idx, "valid_to"] = date_str
                    history_df.at[idx, "is_current"] = False
            else:
                previous_history_payload = {col: previous.get(col) for col in SILVER_COLUMNS}
                previous_history_payload.update({"hash": previous.get("hash"), "valid_to": date_str, "is_current": False})
                previous_history_payload["listing_uid"] = listing_uid
                history_appends.append(previous_history_payload)

            row["first_seen"] = previous.get("first_seen", date_str)
            row["last_seen"] = date_str
            row["is_current"] = True
            latest_df.loc[listing_uid, latest_columns] = [row.get(col) for col in latest_columns]

            history_payload = {col: row.get(col) for col in SILVER_COLUMNS}
            history_payload.update({"hash": row["hash"], "valid_to": None, "is_current": True})
            history_payload["listing_uid"] = listing_uid
            history_appends.append(history_payload)
            upserts_latest += 1
            new_history_rows += 1
        else:
            latest_df.loc[listing_uid, latest_columns] = [row.get(col) for col in latest_columns]
            history_payload = {col: row.get(col) for col in SILVER_COLUMNS}
            history_payload.update({"hash": row["hash"], "valid_to": None, "is_current": True})
            history_payload["listing_uid"] = listing_uid
            history_appends.append(history_payload)
            upserts_latest += 1
            new_history_rows += 1

    if history_appends:
        history_new_df = pd.DataFrame(history_appends)
        history_new_df = history_new_df.reindex(columns=history_columns)
        if history_df.empty:
            history_df = history_new_df
        else:
            history_df = pd.concat(
                [history_df.reset_index(drop=True), history_new_df],
                ignore_index=True,
                sort=False,
            )
            history_df = history_df.reindex(columns=history_columns)

    latest_df = latest_df.reset_index(drop=True)
    history_df = history_df.reset_index(drop=True)

    write_parquet(latest_df, gold_latest_path)
    write_parquet(history_df, gold_history_path)

    return {
        "upserts_latest": upserts_latest,
        "new_history_rows": new_history_rows,
        "latest_count": len(latest_df),
        "history_count": len(history_df),
    }


def build_city_exports(latest_df: pd.DataFrame, exports_root: Path, date_str: str) -> None:
    if latest_df.empty:
        return
    current = latest_df[latest_df["is_current"]].copy()
    if current.empty:
        return

    current["city_slug"] = current["city"].map(make_city_slug)
    current = current[current["city_slug"].astype(bool)]

    for city_value, group in current.groupby("city_slug"):
        city_dir = exports_root / "listings_current" / f"city={city_value}"
        filtered = group[group["city"].map(lambda c: is_city_close_match(c, city_value))]
        if filtered.empty:
            json_path = city_dir / f"{date_str}.json"
            if json_path.exists():
                json_path.unlink()
            try:
                city_dir.rmdir()
            except OSError:
                pass
            continue

        records = []
        for record in filtered.to_dict(orient="records"):
            payload = {
                "listing_uid": record["listing_uid"],
                "title": record["title"],
                "url": record["url"],
                "city": record["city"],
                "region": record["region"],
                "rooms": to_python(record["rooms"]),
                "floor_area_m2": to_python(record["floor_area_m2"]),
                "price_eur": to_python(record["price_eur"]),
                "price_psm_eur": to_python(record["price_psm_eur"]),
                "status": record["status"],
                "energy_cert": record["energy_cert"],
                "last_seen": record["last_seen"],
            }
            records.append(payload)

        ensure_directory(city_dir)
        json_path = city_dir / f"{date_str}.json"
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(records, json_file, ensure_ascii=False, indent=2)


def build_market_stats(latest_df: pd.DataFrame, exports_root: Path, date_str: str, site: str, currency: str) -> None:
    stats_dir = exports_root / "market_stats_daily"
    ensure_directory(stats_dir)

    current = latest_df[latest_df["is_current"]].copy()
    overall = {
        "date": date_str,
        "site": site,
        "currency": currency,
        "total_listings": int(current.shape[0]),
        "avg_price_eur": float(current["price_eur"].mean()) if not current["price_eur"].dropna().empty else None,
        "median_price_eur": float(current["price_eur"].median()) if not current["price_eur"].dropna().empty else None,
        "avg_price_psm_eur": float(current["price_psm_eur"].mean()) if not current["price_psm_eur"].dropna().empty else None,
        "median_price_psm_eur": float(current["price_psm_eur"].median()) if not current["price_psm_eur"].dropna().empty else None,
    }

    city_metrics: List[Dict[str, Any]] = []
    for city_value, group in current.groupby("city"):
        cleaned_city = clean_string(city_value)
        if not cleaned_city:
            cleaned_city = "unknown"
        city_metrics.append(
            {
                "city": cleaned_city,
                "count": int(group.shape[0]),
                "avg_price_eur": float(group["price_eur"].mean()) if not group["price_eur"].dropna().empty else None,
                "median_price_eur": float(group["price_eur"].median()) if not group["price_eur"].dropna().empty else None,
                "avg_price_psm_eur": float(group["price_psm_eur"].mean()) if not group["price_psm_eur"].dropna().empty else None,
                "median_price_psm_eur": float(group["price_psm_eur"].median()) if not group["price_psm_eur"].dropna().empty else None,
                "avg_floor_area_m2": float(group["floor_area_m2"].mean()) if not group["floor_area_m2"].dropna().empty else None,
            }
        )

    output = overall
    output["cities"] = city_metrics

    json_path = stats_dir / f"{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as stats_file:
        json.dump(output, stats_file, ensure_ascii=False, indent=2)


def run_etl(date_str: str, site: str, csv_path: Path, bucket: Path, currency: str) -> Dict[str, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    raw_dir = bucket / "raw" / f"site={site}" / f"dt={date_str}"
    silver_dir = bucket / "silver" / f"site={site}" / f"dt={date_str}"
    gold_dir = bucket / "gold"
    exports_dir = bucket / "exports"

    for directory in (raw_dir, silver_dir, gold_dir, exports_dir):
        ensure_directory(directory)

    raw_gzip_path = raw_dir / "listings.csv.gz"
    gzip_copy(csv_path, raw_gzip_path)

    source_df = pd.read_csv(csv_path, dtype=str).fillna("")
    rows_in_csv = int(source_df.shape[0])

    silver_df = prepare_silver(source_df, site=site, currency=currency, date_str=date_str)
    rows_silver = int(silver_df.shape[0])

    silver_path = silver_dir / "part-000.parquet"
    write_parquet(silver_df, silver_path)

    gold_latest_path = gold_dir / "gold_listings_latest.parquet"
    gold_history_path = gold_dir / "gold_listings_history.parquet"
    gold_stats = upsert_gold(silver_df, gold_latest_path, gold_history_path, date_str)

    latest_df = pd.read_parquet(gold_latest_path)

    build_city_exports(latest_df, exports_dir, date_str)
    build_market_stats(latest_df, exports_dir, date_str, site, currency)

    summary = {
        "rows_in_csv": rows_in_csv,
        "rows_silver": rows_silver,
        "upserts_latest": gold_stats["upserts_latest"],
        "new_history_rows": gold_stats["new_history_rows"],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real estate ETL pipeline.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--currency", default="EUR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bucket_path = normalise_bucket_path(args.bucket)
    csv_path = Path(args.csv_path)
    summary = run_etl(args.date, args.site, csv_path, bucket_path, args.currency)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

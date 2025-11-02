import argparse
import glob
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PRIORITY = ["nehnutelnosti_sk", "reality_sk"]
MIN_SIZE = 10.0
MAX_SIZE = 1000.0
MIN_PRICE = 10_000
SIZE_MERGE_TOL = 0.05
SIZE_SCORE_TOL_STRICT = 0.02
SIZE_SCORE_TOL_LOOSE = 0.03
PRICE_TOL = 0.05
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()


@dataclass
class UnionFind:
    parent: List[int]
    rank: List[int]

    @classmethod
    def create(cls, size: int) -> "UnionFind":
        return cls(list(range(size)), [0] * size)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and merge Slovak property listings.")
    parser.add_argument("--input", required=True, help="Glob pattern for input CSV files")
    parser.add_argument("--out-dir", required=True, help="Output directory for Parquet snapshots")
    parser.add_argument("--run-date", required=True, help="Run date in YYYY-MM-DD format")
    parser.add_argument(
        "--source-priority", default=",".join(DEFAULT_PRIORITY), help="Comma list of portal priority"
    )
    parser.add_argument("--prev-run", default=None, help="Previous run directory (optional)")
    return parser.parse_args()


def load_raw(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files match pattern: {pattern}")
    frames = []
    for path in files:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["__source_path"] = path
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.index.name = "row_id"
    return combined.reset_index()


def ascii_fold(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_transaction(value: str) -> str:
    value = value.strip().lower()
    if value == "predaj":
        return "sale"
    return value or "unknown"


def parse_int(value: str, allow_zero: bool = False) -> Optional[int]:
    clean = re.sub(r"[^0-9]", "", value or "")
    if not clean:
        return None
    num = int(clean)
    if not allow_zero and num == 0:
        return None
    return num


def clamp_price(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value < MIN_PRICE:
        return None
    return value


def parse_float(value: str) -> Optional[float]:
    clean = value.replace(" ", "").replace("\xa0", "").replace(",", ".")
    clean = re.sub(r"[^0-9.]+", "", clean)
    if not clean:
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def clamp_size(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < MIN_SIZE or value > MAX_SIZE:
        return None
    return value


def split_zrea(zrea: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not zrea:
        return None, None
    text = zrea.strip()
    if not text:
        return None, None
    district = None
    region = None
    district_match = re.search(r"okres\s+([\w\s-]+)", text, flags=re.IGNORECASE)
    if district_match:
        district = clean_whitespace(district_match.group(1)).title()
    region_match = re.search(r"([\w\s-]+)\s+kraj", text, flags=re.IGNORECASE)
    if region_match:
        region = clean_whitespace(region_match.group(1)).title()
    return district, region


def infer_rooms_from_title(title: str) -> Optional[int]:
    match = re.search(r"(\d)[-\s]?izb", title.lower())
    if match:
        return int(match.group(1))
    return None


def parse_floor(value: str) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
    if not value:
        return None, None, None
    text = value.lower()
    elevator = None
    if "výťah" in text or "vytah" in text:
        elevator = True
    clean = re.sub(r"\+.*", "", text)
    clean = re.sub(r"výťah|vytah", "", clean)
    clean = clean.strip()
    match = re.search(r"(\d+)\s*/\s*(\d+)", clean)
    if match:
        return int(match.group(1)), int(match.group(2)), elevator
    single = re.search(r"\d+", clean)
    if single:
        return int(single.group()), None, elevator
    return None, None, elevator


def normalize_building_status(value: str) -> str:
    mapping = {
        "novostavba": "new",
        "kompletná rekonštrukcia": "renovated-full",
        "kompletna rekonstrukcia": "renovated-full",
        "čiastočná rekonštrukcia": "renovated-partial",
        "ciastocna rekonstrukcia": "renovated-partial",
    }
    norm = clean_whitespace(value.lower()) if value else ""
    return mapping.get(norm, "unknown")


def normalize_energ_cert(value: str) -> str:
    if not value:
        return "none"
    norm = value.strip().lower()
    if "nie" in norm or norm == "":
        return "none"
    letter = norm[0].upper()
    if letter in "ABCDEFG":
        return letter
    return "unknown"


def extract_photo_tokens(row: pd.Series) -> List[str]:
    tokens: List[str] = []
    for col in ["photo_url_1", "photo_url_2", "photo_url_3"]:
        url = row.get(col, "") or ""
        if not url:
            continue
        token = extract_token_from_url(url)
        if token:
            tokens.append(token)
    return sorted(set(tokens))


def extract_token_from_url(url: str) -> Optional[str]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path
    if not path:
        return None
    name = os.path.basename(path)
    name = name.split("?")[0]
    name = name.replace("_fss", "")
    token = re.sub(r"[^A-Za-z0-9]+", "", name)
    return token or None


def normalize(df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        source = row.get("source", "").strip().lower()
        transaction = normalize_transaction(row.get("transaction", ""))
        property_id = row.get("property_id", "").strip()
        link = row.get("link", "").strip()
        name = clean_whitespace(row.get("property_name", ""))

        price = clamp_price(parse_int(row.get("price", "")))
        size_raw = clamp_size(parse_float(row.get("floor_area", "")))
        if size_raw is None:
            size_raw = None

        street = clean_whitespace(re.sub(r"^Ulica\s+", "", row.get("address_street", ""), flags=re.IGNORECASE))
        if not street:
            street = None
        town_raw = clean_whitespace(row.get("address_town", ""))
        if town_raw.lower() in {"slovensko", "slovakia"}:
            town = None
        else:
            town = town_raw.title() if town_raw else None
        district, region = split_zrea(row.get("address_zrea", ""))
        district = district or None
        region = region or None

        rooms = row.get("room_number", "")
        rooms_val = parse_int(str(rooms)) if rooms != "" else None
        if rooms_val is None:
            rooms_val = infer_rooms_from_title(name)

        floor, floors_total, elevator = parse_floor(row.get("floor", ""))
        building_status = normalize_building_status(row.get("building_status", ""))
        year_built = parse_int(row.get("year_of_construction", ""))
        if year_built and not (1800 <= year_built <= 2100):
            year_built = None
        year_top = parse_int(row.get("year_of_top", ""))
        if year_top and not (1800 <= year_top <= 2100):
            year_top = None
        energ_cert = normalize_energ_cert(row.get("energ_cert", ""))

        size_m2 = size_raw
        price_psm = float(price) / size_m2 if price is not None and size_m2 else None

        tokens = extract_photo_tokens(row)

        address_ascii = ascii_fold(street or "")
        town_ascii = ascii_fold(town or "")
        address_norm = ", ".join(filter(None, [address_ascii, town_ascii]))

        records.append(
            {
                "row_id": row["row_id"],
                "source": source,
                "transaction": transaction,
                "portal_listing_key": property_id,
                "link": link,
                "property_name": name,
                "address_street": street or None,
                "address_town": town or None,
                "district": district,
                "region": region,
                "room_number": rooms_val,
                "size_m2": size_m2,
                "price": price,
                "price_psm": price_psm,
                "floor": floor,
                "floors_total": floors_total,
                "elevator": elevator,
                "building_status": building_status,
                "year_of_construction": year_built,
                "year_of_top": year_top,
                "energ_cert": energ_cert,
                "photo_tokens": tokens,
                "address_norm": address_norm,
                "source_url": link,
                "updated_at": RUN_TIMESTAMP,
                "address_ascii": address_ascii,
                "town_ascii": town_ascii,
                "size_raw": size_raw,
                "price_raw": price,
                "source_path": row.get("__source_path"),
            }
        )

    norm_df = pd.DataFrame(records)
    norm_df["price_psm"] = norm_df.apply(
        lambda r: (r["price"] / r["size_m2"]) if r["price"] and r["size_m2"] else None, axis=1
    )
    return norm_df


def block_candidates(df: pd.DataFrame) -> Dict[str, List[int]]:
    blocks: Dict[str, List[int]] = defaultdict(list)
    for _, row in df.iterrows():
        rid = row["row_id"]
        pid = row["portal_listing_key"]
        if pid:
            blocks[f"pid:{pid}"].append(rid)
        addr = row["address_norm"]
        size = row["size_m2"]
        if addr and size:
            blocks[f"addr:{addr}"].append(rid)
        for token in row["photo_tokens"]:
            blocks[f"photo:{token}"].append(rid)
    return blocks


def score_pair(a: pd.Series, b: pd.Series) -> int:
    score = 0
    if a["portal_listing_key"] and a["portal_listing_key"] == b["portal_listing_key"]:
        score += 8
    if a["address_norm"] and a["address_norm"] == b["address_norm"]:
        score += 4
    size_a, size_b = a["size_m2"], b["size_m2"]
    if size_a and size_b:
        diff = abs(size_a - size_b) / max(size_a, size_b)
        if diff <= SIZE_SCORE_TOL_STRICT:
            score += 3
        elif diff <= SIZE_SCORE_TOL_LOOSE:
            score += 2
    photos_a = set(a["photo_tokens"])
    photos_b = set(b["photo_tokens"])
    if photos_a & photos_b:
        score += 4
    price_a, price_b = a["price"], b["price"]
    if price_a and price_b:
        diff = abs(price_a - price_b) / max(price_a, price_b)
        if diff <= PRICE_TOL:
            score += 1
    return score


def cluster(df: pd.DataFrame, blocks: Dict[str, List[int]]) -> List[List[int]]:
    row_ids = df["row_id"].tolist()
    id_to_pos = {rid: idx for idx, rid in enumerate(row_ids)}
    uf = UnionFind.create(len(row_ids))
    for _, candidates in blocks.items():
        unique = list(dict.fromkeys(candidates))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                rid_i, rid_j = unique[i], unique[j]
                pos_i, pos_j = id_to_pos[rid_i], id_to_pos[rid_j]
                score = score_pair(df.iloc[pos_i], df.iloc[pos_j])
                if score >= 7:
                    uf.union(pos_i, pos_j)
    clusters_map: Dict[int, List[int]] = defaultdict(list)
    for pos, rid in enumerate(row_ids):
        root = uf.find(pos)
        clusters_map[root].append(rid)
    return list(clusters_map.values())


def choose_by_priority(values: List[Tuple[str, Any]], priority: Sequence[str], allow_blank: bool = False) -> Optional[Any]:
    for portal in priority:
        for source, value in values:
            if source == portal and (allow_blank or value not in (None, "")):
                return value if value != "" else None
    for _, value in values:
        if allow_blank or value not in (None, ""):
            return value if value != "" else None
    return None


def median_numeric(values: List[float]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(np.median(nums))


def compute_listing_id(record: Dict[str, Any]) -> str:
    components = [
        ascii_fold(record.get("address_street") or ""),
        ascii_fold(record.get("address_town") or ""),
        f"{record.get('size_m2', 0.0):.1f}",
        str(record.get("floor") or ""),
        str(record.get("floors_total") or ""),
        str(record.get("year_of_construction") or ""),
        str(record.get("energ_cert") or ""),
    ]
    joined = "|".join(components)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def build_golden(cluster_df: pd.DataFrame, priority: Sequence[str], run_date: str) -> Tuple[Dict[str, Any], List[str]]:
    records = cluster_df.to_dict("records")
    conflicts: List[str] = []

    def collect(field: str) -> List[Tuple[str, Any]]:
        return [(rec["source"], rec.get(field)) for rec in records]

    size_values = [rec.get("size_m2") for rec in records if rec.get("size_m2")]
    chosen_size = median_numeric(size_values)
    if chosen_size is None and size_values:
        chosen_size = size_values[0]
    if chosen_size and any(abs((val - chosen_size) / chosen_size) > SIZE_MERGE_TOL for val in size_values if val):
        conflicts.append("size_m2")

    price_record = max(records, key=lambda rec: rec.get("updated_at", RUN_TIMESTAMP))
    chosen_price = price_record.get("price")
    price_values = [rec.get("price") for rec in records if rec.get("price") is not None]
    if price_values and chosen_price is not None:
        if any(abs(val - chosen_price) / chosen_price > PRICE_TOL for val in price_values):
            conflicts.append("price")

    room_values = [rec.get("room_number") for rec in records if rec.get("room_number") is not None]
    if room_values:
        most_common = Counter(room_values).most_common()
        chosen_rooms = most_common[0][0]
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            alt = choose_by_priority(collect("room_number"), priority)
            if alt is not None:
                chosen_rooms = alt
        if len(set(room_values)) > 1:
            conflicts.append("room_number")
    else:
        chosen_rooms = choose_by_priority(collect("room_number"), priority)

    building_values = [rec.get("building_status") for rec in records if rec.get("building_status")]
    if building_values:
        best_building = Counter(building_values).most_common(1)[0][0]
        if len(set(building_values)) > 1:
            conflicts.append("building_status")
    else:
        best_building = choose_by_priority(collect("building_status"), priority, allow_blank=True) or "unknown"

    chosen_floor = choose_by_priority(collect("floor"), priority)
    chosen_floors_total = choose_by_priority(collect("floors_total"), priority)
    elevator_any = any(rec.get("elevator") for rec in records)

    chosen_street = choose_by_priority(collect("address_street"), priority)
    chosen_town = choose_by_priority(collect("address_town"), priority)
    chosen_district = choose_by_priority(collect("district"), priority, allow_blank=True)
    chosen_region = choose_by_priority(collect("region"), priority, allow_blank=True)

    chosen_year_built = choose_by_priority(collect("year_of_construction"), priority)
    chosen_year_top = choose_by_priority(collect("year_of_top"), priority)
    chosen_energ = choose_by_priority(collect("energ_cert"), priority, allow_blank=True) or "unknown"

    photo_union = sorted(set(token for rec in records for token in rec.get("photo_tokens", [])))
    price_psm = (chosen_price / chosen_size) if chosen_price and chosen_size else None

    golden = {
        "listing_id": None,
        "address_street": chosen_street,
        "address_town": chosen_town,
        "district": chosen_district,
        "region": chosen_region,
        "room_number": chosen_rooms,
        "size_m2": chosen_size,
        "price": chosen_price,
        "price_psm": price_psm,
        "floor": chosen_floor,
        "floors_total": chosen_floors_total,
        "elevator": True if elevator_any else None,
        "building_status": best_building,
        "year_of_construction": chosen_year_built,
        "year_of_top": chosen_year_top,
        "energ_cert": chosen_energ,
        "photo_fingerprints": photo_union,
        "updated_at": RUN_TIMESTAMP,
        "provenance_json": None,
        "field_conflict_flags": None,
        "transaction": choose_by_priority(collect("transaction"), priority, allow_blank=True) or "unknown",
    }

    conflict_flags = {field: (field in conflicts) for field in ["size_m2", "price", "room_number", "building_status"]}
    golden["field_conflict_flags"] = json.dumps(conflict_flags)

    def match_value(field: str, value: Any, tolerance: float = 0.0) -> Optional[Dict[str, Any]]:
        for rec in records:
            candidate = rec.get(field)
            if value is None and candidate in (None, ""):
                return {"portal": rec["source"], "url": rec.get("link"), "rule": "priority"}
            if isinstance(value, (int, float)) and isinstance(candidate, (int, float)) and value is not None:
                if tolerance and value:
                    if abs(candidate - value) / max(abs(value), 1) <= tolerance:
                        return {"portal": rec["source"], "url": rec.get("link"), "rule": "priority"}
                elif candidate == value:
                    return {"portal": rec["source"], "url": rec.get("link"), "rule": "priority"}
            elif candidate == value and value not in (None, ""):
                return {"portal": rec["source"], "url": rec.get("link"), "rule": "priority"}
        if records:
            first = records[0]
            return {"portal": first["source"], "url": first.get("link"), "rule": "default"}
        return None

    provenance = {
        "price": {"portal": price_record.get("source"), "url": price_record.get("link"), "rule": "latest"},
        "size_m2": match_value("size_m2", chosen_size, tolerance=SIZE_MERGE_TOL),
        "room_number": match_value("room_number", chosen_rooms),
        "building_status": match_value("building_status", best_building),
        "address_street": match_value("address_street", chosen_street),
        "address_town": match_value("address_town", chosen_town),
    }
    golden["provenance_json"] = json.dumps({k: v for k, v in provenance.items() if v})

    if not chosen_floor and any(rec.get("floor") for rec in records):
        golden["floor"] = choose_by_priority(collect("floor"), priority, allow_blank=True)
    if not chosen_floors_total and any(rec.get("floors_total") for rec in records):
        golden["floors_total"] = choose_by_priority(collect("floors_total"), priority, allow_blank=True)

    golden["listing_id"] = compute_listing_id(golden)
    return golden, conflicts


def load_prev_run(prev_dir: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not prev_dir:
        return None, None
    listings_path = Path(prev_dir) / "gold_listings.parquet"
    sources_path = Path(prev_dir) / "gold_listing_sources.parquet"
    if not listings_path.exists() or not sources_path.exists():
        return None, None
    prev_listings = pd.read_parquet(listings_path)
    prev_sources = pd.read_parquet(sources_path)
    return prev_listings, prev_sources


def write_parquet_tables(out_dir: Path, run_date: str, listings: pd.DataFrame, sources: pd.DataFrame, history: pd.DataFrame) -> None:
    run_dir = out_dir / f"run={run_date}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    schema_listings = pa.schema(
        [
            ("listing_id", pa.string()),
            ("address_street", pa.string()),
            ("address_town", pa.string()),
            ("district", pa.string()),
            ("region", pa.string()),
            ("room_number", pa.int32()),
            ("size_m2", pa.float64()),
            ("price", pa.int64()),
            ("price_psm", pa.float64()),
            ("floor", pa.int32()),
            ("floors_total", pa.int32()),
            ("elevator", pa.bool_()),
            ("building_status", pa.string()),
            ("year_of_construction", pa.int32()),
            ("year_of_top", pa.int32()),
            ("energ_cert", pa.string()),
            ("photo_fingerprints", pa.list_(pa.string())),
            ("updated_at", pa.string()),
            ("provenance_json", pa.string()),
            ("field_conflict_flags", pa.string()),
            ("transaction", pa.string()),
        ]
    )

    table_listings = pa.Table.from_pandas(listings, schema=schema_listings, preserve_index=False)
    pq.write_table(
        table_listings,
        run_dir / "gold_listings.parquet",
        compression="snappy",
        use_dictionary=False,
    )

    schema_sources = pa.schema(
        [
            ("listing_id", pa.string()),
            ("portal", pa.string()),
            ("portal_listing_key", pa.string()),
            ("source_url", pa.string()),
            ("first_seen_at", pa.string()),
            ("last_seen_at", pa.string()),
            ("photo_fingerprints", pa.list_(pa.string())),
        ]
    )
    table_sources = pa.Table.from_pandas(sources, schema=schema_sources, preserve_index=False)
    pq.write_table(table_sources, run_dir / "gold_listing_sources.parquet", compression="snappy")

    schema_history = pa.schema(
        [
            ("listing_id", pa.string()),
            ("run_date", pa.string()),
            ("price", pa.int64()),
            ("status", pa.string()),
            ("fees_monthly", pa.float64()),
            ("photo_changed", pa.bool_()),
            ("price_delta", pa.float64()),
            ("status_delta", pa.string()),
        ]
    )
    table_history = pa.Table.from_pandas(history, schema=schema_history, preserve_index=False)
    pq.write_table(table_history, run_dir / "gold_listing_history.parquet", compression="snappy")


def write_latest(out_dir: Path, listings: pd.DataFrame) -> None:
    latest_dir = out_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(listings, preserve_index=False)
    pq.write_table(table, latest_dir / "gold_listings.parquet", compression="snappy")


def main() -> None:
    args = parse_args()
    run_date = args.run_date
    priority = [p.strip() for p in args.source_priority.split(",") if p.strip()]
    if not priority:
        priority = DEFAULT_PRIORITY

    raw_df = load_raw(args.input)
    norm_df = normalize(raw_df, run_date)

    blocks = block_candidates(norm_df)
    clusters = cluster(norm_df, blocks)

    prev_listings, prev_sources = load_prev_run(args.prev_run)
    prev_listings_map = prev_listings.set_index("listing_id") if prev_listings is not None else None
    prev_first_seen_map = (
        prev_sources.groupby("listing_id")["first_seen_at"].min().to_dict()
        if prev_sources is not None and not prev_sources.empty
        else {}
    )

    gold_rows = []
    source_rows = []
    report_rows = []

    for cluster_ids in clusters:
        subset = norm_df[norm_df["row_id"].isin(cluster_ids)]
        golden, conflicts = build_golden(subset, priority, run_date)
        gold_rows.append(golden)
        listing_id = golden["listing_id"]
        for rec in subset.to_dict("records"):
            source_rows.append(
                {
                    "listing_id": listing_id,
                    "portal": rec["source"],
                    "portal_listing_key": rec["portal_listing_key"],
                    "source_url": rec["link"],
                    "photo_fingerprints": rec["photo_tokens"],
                }
            )
        report_rows.append(
            {
                "listing_id": listing_id,
                "conflict_fields": ",".join(conflicts),
                "source_count": len(subset),
            }
        )

    listing_columns = [
        "listing_id",
        "address_street",
        "address_town",
        "district",
        "region",
        "room_number",
        "size_m2",
        "price",
        "price_psm",
        "floor",
        "floors_total",
        "elevator",
        "building_status",
        "year_of_construction",
        "year_of_top",
        "energ_cert",
        "photo_fingerprints",
        "updated_at",
        "provenance_json",
        "field_conflict_flags",
        "transaction",
    ]

    gold_df = pd.DataFrame(gold_rows)
    if gold_df.empty:
        gold_df = pd.DataFrame(columns=listing_columns)
    for col in listing_columns:
        if col not in gold_df.columns:
            gold_df[col] = None
    gold_df = gold_df[listing_columns]
    if not gold_df.empty:
        gold_df["photo_fingerprints"] = gold_df["photo_fingerprints"].apply(
            lambda v: v if isinstance(v, list) else []
        )
        gold_df["room_number"] = pd.to_numeric(gold_df["room_number"], errors="coerce").astype("Int32")
        gold_df["floor"] = pd.to_numeric(gold_df["floor"], errors="coerce").astype("Int32")
        gold_df["floors_total"] = pd.to_numeric(gold_df["floors_total"], errors="coerce").astype("Int32")
        gold_df["year_of_construction"] = (
            pd.to_numeric(gold_df["year_of_construction"], errors="coerce").astype("Int32")
        )
        gold_df["year_of_top"] = pd.to_numeric(gold_df["year_of_top"], errors="coerce").astype("Int32")
        gold_df["price"] = pd.to_numeric(gold_df["price"], errors="coerce").astype("Int64")
        gold_df["size_m2"] = pd.to_numeric(gold_df["size_m2"], errors="coerce")
        gold_df["price_psm"] = pd.to_numeric(gold_df["price_psm"], errors="coerce")
        gold_df["elevator"] = gold_df["elevator"].astype("boolean")

    sources_columns = [
        "listing_id",
        "portal",
        "portal_listing_key",
        "source_url",
        "photo_fingerprints",
        "first_seen_at",
        "last_seen_at",
    ]
    sources_df = pd.DataFrame(source_rows)
    if sources_df.empty:
        sources_df = pd.DataFrame(columns=sources_columns)
    for col in sources_columns:
        if col not in sources_df.columns:
            sources_df[col] = None
    sources_df = sources_df[sources_columns]
    sources_df["photo_fingerprints"] = sources_df["photo_fingerprints"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    sources_df["first_seen_at"] = sources_df["listing_id"].map(prev_first_seen_map).fillna(run_date)
    sources_df["last_seen_at"] = run_date

    history_rows = []
    if prev_listings_map is not None:
        prev_prices = prev_listings_map["price"] if "price" in prev_listings_map.columns else None
        prev_photos = (
            prev_listings_map["photo_fingerprints"] if "photo_fingerprints" in prev_listings_map.columns else None
        )
    else:
        prev_prices = None
        prev_photos = None

    for _, row in gold_df.iterrows():
        listing_id = row["listing_id"]
        prev_price = None
        prev_photo_tokens = None
        if prev_listings_map is not None and listing_id in prev_listings_map.index:
            prev_price = prev_prices.loc[listing_id] if prev_prices is not None else None
            prev_photo_tokens = prev_photos.loc[listing_id] if prev_photos is not None else None
        price_delta = None
        if prev_price is not None and not pd.isna(prev_price) and row["price"] is not None:
            price_delta = row["price"] - int(prev_price)
        photo_changed = False
        if prev_photo_tokens is not None and not pd.isna(prev_photo_tokens):
            prev_tokens = set(prev_photo_tokens if isinstance(prev_photo_tokens, list) else [])
            photo_changed = prev_tokens != set(row["photo_fingerprints"] or [])
        history_rows.append(
            {
                "listing_id": listing_id,
                "run_date": run_date,
                "price": row["price"],
                "status": None,
                "fees_monthly": None,
                "photo_changed": photo_changed,
                "price_delta": float(price_delta) if price_delta is not None else None,
                "status_delta": None,
            }
        )

    history_df = pd.DataFrame(history_rows)
    if history_df.empty:
        history_df = pd.DataFrame(
            columns=[
                "listing_id",
                "run_date",
                "price",
                "status",
                "fees_monthly",
                "photo_changed",
                "price_delta",
                "status_delta",
            ]
        )
    else:
        history_df["price"] = pd.to_numeric(history_df["price"], errors="coerce").astype("Int64")
        history_df["photo_changed"] = history_df["photo_changed"].astype(bool)
        history_df["price_delta"] = pd.to_numeric(history_df["price_delta"], errors="coerce")

    out_dir = Path(args.out_dir)
    write_parquet_tables(out_dir, run_date, gold_df, sources_df, history_df)
    write_latest(out_dir, gold_df)

    run_dir = out_dir / f"run={run_date}"
    normalized_export = norm_df.copy()
    normalized_export = normalized_export.rename(
        columns={"source": "portal", "photo_tokens": "photo_fingerprints", "room_number": "rooms"}
    )

    def _ensure_token_list(tokens: Any) -> List[str]:
        if tokens is None:
            return []
        if isinstance(tokens, list):
            iterable = tokens
        elif isinstance(tokens, (tuple, set)):
            iterable = list(tokens)
        else:
            iterable = [tokens]
        result: List[str] = []
        for tok in iterable:
            tok_str = str(tok).strip()
            if tok_str:
                result.append(tok_str)
        return result

    normalized_export["photo_fingerprints"] = normalized_export["photo_fingerprints"].apply(_ensure_token_list)
    normalized_export["project_name_norm"] = normalized_export.get("project_name_norm", pd.NA)
    normalized_export["unit_no"] = pd.NA
    normalized_export["lat"] = pd.NA
    normalized_export["lng"] = pd.NA

    normalized_columns = [
        "row_id",
        "portal",
        "portal_listing_key",
        "source_url",
        "updated_at",
        "address_street",
        "address_town",
        "address_norm",
        "address_ascii",
        "project_name_norm",
        "size_m2",
        "photo_fingerprints",
        "lat",
        "lng",
        "unit_no",
        "rooms",
        "floor",
        "floors_total",
        "elevator",
        "year_of_construction",
        "year_of_top",
        "energ_cert",
        "transaction",
    ]
    for column in normalized_columns:
        if column not in normalized_export.columns:
            normalized_export[column] = pd.NA
    normalized_export = normalized_export[normalized_columns]
    normalized_export.to_parquet(run_dir / "normalized_sources.parquet", index=False)

    report_df = pd.DataFrame(report_rows)
    if report_df.empty:
        report_df = pd.DataFrame(columns=["listing_id", "conflict_fields", "source_count"])
    run_dir = out_dir / f"run={run_date}"
    report_df.to_csv(run_dir / "report_normalizer.csv", index=False)


if __name__ == "__main__":
    main()

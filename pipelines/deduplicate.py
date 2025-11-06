"""Deterministic duplicate detection and golden merge for normalized listings."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz as _rapidfuzz

    def token_set_ratio(a: str, b: str) -> int:
        return int(_rapidfuzz.token_set_ratio(a, b))

    RAPIDFUZZ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dependency missing

    def _tokenize(value: Optional[str]) -> List[str]:
        if not value:
            return []
        tokens = re.split(r"[\\s,.;:/\\\\-]+", value.lower())
        return [tok for tok in tokens if tok]

    def token_set_ratio(a: str, b: str) -> int:
        tokens_a = set(_tokenize(a))
        tokens_b = set(_tokenize(b))
        if not tokens_a and not tokens_b:
            return 100
        if not tokens_a or not tokens_b:
            return 0
        joined_a = " ".join(sorted(tokens_a))
        joined_b = " ".join(sorted(tokens_b))
        return int(round(difflib.SequenceMatcher(None, joined_a, joined_b).ratio() * 100))

    RAPIDFUZZ_AVAILABLE = False


logger = logging.getLogger(__name__)

_PREV_SOURCES_CACHE: Dict[int, Dict[Tuple[str, str], Dict[str, Any]]] = {}
_PREV_GOLD_CACHE: Dict[int, Dict[str, Dict[str, Any]]] = {}

# Allow a small bucket neighbourhood so ±3% size differences survive rounding.
_SIZE_BUCKET_OFFSETS = (-2, -1, 0, 1, 2)


class UnionFind:
    """Disjoint-set data structure with path compression."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        """Return canonical parent."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union sets containing x and y. Return True if merged."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False

        rank_x = self.rank[root_x]
        rank_y = self.rank[root_y]

        if rank_x < rank_y:
            self.parent[root_x] = root_y
        elif rank_x > rank_y:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

    def groups(self) -> Dict[int, List[int]]:
        """Return mapping from root -> list of indices."""
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(self.parent)):
            clusters[self.find(idx)].append(idx)
        return clusters


def jaccard_size_close(a: Optional[float], b: Optional[float], tol: float = 0.03) -> bool:
    """Return True if two size values are within the tolerated relative difference."""
    if a is None or b is None:
        return False
    if isinstance(a, float) and math.isnan(a):
        return False
    if isinstance(b, float) and math.isnan(b):
        return False
    if a == b:
        return True
    if a <= 0 or b <= 0:
        return False
    diff = abs(a - b) / max(a, b)
    return diff <= tol


def addr_prefix(value: Optional[str], n: int = 12) -> str:
    """Return the blocking prefix for fuzzy address comparisons."""
    if not value:
        return ""
    return value[:n]


def build_photo_index(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Build inverted index token -> row indices."""
    index: Dict[str, List[int]] = defaultdict(list)
    for row in df.itertuples(index=False):
        tokens: Iterable[str] = getattr(row, "photo_fingerprints", [])
        seen: set[str] = set()
        for token in tokens or []:
            if token is None:
                continue
            token_str = str(token).strip()
            if not token_str or token_str in seen:
                continue
            seen.add(token_str)
            index[token_str].append(getattr(row, "row_id"))
    return index


def build_addr_buckets(df: pd.DataFrame) -> Dict[Tuple[str, int], List[int]]:
    """Return mapping of (address_norm_key, size_bucket) -> row ids."""
    buckets: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for row in df.itertuples(index=False):
        addr_key = getattr(row, "_address_norm_key", None)
        bucket = getattr(row, "_size_bucket", None)
        if addr_key and bucket is not None:
            buckets[(addr_key, bucket)].append(getattr(row, "row_id"))
    return buckets


def build_addrprefix_buckets(df: pd.DataFrame) -> Dict[Tuple[str, str, int], List[int]]:
    """Return mapping of (town_key, addr_prefix, size_bucket) -> row ids."""
    buckets: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for row in df.itertuples(index=False):
        town_key = getattr(row, "_address_town_key", None)
        ascii_key = getattr(row, "_address_ascii_key", None)
        bucket = getattr(row, "_size_bucket", None)
        if town_key and ascii_key and bucket is not None:
            buckets[(town_key, addr_prefix(ascii_key), bucket)].append(getattr(row, "row_id"))
    return buckets


def build_geo_cells(df: pd.DataFrame) -> Dict[Tuple[int, int], List[int]]:
    """Return mapping of geo cell -> row ids."""
    cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for row in df.itertuples(index=False):
        cell_lat = getattr(row, "_cell_lat", None)
        cell_lng = getattr(row, "_cell_lng", None)
        if cell_lat is not None and cell_lng is not None:
            cells[(cell_lat, cell_lng)].append(getattr(row, "row_id"))
    return cells


def neighbors(cell_lat: int, cell_lng: int) -> List[Tuple[int, int]]:
    """Return 3x3 neighbourhood of geo cells."""
    return [
        (cell_lat + d_lat, cell_lng + d_lng)
        for d_lat in (-1, 0, 1)
        for d_lng in (-1, 0, 1)
    ]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return haversine distance in meters."""
    radius = 6_371_000  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def union_list(uf: UnionFind, ids: List[int], can_union: Optional[Callable[[int, int], bool]] = None) -> int:
    """Union every item in ids into the same component respecting optional guard."""
    if len(ids) < 2:
        return 0
    merges = 0
    ordered = sorted(ids)
    pivot = ordered[0]
    for other in ordered[1:]:
        if can_union and not can_union(pivot, other):
            continue
        if uf.union(pivot, other):
            merges += 1
    return merges


def choose_by_priority(
    values_with_portal: List[Dict[str, Any]],
    source_priority: List[str],
) -> Optional[Dict[str, Any]]:
    """Pick the value whose portal ranks highest, breaking ties by recency."""

    def portal_rank(portal: Optional[str]) -> int:
        if portal is None:
            return len(source_priority)
        try:
            return source_priority.index(portal)
        except ValueError:
            return len(source_priority)

    filtered: List[Dict[str, Any]] = []
    for item in values_with_portal:
        value = item.get("value")
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        filtered.append(item)
    if not filtered:
        return None

    def sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
        ts = item.get("updated_at")
        ts_value = float("-inf")
        if ts is not None and not pd.isna(ts):
            ts_value = float(ts.timestamp())
        return (portal_rank(item.get("portal")), -ts_value)

    filtered.sort(key=sort_key)
    return filtered[0]


def _optional_value(value: Any) -> Any:
    """Coerce pandas NA and empty strings to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return value


def _ensure_token_list(raw_value: Any) -> List[str]:
    """Normalize photo_fingerprints into a list[str] without duplicates."""
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []
    if hasattr(raw_value, "tolist") and not isinstance(raw_value, str):
        try:
            raw_value = raw_value.tolist()
        except Exception:  # pragma: no cover - defensive
            pass
    tokens: List[str] = []
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    for tok in parsed:
                        if tok is None:
                            continue
                        tok_str = str(tok).strip()
                        if tok_str:
                            tokens.append(tok_str)
                    return _dedupe_preserve_order(tokens)
            except json.JSONDecodeError:
                pass
        tokens.append(stripped)
        return _dedupe_preserve_order(tokens)
    if isinstance(raw_value, (list, tuple, set)):
        for tok in raw_value:
            if tok is None:
                continue
            tok_str = str(tok).strip()
            if tok_str:
                tokens.append(tok_str)
        return _dedupe_preserve_order(tokens)

    token = str(raw_value).strip()
    return _dedupe_preserve_order([token]) if token else []


def _dedupe_preserve_order(tokens: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _numeric_value(value: Any) -> Optional[float]:
    """Return numeric value as float or None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_value(ts: Any) -> float:
    if ts is None or pd.isna(ts):
        return float("-inf")
    return float(ts.timestamp())


def _priority_index(source_priority: List[str], portal: Optional[str]) -> int:
    if portal is None:
        return len(source_priority)
    try:
        return source_priority.index(portal)
    except ValueError:
        return len(source_priority)


def _compute_listing_id(row: Dict[str, Any]) -> str:
    address = (
        (row.get("address_norm") or row.get("address_ascii") or "")
        .strip()
        .lower()
    )
    size_component = ""
    size_value = row.get("size_m2")
    if size_value is not None and not (isinstance(size_value, float) and math.isnan(size_value)):
        size_component = f"{round(float(size_value), 1):.1f}"

    def clean_component(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip().lower()

    parts = [
        address,
        size_component,
        clean_component(row.get("floor")),
        clean_component(row.get("floors_total")),
        clean_component(row.get("year_of_construction")),
        clean_component(row.get("project_name_norm")),
        clean_component(row.get("energ_cert")),
    ]
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def build_golden_row(
    cluster_df: pd.DataFrame,
    source_priority: List[str],
    run_date: str,
) -> Dict[str, Any]:
    """Construct the merged golden row for a cluster."""
    records = cluster_df.to_dict("records")
    if not records:
        raise ValueError("Cluster is empty")

    records_sorted = sorted(
        records,
        key=lambda rec: (
            -_timestamp_value(rec.get("_updated_at_ts")),
            _priority_index(source_priority, rec.get("portal")),
            rec.get("row_id", 0),
        ),
    )
    primary = records_sorted[0]

    provenance: Dict[str, Any] = {}
    conflicts: Dict[str, bool] = {}

    def record_provenance(field: str, record: Optional[Dict[str, Any]], reason: str) -> None:
        if not record:
            return
        provenance[field] = {
            "portal": record.get("portal"),
            "source_url": record.get("source_url"),
            "reason": reason,
        }

    address_norm = _optional_value(primary.get("address_norm"))
    if address_norm is None:
        for rec in records_sorted:
            candidate = _optional_value(rec.get("address_norm"))
            if candidate:
                address_norm = candidate
                record_provenance("address_norm", rec, "fallback_non_null")
                break
    else:
        record_provenance("address_norm", primary, "primary_recent")

    address_ascii = _optional_value(primary.get("address_ascii"))
    if address_ascii is None:
        for rec in records_sorted:
            candidate = _optional_value(rec.get("address_ascii"))
            if candidate:
                address_ascii = candidate
                record_provenance("address_ascii", rec, "fallback_non_null")
                break
    else:
        record_provenance("address_ascii", primary, "primary_recent")

    town_counter = Counter(
        rec.get("_address_town_key")
        for rec in records
        if rec.get("_address_town_key")
    )
    address_town = None
    if town_counter:
        top_key, _ = town_counter.most_common(1)[0]
        town_record = next(
            (
                rec
                for rec in records_sorted
                if rec.get("_address_town_key") == top_key
                and _optional_value(rec.get("address_town"))
            ),
            None,
        )
        if town_record:
            address_town = _optional_value(town_record.get("address_town"))
            record_provenance("address_town", town_record, "majority")
    if address_town is None:
        address_town = _optional_value(primary.get("address_town"))
        if address_town:
            record_provenance("address_town", primary, "primary_recent")

    project_candidates = [
        {
            "value": _optional_value(rec.get("project_name_norm")),
            "portal": rec.get("portal"),
            "source_url": rec.get("source_url"),
            "updated_at": rec.get("_updated_at_ts"),
            "record": rec,
        }
        for rec in records
    ]
    project_choice = choose_by_priority(project_candidates, source_priority)
    project_name_norm = None
    if project_choice:
        project_name_norm = project_choice["value"]
        record_provenance("project_name_norm", project_choice.get("record"), "priority")

    size_values = [
        float(rec.get("size_m2"))
        for rec in records
        if rec.get("size_m2") is not None and not math.isnan(rec.get("size_m2"))
    ]
    size_m2 = None
    if size_values:
        max_size = max(size_values)
        min_size = min(size_values)
        if max_size > 0:
            diff_ratio = abs(max_size - min_size) / max_size
        else:
            diff_ratio = 0.0
        if diff_ratio <= 0.05:
            # Median is robust against occasional outliers.
            size_values_sorted = sorted(size_values)
            mid = len(size_values_sorted) // 2
            if len(size_values_sorted) % 2 == 1:
                size_m2 = size_values_sorted[mid]
            else:
                size_m2 = (size_values_sorted[mid - 1] + size_values_sorted[mid]) / 2
            record_provenance("size_m2", None, "median")
        else:
            conflicts["size_gt_5pct"] = True
            size_choice = choose_by_priority(
                [
                    {
                        "value": rec.get("size_m2"),
                        "portal": rec.get("portal"),
                        "source_url": rec.get("source_url"),
                        "updated_at": rec.get("_updated_at_ts"),
                        "record": rec,
                    }
                    for rec in records
                ],
                source_priority,
            )
            if size_choice:
                size_m2 = float(size_choice["value"])
                record_provenance("size_m2", size_choice.get("record"), "priority")

    def sort_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def key(item: Dict[str, Any]) -> Tuple[float, int, int]:
            record = item.get("record") or {}
            return (
                -_timestamp_value(item.get("updated_at")),
                _priority_index(source_priority, item.get("portal")),
                record.get("row_id", 0),
            )

        return sorted(items, key=key)

    price_candidates = [
        {
            "value": _numeric_value(rec.get("price")),
            "portal": rec.get("portal"),
            "source_url": rec.get("source_url"),
            "updated_at": rec.get("_updated_at_ts"),
            "record": rec,
        }
        for rec in records
        if _numeric_value(rec.get("price")) is not None
    ]
    price = None
    if price_candidates:
        ordered_prices = sort_candidates(price_candidates)
        price_choice = ordered_prices[0]
        price = float(price_choice["value"]) if price_choice["value"] is not None else None
        record_provenance("price", price_choice.get("record"), "latest")
        baseline = price if price is not None else None
        if baseline and any(
            abs(candidate["value"] - baseline) / max(abs(baseline), 1.0) > 0.05
            for candidate in ordered_prices
            if candidate["value"] is not None
        ):
            conflicts["price_disagree"] = True

    room_values = [
        rec.get("rooms")
        for rec in records
        if rec.get("rooms") is not None and not pd.isna(rec.get("rooms"))
    ]
    rooms = None
    if room_values:
        room_counter = Counter(room_values)
        if len(room_counter) > 1:
            conflicts["rooms_disagree"] = True
        most_common = room_counter.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            target_value = most_common[0][0]
            room_record = next(
                (rec for rec in records_sorted if rec.get("rooms") == target_value),
                None,
            )
            rooms = target_value
            record_provenance("rooms", room_record, "majority" if len(most_common) > 1 else "only_value")
        else:
            rooms_choice = choose_by_priority(
                [
                    {
                        "value": rec.get("rooms"),
                        "portal": rec.get("portal"),
                        "source_url": rec.get("source_url"),
                        "updated_at": rec.get("_updated_at_ts"),
                        "record": rec,
                    }
                    for rec in records
                ],
                source_priority,
            )
            if rooms_choice:
                rooms = rooms_choice["value"]
                record_provenance("rooms", rooms_choice.get("record"), "priority")

    # Floor tuple: prefer entries with both values.
    floor = None
    floors_total = None
    combined_candidates = [
        rec
        for rec in records_sorted
        if rec.get("floor") is not None
        and not pd.isna(rec.get("floor"))
        and rec.get("floors_total") is not None
        and not pd.isna(rec.get("floors_total"))
    ]
    if combined_candidates:
        chosen = combined_candidates[0]
        floor = chosen.get("floor")
        floors_total = chosen.get("floors_total")
        record_provenance("floor", chosen, "floor_tuple")
        record_provenance("floors_total", chosen, "floor_tuple")
    else:
        floor_choice = choose_by_priority(
            [
                {
                    "value": rec.get("floor"),
                    "portal": rec.get("portal"),
                    "source_url": rec.get("source_url"),
                    "updated_at": rec.get("_updated_at_ts"),
                    "record": rec,
                }
                for rec in records
            ],
            source_priority,
        )
        if floor_choice:
            floor = floor_choice["value"]
            record_provenance("floor", floor_choice.get("record"), "priority")

        floors_total_choice = choose_by_priority(
            [
                {
                    "value": rec.get("floors_total"),
                    "portal": rec.get("portal"),
                    "source_url": rec.get("source_url"),
                    "updated_at": rec.get("_updated_at_ts"),
                    "record": rec,
                }
                for rec in records
            ],
            source_priority,
        )
        if floors_total_choice:
            floors_total = floors_total_choice["value"]
            record_provenance("floors_total", floors_total_choice.get("record"), "priority")

    elevator = None
    if any(rec.get("elevator") is True for rec in records):
        elevator = True
        rec_true = next(rec for rec in records if rec.get("elevator") is True)
        record_provenance("elevator", rec_true, "any_true")
    elif any(rec.get("elevator") is False for rec in records):
        elevator = False
        rec_false = next(rec for rec in records if rec.get("elevator") is False)
        record_provenance("elevator", rec_false, "any_false")

    year_choice = choose_by_priority(
        [
            {
                "value": rec.get("year_of_construction"),
                "portal": rec.get("portal"),
                "source_url": rec.get("source_url"),
                "updated_at": rec.get("_updated_at_ts"),
                "record": rec,
            }
            for rec in records
        ],
        source_priority,
    )
    year_of_construction = None
    if year_choice:
        year_of_construction = year_choice["value"]
        record_provenance("year_of_construction", year_choice.get("record"), "priority")

    energ_choice = choose_by_priority(
        [
            {
                "value": _optional_value(rec.get("energ_cert")),
                "portal": rec.get("portal"),
                "source_url": rec.get("source_url"),
                "updated_at": rec.get("_updated_at_ts"),
                "record": rec,
            }
            for rec in records
        ],
        source_priority,
    )
    energ_cert = None
    if energ_choice:
        energ_cert = energ_choice["value"]
        record_provenance("energ_cert", energ_choice.get("record"), "priority")

    unit_choice = choose_by_priority(
        [
            {
                "value": _optional_value(rec.get("unit_no")),
                "portal": rec.get("portal"),
                "source_url": rec.get("source_url"),
                "updated_at": rec.get("_updated_at_ts"),
                "record": rec,
            }
            for rec in records
        ],
        source_priority,
    )
    unit_no = None
    if unit_choice:
        unit_no = unit_choice["value"]
        record_provenance("unit_no", unit_choice.get("record"), "priority")

    lat_values = [
        rec.get("_lat")
        for rec in records
        if rec.get("_lat") is not None and not pd.isna(rec.get("_lat"))
    ]
    lng_values = [
        rec.get("_lng")
        for rec in records
        if rec.get("_lng") is not None and not pd.isna(rec.get("_lng"))
    ]
    lat = sum(lat_values) / len(lat_values) if lat_values else None
    lng = sum(lng_values) / len(lng_values) if lng_values else None
    if lat_values:
        record_provenance("lat", None, "mean")
    if lng_values:
        record_provenance("lng", None, "mean")

    photo_tokens = sorted(
        {
            token
            for rec in records
            for token in rec.get("photo_fingerprints", []) or []
            if token
        }
    )
    record_provenance("photo_fingerprints", None, "union")

    updated_at = primary.get("updated_at")
    record_provenance("updated_at", primary, "primary_recent")

    status_candidates = [
        {
            "value": _optional_value(rec.get("status")),
            "portal": rec.get("portal"),
            "source_url": rec.get("source_url"),
            "updated_at": rec.get("_updated_at_ts"),
            "record": rec,
        }
        for rec in records
        if _optional_value(rec.get("status")) is not None
    ]
    status = None
    if status_candidates:
        ordered_status = sort_candidates(status_candidates)
        status_choice = ordered_status[0]
        status = status_choice["value"]
        record_provenance("status", status_choice.get("record"), "latest")
        if len({item["value"] for item in ordered_status if item["value"] is not None}) > 1:
            conflicts["status_disagree"] = True

    price_psm = None
    if price is not None and size_m2 is not None and not math.isnan(size_m2) and size_m2 != 0:
        price_psm = price / size_m2

    golden_row = {
        "address_norm": address_norm,
        "address_ascii": address_ascii,
        "address_town": address_town,
        "project_name_norm": project_name_norm,
        "unit_no": unit_no,
        "size_m2": size_m2,
        "rooms": rooms,
        "price": price,
        "price_psm": price_psm,
        "floor": floor,
        "floors_total": floors_total,
        "elevator": elevator,
        "year_of_construction": year_of_construction,
        "energ_cert": energ_cert,
        "lat": lat,
        "lng": lng,
        "updated_at": updated_at,
        "status": status,
        "primary_portal": primary.get("portal"),
        "primary_source_url": primary.get("source_url"),
        "photo_fingerprints": photo_tokens,
        "cluster_size": len(cluster_df),
        "provenance_json": json.dumps(provenance, ensure_ascii=True, sort_keys=True),
        "field_conflict_flags": json.dumps(conflicts, ensure_ascii=True, sort_keys=True),
        "run_date": run_date,
    }
    golden_row["listing_id"] = _compute_listing_id(golden_row)
    return golden_row


def _prev_sources_lookup(prev_sources: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cache_key = id(prev_sources)
    if cache_key in _PREV_SOURCES_CACHE:
        return _PREV_SOURCES_CACHE[cache_key]
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in prev_sources.itertuples(index=False):
        portal = getattr(row, "portal", None)
        portal_listing_key = getattr(row, "portal_listing_key", None)
        if portal is None or portal_listing_key is None:
            continue
        mapping[(str(portal), str(portal_listing_key))] = row._asdict()
    _PREV_SOURCES_CACHE[cache_key] = mapping
    return mapping


def _prev_gold_lookup(prev_gold: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    cache_key = id(prev_gold)
    if cache_key in _PREV_GOLD_CACHE:
        return _PREV_GOLD_CACHE[cache_key]
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in prev_gold.itertuples(index=False):
        listing_id = getattr(row, "listing_id", None)
        if listing_id is None:
            continue
        tokens_raw = getattr(row, "photo_fingerprints", None)
        tokens = _ensure_token_list(tokens_raw)
        price_raw = getattr(row, "price", None) if hasattr(row, "price") else None
        price_value = _numeric_value(price_raw)
        status_raw = getattr(row, "status", None) if hasattr(row, "status") else None
        status_value = _optional_value(status_raw)
        mapping[str(listing_id)] = {
            "photo_fingerprints": tokens,
            "price": price_value,
            "status": status_value,
        }
    _PREV_GOLD_CACHE[cache_key] = mapping
    return mapping


def build_sources_rows(
    cluster_df: pd.DataFrame,
    listing_id: str,
    run_date: str,
    prev_sources: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Return per-source provenance rows for the cluster."""
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if prev_sources is not None:
        lookup = _prev_sources_lookup(prev_sources)

    rows: List[Dict[str, Any]] = []
    for rec in cluster_df.to_dict("records"):
        portal = rec.get("portal")
        portal_listing_key = rec.get("portal_listing_key")
        source_url = rec.get("source_url")
        updated_at = rec.get("updated_at")
        key = (str(portal), str(portal_listing_key))
        previous = lookup.get(key, {})
        first_seen = previous.get("first_seen_at") or run_date
        row = {
            "listing_id": listing_id,
            "portal": portal,
            "portal_listing_key": portal_listing_key,
            "source_url": source_url,
            "updated_at": updated_at,
            "first_seen_at": first_seen,
            "last_seen_at": run_date,
            "run_date": run_date,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_history_row(
    listing_id: str,
    run_date: str,
    new_tokens: List[str],
    new_price: Optional[float],
    new_status: Optional[str],
    prev_gold: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Return per-run history snapshot for a listing."""
    prev_lookup: Dict[str, Dict[str, Any]] = {}
    if prev_gold is not None:
        prev_lookup = _prev_gold_lookup(prev_gold)
    prev_entry = prev_lookup.get(listing_id, {})
    prev_tokens = set(prev_entry.get("photo_fingerprints", []))
    new_token_set = set(new_tokens)
    photo_changed = bool(prev_lookup and prev_tokens != new_token_set)
    prev_price = prev_entry.get("price")
    prev_status = prev_entry.get("status")
    price_value = None if new_price is None or (isinstance(new_price, float) and math.isnan(new_price)) else new_price
    price_delta = None
    if price_value is not None and prev_price is not None:
        price_delta = price_value - prev_price
    status_clean = _optional_value(new_status)
    status_changed = False
    if prev_status is not None and status_clean is not None:
        status_changed = prev_status != status_clean
    return {
        "listing_id": listing_id,
        "run_date": run_date,
        "photo_changed": photo_changed,
        "price": price_value if price_value is not None else pd.NA,
        "status": status_clean if status_clean is not None else pd.NA,
        "price_delta": price_delta if price_delta is not None else pd.NA,
        "status_changed": status_changed if prev_status is not None and status_clean is not None else False,
    }


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.reset_index(drop=True)
    df["row_id"] = range(len(df))
    df["photo_fingerprints"] = df.get("photo_fingerprints", pd.Series([], dtype=object)).apply(
        _ensure_token_list
    )
    df["size_m2"] = pd.to_numeric(df.get("size_m2"), errors="coerce")
    df["_size_bucket"] = df["size_m2"].apply(
        lambda v: int(round(v)) if v is not None and not math.isnan(v) else None
    )
    df["_address_norm_key"] = df.get("address_norm").apply(
        lambda v: str(v).strip().lower() if _optional_value(v) else None
    )
    df["_address_ascii_key"] = df.get("address_ascii").apply(
        lambda v: str(v).strip().lower() if _optional_value(v) else None
    )
    df["_address_town_key"] = df.get("address_town").apply(
        lambda v: str(v).strip().lower() if _optional_value(v) else None
    )
    df["_project_name_key"] = df.get("project_name_norm").apply(
        lambda v: str(v).strip().lower() if _optional_value(v) else None
    )
    df["_lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["_lng"] = pd.to_numeric(df.get("lng"), errors="coerce")
    df["_cell_lat"] = df["_lat"].apply(
        lambda v: math.floor(v * 1000) if v is not None and not math.isnan(v) else None
    )
    df["_cell_lng"] = df["_lng"].apply(
        lambda v: math.floor(v * 1000) if v is not None and not math.isnan(v) else None
    )
    df["_updated_at_ts"] = pd.to_datetime(df.get("updated_at"), errors="coerce", utc=False)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    else:
        df["price"] = pd.NA
    if "price_psm" in df.columns:
        df["price_psm"] = pd.to_numeric(df.get("price_psm"), errors="coerce")
    else:
        df["price_psm"] = pd.NA
    if "status" in df.columns:
        df["status"] = df["status"].apply(_optional_value)
    else:
        df["status"] = pd.NA
    return df


def _empty_gold_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": pd.Series(dtype="string"),
            "address_town": pd.Series(dtype="string"),
            "address_norm": pd.Series(dtype="string"),
            "address_ascii": pd.Series(dtype="string"),
            "project_name_norm": pd.Series(dtype="string"),
            "unit_no": pd.Series(dtype="string"),
            "size_m2": pd.Series(dtype="float64"),
            "price": pd.Series(dtype="float64"),
            "price_psm": pd.Series(dtype="float64"),
            "rooms": pd.Series(dtype="float64"),
            "floor": pd.Series(dtype="float64"),
            "floors_total": pd.Series(dtype="float64"),
            "elevator": pd.Series(dtype="boolean"),
            "year_of_construction": pd.Series(dtype="float64"),
            "energ_cert": pd.Series(dtype="string"),
            "lat": pd.Series(dtype="float64"),
            "lng": pd.Series(dtype="float64"),
            "updated_at": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            "primary_portal": pd.Series(dtype="string"),
            "primary_source_url": pd.Series(dtype="string"),
            "photo_fingerprints": pd.Series(dtype="object"),
            "provenance_json": pd.Series(dtype="string"),
            "field_conflict_flags": pd.Series(dtype="string"),
            "cluster_size": pd.Series(dtype="int64"),
            "run_date": pd.Series(dtype="string"),
        }
    )


def _empty_sources_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": pd.Series(dtype="string"),
            "portal": pd.Series(dtype="string"),
            "portal_listing_key": pd.Series(dtype="string"),
            "source_url": pd.Series(dtype="string"),
            "updated_at": pd.Series(dtype="string"),
            "first_seen_at": pd.Series(dtype="string"),
            "last_seen_at": pd.Series(dtype="string"),
            "run_date": pd.Series(dtype="string"),
        }
    )


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": pd.Series(dtype="string"),
            "run_date": pd.Series(dtype="string"),
            "photo_changed": pd.Series(dtype="boolean"),
            "price": pd.Series(dtype="float64"),
            "status": pd.Series(dtype="string"),
            "price_delta": pd.Series(dtype="float64"),
            "status_changed": pd.Series(dtype="boolean"),
        }
    )


def deduplicate_and_merge(
    df: pd.DataFrame,
    run_date: str,
    source_priority: List[str],
    prev_gold: Optional[pd.DataFrame] = None,
    prev_sources: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (gold_listings, gold_listing_sources, gold_listing_history).
    - Uses union-find dedupe via rules P1, A1, A2, I1, G1, B1.
    - Applies Golden merge rules.
    - Preserves first_seen_at/last_seen_at using prev_sources when provided.
    - Computes history row: photo_changed (token set diff vs prev), placeholders for price/status.
    Ensures Arrow-friendly dtypes (list<string> for photo_fingerprints).
    """
    if df is None or df.empty:
        return _empty_gold_df(), _empty_sources_df(), _empty_history_df()

    logger.info("Deduplicating %s normalized listings.", len(df))

    working_df = _prepare_dataframe(df)
    working_df["_portal_priority"] = working_df["portal"].apply(lambda p: _priority_index(source_priority, p))
    working_df = working_df.sort_values(["_portal_priority", "row_id"]).reset_index(drop=True)
    working_df["row_id"] = range(len(working_df))
    records = working_df.to_dict("records")
    priority_lookup = working_df["_portal_priority"].tolist()

    def can_union(idx: int, other: int) -> bool:
        return priority_lookup[idx] <= priority_lookup[other]

    uf = UnionFind(len(working_df))
    merge_counts = {rule: 0 for rule in ("P1", "A1", "A2", "I1", "G1", "B1")}

    # Rule P1: photo fingerprint overlap.
    photo_index = build_photo_index(working_df)
    for ids in photo_index.values():
        merge_counts["P1"] += union_list(uf, ids, can_union=can_union)

    # Rule A1: exact address match within size tolerance.
    addr_buckets = build_addr_buckets(working_df)
    for idx, row in enumerate(records):
        addr_key = row.get("_address_norm_key")
        bucket = row.get("_size_bucket")
        if addr_key is None or bucket is None:
            continue
        size_a = row.get("size_m2")
        if size_a is None or math.isnan(size_a):
            continue
        for offset in _SIZE_BUCKET_OFFSETS:
            neighbor_bucket = bucket + offset
            candidates = addr_buckets.get((addr_key, neighbor_bucket), [])
            for other in candidates:
                if other <= idx:
                    continue
                other_row = records[other]
                if other_row.get("_address_norm_key") != addr_key:
                    continue
                if other_row.get("_address_town_key") != row.get("_address_town_key"):
                    continue
                size_b = other_row.get("size_m2")
                if size_b is None or math.isnan(size_b):
                    continue
                if not jaccard_size_close(size_a, size_b):
                    continue
                if not can_union(idx, other):
                    continue
                if uf.union(idx, other):
                    merge_counts["A1"] += 1

    # Rule A2: very similar address_ascii within same town and size tolerance.
    addrprefix_buckets = build_addrprefix_buckets(working_df)
    for idx, row in enumerate(records):
        town_key = row.get("_address_town_key")
        ascii_key = row.get("_address_ascii_key")
        bucket = row.get("_size_bucket")
        size_a = row.get("size_m2")
        if town_key is None or ascii_key is None or bucket is None:
            continue
        if size_a is None or math.isnan(size_a):
            continue
        prefix = addr_prefix(ascii_key)
        for offset in _SIZE_BUCKET_OFFSETS:
            neighbor_bucket = bucket + offset
            candidates = addrprefix_buckets.get((town_key, prefix, neighbor_bucket), [])
            for other in candidates:
                if other <= idx:
                    continue
                other_row = records[other]
                if other_row.get("_address_town_key") != town_key:
                    continue
                ascii_other = other_row.get("_address_ascii_key")
                if not ascii_other:
                    continue
                size_b = other_row.get("size_m2")
                if size_b is None or math.isnan(size_b):
                    continue
                if not jaccard_size_close(size_a, size_b):
                    continue
                if token_set_ratio(ascii_key, ascii_other) >= 90:
                    if not can_union(idx, other):
                        continue
                    if uf.union(idx, other):
                        merge_counts["A2"] += 1

    # Rule I1: shared portal_listing_key across different portals.
    id_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        portal_listing_key = _optional_value(row.get("portal_listing_key"))
        if portal_listing_key:
            id_groups[str(portal_listing_key)].append(idx)
    for rows in id_groups.values():
        if len(rows) < 2:
            continue
        portals = {
            records[row_idx].get("portal")
            for row_idx in rows
            if records[row_idx].get("portal") is not None
        }
        if len(portals) < 2:
            continue
        merge_counts["I1"] += union_list(uf, rows, can_union=can_union)

    # Rule G1: geo proximity within 60m, same town, size tolerance.
    geo_cells = build_geo_cells(working_df)
    for idx, row in enumerate(records):
        town_key = row.get("_address_town_key")
        lat = row.get("_lat")
        lng = row.get("_lng")
        bucket = row.get("_size_bucket")
        size_a = row.get("size_m2")
        cell_lat = row.get("_cell_lat")
        cell_lng = row.get("_cell_lng")
        if (
            town_key is None
            or lat is None
            or lng is None
            or bucket is None
            or cell_lat is None
            or cell_lng is None
            or size_a is None
            or math.isnan(size_a)
        ):
            continue
        for cell in neighbors(cell_lat, cell_lng):
            for other in geo_cells.get(cell, []):
                if other <= idx:
                    continue
                other_row = records[other]
                if other_row.get("_address_town_key") != town_key:
                    continue
                size_b = other_row.get("size_m2")
                if size_b is None or math.isnan(size_b):
                    continue
                if not jaccard_size_close(size_a, size_b):
                    continue
                lat_b = other_row.get("_lat")
                lng_b = other_row.get("_lng")
                if lat_b is None or lng_b is None:
                    continue
                if haversine_m(lat, lng, lat_b, lng_b) <= 60.0:
                    if not can_union(idx, other):
                        continue
                    if uf.union(idx, other):
                        merge_counts["G1"] += 1

    # Rule B1: same project/building + town + size tolerance.
    project_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        project_key = row.get("_project_name_key")
        town_key = row.get("_address_town_key")
        if project_key and town_key:
            project_groups[(project_key, town_key)].append(idx)
    for ids in project_groups.values():
        if len(ids) < 2:
            continue
        bucket_map: Dict[int, List[int]] = defaultdict(list)
        for row_idx in ids:
            bucket = records[row_idx].get("_size_bucket")
            if bucket is not None:
                bucket_map[bucket].append(row_idx)
        for bucket, rows_in_bucket in bucket_map.items():
            for idx_in_bucket in rows_in_bucket:
                size_a = records[idx_in_bucket].get("size_m2")
                if size_a is None or math.isnan(size_a):
                    continue
                for offset in _SIZE_BUCKET_OFFSETS:
                    neighbor_bucket = bucket + offset
                    for other in bucket_map.get(neighbor_bucket, []):
                        if other <= idx_in_bucket:
                            continue
                        size_b = records[other].get("size_m2")
                        if size_b is None or math.isnan(size_b):
                            continue
                        if jaccard_size_close(size_a, size_b):
                            if not can_union(idx_in_bucket, other):
                                continue
                            if uf.union(idx_in_bucket, other):
                                merge_counts["B1"] += 1

    working_df = working_df.drop(columns=["_portal_priority"])

    clusters = uf.groups()
    logger.info(
        "Formed %s golden clusters from %s sources. Rule merges: %s",
        len(clusters),
        len(working_df),
        merge_counts,
    )

    golden_rows: List[Dict[str, Any]] = []
    sources_frames: List[pd.DataFrame] = []
    history_rows: List[Dict[str, Any]] = []

    for cluster_indices in clusters.values():
        cluster_df = working_df.loc[cluster_indices].reset_index(drop=True)
        golden_row = build_golden_row(cluster_df, source_priority, run_date)
        golden_rows.append(golden_row)
        sources_frames.append(
            build_sources_rows(cluster_df, golden_row["listing_id"], run_date, prev_sources)
        )
        history_rows.append(
            build_history_row(
                golden_row["listing_id"],
                run_date,
                golden_row["photo_fingerprints"],
                golden_row.get("price"),
                golden_row.get("status"),
                prev_gold,
            )
        )

    gold_listings = pd.DataFrame(golden_rows) if golden_rows else _empty_gold_df()
    gold_listing_sources = (
        pd.concat(sources_frames, ignore_index=True) if sources_frames else _empty_sources_df()
    )
    gold_listing_history = pd.DataFrame(history_rows) if history_rows else _empty_history_df()

    # Ensure Arrow-friendly list column.
    if not gold_listings.empty:
        gold_listings["photo_fingerprints"] = gold_listings["photo_fingerprints"].apply(
            lambda vals: list(vals) if isinstance(vals, (list, tuple)) else []
        )
        gold_listings["price"] = pd.to_numeric(gold_listings.get("price"), errors="coerce").astype("Int64")
        gold_listings["price_psm"] = pd.to_numeric(gold_listings.get("price_psm"), errors="coerce")
        gold_listings["status"] = gold_listings["status"].apply(_optional_value).astype("string")

    if not gold_listing_history.empty:
        gold_listing_history["photo_changed"] = gold_listing_history["photo_changed"].astype(bool)
        gold_listing_history["price"] = pd.to_numeric(gold_listing_history.get("price"), errors="coerce").astype("Int64")
        gold_listing_history["price_delta"] = pd.to_numeric(
            gold_listing_history.get("price_delta"), errors="coerce"
        )
        gold_listing_history["status_changed"] = gold_listing_history["status_changed"].fillna(False).astype(bool)
        gold_listing_history["status"] = gold_listing_history["status"].apply(_optional_value).astype("string")

    return gold_listings, gold_listing_sources, gold_listing_history


def _load_parquet_if_exists(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not path.exists():
        logger.warning("Previous dataset %s not found; skipping.", path)
        return None
    return pd.read_parquet(path)


def _default_run_date() -> str:
    return datetime.utcnow().date().isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate normalized listings into golden records.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to normalized listings Parquet file.",
    )
    parser.add_argument(
        "--run-date",
        default=_default_run_date(),
        help="ISO date string representing the current run date.",
    )
    parser.add_argument(
        "--source-priority",
        nargs="*",
        default=[],
        help="Ordered list of portals from highest to lowest priority.",
    )
    parser.add_argument(
        "--prev-gold",
        type=Path,
        help="Optional path to previous golden listings Parquet.",
    )
    parser.add_argument(
        "--prev-sources",
        type=Path,
        help="Optional path to previous listing sources Parquet.",
    )
    return parser.parse_args()


def _summarize_results(
    input_rows: int,
    gold_listings: pd.DataFrame,
    gold_listing_sources: pd.DataFrame,
) -> None:
    logger.info(
        "Merged %s source rows into %s golden listings (%s source mappings).",
        input_rows,
        len(gold_listings),
        len(gold_listing_sources),
    )
    if not gold_listings.empty:
        conflicts = gold_listings["field_conflict_flags"].apply(
            lambda raw: len(json.loads(raw)) if raw else 0
        )
        logger.info("Average conflict flags per listing: %.3f", conflicts.mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    df = pd.read_parquet(args.input)
    prev_gold_df = _load_parquet_if_exists(args.prev_gold)
    prev_sources_df = _load_parquet_if_exists(args.prev_sources)
    gold_listings, gold_sources, gold_history = deduplicate_and_merge(
        df=df,
        run_date=args.run_date,
        source_priority=args.source_priority,
        prev_gold=prev_gold_df,
        prev_sources=prev_sources_df,
    )
    _summarize_results(len(df), gold_listings, gold_sources)
    print("gold_listings:", gold_listings.head())
    print("gold_listing_sources:", gold_sources.head())
    print("gold_listing_history:", gold_history.head())


if __name__ == "__main__":
    main()

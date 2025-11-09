"""
Pipeline reporting utilities.

Generates a quick snapshot showing how many listings were scraped, normalised,
and deduplicated for a given run.  This helps validate the end-to-end funnel.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_PARQUET_DIR = Path("parquet_runs")
DEFAULT_DEDUPE_DIR = Path("dedupe_runs")
DEFAULT_TIMINGS_FILE = DEFAULT_OUTPUT_DIR / "run_timings.json"


@dataclass
class CountSnapshot:
    total: int
    breakdown: Dict[str, int]

    def ratio(self, denominator: int) -> Optional[float]:
        if denominator <= 0 or self.total is None:
            return None
        return self.total / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise listing counts across scrape, normalisation, and dedupe outputs."
    )
    parser.add_argument("--run-date", help="Run date (YYYY-MM-DD). Uses latest run if omitted.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing raw scraper CSV exports.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Normalizer output directory containing run=YYYY-MM-DD folders.",
    )
    parser.add_argument(
        "--dedupe-dir",
        type=Path,
        default=DEFAULT_DEDUPE_DIR,
        help="Deduplicate output directory containing run=YYYY-MM-DD folders.",
    )
    parser.add_argument(
        "--timings-file",
        type=Path,
        default=DEFAULT_TIMINGS_FILE,
        help="JSON file produced by scripts/runner.py containing task durations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def find_run_directory(base_dir: Path, run_date: Optional[str], allow_latest: bool = False) -> Path:
    if run_date:
        candidate = base_dir / f"run={run_date}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Run directory {candidate} not found.")

    if allow_latest:
        latest = base_dir / "latest"
        if latest.exists():
            return latest

    run_dirs = sorted([p for p in base_dir.glob("run=*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run=* directories found under {base_dir}.")
    return run_dirs[-1]


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        # Skip header if present
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def collect_output_counts(output_dir: Path) -> CountSnapshot:
    if not output_dir.exists():
        logging.warning("Output directory %s not found.", output_dir)
        return CountSnapshot(total=0, breakdown={})

    counts: Dict[str, int] = OrderedDict()
    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            counts[csv_path.name] = count_csv_rows(csv_path)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Failed counting %s: %s", csv_path, exc)
    total = sum(counts.values())
    return CountSnapshot(total=total, breakdown=counts)


def load_parquet_counts(parquet_path: Path, column: Optional[str] = None) -> CountSnapshot:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file {parquet_path} not found.")

    columns = [column] if column else None
    df = pd.read_parquet(parquet_path, columns=columns)
    if column:
        series = df[column].fillna("unknown").astype(str)
        counts = series.value_counts().sort_index().to_dict()
        return CountSnapshot(total=int(series.shape[0]), breakdown=counts)
    return CountSnapshot(total=int(df.shape[0]), breakdown={})


CITY_ALIAS_MAP = {
    "vyšné opátske": "Košice",
    "vyšne opatske": "Košice",
    "topoľová": "Košice",
    "topolova": "Košice",
}
CITY_SKIP_SET = {"unknown", "džungľa", "dzungla"}


def _normalize_city(name: str) -> Optional[str]:
    cleaned = name.strip()
    key = cleaned.lower()
    if key in CITY_SKIP_SET:
        return None
    if key in CITY_ALIAS_MAP:
        return CITY_ALIAS_MAP[key]
    return cleaned or None


TRANSACTION_ORDER = ["predaj", "prenajom", "unknown"]


def compute_city_transaction_stats(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file {parquet_path} not found.")

    df = pd.read_parquet(parquet_path, columns=["address_town"])
    if df.empty:
        return pd.DataFrame(columns=["address_town", "transaction", "count"])

    df["address_town"] = df["address_town"].fillna("").astype(str).apply(_normalize_city)
    df = df.dropna(subset=["address_town"])

    # Infer transaction from directory name, file name suffix, or fallback to unknown.
    inferred = parquet_path.parent.name.split("transaction=")[-1]
    inferred = inferred if "transaction=" in parquet_path.parent.name else parquet_path.stem.split("_")[-1]
    inferred = inferred.lower()
    if inferred not in TRANSACTION_ORDER:
        inferred = "unknown"
    df["transaction"] = inferred

    grouped = (
        df.groupby(["address_town", "transaction"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "address_town"], ascending=[False, True])
    )
    return grouped


def _load_city_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Gold listings parquet not found: {path}")
    df = pd.read_parquet(path, columns=["listing_id", "address_town"])
    if df.empty:
        return {}
    df["city"] = (
        df["address_town"]
        .fillna("")
        .astype(str)
        .apply(lambda value: _normalize_city(value) or "Unknown")
    )
    df["listing_id"] = df["listing_id"].astype(str)
    return dict(zip(df["listing_id"], df["city"]))


def _compute_city_deltas(current: Path, previous: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    current_map = _load_city_map(current)
    previous_map = _load_city_map(previous)
    current_ids = set(current_map.keys())
    previous_ids = set(previous_map.keys())

    new_ids = current_ids - previous_ids
    gone_ids = previous_ids - current_ids

    new_counts: Dict[str, int] = defaultdict(int)
    for listing_id in new_ids:
        city = current_map.get(listing_id, "Unknown")
        new_counts[city] += 1

    gone_counts: Dict[str, int] = defaultdict(int)
    for listing_id in gone_ids:
        city = previous_map.get(listing_id, "Unknown")
        gone_counts[city] += 1

    return dict(new_counts), dict(gone_counts)


def _list_run_dirs(base_dir: Path) -> List[Path]:
    return sorted([path for path in base_dir.glob("run=*") if path.is_dir()])


def resolve_run_history(base_dir: Path, run_date: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    run_dirs = _list_run_dirs(base_dir)
    if not run_dirs:
        logging.info("No historical runs found under %s; skipping delta comparison.", base_dir)
        return None, None

    if run_date:
        current_dir = base_dir / f"run={run_date}"
        if not current_dir.exists():
            logging.warning("Requested run date %s not found under %s.", run_date, base_dir)
            return None, None
    else:
        current_dir = run_dirs[-1]

    # Align with list ordering
    idx = None
    for i, path in enumerate(run_dirs):
        if path == current_dir or (path.exists() and current_dir.exists() and path.resolve() == current_dir.resolve()):
            idx = i
            break
    if idx is None:
        run_dirs.append(current_dir)
        run_dirs.sort()
        idx = run_dirs.index(current_dir)

    previous_dir = run_dirs[idx - 1] if idx > 0 else None
    if previous_dir is None:
        logging.info("Previous run not available; skipping delta comparison.")
    return current_dir, previous_dir


def print_listing_deltas(new_counts: Dict[str, int], gone_counts: Dict[str, int]) -> None:
    print("\nListing deltas vs previous run")
    print("------------------------------")
    if not new_counts and not gone_counts:
        print("No differences detected.")
        return

    def _print_block(title: str, data: Dict[str, int]) -> None:
        print(title)
        if not data:
            print("  None")
            return
        for city, count in sorted(data.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {city}: {count:,}")

    _print_block("New listings by city:", new_counts)
    _print_block("Removed listings by city:", gone_counts)


def print_section(title: str, snapshot: CountSnapshot, extra: Optional[Dict[str, float]] = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Total: {snapshot.total:,}")
    if snapshot.breakdown:
        for key, value in snapshot.breakdown.items():
            print(f"  {key}: {value:,}")
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            print(f"  {key}: {value:.2%}")


def compute_ratios(scraped: CountSnapshot, normalized: CountSnapshot, gold_sources: CountSnapshot, gold_listings: CountSnapshot) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    ratios["normalized / scraped"] = normalized.ratio(max(scraped.total, 1))
    ratios["gold sources / normalized"] = gold_sources.ratio(max(normalized.total, 1))
    ratios["gold listings / normalized"] = gold_listings.ratio(max(normalized.total, 1))
    ratios["gold listings / gold sources"] = gold_listings.ratio(max(gold_sources.total, 1))
    return ratios


def print_transaction_summary(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n{label}\n" + "-" * len(label))
        print("  No data.")
        return
    grouped = df.groupby("transaction")["count"].sum()
    print(f"\n{label}")
    print("-" * len(label))
    for txn in TRANSACTION_ORDER + [cat for cat in grouped.index if cat not in TRANSACTION_ORDER]:
        if txn not in grouped:
            continue
        print(f"  {txn}: {grouped[txn]:,}")


def load_timings(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        logging.info("Timings file %s not found; skipping duration summary.", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse timings file %s: %s", path, exc)
        return None
    timings = data.get("timings_seconds")
    if not isinstance(timings, dict):
        logging.warning("Timings file %s missing 'timings_seconds'; skipping.", path)
        return None
    return {
        "generated_at": data.get("generated_at"),
        "run_date": data.get("run_date"),
        "transaction": data.get("transaction"),
        "timings": {str(k): float(v) for k, v in timings.items()},
    }


def print_timings(timing_info: Optional[Dict[str, object]]) -> None:
    if not timing_info:
        return
    timings = timing_info.get("timings", {})
    if not timings:
        return
    print("\nTask durations")
    print("--------------")
    meta_parts = []
    if timing_info.get("run_date"):
        meta_parts.append(f"run={timing_info['run_date']}")
    if timing_info.get("transaction"):
        meta_parts.append(f"transaction={timing_info['transaction']}")
    if timing_info.get("generated_at"):
        meta_parts.append(f"generated={timing_info['generated_at']}")
    if meta_parts:
        print("  " + ", ".join(meta_parts))
    order = [key for key in timings.keys() if key != "total_pipeline"]
    if "total_pipeline" in timings:
        order.append("total_pipeline")
    for key in order:
        seconds = timings[key]
        minutes, secs = divmod(seconds, 60)
        if minutes >= 1:
            display = f"{int(minutes)}m {secs:.1f}s"
        else:
            display = f"{secs:.1f}s"
        print(f"  {key}: {display}")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Scraped CSVs
    scraped_snapshot = collect_output_counts(args.output_dir)

    # Normalized data
    parquet_run_dir = find_run_directory(args.parquet_dir, args.run_date, allow_latest=False)
    normalized_path = parquet_run_dir / "normalized_sources.parquet"
    normalized_snapshot = load_parquet_counts(normalized_path, column="portal")

    # Gold sources and listings
    dedupe_dir = find_run_directory(args.dedupe_dir, args.run_date, allow_latest=True)
    gold_sources_path = dedupe_dir / "gold_listing_sources.parquet"
    gold_listings_path = dedupe_dir / "gold_listings.parquet"

    gold_sources_snapshot = load_parquet_counts(gold_sources_path, column="portal")
    gold_listings_snapshot = load_parquet_counts(gold_listings_path, column="primary_portal")

    city_stats_df = compute_city_transaction_stats(normalized_path)
    print_transaction_summary("Listings by transaction", city_stats_df)

    ratios = compute_ratios(scraped_snapshot, normalized_snapshot, gold_sources_snapshot, gold_listings_snapshot)

    print_section("Scraped CSV rows", scraped_snapshot)
    print_section("Normalized sources", normalized_snapshot)
    print_section("Gold listing sources", gold_sources_snapshot)
    print_section("Gold listings", gold_listings_snapshot)
    print("\nListings by city & transaction")
    print("------------------------------")
    if city_stats_df.empty:
        print("No city data available.")
    else:
        for _, row in city_stats_df.iterrows():
            city = row["address_town"] or "unknown"
            transaction = row["transaction"] or "unknown"
            count = row["count"]
            print(f"{city}: {transaction} -> {count:,}")

    print("\nRatios")
    print("------")
    for label, value in ratios.items():
        if value is None:
            print(f"{label}: n/a")
        else:
            print(f"{label}: {value:.2%}")

    current_run_dir, previous_run_dir = resolve_run_history(args.dedupe_dir, args.run_date)
    if current_run_dir and previous_run_dir:
        current_gold = current_run_dir / "gold_listings.parquet"
        previous_gold = previous_run_dir / "gold_listings.parquet"
        if current_gold.exists() and previous_gold.exists():
            new_counts, gone_counts = _compute_city_deltas(current_gold, previous_gold)
            print_listing_deltas(new_counts, gone_counts)
        else:
            print("\nListing deltas vs previous run")
            print("------------------------------")
            print("Gold listing parquet missing in current or previous run; skipping delta calculation.")

    timing_info = load_timings(args.timings_file)
    print_timings(timing_info)


if __name__ == "__main__":
    main()

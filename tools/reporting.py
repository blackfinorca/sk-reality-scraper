"""
Pipeline reporting utilities.

Generates a quick snapshot showing how many listings were scraped, normalised,
and deduplicated for a given run.  This helps validate the end-to-end funnel.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_PARQUET_DIR = Path("parquet_runs")
DEFAULT_DEDUPE_DIR = Path("dedupe_runs")


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


def compute_city_transaction_stats(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file {parquet_path} not found.")

    df = pd.read_parquet(parquet_path, columns=["address_town", "transaction"])
    if df.empty:
        return pd.DataFrame(columns=["address_town", "transaction", "count"])

    df["address_town"] = (
        df["address_town"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    df["transaction"] = (
        df["transaction"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    transaction_labels = {
        "sale": "predaj",
        "predaj": "predaj",
        "predam": "predaj",
        "rent": "prenajom",
        "lease": "prenajom",
        "unknown": "unknown",
    }
    df["transaction"] = df["transaction"].map(lambda x: transaction_labels.get(x, x))

    grouped = (
        df.groupby(["address_town", "transaction"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "address_town"], ascending=[False, True])
    )
    return grouped


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


if __name__ == "__main__":
    main()

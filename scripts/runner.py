import argparse
import copy
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.deduplicate import deduplicate_and_merge


DEFAULT_SOURCE_PRIORITY = ["nehnutelnosti_sk", "reality_sk", "bazos_sk"]


DEFAULT_CONFIG = {
    "nehnutelnosti": {
        "enabled": True,
        "city": ["trnava", "kosice"],
        "transaction": "predaj",
        "categories": "11,12,300001",
        "csv_output": "output/nehnutelnosti_output.csv",
        "xls_output": "output/nehnutelnosti_output.xls",
        "max_pages": None,
        "limit": None,
        "delay": 1,
        "workers": 4,
        "verbose": False,
    },
    "etl": {
        "enabled": True,
        "date": "2025-10-30",
        "site": "nehnutelnosti_sk",
        "csv_path": "output/nehnutelnosti_output.csv",
        "bucket": "b2://realestate",
        "currency": "EUR",
    },
    "reality": {
        "enabled": True,
        "property_type": "byty",
        "city": ["trnava"],
        "transaction": "predaj",
        "csv_output": "output/reality_output.csv",
        "xls_output": "output/reality_output.xls",
        "max_pages": None,
        "limit": None,
        "delay": 1.0,
        "workers": 1,
        "verbose": False,
    },
    "bazos": {
        "enabled": True,
        "city": "Trnava",
        "transaction": "predam",
        "csv_output": "output/bazos_output.csv",
        "xls_output": "output/bazos_output.xls",
        "max_pages": None,
        "limit": None,
        "delay": 1.5,
        "verbose": False,
    },
    "normalizer": {
        "enabled": True,
        "input": "./output/*.csv",
        "out_dir": "./parquet_runs",
        "source_priority": list(DEFAULT_SOURCE_PRIORITY),
        "prev_run": None,
    },
    "geo_locator": {
        "enabled": True,
        "user_agent": "RealEstate-Investor-Map/1.0 (contact: you@example.com)",
        "country": "Slovakia",
        "cache": "data/geocode_cache.json",
        "max_new": 200,
        "dry_run": False,
        "in_place": True,
    },
    "summarizer": {
        "enabled": True,
        "force": False,
        "dry_run": False,
        "analyze_only": False,
        "source_dir": "output",
        "summary_column": "summary_short_sk",
        "cache": "data/summary_cache.json",
        "in_place": True,
    },
}


def build_nehnutelnosti_command(config: Dict[str, object]) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scrapers" / "nehnutelnosti_scraper.py"),
        "--city",
        ",".join(config["city"]) if isinstance(config["city"], list) else str(config["city"]),
        "--transaction",
        str(config["transaction"]),
        "--categories",
        str(config["categories"]),
        "--csv-output",
        str(config["csv_output"]),
        "--output",
        str(config["xls_output"]),
        "--delay",
        str(config["delay"]),
        "--workers",
        str(config["workers"]),
    ]

    if config.get("max_pages"):
        cmd.extend(["--max-pages", str(config["max_pages"])])

    limit = config.get("limit")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if config.get("verbose"):
        cmd.append("--verbose")

    return cmd


def build_etl_command(config: Dict[str, object]) -> List[str]:
    date_value = config.get("date")
    if not date_value:
        date_value = datetime.now().strftime("%Y-%m-%d")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipelines" / "etl.py"),
        "--date",
        str(date_value),
        "--site",
        str(config["site"]),
        "--csv-path",
        str(config["csv_path"]),
        "--bucket",
        str(config["bucket"]),
        "--currency",
        str(config["currency"]),
    ]
    return cmd


def build_reality_command(config: Dict[str, object]) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scrapers" / "reality_scraper.py"),
        "--property-type",
        str(config["property_type"]),
        "--city",
        ",".join(config["city"]) if isinstance(config["city"], list) else str(config["city"]),
        "--transaction",
        str(config["transaction"]),
        "--csv-output",
        str(config["csv_output"]),
        "--output",
        str(config["xls_output"]),
        "--delay",
        str(config["delay"]),
        "--workers",
        str(config["workers"]),
    ]

    if config.get("max_pages"):
        cmd.extend(["--max-pages", str(config["max_pages"])])

    limit = config.get("limit")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if config.get("verbose"):
        cmd.append("--verbose")

    return cmd


def build_bazos_command(config: Dict[str, object]) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scrapers" / "bazos_reality_scraper.py"),
        "--city",
        str(config["city"]),
        "--transaction",
        str(config["transaction"]),
        "--csv-output",
        str(config["csv_output"]),
        "--output",
        str(config["xls_output"]),
        "--delay",
        str(config["delay"]),
    ]

    if config.get("max_pages"):
        cmd.extend(["--max-pages", str(config["max_pages"])])

    limit = config.get("limit")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if config.get("verbose"):
        cmd.append("--verbose")

    return cmd


def run_deduplicate_pipeline(
    project_root: Path,
    norm_out_dir: Path,
    run_date: str,
    source_priority: List[str],
) -> Optional[Path]:
    normalized_path = norm_out_dir / f"run={run_date}" / "normalized_sources.parquet"
    if not normalized_path.exists():
        logging.warning("Normalized sources parquet not found at %s; skipping deduplicate step.", normalized_path)
        return None

    logging.info("Loading normalized sources from %s", normalized_path)
    df = pd.read_parquet(normalized_path)
    if df.empty:
        logging.info("Normalized dataset is empty; skipping deduplicate step.")
        return None

    priority = list(source_priority) if source_priority else list(DEFAULT_SOURCE_PRIORITY)

    dedupe_root = project_root / "dedupe_runs"
    latest_dir = dedupe_root / "latest"
    prev_gold = prev_sources = None
    prev_gold_path = latest_dir / "gold_listings.parquet"
    prev_sources_path = latest_dir / "gold_listing_sources.parquet"
    if prev_gold_path.exists():
        prev_gold = pd.read_parquet(prev_gold_path)
    if prev_sources_path.exists():
        prev_sources = pd.read_parquet(prev_sources_path)

    logging.info("Running deduplication with %s source rows.", len(df))
    gold, sources, history = deduplicate_and_merge(
        df=df,
        run_date=run_date,
        source_priority=priority,
        prev_gold=prev_gold,
        prev_sources=prev_sources,
    )

    run_dir = dedupe_root / f"run={run_date}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gold.to_parquet(run_dir / "gold_listings.parquet", index=False)
    sources.to_parquet(run_dir / "gold_listing_sources.parquet", index=False)
    history.to_parquet(run_dir / "gold_listing_history.parquet", index=False)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(latest_dir / "gold_listings.parquet", index=False)
    sources.to_parquet(latest_dir / "gold_listing_sources.parquet", index=False)
    history.to_parquet(latest_dir / "gold_listing_history.parquet", index=False)

    cluster_counts = gold["cluster_size"].value_counts().sort_index().to_dict() if not gold.empty else {}
    logging.info(
        "Deduplicate produced %s golden listings (cluster distribution: %s).",
        len(gold),
        cluster_counts,
    )
    return latest_dir / "gold_listings.parquet"


def run_geo_locator(gold_path: Path, config: Dict[str, object], project_root: Path) -> None:
    if not config.get("enabled", True):
        return
    user_agent = config.get("user_agent")
    if not user_agent:
        logging.warning("Geo locator user agent missing; skipping geo enrichment.")
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "geo_locator.py"),
        "--parquet",
        str(gold_path),
    ]

    if config.get("in_place", True):
        cmd.append("--in-place")
    else:
        out_parquet = config.get("out_parquet")
        out_path = Path(out_parquet) if out_parquet else gold_path.with_name("gold_listings_geo.parquet")
        if not out_path.is_absolute():
            out_path = project_root / out_path
        cmd.extend(["--out-parquet", str(out_path)])

    cache_path = Path(config.get("cache", "data/geocode_cache.json"))
    if not cache_path.is_absolute():
        cache_path = project_root / cache_path
    cmd.extend(["--cache", str(cache_path)])

    country = config.get("country")
    if country:
        cmd.extend(["--country", str(country)])

    cmd.extend(["--user-agent", str(user_agent)])

    max_new = config.get("max_new")
    if max_new is not None:
        cmd.extend(["--max-new", str(max_new)])

    dry_run = str(config.get("dry_run", False)).lower() in {"true", "1", "yes"}
    cmd.extend(["--dry-run", "true" if dry_run else "false"])

    run_command("geo locator", cmd, project_root)


def run_summarizer(gold_path: Path, config: Dict[str, object], project_root: Path) -> None:
    if not config.get("enabled", True):
        return
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY not set; skipping summarizer.")
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "sumamrizer.py"),
        "--gold-parquet",
        str(gold_path),
    ]

    if config.get("in_place", True):
        cmd.append("--in-place")
    else:
        out_parquet = config.get("out_parquet")
        out_path = Path(out_parquet) if out_parquet else gold_path.with_name("gold_listings_with_summary.parquet")
        if not out_path.is_absolute():
            out_path = project_root / out_path
        cmd.extend(["--out-parquet", str(out_path)])

    source_dir = config.get("source_dir")
    if source_dir:
        dir_path = Path(source_dir)
        if not dir_path.is_absolute():
            dir_path = project_root / dir_path
        cmd.extend(["--source-dir", str(dir_path)])

    summary_column = config.get("summary_column")
    if summary_column:
        cmd.extend(["--summary-column", str(summary_column)])

    cache_path = config.get("cache")
    if cache_path:
        cache_path = Path(cache_path)
        if not cache_path.is_absolute():
            cache_path = project_root / cache_path
        cmd.extend(["--cache", str(cache_path)])

    if config.get("force"):
        cmd.append("--force")
    if config.get("dry_run"):
        cmd.append("--dry-run")
    if config.get("analyze_only"):
        cmd.append("--analyze-only")

    run_command("summarizer", cmd, project_root)


def run_command(label: str, command: List[str], cwd: Path) -> None:
    logging.info("Running %s command: %s", label, " ".join(command))
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} command failed with exit code {result.returncode}")


def load_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner for scraper and ETL pipeline.")
    parser.add_argument("--skip-scraper", action="store_true", help="Skip running the scraper.")
    parser.add_argument("--skip-nehnutelnosti", action="store_true", help="Skip Nehnutelnosti.sk scraping.")
    parser.add_argument("--skip-reality", action="store_true", help="Skip Reality.sk scraping.")
    parser.add_argument("--skip-bazos", action="store_true", help="Skip Bazos.sk scraping.")
    parser.add_argument(
        "--all-cities",
        help="Apply the same city filter to every scraper (comma-separated list).",
    )
    parser.add_argument(
        "--all-transaction",
        help="Apply the same transaction type to every scraper.",
    )
    parser.add_argument(
        "--all-max-pages",
        type=int,
        help="Apply the same max pages limit to every scraper.",
    )
    parser.add_argument(
        "--all-limit",
        type=int,
        help="Apply the same listings limit to every scraper.",
    )
    parser.add_argument(
        "--all-delay",
        type=float,
        help="Apply the same request delay to every scraper.",
    )
    parser.add_argument(
        "--all-verbose",
        action="store_true",
        help="Enable verbose logging for every scraper.",
    )
    parser.add_argument("--cities", help="Override Nehnutelnosti.sk city slugs (comma-separated).")
    parser.add_argument("--reality-cities", help="Override Reality.sk city slugs (comma-separated).")
    parser.add_argument("--bazos-city", help="Override Bazos.sk city filter.")
    parser.add_argument("--skip-etl", action="store_true", help="Skip running the ETL.")
    parser.add_argument("--skip-normalizer", action="store_true", help="Skip running the normalizer pipeline.")
    parser.add_argument("--config-date", help="Override ETL date (YYYY-MM-DD).")
    parser.add_argument("--config-site", help="Override ETL site.")
    parser.add_argument("--config-bucket", help="Override ETL bucket.")
    parser.add_argument("--config-csv", help="Override ETL CSV path.")
    parser.add_argument("--scraper-limit", type=int, help="Override scraper limit.")
    parser.add_argument("--scraper-workers", type=int, help="Override scraper worker count.")
    parser.add_argument("--scraper-delay", type=float, help="Override scraper delay.")
    parser.add_argument("--scraper-verbose", action="store_true", help="Enable scraper verbose mode.")
    parser.add_argument("--reality-limit", type=int, help="Override Reality.sk scraper limit.")
    parser.add_argument("--reality-delay", type=float, help="Override Reality.sk scraper delay.")
    parser.add_argument("--reality-verbose", action="store_true", help="Enable Reality.sk scraper verbose mode.")
    parser.add_argument("--bazos-limit", type=int, help="Override Bazos.sk scraper limit.")
    parser.add_argument("--bazos-delay", type=float, help="Override Bazos.sk scraper delay.")
    parser.add_argument("--bazos-verbose", action="store_true", help="Enable Bazos.sk scraper verbose mode.")
    parser.add_argument("--skip-geo", action="store_true", help="Skip geo locator enrichment.")
    parser.add_argument("--skip-summarizer", action="store_true", help="Skip summary generation.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = load_arguments()

    nehn_cfg = copy.deepcopy(DEFAULT_CONFIG["nehnutelnosti"])
    reality_cfg = copy.deepcopy(DEFAULT_CONFIG["reality"])
    bazos_cfg = copy.deepcopy(DEFAULT_CONFIG["bazos"])
    etl_cfg = copy.deepcopy(DEFAULT_CONFIG["etl"])
    today_str = datetime.now().strftime("%Y-%m-%d")
    etl_cfg["date"] = today_str
    norm_cfg = copy.deepcopy(DEFAULT_CONFIG["normalizer"])
    geo_cfg = copy.deepcopy(DEFAULT_CONFIG["geo_locator"])
    summarizer_cfg = copy.deepcopy(DEFAULT_CONFIG["summarizer"])

    if args.all_cities:
        shared_cities = [slug.strip() for slug in args.all_cities.split(",") if slug.strip()]
        if shared_cities:
            nehn_cfg["city"] = shared_cities
            reality_cfg["city"] = shared_cities
            bazos_cfg["city"] = ",".join(shared_cities)
    if args.all_transaction:
        transaction_value = str(args.all_transaction)
        nehn_cfg["transaction"] = transaction_value
        reality_cfg["transaction"] = transaction_value
        bazos_cfg["transaction"] = transaction_value
    if args.all_max_pages is not None:
        nehn_cfg["max_pages"] = args.all_max_pages
        reality_cfg["max_pages"] = args.all_max_pages
        bazos_cfg["max_pages"] = args.all_max_pages
    if args.all_limit is not None:
        nehn_cfg["limit"] = args.all_limit
        reality_cfg["limit"] = args.all_limit
        bazos_cfg["limit"] = args.all_limit
    if args.all_delay is not None:
        nehn_cfg["delay"] = args.all_delay
        reality_cfg["delay"] = args.all_delay
        bazos_cfg["delay"] = args.all_delay
    if args.all_verbose:
        nehn_cfg["verbose"] = True
        reality_cfg["verbose"] = True
        bazos_cfg["verbose"] = True

    if args.config_date:
        etl_cfg["date"] = args.config_date
    if args.config_site:
        etl_cfg["site"] = args.config_site
    if args.config_bucket:
        etl_cfg["bucket"] = args.config_bucket
    if args.config_csv:
        etl_cfg["csv_path"] = args.config_csv
        nehn_cfg["csv_output"] = args.config_csv

    if args.cities:
        nehn_cfg["city"] = [slug.strip() for slug in args.cities.split(",") if slug.strip()]
    if args.scraper_limit is not None:
        nehn_cfg["limit"] = args.scraper_limit
    if args.scraper_workers is not None:
        nehn_cfg["workers"] = args.scraper_workers
    if args.scraper_delay is not None:
        nehn_cfg["delay"] = args.scraper_delay
    if args.scraper_verbose:
        nehn_cfg["verbose"] = True

    if args.reality_cities:
        reality_cfg["city"] = [slug.strip() for slug in args.reality_cities.split(",") if slug.strip()]
    if args.reality_limit is not None:
        reality_cfg["limit"] = args.reality_limit
    if args.reality_delay is not None:
        reality_cfg["delay"] = args.reality_delay
    if args.reality_verbose:
        reality_cfg["verbose"] = True

    if args.bazos_city:
        bazos_cfg["city"] = args.bazos_city
    if args.bazos_limit is not None:
        bazos_cfg["limit"] = args.bazos_limit
    if args.bazos_delay is not None:
        bazos_cfg["delay"] = args.bazos_delay
    if args.bazos_verbose:
        bazos_cfg["verbose"] = True

    if args.skip_normalizer:
        norm_cfg["enabled"] = False
    if args.skip_geo:
        geo_cfg["enabled"] = False
    if args.skip_summarizer:
        summarizer_cfg["enabled"] = False
    skip_nehn = args.skip_scraper or args.skip_nehnutelnosti
    skip_reality = args.skip_scraper or args.skip_reality
    skip_bazos = args.skip_scraper or args.skip_bazos

    run_nehn = nehn_cfg["enabled"] and not skip_nehn
    run_reality = reality_cfg["enabled"] and not skip_reality
    run_bazos = bazos_cfg["enabled"] and not skip_bazos
    run_etl = etl_cfg["enabled"] and not args.skip_etl
    run_normalizer = norm_cfg["enabled"] and not args.skip_normalizer

    project_root = PROJECT_ROOT
    default_out_dir = norm_cfg.get("out_dir", "parquet_runs")
    norm_cfg["out_dir"] = str((project_root / default_out_dir).resolve())

    if run_nehn and (not nehn_cfg.get("city")):
        raise ValueError("At least one city must be specified for Nehnutelnosti.sk scraping.")
    if run_reality and (not reality_cfg.get("city")):
        raise ValueError("At least one city must be specified for Reality.sk scraping.")
    if run_bazos and (not bazos_cfg.get("city")):
        raise ValueError("City must be specified for Bazos.sk scraping.")

    if run_nehn:
        nehn_cmd = build_nehnutelnosti_command(nehn_cfg)
        run_command("nehnutelnosti.sk scraper", nehn_cmd, project_root)
        etl_cfg["csv_path"] = nehn_cfg["csv_output"]

    if run_reality:
        reality_cmd = build_reality_command(reality_cfg)
        run_command("reality.sk scraper", reality_cmd, project_root)
    if run_bazos:
        bazos_cmd = build_bazos_command(bazos_cfg)
        run_command("bazos.sk scraper", bazos_cmd, project_root)
    scraped_csvs = []
    if run_nehn:
        scraped_csvs.append(nehn_cfg["csv_output"])
    if run_reality:
        scraped_csvs.append(reality_cfg["csv_output"])
    if run_bazos:
        scraped_csvs.append(bazos_cfg["csv_output"])

    etl_jobs = []
    if run_nehn:
        etl_jobs.append({"csv_path": nehn_cfg["csv_output"], "site": "nehnutelnosti_sk"})
    if run_reality:
        etl_jobs.append({"csv_path": reality_cfg["csv_output"], "site": "reality_sk"})
    if run_bazos:
        etl_jobs.append({"csv_path": bazos_cfg["csv_output"], "site": "bazos_sk"})

    if run_etl:
        for job in etl_jobs or [{"csv_path": etl_cfg["csv_path"], "site": etl_cfg.get("site", "unknown")}]:
            job_cfg = copy.deepcopy(etl_cfg)
            job_cfg["csv_path"] = job["csv_path"]
            job_cfg["site"] = job.get("site", job_cfg.get("site"))
            if args.config_site:
                job_cfg["site"] = args.config_site
            job_cfg["date"] = job_cfg.get("date") or today_str
            csv_path = Path(job_cfg["csv_path"])
            if not csv_path.is_absolute():
                job_cfg["csv_path"] = str(project_root / csv_path)
            etl_cmd = build_etl_command(job_cfg)
            run_command(f"etl ({job_cfg['site']})", etl_cmd, project_root)

    gold_latest_path: Optional[Path] = None

    if run_normalizer:
        norm_cfg["input"] = str(project_root / "output" / "*.csv")
        norm_cfg["run_date"] = datetime.now().strftime("%Y-%m-%d")
        norm_command = [
            sys.executable,
            str(PROJECT_ROOT / "pipelines" / "normalizer.py"),
            "--input",
            norm_cfg["input"],
            "--out-dir",
            norm_cfg["out_dir"],
            "--run-date",
            norm_cfg["run_date"],
            "--source-priority",
            ",".join(norm_cfg.get("source_priority", DEFAULT_SOURCE_PRIORITY)),
        ]
        prev_run = norm_cfg.get("prev_run")
        if prev_run:
            norm_command.extend(["--prev-run", prev_run])
        if geo_cfg.get("enabled", True) and geo_cfg.get("user_agent"):
            norm_command.append("--geo-enabled")
            norm_command.extend(["--geo-user-agent", str(geo_cfg["user_agent"])])
            norm_command.extend(["--geo-country", str(geo_cfg.get("country", "Slovakia"))])
            geo_cache = Path(geo_cfg.get("cache", "data/geocode_cache.json"))
            if not geo_cache.is_absolute():
                geo_cache = project_root / geo_cache
            norm_command.extend(["--geo-cache", str(geo_cache)])
            max_new = geo_cfg.get("max_new")
            if max_new is not None:
                norm_command.extend(["--geo-max-new", str(max_new)])
        run_command("normalizer", norm_command, project_root)
        gold_latest_path = run_deduplicate_pipeline(
            project_root=project_root,
            norm_out_dir=Path(norm_cfg["out_dir"]),
            run_date=norm_cfg["run_date"],
            source_priority=norm_cfg.get("source_priority", DEFAULT_SOURCE_PRIORITY),
        )

    if gold_latest_path is None:
        default_gold = project_root / "dedupe_runs" / "latest" / "gold_listings.parquet"
        if default_gold.exists():
            gold_latest_path = default_gold

    if gold_latest_path and gold_latest_path.exists():
        run_geo_locator(gold_latest_path, geo_cfg, project_root)
        run_summarizer(gold_latest_path, summarizer_cfg, project_root)
    else:
        logging.info("Skipping geo locator and summarizer; gold listings parquet not available.")

    logging.info("Runner completed.")


if __name__ == "__main__":
    main()

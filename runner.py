import argparse
import copy
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_SOURCE_PRIORITY = ["nehnutelnosti_sk", "reality_sk", "bazos_sk"]


DEFAULT_CONFIG = {
    "nehnutelnosti": {
        "enabled": True,
        "city": ["trnava"],
        "transaction": "predaj",
        "categories": "11,12,300001",
        "csv_output": "output/nehnutelnosti_trnava_byty.csv",
        "xls_output": "output/nehnutelnosti_trnava_byty.xls",
        "max_pages": None,
        "limit": 30,
        "delay": 1.2,
        "workers": 1,
        "verbose": False,
    },
    "etl": {
        "enabled": True,
        "date": "2025-10-30",
        "site": "nehnutelnosti_sk",
        "csv_path": "output/nehnutelnosti_trnava_byty.csv",
        "bucket": "b2://realestate",
        "currency": "EUR",
    },
    "reality": {
        "enabled": True,
        "property_type": "byty",
        "city": ["trnava"],
        "transaction": "predaj",
        "csv_output": "output/reality_trnava_byty.csv",
        "xls_output": "output/reality_trnava_byty.xls",
        "max_pages": None,
        "limit": 30,
        "delay": 1.0,
        "workers": 1,
        "verbose": False,
    },
    "bazos": {
        "enabled": True,
        "city": "Trnava",
        "transaction": "predam",
        "csv_output": "output/bazos_trnava_byty.csv",
        "xls_output": "output/bazos_trnava_byty.xls",
        "max_pages": None,
        "limit": 30,
        "delay": 1.5,
        "verbose": False,
    },
    "normalizer": {
        "enabled": True,
        "input": "./output/*byty.csv",
        "out_dir": "./parquet_runs",
        "source_priority": list(DEFAULT_SOURCE_PRIORITY),
        "prev_run": None,
    },
}


def build_nehnutelnosti_command(config: Dict[str, object]) -> List[str]:
    cmd = [
        sys.executable,
        "scraper.py",
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
    cmd = [
        sys.executable,
        "etl.py",
        "--date",
        str(config["date"]),
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
        "reality_scraper.py",
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
        "bazos_reality_trnava.py",
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = load_arguments()

    nehn_cfg = copy.deepcopy(DEFAULT_CONFIG["nehnutelnosti"])
    reality_cfg = copy.deepcopy(DEFAULT_CONFIG["reality"])
    bazos_cfg = copy.deepcopy(DEFAULT_CONFIG["bazos"])
    etl_cfg = copy.deepcopy(DEFAULT_CONFIG["etl"])
    norm_cfg = copy.deepcopy(DEFAULT_CONFIG["normalizer"])

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
    skip_nehn = args.skip_scraper or args.skip_nehnutelnosti
    skip_reality = args.skip_scraper or args.skip_reality
    skip_bazos = args.skip_scraper or args.skip_bazos

    run_nehn = nehn_cfg["enabled"] and not skip_nehn
    run_reality = reality_cfg["enabled"] and not skip_reality
    run_bazos = bazos_cfg["enabled"] and not skip_bazos
    run_etl = etl_cfg["enabled"] and not args.skip_etl
    run_normalizer = norm_cfg["enabled"] and not args.skip_normalizer

    project_root = Path(__file__).resolve().parent
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
            csv_path = Path(job_cfg["csv_path"])
            if not csv_path.is_absolute():
                job_cfg["csv_path"] = str(project_root / csv_path)
            etl_cmd = build_etl_command(job_cfg)
            run_command(f"etl ({job_cfg['site']})", etl_cmd, project_root)

    if run_normalizer:
        norm_cfg["input"] = str(project_root / "output" / "*byty.csv")
        norm_cfg["run_date"] = datetime.now().strftime("%Y-%m-%d")
        norm_command = [
            sys.executable,
            "normalizer.py",
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
        run_command("normalizer", norm_command, project_root)

    logging.info("Runner completed.")


if __name__ == "__main__":
    main()

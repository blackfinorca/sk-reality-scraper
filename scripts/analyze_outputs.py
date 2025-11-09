# Analyze scraped output CSV files and summarize missing values.

import argparse
import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_FILES = [
    "output/nehnutelnosti_predaj_output.csv",
    "output/nehnutelnosti_prenajom_output.csv",
    "output/reality_predaj_output.csv",
    "output/reality_prenajom_output.csv",
    "output/bazos_predaj_output.csv",
    "output/bazos_prenajom_output.csv",
]

KEY_FIELDS = [
    "price",
    "price_area",
    "floor_area",
    "room_number",
    "photo_url_1",
    "photo_url_2",
    "photo_url_3",
]


def load_csv(path: Path):
    with path.open(newline="", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows, reader.fieldnames or []


def main():
    parser = argparse.ArgumentParser(description="Summarize missing values in scraped output files.")
    parser.add_argument(
        "--directory",
        default=Path("output"),
        type=Path,
        help="Directory containing output CSV files (default: ./output)",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=KEY_FIELDS,
        help="Specific fields to inspect (default: price, price_area, floor_area, room_number, photo_url_1-3)",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    fields = args.fields

    print(f"Analyzing output files in {directory} …\n")

    for filename in OUTPUT_FILES:
        path = directory / Path(filename).name
        if not path.exists():
            continue

        rows, headers = load_csv(path)
        total = len(rows)
        if total == 0:
            print(f"{path.name}: no rows found")
            continue

        missing_counts = defaultdict(int)
        for row in rows:
            for field in fields:
                if field not in headers:
                    missing_counts[field] += 1
                else:
                    value = row.get(field, "")
                    if value is None or str(value).strip() == "":
                        missing_counts[field] += 1

        print(f"{path.name}: {total} rows")
        for field in fields:
            missing = missing_counts[field]
            percent = (missing / total) * 100
            status = "missing" if missing else "ok"
            print(f"  - {field:12s}: {status:8s} ({missing}/{total} = {percent:.1f}%)")
        print("")


if __name__ == "__main__":
    main()

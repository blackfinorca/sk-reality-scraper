# Real Estate Scraper

Scrape real estate listings from [nehnutelnosti.sk](https://www.nehnutelnosti.sk/) and export the data to an Excel `.xls` file.

## Features

- Fetches listings for a selected city, transaction type, and list of category IDs.
- Visits every listing detail page to collect structured data (price, price per m², floor, number of rooms, development name, energy certificate, description, etc.).
- Streams results into a CSV file row-by-row (so partial data is saved even if the run stops early).
- Writes the final results to an Excel workbook (`.xls`) that is compatible with legacy spreadsheet software.
- Includes throttling with gentle randomised delays, retry logic, a progress bar, and CLI options for customisation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python scraper.py \
  --city trnava \
  --transaction predaj \
  --categories 11,12,300001 \
  --output trnava_byty.xls \
  --csv-output trnava_byty.csv \
  --limit 5
```

### Useful options

- `--max-pages`: limit how many search result pages to traverse.
- `--limit`: stop after scraping a given number of listings (useful for quick tests).
- `--delay`: seconds to wait between detail requests (default: `1.0`).
- `--verbose`: enable debug logging.

The defaults already match the request for Trnava apartments for sale (categories `11`, `12`, `300001`).

## Notes

- The script relies on page structure that can change; if parsing fails, increase the delay or re-run after inspecting new HTML structure.
- Respect the website's terms of service and avoid making frequent scraping runs.
- When scraping the provided Trnava apartments link the script logs the total count reported by the site (currently 314–315 listings depending on live data) and adjusts automatically.

import argparse
import csv
import logging
import math
import random
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response, Session
from requests.exceptions import HTTPError
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from listing_schema import (
    COLUMN_ORDER,
    COLUMN_RENAMES,
    CSV_FIELDNAMES,
    ListingRecord,
    sanitize_csv_value,
)


LISTING_ID_PATTERN = re.compile(r"/detail/([^/]+)/")
ROOM_COUNT_PATTERN = re.compile(r"(\d+)\s*izb", re.IGNORECASE)
TOTAL_COUNT_PATTERN = re.compile(r'\\"totalCount\\":\s*(\d+)')
EXPECTED_TOTAL_COUNT = 314

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "sk-SK,sk;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}


Listing = ListingRecord


def create_session() -> Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    session.timeout = 30  # type: ignore[attr-defined]
    return session


def build_base_url(city: str, transaction: str) -> str:
    return f"https://www.nehnutelnosti.sk/vysledky/{city}/{transaction}"


def build_query_params(categories: Sequence[str], page: int) -> List[Tuple[str, str]]:
    params: List[Tuple[str, str]] = [("categories", cat.strip()) for cat in categories if cat.strip()]
    if page > 1:
        params.append(("page", str(page)))
    return params


def request_with_retry(session: Session, url: str, params: Optional[Sequence[Tuple[str, str]]] = None,
                       retries: int = 3, backoff: float = 2.0) -> Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logging.warning("Request failed (attempt %s/%s) for %s: %s", attempt, retries, url, exc)
            time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


def gentle_pause(base_delay: float) -> None:
    if base_delay <= 0:
        return
    jitter = random.uniform(-0.2, 0.2)
    time.sleep(max(0.0, base_delay + jitter))


def parse_listing_page(html: str, base_url: str) -> Tuple[List[Tuple[str, str]], Optional[int]]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.select('a[href*="/detail/"]'):
        href = anchor.get("href")
        if not href:
            continue
        if href.startswith("/"):
            href = urljoin(base_url, href)
        match = LISTING_ID_PATTERN.search(href)
        if not match:
            continue
        listing_id = match.group(1)
        if listing_id in seen:
            continue
        seen.add(listing_id)
        links.append((listing_id, href))

    total_count: Optional[int] = None
    match = TOTAL_COUNT_PATTERN.search(html)
    if match:
        try:
            total_count = int(match.group(1))
        except ValueError:
            total_count = None

    return links, total_count


def extract_attributes(soup: BeautifulSoup) -> Dict[str, str]:
    heading = soup.find(lambda tag: tag.name in {"h2", "h3"} and "Vlastnosti nehnuteľnosti" in tag.get_text())
    attributes: Dict[str, str] = {}
    if not heading:
        return attributes

    container = heading
    while container and "MuiBox-root" not in (container.get("class") or []):
        container = container.parent  # type: ignore[assignment]

    if not container:
        return attributes

    for stack in container.find_all("div", class_=lambda c: c and "MuiStack-root" in c):
        texts = stack.find_all("p", attrs={"data-test-id": "text"})
        if len(texts) >= 2:
            label = texts[0].get_text(strip=True).rstrip(":")
            value = texts[1].get_text(strip=True)
            if label:
                attributes[label] = value
        elif len(texts) == 1:
            label = texts[0].get_text(strip=True)
            if label and label not in attributes:
                attributes[label] = ""
    return attributes


def extract_location(soup: BeautifulSoup) -> str:
    for tag in soup.find_all("p", attrs={"data-test-id": "text"}):
        text = tag.get_text(strip=True)
        if not text:
            continue
        lowered = text.lower()
        if "okres" in lowered and "," in text:
            return text
    return ""


def extract_number_of_rooms(type_text: str, attributes: Dict[str, str]) -> str:
    attr_key = next((key for key in attributes if "izieb" in key.lower()), None)
    if attr_key:
        value = attributes[attr_key]
        if value:
            return value

    match = ROOM_COUNT_PATTERN.search(type_text.lower())
    if match:
        return match.group(1)
    return ""


def detect_novostavba(attributes: Dict[str, str], soup: BeautifulSoup) -> bool:
    for key, value in attributes.items():
        combined = f"{key} {value}".lower()
        if "novostavba" in combined:
            return True

    tag = soup.find("p", string=lambda s: isinstance(s, str) and "Novostavba" in s)
    if tag:
        sibling = tag.find_next_sibling("p")
        has_value = sibling is not None and sibling.get_text(strip=True)
        if not has_value:
            return True
    return False


def extract_field(attributes: Dict[str, str], keywords: Sequence[str], *, word_boundary: bool = False) -> str:
    for key, value in attributes.items():
        lowered_key = key.lower()
        for keyword in keywords:
            lowered_keyword = keyword.lower()
            if word_boundary:
                pattern = rf"\b{re.escape(lowered_keyword)}\b"
                if re.search(pattern, lowered_key):
                    return value
            else:
                if lowered_keyword in lowered_key:
                    return value
    return ""


def parse_numeric_value(raw_value: str) -> Optional[int]:
    if not raw_value:
        return None
    cleaned = (
        raw_value.replace("\xa0", "")
        .replace(" ", "")
        .replace("\u202f", "")
        .replace(",", ".")
    )
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    if not cleaned:
        return None
    if cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return int(round(value))


def parse_year_value(raw_value: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", raw_value or "")
    if not match:
        return None
    return int(match.group(0))


def derive_address_parts(location: str, attributes: Dict[str, str], default_city: str) -> Tuple[str, str, str]:
    def looks_like_street(text: str) -> bool:
        lowered = text.lower()
        street_keywords = (
            "ulica",
            "námestie",
            "namestie",
            "cesta",
            "trieda",
            "náb",
            "nabrezie",
            "sídlisko",
            "sidlisko",
            "nám.",
            "nam.",
            "alej",
        )
        if any(keyword in lowered for keyword in street_keywords):
            return True
        return bool(re.search(r"\d", text))

    street = (extract_field(attributes, ("ulica", "adresa")) or "").strip()
    town_attr = (extract_field(attributes, ("mesto", "obec", "lokalita")) or "").strip()
    town = town_attr or default_city
    area = (extract_field(attributes, ("okres",)) or "").strip()

    non_area_segments: List[str] = []
    if location:
        for segment in (seg.strip() for seg in location.split(",") if seg.strip()):
            if "okres" in segment.lower():
                if not area:
                    area = segment
            else:
                non_area_segments.append(segment)

    def split_dash(segment: str) -> List[str]:
        return [part.strip() for part in segment.split("-") if part.strip()]

    if not street and non_area_segments:
        first_segment = non_area_segments[0]
        dash_parts = split_dash(first_segment)
        if len(dash_parts) >= 2:
            candidate_street = dash_parts[0]
            candidate_town = dash_parts[-1]
            if looks_like_street(candidate_street) or candidate_street.lower() != default_city.lower():
                street = candidate_street
            if not town_attr and candidate_town.lower() != default_city.lower() and not looks_like_street(candidate_town):
                town = candidate_town
        else:
            street = first_segment

    if len(non_area_segments) > 1:
        for segment in reversed(non_area_segments[1:]):
            dash_parts = split_dash(segment)
            candidate = dash_parts[-1] if dash_parts else segment
            if not town_attr and candidate.lower() not in {default_city.lower(), (street or "").lower()} and not looks_like_street(candidate):
                town = candidate
                break

    if not street:
        for segment in non_area_segments:
            if looks_like_street(segment):
                street = segment
                break
        if not street and non_area_segments:
            candidate = non_area_segments[0]
            if candidate.lower() != town.lower():
                street = candidate

    if not town:
        town = default_city

    if street and town and street.lower() == town.lower():
        town = default_city

    return street or "", town or "", area or ""


def extract_photo_urls(soup: BeautifulSoup, detail_url: str, limit: int = 3) -> List[str]:
    photo_urls: List[str] = []
    seen: set[str] = set()

    def clean_url(raw_url: Optional[str]) -> Optional[str]:
        if not raw_url:
            return None
        url = raw_url.strip()
        if not url or url.startswith("data:"):
            return None
        if url.startswith("//"):
            url = f"https:{url}"
        elif url.startswith("/"):
            url = urljoin(detail_url, url)
        if url.lower().endswith(".svg"):
            return None
        return url

    def extract_src(img_tag) -> Optional[str]:
        for attr in ("data-src", "data-original", "src", "data-srcset", "srcset"):
            value = img_tag.get(attr)
            if not value:
                continue
            if attr.endswith("srcset"):
                first_entry = value.split(",")[0].strip()
                value = first_entry.split(" ")[0]
            cleaned = clean_url(value)
            if cleaned:
                return cleaned
        return None

    selectors = [
        '[data-testid="gallery-image"] img',
        '[data-testid="gallery"] img',
        'figure img',
        'img',
    ]

    for selector in selectors:
        for img in soup.select(selector):
            candidate = extract_src(img)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            photo_urls.append(candidate)
            if len(photo_urls) >= limit:
                return photo_urls
    return photo_urls


def extract_description(soup: BeautifulSoup) -> str:
    def iter_json_descriptions() -> List[str]:
        descriptions: List[str] = []

        def visit(node) -> None:
            if isinstance(node, dict):
                value = node.get("description")
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned:
                        descriptions.append(cleaned)
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        for script in soup.find_all("script", type="application/ld+json"):
            text = script.string or ""
            if not text.strip():
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            visit(data)
            if descriptions:
                break
        return descriptions

    json_descriptions = iter_json_descriptions()
    if json_descriptions:
        return json_descriptions[0]

    selectors = [
        '[itemprop="description"]',
        '[data-test-id="description"]',
        '[data-testid="description"]',
        'section[data-testid="property-description"]',
        'section[data-test-id="property-description"]',
    ]
    for selector in selectors:
        container = soup.select_one(selector)
        if container:
            text = container.get_text("\n", strip=True)
            if text:
                return text

    fallback_container = soup.find("article")
    if fallback_container:
        text = fallback_container.get_text("\n", strip=True)
        if text:
            return text
    return ""


def parse_listing_detail(listing_id: str, url: str, html: str, city_slug: str, transaction: str) -> Listing:
    soup = BeautifulSoup(html, "html.parser")
    type_heading = soup.find("h2")
    type_text = type_heading.get_text(strip=True) if type_heading else ""
    cleaned_type = re.sub(r"\s+na\s+predaj$", "", type_text, flags=re.IGNORECASE)
    cleaned_type = re.sub(r"\s+na\s+prenájom$", "", cleaned_type, flags=re.IGNORECASE)

    title_tag = soup.find("h1")
    property_name = title_tag.get_text(strip=True) if title_tag else ""
    if not property_name:
        property_name = cleaned_type or type_text

    attributes = extract_attributes(soup)
    room_number_str = extract_number_of_rooms(cleaned_type or type_text, attributes)
    room_number = parse_numeric_value(room_number_str)

    price_tag = soup.find("p", class_=lambda c: c and "MuiTypography-h3" in c)
    price_text = price_tag.get_text(strip=True) if price_tag else ""
    price = parse_numeric_value(price_text)

    unit_tag = soup.find("p", string=lambda s: isinstance(s, str) and "€/m²" in s)
    price_per_m2_text = unit_tag.get_text(strip=True) if unit_tag else ""
    price_area = parse_numeric_value(price_per_m2_text)

    location = extract_location(soup)
    city_name = " ".join(part.capitalize() for part in re.split(r"[-_]", city_slug) if part)
    address_street, address_town, address_area = derive_address_parts(location, attributes, city_name)

    floor_area_value = extract_field(attributes, ("úžitkov", "uzitkov", "podlah", "plocha bytu", "plocha"))
    floor_area = parse_numeric_value(floor_area_value)

    building_status = extract_field(attributes, ("stav", "status", "vyhotovenie"), word_boundary=True)
    if not building_status and detect_novostavba(attributes, soup):
        building_status = "Novostavba"

    year_of_construction_value = extract_field(
        attributes,
        ("rok výstavby", "rok vystavby", "rok rekonštrukcie", "rok prestavby"),
    )
    year_of_construction = parse_year_value(year_of_construction_value)

    year_of_top_value = extract_field(attributes, ("rok kolaud", "kolaud"))
    year_of_top = parse_year_value(year_of_top_value)

    floor = extract_field(attributes, ("podlaž", "poschod"))
    energ_cert = extract_field(attributes, ("energet", "certifik"))
    photo_urls = extract_photo_urls(soup, url, limit=3)
    description = extract_description(soup)
    if not description and property_name:
        description = property_name

    return Listing(
        source="nehnutelnosti_sk",
        transaction=transaction,
        property_id=listing_id,
        link=url,
        property_name=property_name,
        description=description,
        address_street=address_street,
        address_town=address_town,
        address_zrea=address_area,
        room_number=room_number,
        floor_area=floor_area,
        building_status=building_status,
        year_of_construction=year_of_construction,
        year_of_top=year_of_top,
        floor=floor,
        energ_cert=energ_cert,
        price=price,
        price_area=price_area,
        photo_url_1=photo_urls[0] if len(photo_urls) > 0 else "",
        photo_url_2=photo_urls[1] if len(photo_urls) > 1 else "",
        photo_url_3=photo_urls[2] if len(photo_urls) > 2 else "",
    )
def write_listing_csv_row(listing: Listing, csv_writer: csv.DictWriter, csv_file) -> None:
    row_dict = asdict(listing)
    mapped_row = {}
    for key in COLUMN_ORDER:
        value = row_dict.get(key, "")
        if key == "description":
            mapped_row[COLUMN_RENAMES[key]] = "" if value is None else str(value)
        else:
            mapped_row[COLUMN_RENAMES[key]] = sanitize_csv_value(value)
    csv_writer.writerow(mapped_row)
    csv_file.flush()


def scrape_listings(city: str, transaction: str, categories: Sequence[str], max_pages: Optional[int],
                    delay: float, limit: Optional[int], workers: int,
                    csv_writer: Optional[csv.DictWriter] = None, csv_file=None) -> List[Listing]:
    session = create_session()
    base_url = build_base_url(city, transaction)
    all_listings: List[Listing] = []
    seen_ids: set[str] = set()
    progress_bar: Optional[tqdm] = None
    reported_total: Optional[int] = None
    estimated_pages: Optional[int] = None
    results_per_page: Optional[int] = None

    worker_count = max(1, workers)

    def handle_listing(listing: Listing) -> None:
        all_listings.append(listing)
        seen_ids.add(listing.property_id)
        if progress_bar:
            progress_bar.update(1)
        if csv_writer and csv_file:
            write_listing_csv_row(listing, csv_writer, csv_file)
        logging.info("Scraped %s listings so far.", len(all_listings))

    def fetch_listing_detail_concurrent(listing_id: str, link: str) -> Listing:
        logging.info("Fetching detail for %s", listing_id)
        local_session = create_session()
        try:
            detail_response = request_with_retry(local_session, link)
        finally:
            local_session.close()
        listing = parse_listing_detail(listing_id, link, detail_response.text, city, transaction)
        gentle_pause(delay)
        return listing

    page = 1
    while True:
        if max_pages and page > max_pages:
            break

        params = build_query_params(categories, page)
        logging.info("Fetching search page %s", page)
        try:
            response = request_with_retry(session, base_url, params=params)
        except HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                logging.info("Search page %s returned 404, assuming no more pages.", page)
                break
            raise
        listings, total_count = parse_listing_page(response.text, base_url)

        if reported_total is None and total_count is not None:
            reported_total = total_count
            logging.info("Search reports %s total listings.", reported_total)
            if reported_total == EXPECTED_TOTAL_COUNT:
                logging.info("Confirmed expected total of %s listings.", EXPECTED_TOTAL_COUNT)
            else:
                logging.warning(
                    "Reported total %s differs from expected %s listings.",
                    reported_total,
                    EXPECTED_TOTAL_COUNT,
                )
            if progress_bar is None:
                progress_total = limit if limit else reported_total
                progress_bar = tqdm(total=progress_total, unit="listing", desc=f"Scraping {city}")
        elif progress_bar is None and limit:
            progress_bar = tqdm(total=limit, unit="listing", desc=f"Scraping {city}")

        new_links = [(lid, link) for lid, link in listings if lid not in seen_ids]
        if progress_bar is None and new_links:
            progress_total = limit if limit else reported_total
            progress_bar = tqdm(total=progress_total, unit="listing", desc=f"Scraping {city}")
        if not new_links:
            logging.info("No new listings found on page %s, stopping.", page)
            break

        if results_per_page is None and listings:
            results_per_page = len(listings)
            if reported_total and results_per_page:
                estimated_pages = math.ceil(reported_total / results_per_page)

        if limit:
            remaining = limit - len(all_listings)
            if remaining <= 0:
                logging.info("Reached listing limit (%s).", limit)
                if progress_bar:
                    progress_bar.close()
                return all_listings
            new_links = new_links[:remaining]

        if not new_links:
            logging.info("No listings left to fetch after applying limit.")
            break

        if worker_count == 1:
            for listing_id, link in new_links:
                logging.info("Fetching detail for %s", listing_id)
                detail_response = request_with_retry(session, link)
                listing = parse_listing_detail(listing_id, link, detail_response.text, city, transaction)
                handle_listing(listing)
                gentle_pause(delay)
        else:
            indexed_links = list(enumerate(new_links))
            results: Dict[int, Listing] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_idx = {
                    executor.submit(fetch_listing_detail_concurrent, listing_id, link): idx
                    for idx, (listing_id, link) in indexed_links
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    listing = future.result()
                    results[idx] = listing
            for idx in sorted(results):
                handle_listing(results[idx])

        page += 1
        if estimated_pages and page > estimated_pages:
            logging.info("Reached estimated last page (%s).", estimated_pages)
            break
        gentle_pause(delay)

    if progress_bar:
        progress_bar.close()
    logging.info("Finished scraping %s listings.", len(all_listings))
    return all_listings


def listings_to_dataframe(listings: Sequence[Listing]) -> pd.DataFrame:
    rows = [asdict(listing) for listing in listings]
    df = pd.DataFrame(rows)
    for column in COLUMN_ORDER:
        if column not in df.columns:
            df[column] = ""
    df = df[COLUMN_ORDER]
    df.rename(columns=COLUMN_RENAMES, inplace=True)
    return df


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Nehnutelnosti.sk real estate listings and export to XLS."
    )
    parser.add_argument(
        "--city",
        default="trnava",
        help="Comma-separated city slugs to scrape (default: trnava)",
    )
    parser.add_argument(
        "--transaction",
        choices=["predaj", "prenajom"],
        default="predaj",
        help="Transaction type (default: predaj)",
    )
    parser.add_argument(
        "--categories",
        default="11,12,300001",
        help="Comma separated category IDs to include (default: 11,12,300001)",
    )
    parser.add_argument(
        "--output",
        default="output/nehnutelnosti_trnava_byty.xls",
        help="Output XLS filename (default: output/nehnutelnosti_trnava_byty.xls)",
    )
    parser.add_argument(
        "--csv-output",
        default="output/nehnutelnosti_trnava_byty.csv",
        help="Output CSV filename for incremental saves (default: output/nehnutelnosti_trnava_byty.csv)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of search result pages to crawl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of listings to scrape (default: 3, use 0 for all).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay (seconds) between detail requests (default: 1.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent detail fetchers (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    start_time = time.perf_counter()
    categories = [cat.strip() for cat in args.categories.split(",") if cat.strip()]
    if not categories:
        raise ValueError("At least one category must be specified.")
    city_slugs = [city.strip() for city in args.city.split(",") if city.strip()]
    if not city_slugs:
        raise ValueError("At least one city must be specified.")
    listing_limit = args.limit
    if listing_limit is not None and listing_limit <= 0:
        listing_limit = None

    csv_path = Path(args.csv_output)
    if csv_path.parent:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    xls_path = Path(args.output)
    if xls_path.parent:
        xls_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()
            combined_listings: List[Listing] = []
            for city_slug in city_slugs:
                logging.info("Starting scrape for city '%s'", city_slug)
                city_listings = scrape_listings(
                    city=city_slug,
                    transaction=args.transaction,
                    categories=categories,
                    max_pages=args.max_pages,
                    delay=args.delay,
                    limit=listing_limit,
                    workers=args.workers,
                    csv_writer=csv_writer,
                    csv_file=csv_file,
                )
                combined_listings.extend(city_listings)

            if not combined_listings:
                logging.warning("No listings scraped across requested cities. Nothing to export.")
                return

            df = listings_to_dataframe(combined_listings)
            logging.info("Writing %s listings to %s", len(df), xls_path)
            try:
                df.to_excel(xls_path, index=False, engine="xlwt")
            except ValueError as exc:
                if "No Excel writer 'xlwt'" in str(exc):
                    logging.error(
                        "Excel writer 'xlwt' is not available. Install it with 'pip install xlwt' "
                        "or run 'pip install -r requirements.txt' inside a virtual environment."
                    )
                    logging.info("Skipping XLS export; CSV output at %s is ready.", csv_path)
                else:
                    raise
    finally:
        elapsed = time.perf_counter() - start_time
        logging.info("Total scrape time: %.2f seconds", elapsed)


if __name__ == "__main__":
    main()

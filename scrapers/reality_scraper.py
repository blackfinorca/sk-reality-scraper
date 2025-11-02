import argparse
import csv
import json
import logging
import math
import re
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode, urlparse

from html import unescape

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup
from requests import Response, Session
from requests.exceptions import HTTPError
from tqdm import tqdm

from listing_schema import (
    COLUMN_ORDER,
    COLUMN_RENAMES,
    CSV_FIELDNAMES,
    ListingRecord,
    sanitize_csv_value,
)


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "sk-SK,sk;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

AMENITY_KEY_MAP = {
    "floor_area": ["uzitkova plocha", "podlahova plocha"],
    "floor": ["podlazie"],
    "building_status": ["stav nehnutelnosti"],
    "year_of_construction": ["rok vystavby"],
    "year_of_top": ["rok kolaudacie"],
    "energy_cert": ["energeticky certifikat", "energeticky certifikat budovy"],
    "room_number": ["pocet izieb", "pocet izieb / miestnosti", "pocet spalni"],
    "price_area": ["cena za m2", "cena za m^2", "cena za m²"],
}


def normalise_label(value: str) -> str:
    normalised = unicodedata.normalize("NFKD", value or "")
    ascii_only = "".join(ch for ch in normalised if not unicodedata.combining(ch))
    return ascii_only.lower().strip()


def create_session() -> Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    session.timeout = 30  # type: ignore[attr-defined]
    return session


def request_with_retry(session: Session, url: str, params: Optional[Dict[str, str]] = None,
                       retries: int = 3, backoff: float = 2.0) -> Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logging.warning(
                "Request failed (attempt %s/%s) for %s: %s",
                attempt,
                retries,
                response_url(url, params),
                exc,
            )
            time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


def response_url(url: str, params: Optional[Dict[str, str]]) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(params)}"


def build_search_url(property_type: str, city: str, transaction: str) -> str:
    return f"https://www.reality.sk/{property_type}/{city}/{transaction}/"


def parse_item_list(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}", strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            types = data.get("@type")
            if isinstance(types, list) and "itemlist" in {normalise_label(t) for t in types}:
                return extract_urls_from_item_list(data)
            if isinstance(types, str) and normalise_label(types) == "itemlist":
                return extract_urls_from_item_list(data)
    return []


def extract_urls_from_item_list(data: Dict) -> List[str]:
    urls: List[str] = []
    for element in data.get("itemListElement", []):
        if isinstance(element, dict):
            main = element.get("mainEntity") if isinstance(element.get("mainEntity"), dict) else {}
            candidate = main.get("url") if isinstance(main, dict) else None
            if not candidate:
                candidate = element.get("url")
            if candidate:
                urls.append(candidate)
    return urls


def extract_listing_id(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        return url
    last_segment = path.split("/")[-1]
    last_segment = last_segment.split("?")[0]
    return last_segment or url


def parse_numeric(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(round(float(value)))
        except (ValueError, TypeError):
            return None
    cleaned = value.replace("\xa0", " ").replace("\u202f", " ")
    cleaned = re.sub(r"[^0-9,\.]+", "", cleaned)
    cleaned = cleaned.replace(",", ".")
    if not cleaned:
        return None
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    parsed = parse_numeric(value)
    return float(parsed) if parsed is not None else None


def build_amenity_lookup(items: Iterable[Dict[str, str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        val = item.get("value")
        if not name or val is None:
            continue
        lookup[normalise_label(name)] = str(val).strip()
    return lookup


def extract_from_amenities(lookup: Dict[str, str], key: str) -> Optional[str]:
    targets = AMENITY_KEY_MAP.get(key, [])
    for target in targets:
        if target in lookup:
            return lookup[target]
    return None


def extract_photo_urls(product_data: Dict) -> Tuple[str, str, str]:
    images: List[str] = []
    raw_images = product_data.get("image") if isinstance(product_data, dict) else []
    if isinstance(raw_images, list):
        images.extend(unescape(str(img)) for img in raw_images if img)
    elif isinstance(raw_images, str):
        images.append(unescape(raw_images))
    while len(images) < 3:
        images.append("")
    return images[0], images[1], images[2]


def extract_description(soup: BeautifulSoup, json_docs: List[Dict]) -> str:
    def find_in_node(node) -> Optional[str]:
        if isinstance(node, dict):
            candidate = node.get("description")
            if isinstance(candidate, str):
                cleaned = candidate.strip()
                if cleaned:
                    return cleaned
            for value in node.values():
                result = find_in_node(value)
                if result:
                    return result
        elif isinstance(node, list):
            for value in node:
                result = find_in_node(value)
                if result:
                    return result
        return None

    for doc in json_docs:
        result = find_in_node(doc)
        if result:
            return result

    selectors = [
        '[itemprop="description"]',
        '[data-testid="description"]',
        '[data-test-id="description"]',
        'section.detail-text',
    ]
    for selector in selectors:
        container = soup.select_one(selector)
        if container:
            text = container.get_text("\n", strip=True)
            if text:
                return text

    fallback = soup.find("article")
    if fallback:
        text = fallback.get_text("\n", strip=True)
        if text:
            return text
    return ""


def parse_listing_detail(session: Session, url: str, transaction: str) -> ListingRecord:
    response = request_with_retry(session, url)
    soup = BeautifulSoup(response.text, "html.parser")
    json_docs: List[Dict] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}", strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            json_docs.append(data)

    product = next((doc for doc in json_docs if "product" in _ensure_type_list(doc.get("@type"))), {})
    residence = next((doc for doc in json_docs if _ensure_type_list(doc.get("@type")) & {
        "residence",
        "apartment",
        "singlefamilyresidence",
    }), {})

    address = {}
    for candidate in (residence.get("address"), product.get("address")):
        if isinstance(candidate, dict) and candidate:
            address = candidate
            break

    amenities = []
    for candidate in (
        residence.get("amenityFeature"),
        product.get("amenityFeature"),
    ):
        if isinstance(candidate, list):
            amenities.extend(candidate)
    amenity_lookup = build_amenity_lookup(amenities)

    floor_area = None
    floor_size = residence.get("floorSize") if isinstance(residence, dict) else None
    if isinstance(floor_size, dict):
        floor_area = parse_numeric(str(floor_size.get("value")))
    if floor_area is None:
        value = extract_from_amenities(amenity_lookup, "floor_area")
        floor_area = parse_numeric(value) if value else None

    room_value = extract_from_amenities(amenity_lookup, "room_number")
    room_number = parse_numeric(room_value) if room_value else None

    floor_raw = extract_from_amenities(amenity_lookup, "floor") or ""
    building_status = extract_from_amenities(amenity_lookup, "building_status") or ""
    year_of_construction = extract_from_amenities(amenity_lookup, "year_of_construction")
    year_of_top = extract_from_amenities(amenity_lookup, "year_of_top")
    energy_cert = extract_from_amenities(amenity_lookup, "energy_cert") or ""
    price_per_m2_raw = extract_from_amenities(amenity_lookup, "price_area")

    price = None
    price_info = product.get("offers") if isinstance(product, dict) else {}
    if isinstance(price_info, dict):
        price = parse_numeric(str(price_info.get("price")))

    if price_per_m2_raw:
        price_per_m2 = parse_numeric(price_per_m2_raw)
    elif price is not None and floor_area and floor_area > 0:
        price_per_m2 = int(round(price / floor_area))
    else:
        price_per_m2 = None

    photo_url_1, photo_url_2, photo_url_3 = extract_photo_urls(product)
    description = extract_description(soup, json_docs)

    listing = ListingRecord(
        source="reality_sk",
        transaction=transaction,
        property_id=extract_listing_id(url),
        link=url,
        property_name=str(product.get("name", "")).strip(),
        description=description,
        address_street=str(address.get("streetAddress", "")),
        address_town=str(address.get("addressLocality", "")),
        address_zrea=str(address.get("addressRegion", "")),
        room_number=room_number,
        floor_area=floor_area,
        building_status=building_status,
        year_of_construction=parse_numeric(year_of_construction) if year_of_construction else None,
        year_of_top=parse_numeric(year_of_top) if year_of_top else None,
        floor=floor_raw,
        energ_cert=energy_cert,
        price=price,
        price_area=price_per_m2,
        photo_url_1=photo_url_1,
        photo_url_2=photo_url_2,
        photo_url_3=photo_url_3,
    )
    return listing


def _ensure_type_list(value) -> set:
    if isinstance(value, list):
        return {normalise_label(str(v)) for v in value}
    if isinstance(value, str):
        return {normalise_label(value)}
    return set()


def write_listing_csv_row(listing: ListingRecord, csv_writer: csv.DictWriter, csv_file) -> None:
    row_dict = asdict(listing)
    mapped = {}
    for key in COLUMN_ORDER:
        value = row_dict.get(key, "")
        if key == "description":
            mapped[COLUMN_RENAMES[key]] = "" if value is None else str(value)
        else:
            mapped[COLUMN_RENAMES[key]] = sanitize_csv_value(value)
    csv_writer.writerow(mapped)
    csv_file.flush()


def scrape_listings(property_type: str, city: str, transaction: str, max_pages: Optional[int],
                    delay: float, limit: Optional[int], workers: int,
                    csv_writer: Optional[csv.DictWriter] = None, csv_file=None) -> List[ListingRecord]:
    session = create_session()
    base_url = build_search_url(property_type, city, transaction)
    all_urls: List[str] = []
    page = 1
    progress_bar = tqdm(total=limit, unit="listing", desc=f"Collecting URLs ({city})") if limit else None

    try:
        while True:
            if max_pages and page > max_pages:
                break
            params = {"page": str(page)} if page > 1 else None
            logging.info("Fetching search page %s", page)
            response = request_with_retry(session, base_url, params=params)
            urls = parse_item_list(response.text)
            if not urls:
                logging.info("No listings found on page %s, stopping.", page)
                break
            for url in urls:
                if url not in all_urls:
                    all_urls.append(url)
                    if progress_bar:
                        progress_bar.update(1)
                    if limit and len(all_urls) >= limit:
                        break
            if limit and len(all_urls) >= limit:
                break
            page += 1
            if limit is None and len(urls) == 0:
                break
            time.sleep(delay)
    finally:
        if progress_bar:
            progress_bar.close()

    listings: List[ListingRecord] = []
    listing_bar = tqdm(total=len(all_urls), unit="listing", desc=f"Scraping {city}") if all_urls else None
    for url in all_urls:
        try:
            listing = parse_listing_detail(session, url, transaction)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to parse %s: %s", url, exc)
            continue
        listings.append(listing)
        if csv_writer and csv_file:
            write_listing_csv_row(listing, csv_writer, csv_file)
        if listing_bar:
            listing_bar.update(1)
        time.sleep(delay)
    if listing_bar:
        listing_bar.close()
    logging.info("Finished scraping %s listings.", len(listings))
    return listings


def listings_to_dataframe(listings: Sequence[ListingRecord]):
    import pandas as pd  # Local import to avoid hard dependency if unused

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
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Reality.sk listings and export to CSV/XLS.")
    parser.add_argument("--property-type", default="byty", help="Property type slug (default: byty)")
    parser.add_argument("--city", default="trnava", help="Comma-separated city slugs (default: trnava)")
    parser.add_argument("--transaction", default="predaj", help="Transaction type (predaj/prenajom)")
    parser.add_argument(
        "--csv-output",
        default="output/reality_trnava_byty.csv",
        help="CSV output filename (default: output/reality_trnava_byty.csv)",
    )
    parser.add_argument(
        "--output",
        default="output/reality_trnava_byty.xls",
        help="XLS output filename (default: output/reality_trnava_byty.xls)",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Max search pages to traverse")
    parser.add_argument("--limit", type=int, default=3, help="Max listings to scrape (default: 3, use 0 for all)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future concurrency support")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    listing_limit = args.limit
    if listing_limit is not None and listing_limit <= 0:
        listing_limit = None

    city_slugs = [slug.strip() for slug in args.city.split(",") if slug.strip()]
    if not city_slugs:
        raise ValueError("At least one city must be specified.")

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xls_path = Path(args.output)
    xls_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        combined: List[ListingRecord] = []
        for city_slug in city_slugs:
            logging.info("Starting scrape for city '%s'", city_slug)
            city_listings = scrape_listings(
                property_type=args.property_type,
                city=city_slug,
                transaction=args.transaction,
                max_pages=args.max_pages,
                delay=args.delay,
                limit=listing_limit,
                workers=args.workers,
                csv_writer=csv_writer,
                csv_file=csv_file,
            )
            combined.extend(city_listings)

    if not combined:
        logging.warning("No listings scraped. Nothing to export.")
        return

    try:
        df = listings_to_dataframe(combined)
        df.to_excel(xls_path, index=False, engine="xlwt")
        logging.info("Exported %s listings to %s", len(df), xls_path)
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to export XLS: %s", exc)


if __name__ == "__main__":
    main()

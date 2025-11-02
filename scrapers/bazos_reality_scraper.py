import argparse
import csv
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin

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

CUSTOM_SEARCH_URLS = {
    "trnava": "https://reality.bazos.sk/predam/byt/1120/?hledat=&rubriky=reality&hlokalita=91701&humkreis=10&cenaod=&cenado=&order=&crp=&kitx=ano",
}

LISTING_LINK_PATTERN = re.compile(r"/inzerat/(\d+)/")
ROOM_PATTERN = re.compile(r"(\d+)\s*izb", re.IGNORECASE)
AREA_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(m2|m²)", re.IGNORECASE)
FLOOR_FRACTION_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)\s*(?:posch|np)?", re.IGNORECASE)
FLOOR_WORD_PATTERN = re.compile(r"na\s*(\d+)\.\s*(?:poschod[ie]|np)", re.IGNORECASE)
FLOOR_TOTAL_PATTERN = re.compile(r"z\s*(\d+)\s*(?:posch|np)", re.IGNORECASE)
GROUND_PATTERN = re.compile(r"pr[ií]zem", re.IGNORECASE)
PRICE_PER_M2_PATTERN = re.compile(r"(\d{3,5})\s*(?:€|eur)\s*/\s*(?:m2|m²)", re.IGNORECASE)

STATUS_KEYWORDS = [
    ("novostav", "Novostavba"),
    ("komplet", "Kompletná rekonštrukcia"),
    ("rekonstr", "Rekonštrukcia"),
    ("rekonštr", "Rekonštrukcia"),
    ("čiastoč", "Čiastočná rekonštrukcia"),
    ("ciastoč", "Čiastočná rekonštrukcia"),
    ("čiastoc", "Čiastočná rekonštrukcia"),
    ("ciastoc", "Čiastočná rekonštrukcia"),
    ("pôvodn", "Pôvodný stav"),
    ("povodn", "Pôvodný stav"),
]


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
            response.encoding = response.apparent_encoding or "utf-8"
            return response
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logging.warning(
                "Request failed (attempt %s/%s) for %s: %s",
                attempt,
                retries,
                response.url if "response" in locals() else url,
                exc,
            )
            time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


def build_search_url(city: str, transaction: str) -> str:
    city_key = city.strip().lower()
    if not city_key and transaction:
        city_key = transaction.strip().lower()
    if city_key in CUSTOM_SEARCH_URLS:
        return CUSTOM_SEARCH_URLS[city_key]
    return f"https://reality.bazos.sk/{transaction}/byt/"


def parse_search_page(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    for anchor in soup.select("a.nadpis"):
        href = anchor.get("href")
        if not href:
            continue
        full_url = urljoin(base_url, href)
        if LISTING_LINK_PATTERN.search(full_url):
            urls.append(full_url)
    if not urls:
        # Fallback for alternative markup
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            full_url = urljoin(base_url, href)
            if LISTING_LINK_PATTERN.search(full_url):
                urls.append(full_url)
    seen: set[str] = set()
    unique_urls: List[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def extract_listing_id(url: str) -> str:
    match = LISTING_LINK_PATTERN.search(url)
    if match:
        return match.group(1)
    return url


def normalise_text(value: Optional[str]) -> str:
    return (value or "").strip()


def build_attribute_lookup(soup: BeautifulSoup) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    tables = soup.select("table")
    for table in tables:
        for row in table.select("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(" ", strip=True).lower()
            value = cells[1].get_text(" ", strip=True)
            if label and label not in lookup:
                lookup[label] = value
    for definition in soup.select("dl"):
        terms = definition.find_all("dt")
        descriptions = definition.find_all("dd")
        for term, desc in zip(terms, descriptions):
            label = term.get_text(" ", strip=True).lower()
            value = desc.get_text(" ", strip=True)
            if label and label not in lookup:
                lookup[label] = value
    return lookup


def extract_from_lookup(lookup: Dict[str, str], keywords: Iterable[str]) -> str:
    for keyword in keywords:
        lowered = keyword.lower()
        for label, value in lookup.items():
            if lowered in label:
                return normalise_text(value)
    return ""


def parse_numeric(raw_value: str) -> Optional[int]:
    if not raw_value:
        return None
    cleaned = raw_value.replace("\xa0", " ").replace("\u202f", " ")
    cleaned = re.sub(r"[^\d,\.]+", "", cleaned)
    cleaned = cleaned.replace(",", ".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return int(round(value))


def parse_room_count(*candidates: str) -> Optional[int]:
    for candidate in candidates:
        if not candidate:
            continue
        match = ROOM_PATTERN.search(candidate)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def parse_area_value(*candidates: str) -> Optional[int]:
    for candidate in candidates:
        if not candidate:
            continue
        match = AREA_PATTERN.search(candidate)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:
            continue
        return int(round(value))
    return None


# Heuristic fallbacks that mine structured values from free-form description text.
def infer_floor_from_text(text: str) -> str:
    if not text:
        return ""
    fraction_match = FLOOR_FRACTION_PATTERN.search(text)
    if fraction_match:
        current, total = fraction_match.groups()
        if total:
            return f"{current}/{total}"
        return current
    word_match = FLOOR_WORD_PATTERN.search(text)
    if word_match:
        current = word_match.group(1)
        total_match = FLOOR_TOTAL_PATTERN.search(text)
        if total_match:
            return f"{current}/{total_match.group(1)}"
        return current
    if GROUND_PATTERN.search(text):
        total_match = FLOOR_TOTAL_PATTERN.search(text)
        if total_match:
            return f"0/{total_match.group(1)}"
        return "0"
    return ""


def infer_status_from_text(text: str) -> str:
    lowered = text.lower()
    for keyword, value in STATUS_KEYWORDS:
        if keyword in lowered:
            return value
    return ""


def extract_price_per_m2(text: str) -> Optional[int]:
    if not text:
        return None
    match = PRICE_PER_M2_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_year_from_text(text: str, keywords: Sequence[str]) -> Optional[int]:
    if not text:
        return None
    for keyword in keywords:
        pattern = re.compile(rf"{keyword}[^\d]*(19|20)\d{{2}}", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue
        year_match = re.search(r"(19|20)\d{2}", match.group(0))
        if year_match:
            try:
                return int(year_match.group(0))
            except ValueError:
                continue
    return None


def extract_photo_urls(soup: BeautifulSoup, detail_url: str, limit: int = 3) -> List[str]:
    photo_urls: List[str] = []
    seen: set[str] = set()

    def register(url: Optional[str]) -> None:
        if not url:
            return
        clean = url.strip()
        if not clean or clean in seen:
            return
        if clean.startswith("//"):
            clean = f"https:{clean}"
        elif clean.startswith("/"):
            clean = urljoin(detail_url, clean)
        if clean.lower().endswith(".svg"):
            return
        seen.add(clean)
        photo_urls.append(clean)

    selectors = [
        "div#imglist img",
        "div.detail-foto img",
        "div.foto img",
        "img",
    ]
    for selector in selectors:
        for img in soup.select(selector):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            register(src)
            if len(photo_urls) >= limit:
                return photo_urls
    return photo_urls[:limit]


def parse_listing_detail(session: Session, url: str, transaction: str, default_city: str) -> ListingRecord:
    response = request_with_retry(session, url)
    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.select_one("span.nadpisdetail") or soup.select_one("h1")
    title = normalise_text(title_tag.get_text(strip=True) if title_tag else "")

    price_tag = soup.select_one("span.cena") or soup.find("div", class_="inzeratycena")
    price_text = normalise_text(price_tag.get_text(" ", strip=True) if price_tag else "")
    price = parse_numeric(price_text)

    attributes = build_attribute_lookup(soup)
    room_attr = extract_from_lookup(attributes, ["izby", "pocet izieb", "dispozicia"])
    floor_area_attr = extract_from_lookup(attributes, ["plocha", "uzitkova plocha", "podlahova plocha"])
    floor_attr = extract_from_lookup(attributes, ["poschodie"])
    status_attr = extract_from_lookup(attributes, ["stav"])
    year_build_attr = extract_from_lookup(attributes, ["rok vystavby"])
    year_top_attr = extract_from_lookup(attributes, ["rok kolaudacie", "rok kolaud"])
    energy_attr = extract_from_lookup(attributes, ["energeticky certifikat", "energeticka trieda"])
    price_area_attr = extract_from_lookup(attributes, ["cena za m2", "cena za m"])
    street_attr = extract_from_lookup(attributes, ["ulica"])
    town_attr = extract_from_lookup(attributes, ["obec", "mesto"])
    district_attr = extract_from_lookup(attributes, ["okres"])
    region_attr = extract_from_lookup(attributes, ["kraj"])

    description_block = soup.select_one("div.popisdetail")
    description_text = normalise_text(description_block.get_text(" ", strip=True) if description_block else "")

    room_number = parse_room_count(room_attr, title, description_text)
    floor_area = parse_area_value(floor_area_attr, description_text)
    if not floor_attr:
        floor_attr = infer_floor_from_text(description_text)
    if not status_attr:
        status_attr = infer_status_from_text(description_text)

    if not town_attr:
        town_attr = default_city

    address_zrea_parts = [part for part in [district_attr, region_attr] if part]
    address_zrea = ", ".join(address_zrea_parts)

    year_of_construction = parse_numeric(year_build_attr) if year_build_attr else None
    year_of_top = parse_numeric(year_top_attr) if year_top_attr else None
    price_area = parse_numeric(price_area_attr) if price_area_attr else None
    if price_area is None:
        price_area = extract_price_per_m2(description_text)
    if year_of_construction is None:
        year_of_construction = extract_year_from_text(description_text, ["rok vystavby", "postaven", "postavený", "postavena"])
    if year_of_top is None:
        year_of_top = extract_year_from_text(description_text, ["rok kolaud", "kolaud", "rekonštruk", "rekonstru"])
    if price_area is None and price is not None and floor_area and floor_area > 0:
        price_area = int(round(price / floor_area))

    photos = extract_photo_urls(soup, url, limit=3)

    listing = ListingRecord(
        source="bazos_sk",
        transaction=transaction,
        property_id=extract_listing_id(url),
        link=url,
        property_name=title,
        description=description_text,
        address_street=street_attr,
        address_town=town_attr,
        address_zrea=address_zrea,
        room_number=room_number,
        floor_area=floor_area,
        building_status=status_attr,
        year_of_construction=year_of_construction,
        year_of_top=year_of_top,
        floor=floor_attr,
        energ_cert=energy_attr,
        price=price,
        price_area=price_area,
        photo_url_1=photos[0] if len(photos) > 0 else "",
        photo_url_2=photos[1] if len(photos) > 1 else "",
        photo_url_3=photos[2] if len(photos) > 2 else "",
    )
    return listing


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


def scrape_listings(city: str, transaction: str, max_pages: Optional[int],
                    delay: float, limit: Optional[int],
                    csv_writer: Optional[csv.DictWriter] = None, csv_file=None) -> List[ListingRecord]:
    session = create_session()
    base_url = build_search_url(city, transaction)
    detail_urls: List[str] = []
    _ = max_pages  # Deprecated: retained for CLI compatibility

    if limit is not None and limit <= 0:
        limit = None

    logging.info("Fetching search page for %s", city or "selected area")
    try:
        response = request_with_retry(session, base_url)
    except HTTPError as exc:
        logging.error("Search page request failed: %s", exc)
        session.close()
        return []

    urls = parse_search_page(response.text, base_url)
    if not urls:
        logging.info("No listings found at %s", base_url)
    for url in urls:
        if url in detail_urls:
            continue
        detail_urls.append(url)
        if limit and len(detail_urls) >= limit:
            break

    listings: List[ListingRecord] = []
    detail_bar = tqdm(total=len(detail_urls), unit="listing", desc=f"Scraping {city}") if detail_urls else None
    try:
        for url in detail_urls:
            try:
                listing = parse_listing_detail(session, url, transaction, default_city=city)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to parse %s: %s", url, exc)
                continue
            listings.append(listing)
            if csv_writer and csv_file:
                write_listing_csv_row(listing, csv_writer, csv_file)
            if detail_bar:
                detail_bar.update(1)
            time.sleep(delay)
    finally:
        if detail_bar:
            detail_bar.close()
        session.close()
    logging.info("Finished scraping %s listings from Bazos.sk.", len(listings))
    return listings


def listings_to_dataframe(listings: Sequence[ListingRecord]):
    import pandas as pd  # Delay heavy import until needed

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
    parser = argparse.ArgumentParser(description="Scrape Bazos.sk real estate listings and export to CSV/XLS.")
    parser.add_argument("--city", default="Trnava", help="City filter used for the hlokalita search parameter (default: Trnava)")
    parser.add_argument("--transaction", default="predam", help="Transaction path segment (e.g. predam, kupim)")
    parser.add_argument(
        "--csv-output",
        default="output/bazos_trnava_byty.csv",
        help="CSV output filename (default: output/bazos_trnava_byty.csv)",
    )
    parser.add_argument(
        "--output",
        default="output/bazos_trnava_byty.xls",
        help="XLS output filename (default: output/bazos_trnava_byty.xls)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Deprecated; the scraper now fetches a single filtered search page.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Max listings to scrape (default: 30, use 0 for all)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    listing_limit = args.limit
    if listing_limit is not None and listing_limit <= 0:
        listing_limit = None

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xls_path = Path(args.output)
    xls_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        listings = scrape_listings(
            city=args.city,
            transaction=args.transaction,
            max_pages=args.max_pages,
            delay=args.delay,
            limit=listing_limit,
            csv_writer=csv_writer,
            csv_file=csv_file,
        )

    if not listings:
        logging.warning("No listings scraped from Bazos.sk. Nothing to export.")
        return

    try:
        df = listings_to_dataframe(listings)
        df.to_excel(xls_path, index=False, engine="xlwt")
        logging.info("Exported %s listings to %s", len(df), xls_path)
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to export XLS: %s", exc)


if __name__ == "__main__":
    main()

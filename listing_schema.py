from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ListingRecord:
    source: str
    transaction: str
    property_id: str
    link: str
    property_name: str
    description: str
    address_street: str
    address_town: str
    address_zrea: str
    room_number: Optional[int]
    floor_area: Optional[int]
    building_status: str
    year_of_construction: Optional[int]
    year_of_top: Optional[int]
    floor: str
    energ_cert: str
    price: Optional[int]
    price_area: Optional[int]
    photo_url_1: str
    photo_url_2: str
    photo_url_3: str


COLUMN_ORDER = [
    "source",
    "transaction",
    "property_id",
    "link",
    "property_name",
    "description",
    "address_street",
    "address_town",
    "address_zrea",
    "room_number",
    "floor_area",
    "building_status",
    "year_of_construction",
    "year_of_top",
    "floor",
    "energ_cert",
    "price",
    "price_area",
    "photo_url_1",
    "photo_url_2",
    "photo_url_3",
]

COLUMN_RENAMES = {column: column for column in COLUMN_ORDER}

CSV_FIELDNAMES = [COLUMN_RENAMES[column] for column in COLUMN_ORDER]


def sanitize_csv_value(value) -> str:
    if isinstance(value, str):
        cleaned = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()
    if value is None:
        return ""
    return str(value)

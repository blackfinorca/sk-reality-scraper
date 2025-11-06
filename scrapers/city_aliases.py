from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple


def _normalize_token(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_only).strip().lower()


CITY_NAME_OVERRIDES = {
    "bratislava": "Bratislava",
    "kosice": "Košice",
}


DISTRICT_ALIAS_MAP = {
    "bratislava": {
        "stare mesto",
        "ruzinov",
        "nove mesto",
        "karlova ves",
        "petrzalka",
        "raca",
        "vajnory",
        "dubravka",
        "lamac",
        "devin",
        "devinska nova ves",
        "zahorska bystrica",
        "rusovce",
        "jarovce",
        "cunovo",
        "podunajske biskupice",
        "bratislava i",
        "bratislava ii",
        "bratislava iii",
        "bratislava iv",
        "bratislava v",
    },
    "kosice": {
        "stare mesto",
        "sever",
        "zapad",
        "juh",
        "sidlisko kvp",
        "nad jazerom",
        "dargovskych hrdinov",
        "myslava",
        "peres",
        "lorincik",
        "polov",
        "barca",
        "kosicka nova ves",
        "kavecany",
        "saca",
        "sebastovce",
        "tahanovce",
        "topolova",
    },
}


def canonical_city_from_slug(slug: str) -> str:
    cleaned = (slug or "").strip()
    if not cleaned:
        return ""
    parts = [part for part in re.split(r"[-_/]", cleaned) if part]
    base = " ".join(parts)
    normalized = _normalize_token(base)
    if normalized in CITY_NAME_OVERRIDES:
        return CITY_NAME_OVERRIDES[normalized]
    return " ".join(part.capitalize() for part in parts) if parts else cleaned


def normalize_town_for_city(
    city: str, town: Optional[str], area: Optional[str] = None
) -> Tuple[str, str]:
    city = city or ""
    town = (town or "").strip()
    area = (area or "").strip()

    if not city:
        return town or "", area

    city_norm = _normalize_token(city)
    town_norm = _normalize_token(town)
    if not town:
        return city, area
    if town_norm == city_norm:
        return city, area

    alias_set = DISTRICT_ALIAS_MAP.get(city_norm)
    if alias_set:
        candidates = [town]
        for separator in (",", "-", "/", "|"):
            if separator in town:
                candidates.extend(part.strip() for part in town.split(separator) if part.strip())
        for candidate in candidates:
            candidate_norm = _normalize_token(candidate)
            if candidate_norm in alias_set:
                alias_name = candidate or town
                if not area:
                    area = alias_name
                elif _normalize_token(area) != candidate_norm:
                    area = ", ".join(filter(None, [alias_name, area]))
                return city, area

    return town, area

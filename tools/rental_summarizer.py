"""Augment rental (prenájom) CSV outputs with operating/energy cost estimates via OpenAI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from openai import APIConnectionError, APIError, OpenAI

BATCH_HEADER = (
    "Si finančný analytik. Pre každý inzerát nižšie priprav JSON objekt s kľúčmi "
    '"id", "energy_costs". Hodnota musí byť číslo v EUR alebo null (zahŕňa energie/služby spojené s nájmom).\n'
    "Vráť čisté JSON pole.\n\n"
)

DEFAULT_PATTERNS = [
    "output/*prenajom*_output.csv",
    "output/*_prenajom_output.csv",
]

BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rental operating/energy costs from descriptions.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_PATTERNS,
        help="Glob pattern(s) for prenajom CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--cache",
        default="data/rental_cost_cache.json",
        help="Path to cache file (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute costs even if columns already populated.",
    )
    return parser.parse_args()


def resolve_files(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        for path in glob(pattern):
            candidate = Path(path)
            if candidate.exists():
                files.append(candidate)
    return sorted(set(files))


def cache_key(text: str) -> str:
    normalized = text.strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        logging.warning("Failed to parse cache %s; starting empty.", path)
        return {}
    cache: Dict[str, Optional[float]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            cache[key] = parse_number(value)
    return cache


def save_cache(path: Path, cache: Dict[str, Optional[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)


def build_batch_prompt(items: List[Dict[str, str]]) -> str:
    sections = [BATCH_HEADER]
    for item in items:
        sections.append(f'ID: {item["row_id"]}\nTEXT: """{item["text"]}"""')
    sections.append("JSON OUTPUT:")
    return "\n\n".join(sections)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[stripped.find("\n") + 1 :] if "\n" in stripped else stripped.lstrip("`")
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def parse_number(value: object) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def request_batch(items: List[Dict[str, str]], client: OpenAI, model: str) -> Dict[str, Optional[float]]:
    if not items:
        return {}
    prompt = build_batch_prompt(items)
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )
    except (APIConnectionError, APIError) as exc:
        logging.warning("Rental cost batch request failed: %s", exc)
        return {}

    text = response.output[0].content[0].text.strip()  # type: ignore[attr-defined]
    text = _strip_code_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse rental cost JSON: %s", exc)
        return {}

    entries = []
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            entries = data["items"]
        else:
            entries = [data]
    elif isinstance(data, list):
        entries = data

    results: Dict[str, Optional[float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row_id = entry.get("id") or entry.get("row_id")
        if not row_id:
            continue
        results[str(row_id)] = parse_number(entry.get("energy_costs"))
    return results


def ensure_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        df[column] = pd.NA


def enrich_csv(
    csv_path: Path,
    client: OpenAI,
    model: str,
    cache: Dict[str, Optional[float]],
    force: bool,
) -> bool:
    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    ensure_column(df, "description")
    ensure_column(df, "energy_costs")

    pending: List[Dict[str, object]] = []
    updated = False
    for idx, row in df.iterrows():
        desc = str(row.get("description") or "").strip()
        if not desc:
            continue
        needs_energy = force or pd.isna(row.get("energy_costs"))
        if not needs_energy:
            continue

        combined_text = desc
        title = str(row.get("property_name") or "").strip()
        if title and title not in combined_text:
            combined_text = f"{title}\n{combined_text}"

        key = cache_key(combined_text)
        cached_value = cache.get(key)
        if cached_value is not None:
            df.at[idx, "energy_costs"] = cached_value
            updated = True
            continue

        pending.append(
            {
                "df_idx": idx,
                "row_id": str(row.get("property_id") or row.get("listing_id") or f"row-{idx}"),
                "text": combined_text,
                "cache_key": key,
            }
        )

    for start in range(0, len(pending), BATCH_SIZE):
        chunk = pending[start : start + BATCH_SIZE]
        batch_items = [{"row_id": item["row_id"], "text": item["text"]} for item in chunk]
        batch_results = request_batch(batch_items, client, model)
        for item in chunk:
            attrs = batch_results.get(item["row_id"])
            if not attrs:
                continue
            cache[item["cache_key"]] = attrs
            if attrs is not None:
                df.at[item["df_idx"], "energy_costs"] = attrs
                updated = True

    if updated:
        df.to_csv(csv_path, index=False)
    return updated


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the rental summarizer.")

    files = resolve_files(args.inputs)
    if not files:
        raise FileNotFoundError(f"No rental CSV files matched patterns: {args.inputs}")

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    client = OpenAI(api_key=api_key)

    total_updates = 0
    for csv_file in files:
        try:
            if enrich_csv(csv_file, client, args.model, cache, args.force):
                logging.info("Updated rental costs in %s", csv_file)
                total_updates += 1
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Rental summarizer failed for %s: %s", csv_file, exc)

    save_cache(cache_path, cache)
    logging.info("Rental summarizer finished. Files updated: %s", total_updates)


if __name__ == "__main__":
    main()

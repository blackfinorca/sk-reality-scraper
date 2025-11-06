"""Generate concise Slovak summaries of real-estate descriptions using OpenAI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import APIConnectionError, APIError, OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE = (
    'Si stručný sumarizátor realitných ponúk pre investorov. Použi LEN text z poľa DESCRIPTION.\n'
    "Cieľ: vyprodukuj 1–2 krátke vety v slovenčine s investične dôležitými faktami.\n\n"
    "Pravidlá:\n"
    "- Ak údaj v DESCRIPTION nie je, NEUVÁDZAJ ho. Nič nevymýšľaj, žiadne superlatívy.\n"
    "- VÝSTUP = 1–2 vety, spolu ≤ 35 slov. Žiadne ceny, odkazy, kontakty.\n"
    "- Uprednostni: počet izieb, výmera (m²), stav (novostavba/kompletná/čiastočná rekonštrukcia/bez nutnej), poschodie + výťah, vonkajší priestor (balkón/loggia/terasa/záhrada), parkovanie, pivnica/komora/technická miestnosť, vykurovanie/klíma/rekuperácia/podlahové kúrenie, mesačné náklady/poplatky, pešia dostupnosť/čas k centru/MHD/žst, špecifiká projektu/lokality a nieco co je zaujimave pre investorov.\n"
    "- Jazyk vecný a úsporný; Pouzi cele vety ak sa neda, informácie môžeš oddeliť bodkočiarkami.\n\n"
    "FORMÁT:\n"
    'INPUT DESCRIPTION: """{description}"""\n\n'
    'OUTPUT:\n'
    'summary_short_sk: "<1–2 vety v slovenčine, podľa pravidiel vyššie>"\n'
)

BATCH_SIZE = 10
BATCH_PROMPT_HEADER = (
    "Si stručný sumarizátor realitných ponúk pre investorov. "
    "Pre každý inzerát nižšie vytvor 1–2 vety so zameraním na investične dôležité informácie "
    "(izby, výmera, stav, poschodie/výťah, balkón/loggia/terasa/záhrada, parkovanie, pivnica, technika, "
    "náklady a dostupnosť). Nepoužívaj superlatívy ani vymyslené údaje. "
    "Výstup musí byť čisté JSON pole, kde každý objekt obsahuje polia \"id\" a \"summary_short_sk\". "
    "Drž sa formátu JSON, žiadne komentáre ani úvodný text.\n\n"
)


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str).fillna("")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def build_prompt(description: str) -> str:
    clean_description = description.strip()
    return PROMPT_TEMPLATE.format(description=clean_description)


def summarise(description: str, client: OpenAI, model: str) -> Optional[str]:
    prompt = build_prompt(description)
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )
    except (APIConnectionError, APIError) as exc:
        logging.warning("OpenAI request failed: %s", exc)
        return None
    text = response.output[0].content[0].text.strip()  # type: ignore[attr-defined]
    if text.lower().startswith("summary_short_sk:"):
        text = text.split(":", 1)[1].strip()
    return text.strip('" ')


def summarise_batch(items: List[Dict[str, object]], client: OpenAI, model: str) -> Dict[str, str]:
    if not items:
        return {}
    prompt = build_batch_prompt(
        [
            {
                "listing_id": str(item["listing_id"]),
                "description": str(item["description"]),
            }
            for item in items
        ]
    )
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    text = response.output[0].content[0].text.strip()  # type: ignore[attr-defined]
    text = _strip_code_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse batch JSON: {exc}") from exc

    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            entries = data["items"]
        else:
            raise ValueError("Batch response JSON does not contain expected 'items' array.")
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Batch response JSON must be an object or list.")

    results: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        listing_id = entry.get("id") or entry.get("listing_id")
        summary = entry.get("summary_short_sk") or entry.get("summary")
        if listing_id and summary:
            results[str(listing_id)] = str(summary).strip().strip('" ')
    return results


def load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        logging.warning("Failed to parse cache file %s; starting with empty cache.", path)
    return {}


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def cache_key(description: str) -> str:
    normalized = description.strip()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest

def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[stripped.find("\n") + 1 :] if "\n" in stripped else stripped.lstrip("`")
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def build_batch_prompt(items: List[Dict[str, str]]) -> str:
    sections = [BATCH_PROMPT_HEADER]
    for item in items:
        listing_id = item["listing_id"]
        description = item["description"].strip()
        sections.append(f'ID: {listing_id}\nDESCRIPTION: """{description}"""')
    sections.append("JSON OUTPUT:")
    return "\n\n".join(sections)


def build_source_index(
    source_dir: Path,
    source_column: str,
    property_id_column: str,
    description_column: str,
) -> Tuple[Dict[str, Dict[str, str]], Set[Tuple[str, str]]]:
    """Load all CSVs in source_dir and index descriptions by portal -> property_id."""
    mapping: Dict[str, Dict[str, str]] = defaultdict(dict)
    blanks: Set[Tuple[str, str]] = set()
    for csv_path in sorted(source_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
        except FileNotFoundError:
            continue
        if source_column not in df.columns or property_id_column not in df.columns:
            continue
        descriptions = df.get(description_column)
        if descriptions is None:
            continue
        for portal, prop_id, descr in zip(
            df[source_column],
            df[property_id_column],
            descriptions,
        ):
            portal_key = (portal or "").strip().lower()
            property_key = (prop_id or "").strip()
            description = (descr or "").strip()
            if not portal_key or not property_key:
                continue
            if not description:
                blanks.add((portal_key, property_key))
                continue
            mapping[portal_key][property_key] = description
    return mapping, blanks


def build_listing_sources(sources_df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    listing_sources: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in sources_df.to_dict(orient="records"):
        listing_id = row.get("listing_id")
        portal = (row.get("portal") or "").strip().lower()
        portal_listing_key = (row.get("portal_listing_key") or "").strip()
        if not listing_id or not portal or not portal_listing_key:
            continue
        listing_sources[listing_id].append(
            {
                "portal": portal,
                "portal_listing_key": portal_listing_key,
                "first_seen_at": row.get("first_seen_at") or "",
            }
        )
    return listing_sources


def select_description(
    sources: List[Dict[str, str]],
    description_index: Dict[str, Dict[str, str]],
    source_priority: List[str],
    blanks: Set[Tuple[str, str]],
) -> Optional[str]:
    """Pick the best available description among sources following priority order."""
    if not sources:
        return None

    priority_map = {portal: rank for rank, portal in enumerate(source_priority)}

    def sort_key(item: Dict[str, str]) -> tuple[int, str]:
        portal = item["portal"]
        priority = priority_map.get(portal, len(source_priority))
        first_seen = item.get("first_seen_at") or ""
        return priority, first_seen

    for item in sorted(sources, key=sort_key):
        portal = item["portal"]
        property_id = item["portal_listing_key"]
        description = description_index.get(portal, {}).get(property_id)
        if description:
            return description
        if (portal, property_id) in blanks:
            continue
    return None


def summarise_gold(
    gold_df: pd.DataFrame,
    listing_sources: Dict[str, List[Dict[str, str]]],
    description_index: Dict[str, Dict[str, str]],
    blanks: Set[Tuple[str, str]],
    client: Optional[OpenAI],
    model: str,
    cache_path: Path,
    summary_column: str,
    source_priority: List[str],
    force: bool,
    analysis_only: bool,
) -> pd.DataFrame:
    cache = load_cache(cache_path)

    if summary_column not in gold_df.columns:
        gold_df[summary_column] = ""

    stats = {
        "listings_total": len(gold_df),
        "summaries_written": 0,
        "cache_hits": 0,
        "generated": 0,
        "missing_description": 0,
        "api_failures": 0,
    }
    missing_details: List[Dict[str, object]] = []
    pending_items: List[Dict[str, object]] = []

    for idx, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Summarizing listings", unit="listing"):
        existing_summary = str(row.get(summary_column, "") or "").strip()
        if existing_summary and not force:
            continue

        listing_id = row["listing_id"]
        sources = listing_sources.get(listing_id, [])
        description = select_description(
            sources,
            description_index,
            source_priority,
            blanks,
        )
        if not description:
            stats["missing_description"] += 1
            reason_entries: List[str] = []
            if not sources:
                reason_entries.append("no sources linked to listing")
            else:
                for source in sources:
                    portal = source["portal"]
                    property_id = source["portal_listing_key"]
                    portal_index = description_index.get(portal)
                    if (portal, property_id) in blanks:
                        reason_entries.append(f"{portal}:{property_id} empty description in CSV")
                    elif portal_index is None:
                        reason_entries.append(f"{portal}:{property_id} missing portal CSV")
                    elif property_id not in portal_index:
                        reason_entries.append(f"{portal}:{property_id} missing property_id")
            missing_details.append(
                {
                    "listing_id": listing_id,
                    "sources": sources,
                    "reasons": reason_entries or ["unknown"],
                }
            )
            continue

        key = cache_key(description)
        summary = cache.get(key)
        if summary:
            stats["cache_hits"] += 1
            gold_df.at[idx, summary_column] = summary
            stats["summaries_written"] += 1
            continue

        if client is None or analysis_only:
            continue

        pending_items.append(
            {
                "df_idx": idx,
                "listing_id": listing_id,
                "description": description,
                "cache_key": key,
            }
        )

    while pending_items:
        chunk = pending_items[:BATCH_SIZE]
        pending_items = pending_items[BATCH_SIZE:]
        batch_results: Dict[str, str] = {}
        if client is not None and not analysis_only:
            try:
                batch_results = summarise_batch(chunk, client, model)
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Batch summarization failed (%s); falling back to single requests.", exc)
                batch_results = {}

        for item in chunk:
            listing_id = item["listing_id"]
            cache_key_value = item["cache_key"]
            summary = batch_results.get(listing_id)
            if summary is None and client is not None and not analysis_only:
                summary = summarise(item["description"], client, model)
            if summary is None:
                stats["api_failures"] += 1
                continue
            cache[cache_key_value] = summary
            gold_df.at[item["df_idx"], summary_column] = summary
            stats["generated"] += 1
            stats["summaries_written"] += 1

    logging.info(
        "Listings: %s | Summaries written: %s | Cache hits: %s | Generated: %s | Missing descriptions: %s | API failures: %s",
        stats["listings_total"],
        stats["summaries_written"],
        stats["cache_hits"],
        stats["generated"],
        stats["missing_description"],
        stats["api_failures"],
    )
    if missing_details:
        logging.info("Listings with missing descriptions: %s", len(missing_details))
        for entry in missing_details[:10]:
            logging.info(
                "Listing %s missing description; reasons: %s",
                entry["listing_id"],
                ", ".join(entry["reasons"]),
            )
        if len(missing_details) > 10:
            logging.info("...and %s more listings without descriptions.", len(missing_details) - 10)

    save_cache(cache_path, cache)
    return gold_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real estate listings into Slovak investor-friendly blurbs.")
    parser.add_argument(
        "--gold-parquet",
        default="parquet_runs/latest/gold_listings.parquet",
        help="Path to golden listings Parquet (default: parquet_runs/latest/gold_listings.parquet).",
    )
    parser.add_argument(
        "--sources-parquet",
        default=None,
        help="Path to gold_listing_sources.parquet (default: alongside the gold parquet).",
    )
    parser.add_argument(
        "--source-dir",
        default="output",
        help="Directory containing source CSV files (default: output).",
    )
    parser.add_argument(
        "--out-parquet",
        help="Destination Parquet path (defaults to parquet_runs/latest/gold_listings_with_summary.parquet unless --in-place is set).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input gold parquet with summaries.",
    )
    parser.add_argument(
        "--description-column",
        default="description",
        help="Column name in source CSVs containing descriptions (default: description).",
    )
    parser.add_argument(
        "--property-id-column",
        default="property_id",
        help="Column name in source CSVs matching portal_listing_key (default: property_id).",
    )
    parser.add_argument(
        "--source-column",
        default="source",
        help="Column name in source CSVs identifying the portal (default: source).",
    )
    parser.add_argument(
        "--summary-column",
        default="summary_short_sk",
        help="Column name to store summaries in the gold parquet (default: summary_short_sk).",
    )
    parser.add_argument(
        "--source-priority",
        default=None,
        help="Comma-separated list of portals in priority order (default: order found in sources file).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--cache",
        default="data/summary_cache.json",
        help="Path to cache JSON file (default: data/summary_cache.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute summaries even if the summary column already has content.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing the output parquet.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip OpenAI calls; only report description availability diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.analyze_only:
        raise RuntimeError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")

    if args.analyze_only:
        if not args.dry_run:
            logging.info("Analyze-only mode implies --dry-run; skipping output write.")
        client = None
    else:
        client = OpenAI(api_key=api_key)

    gold_path = Path(args.gold_parquet)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold parquet not found: {gold_path}")

    sources_paths: List[Path]
    if args.sources_parquet:
        sources_path = Path(args.sources_parquet)
        sources_paths = [sources_path]
    else:
        candidate = gold_path.with_name("gold_listing_sources.parquet")
        if candidate.exists():
            sources_paths = [candidate]
        else:
            runs_root = gold_path.parent.parent if gold_path.parent.name == "latest" else gold_path.parent
            run_candidates = sorted(runs_root.glob("run=*/gold_listing_sources.parquet"))
            if run_candidates:
                sources_paths = run_candidates
                logging.info(
                    "Falling back to %s source parquet files (latest: %s)",
                    len(run_candidates),
                    run_candidates[-1],
                )
            else:
                raise FileNotFoundError(
                    f"Sources parquet not found near {gold_path}. Set --sources-parquet explicitly."
                )

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    cache_path = Path(args.cache)

    gold_df = pd.read_parquet(gold_path)
    sources_frames = []
    for path in sources_paths:
        try:
            frame = pd.read_parquet(path)
            sources_frames.append(frame)
        except FileNotFoundError:
            logging.warning("Sources parquet %s not found; skipping.", path)
    if not sources_frames:
        raise FileNotFoundError("No valid sources parquet files could be loaded.")

    sources_df = pd.concat(sources_frames, ignore_index=True)
    dedupe_columns = [
        col
        for col in ["listing_id", "portal", "portal_listing_key", "source_url", "first_seen_at", "last_seen_at"]
        if col in sources_df.columns
    ]
    if dedupe_columns:
        sources_df = sources_df.drop_duplicates(subset=dedupe_columns)

    description_index, missing_entries = build_source_index(
        source_dir=source_dir,
        source_column=args.source_column,
        property_id_column=args.property_id_column,
        description_column=args.description_column,
    )

    if not description_index:
        logging.warning("No descriptions found in %s; summaries may not be generated.", source_dir)

    if args.source_priority:
        source_priority = [portal.strip().lower() for portal in args.source_priority.split(",") if portal.strip()]
    else:
        ordered = [str(p or "").strip().lower() for p in sources_df["portal"].tolist()]
        source_priority = list(dict.fromkeys(filter(None, ordered)))

    listing_sources = build_listing_sources(sources_df)

    updated_df = summarise_gold(
        gold_df,
        listing_sources=listing_sources,
        description_index=description_index,
        blanks=missing_entries,
        client=client,
        model=args.model,
        cache_path=cache_path,
        summary_column=args.summary_column,
        source_priority=source_priority,
        force=args.force,
        analysis_only=args.analyze_only,
    )

    if args.dry_run:
        logging.info("Dry-run mode enabled; not writing output.")
        return

    default_output = gold_path.with_name(f"{gold_path.stem}_with_summary{gold_path.suffix}")
    if gold_path.parent.name == "latest":
        default_output = gold_path.parent.parent / "latest" / default_output.name
    if args.in_place:
        if args.out_parquet:
            logging.info("Ignoring --out-parquet because --in-place was specified.")
        output_path = gold_path
    else:
        output_path = Path(args.out_parquet) if args.out_parquet else default_output

    write_dataframe(updated_df, output_path)
    logging.info("Summaries written to %s", output_path)


if __name__ == "__main__":
    main()

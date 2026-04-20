"""Export P/I/O tabular schemas from cleaned data or extraction results.

This script creates one row per PMID for the current project scope:
- gold table rows from `cleaned_data/*.json`
- prediction table rows from `extraction_pipeline_results.json`

Each row stores P/I/O spans as JSON-encoded lists so the export stays both
human-readable and machine-friendly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ELEMENTS = ["participants", "interventions", "outcomes"]
DEFAULT_CLEANED_JSON = Path("cleaned_data") / "cleaned_dataset.json"
DEFAULT_RESULTS_JSON = Path("extraction_pipeline_results.json")
DEFAULT_OUTPUT_DIR = Path("structured_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one-row-per-PMID P/I/O tables from cleaned data and/or extraction results."
    )
    parser.add_argument(
        "--cleaned-json",
        type=Path,
        default=DEFAULT_CLEANED_JSON,
        help="Path to cleaned_dataset.json (or a combined train/test JSON export).",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="Path to extraction_pipeline_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where tabular exports will be written.",
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Do not export the gold table from cleaned data.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not export prediction tables from extraction results.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_span_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return normalize_span_list(parsed)
            except json.JSONDecodeError:
                pass
        if " | " in text:
            parts = [part.strip() for part in text.split("|")]
            return [part for part in parts if part]
        return [text]
    return [str(value).strip()] if str(value).strip() else []


def as_json_cell(value: Any) -> str:
    return json.dumps(normalize_span_list(value), ensure_ascii=False)


def load_cleaned_docs(cleaned_json: Path) -> List[Dict[str, Any]]:
    if cleaned_json.exists():
        data = read_json(cleaned_json)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a list in {cleaned_json}, got {type(data).__name__}")

    # Fallback to train/test files when the combined export is unavailable.
    cleaned_dir = cleaned_json.parent
    docs: List[Dict[str, Any]] = []
    for filename in ("train.json", "test.json"):
        path = cleaned_dir / filename
        if path.exists():
            payload = read_json(path)
            if isinstance(payload, list):
                docs.extend(payload)
    return docs


def build_doc_index(cleaned_docs: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    index: Dict[str, Mapping[str, Any]] = {}
    for doc in cleaned_docs:
        pmid = str(doc.get("pmid", "")).strip()
        if pmid:
            index[pmid] = doc
    return index


def build_gold_rows(cleaned_docs: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc in cleaned_docs:
        pmid = str(doc.get("pmid", "")).strip()
        if not pmid:
            continue

        spans = doc.get("spans", {}) if isinstance(doc.get("spans", {}), Mapping) else {}
        row: Dict[str, Any] = OrderedDict()
        row["pmid"] = pmid
        row["split"] = doc.get("split")
        row["source"] = "gold"
        for elem in ELEMENTS:
            elem_spans = normalize_span_list(spans.get(elem, [])) if isinstance(spans, Mapping) else []
            row[elem] = as_json_cell(elem_spans)
            row[f"{elem}_count"] = len(elem_spans)
        rows.append(row)
    return rows


def build_prediction_rows(
    predictions: Iterable[Mapping[str, Any]],
    doc_index: Mapping[str, Mapping[str, Any]],
    pipeline_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in predictions:
        pmid = str(record.get("pmid", "")).strip()
        if not pmid:
            continue

        gold = record.get("gold", {}) if isinstance(record.get("gold", {}), Mapping) else {}
        preds = record.get("predictions", {}) if isinstance(record.get("predictions", {}), Mapping) else {}
        doc = doc_index.get(pmid, {})

        row: Dict[str, Any] = OrderedDict()
        row["pmid"] = pmid
        row["split"] = doc.get("split")
        row["source"] = "prediction"
        row["pipeline"] = pipeline_name

        for elem in ELEMENTS:
            pred_spans = normalize_span_list(preds.get(elem, []))
            gold_spans = normalize_span_list(gold.get(elem, []))
            row[f"{elem}_pred"] = as_json_cell(pred_spans)
            row[f"{elem}_gold"] = as_json_cell(gold_spans)
            row[f"{elem}_pred_count"] = len(pred_spans)
            row[f"{elem}_gold_count"] = len(gold_spans)

        rows.append(row)

    return rows


def export_rows(rows: List[Dict[str, Any]], output_dir: Path, stem: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["pmid", "split", "source"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(json_path, rows)

    return {"csv": csv_path, "json": json_path}


def export_manifest(
    output_dir: Path,
    cleaned_json: Optional[Path],
    results_json: Optional[Path],
    gold_rows: int,
    prediction_rows: Dict[str, int],
) -> Path:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cleaned_json": str(cleaned_json) if cleaned_json else None,
        "results_json": str(results_json) if results_json else None,
        "gold_rows": gold_rows,
        "prediction_rows": prediction_rows,
        "elements": ELEMENTS,
    }
    path = output_dir / "schema_export_manifest.json"
    write_json(path, manifest)
    return path


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_docs: List[Dict[str, Any]] = []
    doc_index: Dict[str, Mapping[str, Any]] = {}
    gold_paths: Optional[Dict[str, Path]] = None
    gold_rows: List[Dict[str, Any]] = []

    if not args.skip_gold:
        cleaned_docs = load_cleaned_docs(args.cleaned_json)
        if cleaned_docs:
            doc_index = build_doc_index(cleaned_docs)
            gold_rows = build_gold_rows(cleaned_docs)
            gold_paths = export_rows(gold_rows, output_dir, "schema_gold_table")
            print(f"Exported gold schema table: {gold_paths['csv']}")
        else:
            print("No cleaned documents found; skipping gold table export.")

    prediction_counts: Dict[str, int] = {}
    if not args.skip_predictions:
        if args.results_json.exists():
            results = read_json(args.results_json)
            predictions = results.get("predictions", {}) if isinstance(results, Mapping) else {}
            if not doc_index and cleaned_docs:
                doc_index = build_doc_index(cleaned_docs)

            for pipeline_name, records in predictions.items():
                if not isinstance(records, list):
                    continue
                rows = build_prediction_rows(records, doc_index, pipeline_name)
                prediction_counts[pipeline_name] = len(rows)
                stem = f"schema_predictions_{_slugify(pipeline_name)}"
                paths = export_rows(rows, output_dir, stem)
                print(f"Exported prediction table for {pipeline_name}: {paths['csv']}")
        else:
            print(f"Results file not found: {args.results_json}; skipping prediction export.")

    export_manifest(
        output_dir=output_dir,
        cleaned_json=args.cleaned_json if args.cleaned_json.exists() else None,
        results_json=args.results_json if args.results_json.exists() else None,
        gold_rows=len(gold_rows),
        prediction_rows=prediction_counts,
    )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "pipeline"


if __name__ == "__main__":
    main()


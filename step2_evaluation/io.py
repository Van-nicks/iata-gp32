from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ELEMENTS = ["participants", "interventions", "outcomes"]


@dataclass(frozen=True)
class SchemaRecord:
    pmid: str
    pipeline: str
    split: Optional[str]
    gold: Dict[str, List[str]]
    pred: Dict[str, List[str]]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_span_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\n\r\"'`.,;:()[]{}")
    return value


def normalize_span_list(value: Any) -> List[str]:
    if value is None:
        return []
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
            return [part.strip() for part in text.split("|") if part.strip()]
        return [text]
    if isinstance(value, Mapping):
        return [normalize_span_text(v) for v in value.values() if normalize_span_text(v)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [normalize_span_text(item) for item in value if normalize_span_text(item)]
    normalized = normalize_span_text(value)
    return [normalized] if normalized else []


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list at {path}, got {type(payload).__name__}")
    if payload and not isinstance(payload[0], Mapping):
        raise ValueError(f"Expected list items to be objects in {path}")
    rows: List[Dict[str, Any]] = []
    for row in payload:
        rows.append(dict(row))
    return rows


def load_mapping_by_pmid(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_rows(path)
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pmid = str(row.get("pmid", "")).strip()
        if pmid:
            mapping[pmid] = row
    return mapping


def _field_from_row(row: Mapping[str, Any], field: str, role: str) -> List[str]:
    if role == "gold":
        candidates = [field, f"{field}_gold"]
    else:
        candidates = [f"{field}_pred", field, f"{field}_gold"]

    nested_key = role
    nested = row.get(nested_key)
    if isinstance(nested, Mapping):
        for candidate in candidates:
            if candidate in nested:
                return normalize_span_list(nested.get(candidate))

    for candidate in candidates:
        if candidate in row:
            return normalize_span_list(row.get(candidate))

    return []


def schema_records_from_tables(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    pipeline_name: Optional[str] = None,
) -> List[SchemaRecord]:
    gold_by_pmid = {str(row.get("pmid", "")).strip(): row for row in gold_rows if str(row.get("pmid", "")).strip()}
    records: List[SchemaRecord] = []

    for row in prediction_rows:
        pmid = str(row.get("pmid", "")).strip()
        if not pmid:
            continue
        gold_row = gold_by_pmid.get(pmid, row)
        pipeline = pipeline_name or str(row.get("pipeline", "pipeline")).strip() or "pipeline"
        split = row.get("split", gold_row.get("split"))

        gold = {elem: _field_from_row(gold_row, elem, "gold") for elem in ELEMENTS}
        pred = {elem: _field_from_row(row, elem, "pred") for elem in ELEMENTS}
        records.append(SchemaRecord(pmid=pmid, pipeline=pipeline, split=split, gold=gold, pred=pred))

    return records


def schema_records_from_results(results_path: Path) -> List[SchemaRecord]:
    payload = read_json(results_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object at {results_path}")

    records: List[SchemaRecord] = []
    predictions = payload.get("predictions", {})
    if not isinstance(predictions, Mapping):
        return records

    for pipeline_name, pipeline_rows in predictions.items():
        if not isinstance(pipeline_rows, list):
            continue
        for row in pipeline_rows:
            if not isinstance(row, Mapping):
                continue
            pmid = str(row.get("pmid", "")).strip()
            if not pmid:
                continue
            gold = {elem: normalize_span_list(row.get("gold", {}).get(elem, [])) if isinstance(row.get("gold", {}), Mapping) else [] for elem in ELEMENTS}
            pred = {elem: normalize_span_list(row.get("predictions", {}).get(elem, [])) if isinstance(row.get("predictions", {}), Mapping) else [] for elem in ELEMENTS}
            records.append(
                SchemaRecord(
                    pmid=pmid,
                    pipeline=str(pipeline_name),
                    split=None,
                    gold=gold,
                    pred=pred,
                )
            )

    return records


def top_k_spans(spans: Sequence[str], k: Optional[int]) -> List[str]:
    if k is None:
        return list(spans)
    if k <= 0:
        return []
    return list(spans[:k])



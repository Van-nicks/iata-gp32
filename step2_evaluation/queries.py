from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

from .io import SchemaRecord, normalize_span_text


@dataclass(frozen=True)
class QuerySpec:
    name: str
    field: str
    keywords: List[str]
    description: str = ""


@dataclass(frozen=True)
class QueryScore:
    pipeline: str
    query: str
    field: str
    gold_hits: int
    pred_hits: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_query_specs(path: Path) -> List[QuerySpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", []) if isinstance(payload, Mapping) else []
    specs: List[QuerySpec] = []
    for item in queries:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        field = str(item.get("field", "")).strip()
        keywords = [normalize_span_text(k) for k in item.get("keywords", []) if normalize_span_text(k)]
        description = str(item.get("description", "")).strip()
        if name and field and keywords:
            specs.append(QuerySpec(name=name, field=field, keywords=keywords, description=description))
    return specs


def _record_hits(records: Sequence[SchemaRecord], field: str, keywords: Sequence[str], use_pred: bool) -> Set[str]:
    hits: Set[str] = set()
    keyword_set = {normalize_span_text(k) for k in keywords if normalize_span_text(k)}
    for record in records:
        spans = record.pred.get(field, []) if use_pred else record.gold.get(field, [])
        text = " ".join(normalize_span_text(span) for span in spans)
        if any(keyword in text for keyword in keyword_set):
            hits.add(record.pmid)
    return hits


def score_queries(
    records_by_pipeline: Dict[str, Sequence[SchemaRecord]],
    query_specs: Sequence[QuerySpec],
) -> List[QueryScore]:
    rows: List[QueryScore] = []
    for pipeline_name, records in records_by_pipeline.items():
        for spec in query_specs:
            gold_hits = _record_hits(records, spec.field, spec.keywords, use_pred=False)
            pred_hits = _record_hits(records, spec.field, spec.keywords, use_pred=True)
            tp = len(gold_hits & pred_hits)
            fp = len(pred_hits - gold_hits)
            fn = len(gold_hits - pred_hits)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            rows.append(
                QueryScore(
                    pipeline=pipeline_name,
                    query=spec.name,
                    field=spec.field,
                    gold_hits=len(gold_hits),
                    pred_hits=len(pred_hits),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    f1=round(f1, 4),
                )
            )
    return rows



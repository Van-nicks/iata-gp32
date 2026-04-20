from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from .field_metrics import evaluate_pipeline_fields
from .io import SchemaRecord


def coverage_precision_sweep(
    records_by_pipeline: Dict[str, Sequence[SchemaRecord]],
    span_caps: Sequence[Optional[int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pipeline_name, records in records_by_pipeline.items():
        for cap in span_caps:
            field_scores, _ = evaluate_pipeline_fields(records, max_spans=cap)
            for field, score in field_scores.items():
                row = asdict(score)
                row["pipeline"] = pipeline_name
                row["tradeoff_cap"] = "all" if cap is None else cap
                rows.append(row)
    return rows



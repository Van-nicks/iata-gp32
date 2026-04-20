from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .field_metrics import FieldScore
from .queries import QueryScore


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def build_manifest(
    *,
    run_name: str,
    input_sources: Dict[str, str],
    span_caps: Sequence[Any],
    query_config: str,
    record_counts: Dict[str, int],
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sources": input_sources,
        "span_caps": ["all" if cap is None else cap for cap in span_caps],
        "query_config": query_config,
        "record_counts": record_counts,
        "notes": notes,
    }


def field_scores_to_rows(field_scores_by_pipeline: Dict[str, Dict[str, FieldScore]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pipeline, field_scores in field_scores_by_pipeline.items():
        for field, score in field_scores.items():
            row = asdict(score)
            row["pipeline"] = pipeline
            rows.append(row)
    return rows


def query_scores_to_rows(query_scores: Sequence[QueryScore]) -> List[Dict[str, Any]]:
    return [score.to_dict() for score in query_scores]



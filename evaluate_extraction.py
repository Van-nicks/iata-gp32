from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from step2_evaluation.field_metrics import FieldScore, evaluate_pipeline_fields
from step2_evaluation.io import (
    SchemaRecord,
    load_rows,
    schema_records_from_results,
    schema_records_from_tables,
)
from step2_evaluation.queries import load_query_specs, score_queries
from step2_evaluation.report import (
    build_manifest,
    field_scores_to_rows,
    query_scores_to_rows,
    write_csv,
    write_json,
)
from step2_evaluation.tradeoff import coverage_precision_sweep

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_GOLD_CANDIDATES = [
    Path("structured_tables") / "schema_gold_table.json",
    Path("structured_tables_test") / "schema_gold_table.json",
]
DEFAULT_PRED_CANDIDATES = [
    Path("structured_tables") / "schema_predictions_rule_based.json",
    Path("structured_tables_test") / "schema_predictions_rule_based.json",
]
DEFAULT_RESULTS_JSON = Path("extraction_pipeline_results.json")
DEFAULT_QUERY_CONFIG = PACKAGE_ROOT / "step2_evaluation" / "configs" / "pio_queries.json"
DEFAULT_OUTPUT_DIR = Path("evaluation_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardized P/I/O extraction evaluation for schema tables and pipeline outputs."
    )
    parser.add_argument(
        "--gold-table",
        type=Path,
        default=None,
        help="Gold schema table JSON (one row per PMID). If omitted, the script searches common locations.",
    )
    parser.add_argument(
        "--prediction-table",
        type=Path,
        action="append",
        default=None,
        help="Prediction schema table JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        action="append",
        default=None,
        help="Raw extraction results JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--query-config",
        type=Path,
        default=DEFAULT_QUERY_CONFIG,
        help="JSON file with downstream P/I/O query definitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation outputs will be saved.",
    )
    parser.add_argument(
        "--span-caps",
        default="1,2,3,all",
        help="Comma-separated list of truncation budgets for the coverage/precision sweep. Use 'all' for no truncation.",
    )
    parser.add_argument(
        "--run-name",
        default="step2_evaluation",
        help="Human-readable run label stored in the manifest.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes saved in the manifest.",
    )
    return parser.parse_args()


def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_span_caps(raw: str) -> List[Optional[int]]:
    caps: List[Optional[int]] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"all", "none", "full"}:
            caps.append(None)
        else:
            caps.append(int(token))
    return caps or [None]


def build_records_from_tables(gold_table: Path, prediction_tables: Sequence[Path]) -> Dict[str, List[SchemaRecord]]:
    gold_rows = load_rows(gold_table)
    records_by_pipeline: Dict[str, List[SchemaRecord]] = {}
    for pred_table in prediction_tables:
        pred_rows = load_rows(pred_table)
        pipeline_name = str(pred_rows[0].get("pipeline") or pred_table.stem) if pred_rows else pred_table.stem
        records = schema_records_from_tables(gold_rows, pred_rows, pipeline_name=pipeline_name)
        records_by_pipeline[pipeline_name] = records
    return records_by_pipeline


def build_records_from_results(results_jsons: Sequence[Path]) -> Dict[str, List[SchemaRecord]]:
    records_by_pipeline: Dict[str, List[SchemaRecord]] = {}
    for results_json in results_jsons:
        records = schema_records_from_results(results_json)
        if not records:
            continue
        grouped: Dict[str, List[SchemaRecord]] = {}
        for record in records:
            grouped.setdefault(record.pipeline, []).append(record)
        for pipeline_name, pipeline_records in grouped.items():
            records_by_pipeline[pipeline_name] = pipeline_records
    return records_by_pipeline


def main() -> None:
    args = parse_args()
    span_caps = parse_span_caps(args.span_caps)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_tables = args.prediction_table or []
    results_jsons = args.results_json or []

    gold_table = args.gold_table or first_existing(DEFAULT_GOLD_CANDIDATES)
    if gold_table is None and not results_jsons:
        raise FileNotFoundError(
            "No gold schema table found. Pass --gold-table or generate schema_export outputs first."
        )

    records_by_pipeline: Dict[str, List[SchemaRecord]] = {}
    input_sources: Dict[str, str] = {}

    if results_jsons:
        records_by_pipeline.update(build_records_from_results(results_jsons))
        for results_json in results_jsons:
            input_sources[f"results:{results_json.stem}"] = str(results_json)
    else:
        if not prediction_tables:
            default_pred = first_existing(DEFAULT_PRED_CANDIDATES)
            if default_pred is None:
                raise FileNotFoundError(
                    "No prediction table found. Pass --prediction-table or generate schema_export outputs first."
                )
            prediction_tables = [default_pred]

        if gold_table is None:
            raise FileNotFoundError("Gold table is required when evaluating schema tables.")

        records_by_pipeline.update(build_records_from_tables(gold_table, prediction_tables))
        input_sources["gold_table"] = str(gold_table)
        for pred_table in prediction_tables:
            input_sources[f"prediction:{pred_table.stem}"] = str(pred_table)

    if not records_by_pipeline:
        raise ValueError("No evaluation records could be built from the provided inputs.")

    query_specs = load_query_specs(args.query_config)

    field_scores_by_pipeline: Dict[str, Dict[str, FieldScore]] = {}
    field_rows: List[Dict[str, object]] = []
    tradeoff_rows = coverage_precision_sweep(records_by_pipeline, span_caps)
    query_scores = score_queries(records_by_pipeline, query_specs)

    for pipeline, records in records_by_pipeline.items():
        field_scores, _ = evaluate_pipeline_fields(records, max_spans=None)
        field_scores_by_pipeline[pipeline] = field_scores
        field_rows.extend(field_scores_to_rows({pipeline: field_scores}))

    summary_payload = {
        "run_name": args.run_name,
        "records": {pipeline: len(records) for pipeline, records in records_by_pipeline.items()},
        "field_scores": {
            pipeline: {field: score.to_dict() for field, score in scores.items()}
            for pipeline, scores in field_scores_by_pipeline.items()
        },
        "tradeoff_curve": tradeoff_rows,
        "query_scores": query_scores_to_rows(query_scores),
    }

    manifest = build_manifest(
        run_name=args.run_name,
        input_sources=input_sources,
        span_caps=span_caps,
        query_config=str(args.query_config),
        record_counts={pipeline: len(records) for pipeline, records in records_by_pipeline.items()},
        notes=args.notes,
    )
    summary_payload["manifest"] = manifest

    write_csv(output_dir / "field_scores.csv", field_rows)
    write_json(output_dir / "field_scores.json", summary_payload["field_scores"])
    write_csv(output_dir / "tradeoff_curve.csv", tradeoff_rows)
    write_json(output_dir / "tradeoff_curve.json", tradeoff_rows)
    write_csv(output_dir / "query_scores.csv", query_scores_to_rows(query_scores))
    write_json(output_dir / "query_scores.json", summary_payload["query_scores"])
    write_json(output_dir / "evaluation_summary.json", summary_payload)
    write_json(output_dir / "evaluation_manifest.json", manifest)

    print(f"Saved evaluation outputs to: {output_dir}")
    for pipeline, count in manifest["record_counts"].items():
        print(f"  - {pipeline}: {count} records")


if __name__ == "__main__":
    main()



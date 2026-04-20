from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

from .io import ELEMENTS, SchemaRecord, normalize_span_text, top_k_spans


@dataclass(frozen=True)
class FieldScore:
    field: str
    max_spans: Optional[int]
    n_docs: int
    support_docs: int
    gold_docs: int
    pred_docs: int
    covered_gold_docs: int
    micro_tp: int
    micro_fp: int
    micro_fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    coverage_gold: float
    coverage_all: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocMatch:
    pmid: str
    field: str
    max_spans: Optional[int]
    gold_spans: List[str]
    pred_spans: List[str]
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _normalize_set(spans: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for span in spans:
        norm = normalize_span_text(span)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)
    return cleaned


def _score_pair(gold_spans: Sequence[str], pred_spans: Sequence[str]) -> Dict[str, float]:
    gold_set = set(_normalize_set(gold_spans))
    pred_set = set(_normalize_set(pred_spans))
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_records_for_field(
    records: Sequence[SchemaRecord],
    field: str,
    max_spans: Optional[int] = None,
) -> tuple[FieldScore, List[DocMatch]]:
    doc_matches: List[DocMatch] = []
    micro_tp = micro_fp = micro_fn = 0
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    gold_docs = pred_docs = covered_gold_docs = support_docs = 0

    for record in records:
        gold_spans = _normalize_set(record.gold.get(field, []))
        pred_spans = _normalize_set(top_k_spans(record.pred.get(field, []), max_spans))

        if gold_spans:
            gold_docs += 1
        if pred_spans:
            pred_docs += 1
        if gold_spans and pred_spans:
            covered_gold_docs += 1

        pair = _score_pair(gold_spans, pred_spans)
        micro_tp += int(pair["tp"])
        micro_fp += int(pair["fp"])
        micro_fn += int(pair["fn"])

        if gold_spans or pred_spans:
            support_docs += 1
            precision_values.append(float(pair["precision"]))
            recall_values.append(float(pair["recall"]))
            f1_values.append(float(pair["f1"]))

        doc_matches.append(
            DocMatch(
                pmid=record.pmid,
                field=field,
                max_spans=max_spans,
                gold_spans=list(gold_spans),
                pred_spans=list(pred_spans),
                tp=int(pair["tp"]),
                fp=int(pair["fp"]),
                fn=int(pair["fn"]),
                precision=float(pair["precision"]),
                recall=float(pair["recall"]),
                f1=float(pair["f1"]),
            )
        )

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    macro_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
    macro_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    coverage_gold = covered_gold_docs / gold_docs if gold_docs else 0.0
    coverage_all = pred_docs / len(records) if records else 0.0

    return (
        FieldScore(
            field=field,
            max_spans=max_spans,
            n_docs=len(records),
            support_docs=support_docs,
            gold_docs=gold_docs,
            pred_docs=pred_docs,
            covered_gold_docs=covered_gold_docs,
            micro_tp=micro_tp,
            micro_fp=micro_fp,
            micro_fn=micro_fn,
            micro_precision=round(micro_precision, 4),
            micro_recall=round(micro_recall, 4),
            micro_f1=round(micro_f1, 4),
            macro_precision=round(macro_precision, 4),
            macro_recall=round(macro_recall, 4),
            macro_f1=round(macro_f1, 4),
            coverage_gold=round(coverage_gold, 4),
            coverage_all=round(coverage_all, 4),
        ),
        doc_matches,
    )


def evaluate_pipeline_fields(
    records: Sequence[SchemaRecord],
    max_spans: Optional[int] = None,
) -> tuple[Dict[str, FieldScore], List[DocMatch]]:
    per_field: Dict[str, FieldScore] = {}
    doc_matches: List[DocMatch] = []
    for field in ELEMENTS:
        score, matches = score_records_for_field(records, field, max_spans=max_spans)
        per_field[field] = score
        doc_matches.extend(matches)
    return per_field, doc_matches



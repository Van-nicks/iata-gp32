import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


DEFAULT_DATA_DIR = Path("ebm_nlp_2_00") / "documents"
DEFAULT_OUTPUT_DIR = Path("clustering_results")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ClusterRun:
    method: str
    k: int
    silhouette_cosine: float
    calinski_harabasz: float
    davies_bouldin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sentence-embedding clustering (KMeans and HAC) over a k-range."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing abstract .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/JSON outputs are saved.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence-Transformers model name.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum k value (inclusive).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum k value (inclusive).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap on number of abstracts for quick runs (0 = all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def split_sentences(text: str) -> List[str]:
    # Use NLTK if available, otherwise fall back to a regex splitter.
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s and s.strip()]


def load_sentences(docs_dir: Path, max_docs: int = 0) -> List[Dict[str, object]]:
    txt_files = sorted(docs_dir.glob("*.txt"))
    if max_docs > 0:
        txt_files = txt_files[:max_docs]

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {docs_dir}. Check the path and file extension."
        )

    rows: List[Dict[str, object]] = []
    for file_path in txt_files:
        pmid = file_path.stem
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for sent_id, sentence in enumerate(split_sentences(text)):
            rows.append(
                {
                    "pmid": pmid,
                    "sentence_id": sent_id,
                    "sentence": sentence,
                }
            )

    if not rows:
        raise ValueError("No non-empty sentences found after tokenization.")

    return rows


def encode_sentences(sentences: Sequence[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(sentences),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def fit_hac(embeddings: np.ndarray, k: int) -> np.ndarray:
    # Support both older and newer scikit-learn APIs.
    try:
        model = AgglomerativeClustering(
            **{"n_clusters": k, "metric": "cosine", "linkage": "average"}
        )
    except TypeError:
        model = AgglomerativeClustering(
            **{"n_clusters": k, "affinity": "cosine", "linkage": "average"}
        )
    return model.fit_predict(embeddings)


def compute_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))
    ch = float(calinski_harabasz_score(embeddings, labels))
    db = float(davies_bouldin_score(embeddings, labels))
    return silhouette, ch, db


def evaluate_k_range(embeddings: np.ndarray, k_values: Sequence[int]) -> Tuple[List[ClusterRun], Dict[Tuple[str, int], np.ndarray]]:
    results: List[ClusterRun] = []
    labels_store: Dict[Tuple[str, int], np.ndarray] = {}

    n_sentences = embeddings.shape[0]
    valid_k = [k for k in k_values if 2 <= k < n_sentences]
    if not valid_k:
        raise ValueError(
            f"No valid k values for {n_sentences} samples. Try lowering k range."
        )

    for k in valid_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km_labels = km.fit_predict(embeddings)
        s, ch, db = compute_metrics(embeddings, km_labels)
        results.append(
            ClusterRun(
                method="kmeans",
                k=k,
                silhouette_cosine=s,
                calinski_harabasz=ch,
                davies_bouldin=db,
            )
        )
        labels_store[("kmeans", k)] = km_labels

        hac_labels = fit_hac(embeddings, k)
        s, ch, db = compute_metrics(embeddings, hac_labels)
        results.append(
            ClusterRun(
                method="hac",
                k=k,
                silhouette_cosine=s,
                calinski_harabasz=ch,
                davies_bouldin=db,
            )
        )
        labels_store[("hac", k)] = hac_labels

    return results, labels_store


def pick_best_by_silhouette(runs: Sequence[ClusterRun]) -> Dict[str, ClusterRun]:
    best: Dict[str, ClusterRun] = {}
    for method in sorted({r.method for r in runs}):
        method_runs = [r for r in runs if r.method == method]
        best[method] = max(method_runs, key=lambda r: r.silhouette_cosine)
    return best


def save_metrics(runs: Sequence[ClusterRun], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "k_sweep_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "k",
            "silhouette_cosine",
            "calinski_harabasz",
            "davies_bouldin",
        ])
        for r in runs:
            writer.writerow([
                r.method,
                r.k,
                f"{r.silhouette_cosine:.6f}",
                f"{r.calinski_harabasz:.6f}",
                f"{r.davies_bouldin:.6f}",
            ])
    return path


def save_best_summary(best_runs: Dict[str, ClusterRun], output_dir: Path, args: argparse.Namespace, n_sentences: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best_k_summary.json"
    payload = {
        "config": {
            "docs_dir": str(args.docs_dir),
            "model": args.model,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "max_docs": args.max_docs,
            "batch_size": args.batch_size,
            "n_sentences": n_sentences,
        },
        "best_by_method": {
            method: {
                "k": run.k,
                "silhouette_cosine": run.silhouette_cosine,
                "calinski_harabasz": run.calinski_harabasz,
                "davies_bouldin": run.davies_bouldin,
            }
            for method, run in best_runs.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_labels(
    rows: Sequence[Dict[str, object]],
    labels_store: Dict[Tuple[str, int], np.ndarray],
    best_runs: Dict[str, ClusterRun],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for method, run in best_runs.items():
        labels = labels_store[(method, run.k)]
        path = output_dir / f"best_labels_{method}_k{run.k}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pmid", "sentence_id", "sentence", "cluster"])
            for row, label in zip(rows, labels):
                writer.writerow([
                    row["pmid"],
                    row["sentence_id"],
                    row["sentence"],
                    int(label),
                ])
        output_paths.append(path)

    return output_paths


def print_leaderboard(runs: Sequence[ClusterRun]) -> None:
    print("\nTop runs by silhouette (higher is better):")
    sorted_runs = sorted(runs, key=lambda r: r.silhouette_cosine, reverse=True)
    for rank, run in enumerate(sorted_runs[:10], start=1):
        print(
            f"  {rank:2d}. {run.method:6s} k={run.k:2d} "
            f"sil={run.silhouette_cosine:.4f} "
            f"CH={run.calinski_harabasz:.1f} DB={run.davies_bouldin:.4f}"
        )


def main() -> None:
    args = parse_args()

    if args.k_min > args.k_max:
        raise ValueError("k-min must be <= k-max")

    rows = load_sentences(args.docs_dir, args.max_docs)
    sentences = [str(row["sentence"]) for row in rows]

    print(f"Loaded {len(rows)} sentences from {args.docs_dir}")
    embeddings = encode_sentences(sentences, args.model, args.batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    k_values = list(range(args.k_min, args.k_max + 1))
    runs, labels_store = evaluate_k_range(embeddings, k_values)
    best_runs = pick_best_by_silhouette(runs)

    metrics_path = save_metrics(runs, args.output_dir)
    summary_path = save_best_summary(best_runs, args.output_dir, args, len(rows))
    label_paths = save_labels(rows, labels_store, best_runs, args.output_dir)

    print_leaderboard(runs)
    print("\nBest k by method:")
    for method, run in best_runs.items():
        print(
            f"  - {method}: k={run.k}, silhouette={run.silhouette_cosine:.4f}, "
            f"CH={run.calinski_harabasz:.1f}, DB={run.davies_bouldin:.4f}"
        )

    print("\nSaved outputs:")
    print(f"  - {metrics_path}")
    print(f"  - {summary_path}")
    for p in label_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()



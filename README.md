# Copilot Bundle: Combined Notes

This folder collects directories/files produced or updated during the assistant-led implementation flow.

## Included items

- `step1_clustering_experiment.py`
- `requirements.txt`
- `schema_export.py`
- `evaluate_extraction.py`
- `step2_evaluation/`
- `structured_tables_test/`
- `evaluation_results_test/`

## Notes

- Items are copied here for review; originals remain in the project root.
- `structured_tables_test/` and `evaluation_results_test/` are generated test outputs from verification runs.

---

## Step 1: Sentence Clustering Sweep (KMeans + HAC)

This script runs clustering pre-analysis for EBM-NLP abstracts and compares `k` values in a range (default `3..10`) for both KMeans and HAC.

### File

- `step1_clustering_experiment.py`

### What it does

1. Loads abstract text files from `ebm_nlp_2_00/documents/*.txt`
2. Splits abstracts into sentences
3. Creates sentence embeddings using Sentence-Transformers
4. Runs both clustering methods for each `k` in range
5. Computes metrics:
   - cosine silhouette (higher is better)
   - Calinski-Harabasz (higher is better)
   - Davies-Bouldin (lower is better)
6. Selects best `k` per method by silhouette
7. Saves metrics and best-cluster labels

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Run (full data, default `k=3..10`)

```bash
python3 step1_clustering_experiment.py
```

### Run (quick sanity check)

```bash
python3 step1_clustering_experiment.py --max-docs 20 --k-min 3 --k-max 6
```

### Output files

Saved under `clustering_results/`:

- `k_sweep_metrics.csv`
- `best_k_summary.json`
- `best_labels_kmeans_k<k>.csv`
- `best_labels_hac_k<k>.csv`

---

## Step 2: P/I/O Schema Export

This step turns the cleaned dataset and extraction outputs into a tabular schema with one row per PMID.

### What it exports

- `schema_gold_table.csv` / `schema_gold_table.json`
  - one row per PMID from the cleaned dataset
  - fields: `participants`, `interventions`, `outcomes`

- `schema_predictions_<pipeline>.csv` / `schema_predictions_<pipeline>.json`
  - one row per PMID for each extraction pipeline
  - predicted P/I/O spans plus gold spans for comparison

- `schema_export_manifest.json`
  - export metadata for reproducibility

### Run it

```bash
python3 schema_export.py --output-dir structured_tables
```

Only gold table:

```bash
python3 schema_export.py --skip-predictions
```

Only prediction tables:

```bash
python3 schema_export.py --skip-gold
```

### Why this matters

This is the first step toward a final structured dataset for:

- field-level extraction evaluation
- downstream query testing
- comparison of future extraction strategies

---

## Step 2: Standardized Evaluation Suite

This step evaluates P/I/O extraction outputs in a consistent way across pipelines.

### What it measures

- field-level precision / recall / F1 for:
  - `participants`
  - `interventions`
  - `outcomes`
- coverage vs precision tradeoff by truncating predicted spans
- downstream query success for simple P/I/O tasks
- reproducibility through a saved manifest

### Main entrypoint

```bash
python3 evaluate_extraction.py --output-dir evaluation_results
```

By default, the script looks for:

- gold schema table:
  - `structured_tables/schema_gold_table.json`
  - or `structured_tables_test/schema_gold_table.json`
- prediction table:
  - `structured_tables/schema_predictions_rule_based.json`
  - or `structured_tables_test/schema_predictions_rule_based.json`

Explicit file paths:

```bash
python3 evaluate_extraction.py \
  --gold-table structured_tables_test/schema_gold_table.json \
  --prediction-table structured_tables_test/schema_predictions_rule_based.json
```

### Output files

- `field_scores.csv` / `field_scores.json`
- `tradeoff_curve.csv` / `tradeoff_curve.json`
- `query_scores.csv` / `query_scores.json`
- `evaluation_summary.json`
- `evaluation_manifest.json`

### Query configuration

- `step2_evaluation/configs/pio_queries.json`

### Why this matters

This gives a standardized way to compare extraction methods, instead of only checking raw spans.


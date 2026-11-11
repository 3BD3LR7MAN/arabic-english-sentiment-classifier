# Arabic / English Sentiment Classifier

A three-class sentiment project for Arabic and English text. It compares a word/character TF-IDF + Logistic Regression baseline with pretrained multilingual XLM-R inference.

## Privacy-safe repository contents

This repository intentionally excludes the source-text sample, row-level predictions, error examples, and serialized fitted model. The upstream corpus aggregates third-party text, and some sampled records contain long numeric identifiers. Only code and aggregate evaluation artifacts are published. Do not commit locally generated text data, row-level outputs, or fitted models unless source permissions and privacy have been reviewed.

## Results: two different evaluation settings

### Existing official test split (sources overlap)

| Model | Test accuracy | Test macro-F1 | Neutral recall |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.648 | 0.648 | 0.637 |
| Pretrained XLM-R, no fine-tuning | 0.627 | 0.616 | 0.403 |

Both models use the same 900 held-out rows. TF-IDF regularization was selected on validation; XLM-R was applied without fine-tuning, so the tuning budgets differ. The sample is balanced and educational, not representative of production sentiment prevalence. All 17 test domains and 27 test sources also occur in training, so this split does not evaluate unseen-source or unseen-domain generalization. Aggregate artifacts are under `reports/`.

### Additional source-heldout estimate

A nested 3-outer-fold × 3-inner-fold grouped evaluation uses only the 3,600 rows from the original `train` portion and holds out complete `source` groups in each outer fold. It produced **accuracy 0.391 and macro-F1 0.378** across 29 source groups. Per-language macro-F1 was 0.382 for Arabic and 0.358 for English. Fold macro-F1 values were 0.463, 0.243, and 0.401, showing substantial variation across held-out sources.

This is an internal cross-validation diagnostic, not a replacement for an independent external benchmark. It is not directly comparable to the official test score: the data rows and split protocol differ. The sharp drop is evidence that the baseline generalizes poorly to some unseen sources. The report and aggregate-only outputs are in [`reports/Source_Heldout_Evaluation.md`](reports/Source_Heldout_Evaluation.md), [`reports/source_heldout_metrics.json`](reports/source_heldout_metrics.json), and [`reports/source_heldout_folds.csv`](reports/source_heldout_folds.csv). No source names or row-level examples are included.

## Reproduce locally

The dataset is intentionally not bundled. After reviewing [`DATA_SOURCES.md`](DATA_SOURCES.md) and upstream terms, prepare a local sample. This downloads roughly 0.9 GB of Parquet data and writes the generated text sample under an ignored path.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/prepare_dataset.py
```

The classical baseline does not need PyTorch or Transformers:

```bash
python src/train_evaluate.py --data data/processed/bilingual_sentiment_sample.csv --output . --skip-transformer
```

For the optional pretrained XLM-R comparison, install the larger transformer stack and run without `--skip-transformer`:

```bash
python -m pip install -r requirements-transformer.txt
python src/train_evaluate.py --data data/processed/bilingual_sentiment_sample.csv --output .
```

To run the nested source-heldout diagnostic and the automated checks:

```bash
python src/evaluate_source_heldout.py --data data/processed/bilingual_sentiment_sample.csv --output .
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The source-heldout evaluator uses only the original training rows, fits vectorizers within each training fold, tunes regularization inside grouped validation folds, and writes aggregate metrics only. Training the regular pipeline creates local row-level prediction/error files and a fitted model; `.gitignore` excludes them. Keep those artifacts private unless their rights and privacy have been reviewed.

## Methods and responsible use

The classical baseline tunes regularization on validation macro-F1 in the ordinary split and uses nested group-aware tuning for the source-heldout diagnostic. The XLM-R checkpoint is applied without fine-tuning. Public-source overlap with its pretraining or fine-tuning data cannot be ruled out. Sentiment labels are noisy; models may fail on sarcasm, dialect, code-switching, negation, short text, and domain-specific language. These models are educational examples, not production decision systems; human review is required before consequential use.

A lightweight CI workflow runs syntax, notebook-JSON, and unit checks without downloading the dataset or model weights.

## Licensing

**No open-source license is granted for this project's code at this time.** Public visibility does not grant permission to reuse or redistribute it. The dataset and pretrained model are third-party materials with separate terms; this repository does not grant rights to them. Add a license only after the copyright holder and applicable permissions are confirmed.

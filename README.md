# Arabic / English Sentiment Classifier

A three-class sentiment project for Arabic and English text. It compares a word/character TF-IDF + Logistic Regression baseline with pretrained multilingual XLM-R inference.

## Privacy-safe repository contents

This clean handoff intentionally does **not** include the source-text sample, record-level predictions, error examples, or serialized fitted model. The upstream corpus aggregates third-party text, and some sampled records contain long numeric identifiers. The repository therefore keeps only code and aggregate evaluation artifacts. Do not commit locally generated text data, row-level outputs, or models unless the underlying source permissions and privacy have been reviewed.

## Aggregate results from the documented evaluation run

| Model | Test accuracy | Test macro-F1 | Neutral recall |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.648 | 0.648 | 0.637 |
| Pretrained XLM-R | 0.627 | 0.616 | 0.403 |

Both models used the same held-out rows. The classical baseline's regularization was selected on validation; XLM-R was applied without fine-tuning, so the training/tuning budgets differ. These scores are from a balanced educational sample, not a production benchmark. All 17 test domains and 27 test sources also occur in training, so the evaluation does not measure generalization to unseen sources or domains.

The aggregate report, classification summaries, confusion matrices, and plots are under `reports/`. Row-level text is not included.

## Reproduce locally

The dataset is intentionally not bundled. After reviewing the source and upstream terms in [`DATA_SOURCES.md`](DATA_SOURCES.md), prepare a local sample. This downloads roughly 0.9 GB of Parquet data and writes the generated text sample under an ignored path.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/prepare_dataset.py
python src/train_evaluate.py --data data/processed/bilingual_sentiment_sample.csv --output .
```

Training creates local model and record-level report files; `.gitignore` excludes these outputs. Do not publish them without source permission and a privacy review. The pretrained Cardiff NLP checkpoint is downloaded on first transformer run; use `--skip-transformer` for a faster classical-only run.

For a local inference example after training:

```bash
python src/predict.py --model classical --text "الخدمة ممتازة وسريعة" --language ar
python src/predict.py --model classical --text "Great service and product, I love it." --language en
```

## Methods and responsible use

The classical baseline tunes regularization on validation macro-F1. The XLM-R checkpoint is evaluated without fine-tuning on the same test rows. Public-source overlap with the pretrained model's original training data cannot be ruled out. Sentiment labels are noisy; models may fail on sarcasm, dialect, code-switching, negation, short text, and domain-specific language. Human review is required before consequential use.

## Licensing

**No open-source license is granted for this project's code at this time.** Public visibility does not grant permission to reuse or redistribute it. The dataset and pretrained model are third-party materials with separate terms; this repository does not grant rights to them. Add a license only after the copyright holder and applicable permissions are confirmed.

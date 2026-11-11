# Source-Heldout Evaluation

## Purpose

The ordinary held-out test split shares sources and domains with training. This additional evaluation estimates how the classical baseline behaves when its test rows come from source groups absent from the corresponding model's training fold.

## Protocol

- Input scope: only the 3,600 rows marked `train` in the locally generated sample; the original validation and test rows are not used.
- Grouping: complete `source` groups are assigned to outer folds. The 29 groups are disjoint between each outer training and held-out fold.
- Nested selection: 3 outer grouped folds and 3 inner grouped folds; Logistic Regression `C` is selected from 0.5, 1.0, and 2.0 using inner-fold macro-F1.
- Leakage controls: text normalization and TF-IDF fitting occur within each training fold. Conflicting normalized texts are removed, and duplicate normalized texts are deduplicated before splitting.
- Reporting: only aggregate out-of-fold metrics, fold counts, and selected hyperparameters are saved. No text, source names, or row-level predictions are published.

## Results

| Outer fold | Held-out rows | Held-out source groups | Selected C | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,445 | 12 | 0.5 | 0.469 | 0.463 |
| 2 | 1,193 | 8 | 0.5 | 0.244 | 0.243 |
| 3 | 962 | 9 | 0.5 | 0.457 | 0.401 |
| **Out-of-fold aggregate** | **3,600** | **29** | — | **0.391** | **0.378** |

Out-of-fold macro-F1 was 0.382 for Arabic and 0.358 for English. The fold-to-fold spread is material; the pooled aggregate should not hide the weak performance on the second fold.

## Interpretation and limitations

The source-heldout macro-F1 is substantially below the 0.648 result on the ordinary test split. These estimates are not directly comparable because they use different rows and split protocols. The grouped result suggests substantial source dependence and weak generalization to some sources. It is an internal cross-validation diagnostic on the upstream training portion, not an independent external benchmark. Source groups are not necessarily equivalent to domains, and the pretrained XLM-R model was not evaluated with this grouped protocol.

## Reproduce

After preparing the local sample as described in [`../DATA_SOURCES.md`](../DATA_SOURCES.md), install core dependencies and run:

```bash
python src/evaluate_source_heldout.py --data data/processed/bilingual_sentiment_sample.csv --output .
```

The aggregate outputs are [`source_heldout_metrics.json`](source_heldout_metrics.json) and [`source_heldout_folds.csv`](source_heldout_folds.csv). The text dataset is not included in this repository.

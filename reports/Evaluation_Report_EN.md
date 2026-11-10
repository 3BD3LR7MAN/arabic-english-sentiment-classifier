# Arabic / English Sentiment Classifier — Evaluation Report

## Dataset and protocol

This experiment uses a balanced sample of 5,100 Arabic and English records from `clapAI/MultiLingualSentiment` (original labels: negative, neutral, positive). Sample sizes: train 3,600, validation 600, test 900. Data are grouped by language and label for the sample; source/domain columns are retained. In this sample all 17 test domains and all 27 test sources also occur in train, so the split does not measure unseen-source or unseen-domain generalization. The dataset is not a domain-controlled Arabic-vs-English benchmark.

The classical model's regularization strength is selected on validation macro-F1. The XLM-R option is an existing multilingual Twitter sentiment checkpoint applied without fine-tuning to this sample. Both are evaluated on the same held-out test rows, but the tuning budgets are not identical. Results are descriptive; source overlap with upstream/pretraining data cannot be ruled out.

## Results

- TF-IDF + Logistic Regression — validation: accuracy 0.642, macro-F1 0.643, weighted-F1 0.643 (n=600).
- TF-IDF + Logistic Regression — test: accuracy 0.648, macro-F1 0.648, weighted-F1 0.648 (n=900).
- CardiffNLP XLM-R pretrained (no fine-tuning on this sample) — validation: accuracy 0.623, macro-F1 0.613, weighted-F1 0.613 (n=600).
- CardiffNLP XLM-R pretrained (no fine-tuning on this sample) — test: accuracy 0.627, macro-F1 0.616, weighted-F1 0.616 (n=900).

## Per-language test results

- TF-IDF + Logistic Regression — ar: accuracy 0.647, macro-F1 0.646, macro-recall 0.647 (n=450).
- CardiffNLP XLM-R pretrained — ar: accuracy 0.618, macro-F1 0.608, macro-recall 0.618 (n=450).
- TF-IDF + Logistic Regression — en: accuracy 0.649, macro-F1 0.648, macro-recall 0.649 (n=450).
- CardiffNLP XLM-R pretrained — en: accuracy 0.636, macro-F1 0.622, macro-recall 0.636 (n=450).

## Error analysis and limitations

The row-level error table from the original experiment is intentionally omitted from this shareable copy because it contains source text. XLM-R's neutral-class recall is 0.403 (121/300), versus 0.637 (191/300) for TF-IDF; review the aggregate confusion matrices. Inspect Arabic dialect, sarcasm, mixed-language text, negation, and domain vocabulary before using the model. The pretrained model is not fine-tuned on this sample. The balanced subset changes the class prevalence and must not be used to infer production prevalence or calibrated probabilities. Do not make consequential customer decisions solely from sentiment labels.

## References

[1]: https://huggingface.co/datasets/clapAI/MultiLingualSentiment "MultilingualSentiment dataset card"
[2]: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment "Cardiff NLP XLM-R Twitter sentiment model card"
[3]: https://arxiv.org/abs/2104.12250 "XLM-T multilingual Twitter language model paper"

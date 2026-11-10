# Data and model sources

## Dataset

- **Source:** [`clapAI/MultiLingualSentiment`](https://huggingface.co/datasets/clapAI/MultiLingualSentiment)
- **Dataset card:** The card declares Apache-2.0 for the aggregate as represented there and describes Arabic/English sentiment labels.
- **Important rights caveat:** The aggregate combines multiple upstream Hugging Face and Kaggle sources. Their original redistribution terms may differ; this project has not independently verified every upstream grant.
- **Distribution choice:** No source Parquet, sampled review text, row-level predictions, error examples, or fitted model is included in this clean repository copy. Some source records may contain numeric identifiers. Keep locally generated text data private unless source permissions and privacy review explicitly allow redistribution.
- **Recreation:** `python src/prepare_dataset.py` downloads the pinned Parquet conversion (roughly 0.9 GB total), verifies SHA-256 hashes, and writes a local balanced sample. Running that script does not grant permission to publish its outputs.
- **Fields used by the pipeline:** `text`, `label`, `source`, `domain`, and `language`.
- **Sample design from the documented run:** 3,600 train, 600 validation, and 900 test rows, balanced by language and label. Every test domain (17) and source (27) also occurs in train; this is not a source- or domain-held-out evaluation.
- **Pinned Parquet conversion revision:** `a51e836ffc377d55c2278b0556c939ad9a3ca09e`.

| Parquet file | SHA-256 |
|---|---|
| `train_0000.parquet` | `08b904f9acd92b32a92d455e51676a29c3569900ec96478637f4e1ccc31f7cf8` |
| `train_0001.parquet` | `626dd6e46725e528306325cec0e047a9f727811ca36ef761bd146c713f00c5e0` |
| `train_0002.parquet` | `0874cb9fe7310ea1e815acceb292421de756c44e67b84e441f6cd8845efcecc9` |
| `test_0000.parquet` | `aa84e2c86f0150673cf44c9290536ef50a9886b7e3f8eb9b00333bc6b1910952` |

## Pretrained model

- **Model:** [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
- **Paper:** [XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond](https://arxiv.org/abs/2104.12250)
- **Use in this project:** direct inference without fine-tuning. The checkpoint is fetched on first use and is not redistributed here. Review the model card and upstream terms before use.

## Citation suggestions

```bibtex
@dataset{clapAI2024multilingualsentiment,
  title  = {MultilingualSentiment: A Multilingual Sentiment Classification Dataset},
  author = {clapAI},
  year   = {2024},
  url    = {https://huggingface.co/datasets/clapAI/MultiLingualSentiment}
}

@inproceedings{barbieri-etal-2022-xlm,
  title = {XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond},
  author = {Barbieri, Francesco and Espinosa Anke, Luis and Camacho-Collados, Jose},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  year = {2022},
  pages = {258--266},
  url = {https://aclanthology.org/2022.lrec-1.27}
}
```

## Project code license

No project-code license is included. No permission to reuse third-party dataset or model materials is granted by this repository.

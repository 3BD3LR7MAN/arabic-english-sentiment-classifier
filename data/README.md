# Local data only

The source Parquet files and sampled review text are intentionally not distributed in this repository. After reviewing upstream permissions, run `python src/prepare_dataset.py` to create local files. Do not commit generated raw text, row-level predictions, error examples, or fitted models; the repository `.gitignore` excludes these paths.

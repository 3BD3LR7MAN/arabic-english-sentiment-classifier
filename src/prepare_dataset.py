#!/usr/bin/env python3
"""Download source Parquet shards and create a balanced Arabic/English sentiment sample."""
from __future__ import annotations
import argparse
import hashlib
import random
from pathlib import Path
import requests
import pandas as pd
import pyarrow.parquet as pq

REVISION = 'a51e836ffc377d55c2278b0556c939ad9a3ca09e'
BASE = f'https://huggingface.co/datasets/clapAI/MultiLingualSentiment/resolve/{REVISION}/default'
TRAIN_FILES = [f'{BASE}/train/{i:04d}.parquet' for i in range(3)]
TEST_FILE = f'{BASE}/test/0000.parquet'
SHA256 = {
    'train_0000.parquet': '08b904f9acd92b32a92d455e51676a29c3569900ec96478637f4e1ccc31f7cf8',
    'train_0001.parquet': '626dd6e46725e528306325cec0e047a9f727811ca36ef761bd146c713f00c5e0',
    'train_0002.parquet': '0874cb9fe7310ea1e815acceb292421de756c44e67b84e441f6cd8845efcecc9',
    'test_0000.parquet': 'aa84e2c86f0150673cf44c9290536ef50a9886b7e3f8eb9b00333bc6b1910952',
}
LABELS = ('negative', 'neutral', 'positive')
LANGS = ('ar', 'en')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected_sha256: str):
    if path.exists() and path.stat().st_size > 0 and sha256(path) == expected_sha256:
        return
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for block in r.iter_content(1024 * 1024):
                if block:
                    f.write(block)
                    digest.update(block)
    if digest.hexdigest() != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'SHA-256 mismatch for {path.name}; source may have changed.')
    tmp.replace(path)


def reservoir_from_parquet(paths, split: str, k_per_class_lang: int, seed: int):
    rng = random.Random(seed)
    reservoir = {(lang, label): [] for lang in LANGS for label in LABELS}
    seen = {(lang, label): 0 for lang in LANGS for label in LABELS}
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=40_000, columns=['text', 'label', 'source', 'domain', 'language']):
            cols = [batch.column(i).to_pylist() for i in range(5)]
            for text, label, source, domain, lang in zip(*cols):
                if lang not in LANGS or label not in LABELS or not isinstance(text, str) or not text.strip():
                    continue
                key = (lang, label)
                seen[key] += 1
                item = {'text': text.strip(), 'label': label, 'language': lang,
                        'source': source or '', 'domain': domain or '', 'split': split}
                n = seen[key]
                bucket = reservoir[key]
                if len(bucket) < k_per_class_lang:
                    bucket.append(item)
                else:
                    j = rng.randrange(n)
                    if j < k_per_class_lang:
                        bucket[j] = item
    frames = []
    for key, records in reservoir.items():
        if len(records) < k_per_class_lang:
            print(f'Warning: {split} {key} only has {len(records)} of {k_per_class_lang} desired rows')
        frames.extend(records)
    return pd.DataFrame(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='data/processed/bilingual_sentiment_sample.csv')
    ap.add_argument('--train-per-class-language', type=int, default=600)
    ap.add_argument('--validation-per-class-language', type=int, default=100)
    ap.add_argument('--test-per-class-language', type=int, default=150)
    ap.add_argument('--raw-dir', default='data/raw')
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    train_paths = []
    for i, url in enumerate(TRAIN_FILES):
        p = raw_dir / f'train_{i:04d}.parquet'
        print(f'Checking/downloading {p.name}', flush=True)
        download(url, p, SHA256[p.name])
        train_paths.append(p)
    test_path = raw_dir / 'test_0000.parquet'
    print('Checking/downloading test_0000.parquet', flush=True)
    download(TEST_FILE, test_path, SHA256[test_path.name])

    train_total = args.train_per_class_language + args.validation_per_class_language
    train_pool = reservoir_from_parquet(train_paths, 'train_pool', train_total * 3, seed=20261108)
    conflicts = train_pool.groupby(['language', 'text'])['label'].nunique()
    conflicting = set(conflicts[conflicts > 1].index.tolist())
    if conflicting:
        train_pool = train_pool[~train_pool.apply(lambda r: (r['language'], r['text']) in conflicting, axis=1)]
    train_pool = train_pool.drop_duplicates(['language', 'text']).copy()
    train_parts, val_parts = [], []
    rng = random.Random(42)
    for (lang, label), group in train_pool.groupby(['language', 'label'], sort=True):
        if len(group) < train_total:
            raise ValueError(f'Not enough unique rows for {lang}/{label}: {len(group)} < {train_total}; increase reservoir size.')
        group = group.sample(n=train_total, random_state=42).reset_index(drop=True)
        idx = list(range(len(group)))
        rng.shuffle(idx)
        val_idx = idx[:args.validation_per_class_language]
        train_idx = idx[args.validation_per_class_language:args.validation_per_class_language + args.train_per_class_language]
        train_parts.append(group.iloc[train_idx].assign(split='train'))
        val_parts.append(group.iloc[val_idx].assign(split='validation'))
    test_df = reservoir_from_parquet([test_path], 'test', args.test_per_class_language * 3, seed=20261109)
    conflicts = test_df.groupby(['language', 'text'])['label'].nunique()
    conflicting = set(conflicts[conflicts > 1].index.tolist())
    if conflicting:
        test_df = test_df[~test_df.apply(lambda r: (r['language'], r['text']) in conflicting, axis=1)]
    test_df = test_df.drop_duplicates(['language', 'text']).copy()
    test_parts=[]
    for (lang, label), group in test_df.groupby(['language','label'], sort=True):
        if len(group) < args.test_per_class_language:
            raise ValueError(f'Not enough unique test rows for {lang}/{label}: {len(group)} < {args.test_per_class_language}; increase reservoir size.')
        test_parts.append(group.sample(n=args.test_per_class_language, random_state=43))
    test_df = pd.concat(test_parts, ignore_index=True)
    result = pd.concat(train_parts + val_parts + [test_df], ignore_index=True)
    result['text'] = result['text'].astype(str).str.strip()
    result = result[result['text'].str.len() > 0].drop_duplicates(subset=['language', 'text', 'split'])
    # For official test, avoid exact text duplicates found in the sampled train/validation pool.
    train_texts = set(result.loc[result['split'] != 'test', ['language', 'text']].itertuples(index=False, name=None))
    test_mask = result['split'].eq('test')
    dup_test = result.loc[test_mask].apply(lambda r: (r['language'], r['text']) in train_texts, axis=1)
    if dup_test.any():
        result = result.loc[~(test_mask & dup_test)].copy()
        print(f'Removed {int(dup_test.sum())} exact cross-split text duplicates from test sample.')
    expected={'train':args.train_per_class_language,'validation':args.validation_per_class_language,'test':args.test_per_class_language}
    for split, target in expected.items():
        counts=result[result['split']==split].groupby(['language','label']).size()
        if len(counts)!=len(LANGS)*len(LABELS) or not (counts==target).all():
            raise ValueError(f'{split} is not balanced at {target} examples per language-label pair: {counts.to_dict()}')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding='utf-8')
    counts = result.groupby(['split', 'language', 'label']).size().rename('count').reset_index()
    counts.to_csv(out.with_name('sample_counts.csv'), index=False)
    print('Saved', out.resolve(), 'rows=', len(result))
    print(counts.to_string(index=False))

if __name__ == '__main__':
    main()

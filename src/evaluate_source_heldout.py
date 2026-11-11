#!/usr/bin/env python3
"""Nested source-held-out cross-validation for the classical sentiment baseline.

Uses only rows marked split=train. Source groups are disjoint between every
outer train/held-out fold. Inner grouped CV chooses C without seeing the outer
held-out groups. Only aggregate metrics are written; no text, source names, or
row-level predictions are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

try:
    from .train_evaluate import LABELS, metric_block, normalize_text
except ImportError:  # direct invocation: python src/evaluate_source_heldout.py
    from train_evaluate import LABELS, metric_block, normalize_text

C_GRID = (0.5, 1.0, 2.0)


def make_vectorizer() -> FeatureUnion:
    """Match the published word/character TF-IDF feature construction."""
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
                                 ngram_range=(1, 2), min_df=2,
                                 max_features=180_000, sublinear_tf=True,
                                 strip_accents=None)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5),
                                 min_df=2, max_features=280_000,
                                 sublinear_tf=True)),
    ])


def make_classifier(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight='balanced', max_iter=1200,
                              solver='lbfgs', random_state=42)


def prepare_training_rows(data: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, dict]:
    required = {'text', 'label', 'language', 'split', group_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Data is missing required columns: {sorted(missing)}')
    frame = data.loc[data['split'].eq('train')].copy()
    frame = frame[frame['language'].isin(['ar', 'en']) & frame['label'].isin(LABELS)].copy()
    frame['normalized_text'] = frame['text'].astype(str).map(normalize_text)
    frame = frame[frame['normalized_text'].str.len() >= 2].copy()
    frame['group'] = frame[group_column].fillna('').astype(str).str.strip()
    if frame['group'].eq('').any():
        raise ValueError(f'{group_column} has missing/blank values in the training rows; source grouping would be ambiguous.')

    # Remove conflicting normalized text keys, then de-duplicate across groups
    # so the same normalized text cannot leak between source-held-out folds.
    conflict = frame.groupby(['language', 'normalized_text'])['label'].transform('nunique').gt(1)
    conflict_rows = int(conflict.sum())
    frame = frame.loc[~conflict].copy()
    duplicate = frame.duplicated(['language', 'normalized_text'], keep='first')
    duplicate_rows = int(duplicate.sum())
    frame = frame.loc[~duplicate].reset_index(drop=True)
    frame['stratum'] = frame['language'].astype(str) + '|' + frame['label'].astype(str)
    audit = {
        'input_train_rows': int(data['split'].eq('train').sum()),
        'rows_removed_for_conflicting_normalized_labels': conflict_rows,
        'rows_removed_as_normalized_duplicates': duplicate_rows,
        'rows_used': int(len(frame)),
        'group_count': int(frame['group'].nunique()),
    }
    return frame, audit


def evaluate(data: pd.DataFrame, group_column: str = 'source', outer_folds: int = 3,
             inner_folds: int = 3, seed: int = 42) -> dict:
    frame, audit = prepare_training_rows(data, group_column)
    groups = frame['group'].to_numpy()
    strata = frame['stratum'].to_numpy()
    if audit['group_count'] < outer_folds:
        raise ValueError('There are fewer groups than requested outer folds.')

    outer_cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True,
                                    random_state=seed)
    y_oof, pred_oof, lang_oof = [], [], []
    fold_rows = []
    for fold_index, (outer_train_idx, outer_test_idx) in enumerate(
            outer_cv.split(frame, strata, groups), start=1):
        outer_train = frame.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = frame.iloc[outer_test_idx].reset_index(drop=True)
        train_groups = set(outer_train['group'])
        test_groups = set(outer_test['group'])
        overlap_count = len(train_groups & test_groups)
        if overlap_count:
            raise AssertionError(f'Outer fold {fold_index} has group leakage.')
        if set(outer_train['label']) != set(LABELS):
            raise ValueError(f'Outer training fold {fold_index} does not contain all sentiment labels.')

        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True,
                                        random_state=seed + fold_index)
        inner_truth, inner_predictions = [], {c: [] for c in C_GRID}
        inner_strata = outer_train['stratum'].to_numpy()
        inner_groups = outer_train['group'].to_numpy()
        for inner_train_idx, inner_valid_idx in inner_cv.split(
                outer_train, inner_strata, inner_groups):
            inner_train = outer_train.iloc[inner_train_idx]
            inner_valid = outer_train.iloc[inner_valid_idx]
            if set(inner_train['group']) & set(inner_valid['group']):
                raise AssertionError('Inner fold has group leakage.')
            if set(inner_train['label']) != set(LABELS):
                raise ValueError(f'An inner training fold inside outer fold {fold_index} misses a sentiment label.')
            vectorizer = make_vectorizer()
            x_train = vectorizer.fit_transform(inner_train['normalized_text'])
            x_valid = vectorizer.transform(inner_valid['normalized_text'])
            y_valid = inner_valid['label'].to_numpy()
            inner_truth.append(y_valid)
            for c in C_GRID:
                classifier = make_classifier(c)
                classifier.fit(x_train, inner_train['label'])
                inner_predictions[c].append(classifier.predict(x_valid))
            del vectorizer, x_train, x_valid

        truth = np.concatenate(inner_truth)
        c_scores = {
            c: float(f1_score(truth, np.concatenate(inner_predictions[c]),
                              labels=LABELS, average='macro', zero_division=0))
            for c in C_GRID
        }
        best_c = max(C_GRID, key=lambda c: (c_scores[c], -c))

        final_vectorizer = make_vectorizer()
        x_outer_train = final_vectorizer.fit_transform(outer_train['normalized_text'])
        x_outer_test = final_vectorizer.transform(outer_test['normalized_text'])
        final_classifier = make_classifier(best_c)
        final_classifier.fit(x_outer_train, outer_train['label'])
        outer_pred = final_classifier.predict(x_outer_test)
        y_oof.extend(outer_test['label'].tolist())
        pred_oof.extend(outer_pred.tolist())
        lang_oof.extend(outer_test['language'].tolist())
        fold_rows.append({
            'fold': fold_index,
            'n_train': int(len(outer_train)),
            'n_heldout': int(len(outer_test)),
            'train_groups': int(len(train_groups)),
            'heldout_groups': int(len(test_groups)),
            'group_overlap': overlap_count,
            'selected_C': float(best_c),
            'inner_macro_f1_by_C': {str(c): c_scores[c] for c in C_GRID},
            'heldout_metrics': metric_block(outer_test['label'], outer_pred),
        })
        print(f'Completed outer fold {fold_index}/{outer_folds}; '
              f'held-out groups={len(test_groups)}, selected C={best_c}', flush=True)

    y_oof = np.asarray(y_oof)
    pred_oof = np.asarray(pred_oof)
    lang_oof = np.asarray(lang_oof)
    per_language = []
    for language in ('ar', 'en'):
        mask = lang_oof == language
        per_language.append({
            'language': language,
            **metric_block(y_oof[mask], pred_oof[mask]),
        })
    return {
        'evaluation': f'nested stratified {group_column}-held-out cross-validation',
        'group_column': group_column,
        'outer_folds': outer_folds,
        'inner_folds': inner_folds,
        'seed': seed,
        'C_grid': list(C_GRID),
        'scope_note': (f'Uses only rows whose original split is train. Each outer '
                       f'held-out {group_column} group is absent from that fold training; '
                       'this is not a replacement for an external benchmark.'),
        'audit': audit,
        'outer_fold_results': fold_rows,
        'aggregate_oof_metrics': metric_block(y_oof, pred_oof),
        'per_language_oof_metrics': per_language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/bilingual_sentiment_sample.csv')
    parser.add_argument('--output', default='.')
    parser.add_argument('--group-column', choices=['source', 'domain'], default='source')
    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.data), args.group_column,
                      args.outer_folds, args.inner_folds, args.seed)
    reports = Path(args.output).resolve() / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f'{args.group_column}_heldout_metrics.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pd.DataFrame([
        {k: v for k, v in fold.items() if k != 'inner_macro_f1_by_C'}
        for fold in result['outer_fold_results']
    ]).to_csv(reports / f'{args.group_column}_heldout_folds.csv', index=False)
    print(json.dumps({
        'aggregate_oof_metrics': result['aggregate_oof_metrics'],
        'per_language_oof_metrics': result['per_language_oof_metrics'],
        'audit': result['audit'],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

import pandas as pd

from src.evaluate_source_heldout import prepare_training_rows


def test_source_heldout_prep_uses_train_rows_and_removes_text_overlap():
    data = pd.DataFrame([
        {'text': 'Great service!', 'label': 'positive', 'language': 'en', 'split': 'train', 'source': 'a'},
        {'text': ' GREAT SERVICE! ', 'label': 'positive', 'language': 'en', 'split': 'train', 'source': 'b'},
        {'text': 'Bad service!', 'label': 'negative', 'language': 'en', 'split': 'train', 'source': 'a'},
        {'text': 'BAD SERVICE!', 'label': 'positive', 'language': 'en', 'split': 'train', 'source': 'b'},
        {'text': 'Validation-only row', 'label': 'neutral', 'language': 'ar', 'split': 'validation', 'source': 'c'},
    ])

    prepared, audit = prepare_training_rows(data, 'source')

    assert audit['input_train_rows'] == 4
    assert audit['rows_removed_for_conflicting_normalized_labels'] == 2
    assert audit['rows_removed_as_normalized_duplicates'] == 1
    assert audit['rows_used'] == 1
    assert set(prepared['split']) == {'train'}
    assert prepared.iloc[0]['normalized_text'] == 'great service!'

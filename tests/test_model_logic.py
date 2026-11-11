import subprocess
import sys
from pathlib import Path

import pytest

from src.train_evaluate import metric_block, normalize_text, predict_transformer


def test_normalize_text_handles_arabic_diacritics_urls_and_mentions():
    result = normalize_text('  إِنَّهُ ممتازٌ! https://example.com @ali  ')
    assert result == 'انه ممتاز! url user'


def test_metric_block_returns_expected_aggregate_values():
    result = metric_block(
        ['negative', 'negative', 'neutral', 'positive'],
        ['negative', 'neutral', 'neutral', 'positive'],
    )
    assert result['accuracy'] == pytest.approx(0.75)
    assert result['macro_f1'] == pytest.approx(7 / 9)
    assert result['n'] == 4


def test_module_import_does_not_load_transformer_stack():
    project = Path(__file__).resolve().parents[1]
    code = '''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("transformers"):
        raise AssertionError("optional transformer dependency imported eagerly")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import src.train_evaluate
'''
    subprocess.run([sys.executable, '-c', code], cwd=project, check=True)


def test_transformer_path_explains_optional_install_if_missing(monkeypatch):
    real_import = __import__

    def block_transformer(name, *args, **kwargs):
        if name == 'torch' or name.startswith('transformers'):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', block_transformer)
    with pytest.raises(RuntimeError, match='requirements-transformer.txt'):
        predict_transformer(['illustrative text'])

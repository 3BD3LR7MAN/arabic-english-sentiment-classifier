#!/usr/bin/env python3
"""Classify one Arabic or English text with the trained baseline or pretrained transformer."""
import argparse, json
from pathlib import Path
import joblib
from train_evaluate import normalize_text, predict_transformer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['classical','transformer'],default='classical')
    ap.add_argument('--language',choices=['ar','en'],default='en')
    ap.add_argument('--text',required=True)
    ap.add_argument('--model-path',default='models/tfidf_logistic_regression.joblib')
    args=ap.parse_args()
    text=normalize_text(args.text)
    if args.model=='classical':
        path=Path(args.model_path)
        if not path.exists():
            raise SystemExit(f'Model file not found: {path}. Run python src/train_evaluate.py first.')
        bundle=joblib.load(path)
        pred=bundle['classifier'].predict(bundle['vectorizer'].transform([text]))[0]
        probs=bundle['classifier'].predict_proba(bundle['vectorizer'].transform([text]))[0]
        result={'language':args.language,'prediction':str(pred),'scores':{str(k):float(v) for k,v in zip(bundle['classifier'].classes_,probs)},
                'model':'TF-IDF + Logistic Regression'}
    else:
        pred,probs,_,device=predict_transformer([text],batch_size=1)
        result={'language':args.language,'prediction':str(pred[0]),'scores':probs[0],
                'model':'cardiffnlp/twitter-xlm-roberta-base-sentiment','runtime_device':device}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

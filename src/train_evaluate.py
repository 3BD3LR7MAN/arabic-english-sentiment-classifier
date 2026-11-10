#!/usr/bin/env python3
"""Train a classical baseline and compare with pretrained multilingual XLM-R sentiment."""
from __future__ import annotations
import argparse, json, re, html, unicodedata
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

LABELS = ['negative', 'neutral', 'positive']
MODEL_ID = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'


def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'(?<!\w)@\w+', ' USER ', text)
    text = text.replace('\u0640', '')  # Arabic tatweel
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # Arabic diacritics
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = text.replace('ى', 'ي')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def metric_block(y_true, y_pred):
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(pr),
        'macro_recall': float(rc),
        'macro_f1': float(f1),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0)),
        'n': int(len(y_true)),
    }


def predict_transformer(texts, batch_size=12, max_length=160):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f'XLM-R inference device: {device}', flush=True)
    id_to_label = {int(k): str(v).lower() for k, v in config.id2label.items()}
    mapped = {}
    for idx, lab in id_to_label.items():
        if 'neg' in lab:
            mapped[idx] = 'negative'
        elif 'neu' in lab:
            mapped[idx] = 'neutral'
        elif 'pos' in lab:
            mapped[idx] = 'positive'
    if set(mapped.values()) != set(LABELS):
        raise ValueError(f'Unexpected labels from {MODEL_ID}: {id_to_label}')
    preds, all_probs = [], []
    for start in range(0, len(texts), batch_size):
        batch = [str(t)[:4000] for t in texts[start:start+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits.detach().cpu().numpy()
        probs = softmax(logits, axis=1)
        for row in probs:
            all_probs.append({mapped[i]: float(row[i]) for i in mapped})
            preds.append(mapped[int(row.argmax())])
        if start and start % (batch_size*25) == 0:
            print(f'XLM-R inference: {min(start+batch_size,len(texts))}/{len(texts)}', flush=True)
    del model, tokenizer
    return np.array(preds), all_probs, id_to_label, str(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/bilingual_sentiment_sample.csv')
    ap.add_argument('--output', default='.')
    ap.add_argument('--skip-transformer', action='store_true', help='Run only the TF-IDF baseline')
    ap.add_argument('--batch-size', type=int, default=12)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    reports = root / 'reports'; figures = reports / 'figures'; models = root / 'models'
    for p in (reports, figures, models): p.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    required = {'text','label','language','split'}
    if not required.issubset(df.columns):
        raise ValueError(f'Data must contain {sorted(required)}')
    df = df[df['language'].isin(['ar','en']) & df['label'].isin(LABELS)].copy()
    df['text'] = df['text'].astype(str).map(normalize_text)
    df = df[df['text'].str.len() >= 2]
    train = df[df['split']=='train'].copy()
    val = df[df['split']=='validation'].copy()
    test = df[df['split']=='test'].copy()
    for name, part in [('train',train),('validation',val),('test',test)]:
        if set(part['label'].unique()) != set(LABELS) or set(part['language'].unique()) != {'ar','en'}:
            raise ValueError(f'{name} must include both languages and all three labels.')
        counts = part.groupby(['language','label']).size()
        if len(counts) != 6 or counts.nunique() != 1:
            raise ValueError(f'{name} is not balanced across both languages and all labels: {counts.to_dict()}')

    # Language-aware word and character n-grams keep Arabic and English features without external tokenizers.
    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=(1,2),
                                 min_df=2, max_features=180_000, sublinear_tf=True, strip_accents=None)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=2,
                                 max_features=280_000, sublinear_tf=True)),
    ])
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    X_test = vectorizer.transform(test['text'])
    best = None; validation_grid=[]
    for c in [0.5, 1.0, 2.0]:
        clf = LogisticRegression(C=c, class_weight='balanced', max_iter=1200, solver='lbfgs', random_state=42)
        clf.fit(X_train, train['label'])
        pred = clf.predict(X_val)
        score = f1_score(val['label'], pred, labels=LABELS, average='macro', zero_division=0)
        validation_grid.append({'C':c,'validation_macro_f1':float(score)})
        print(f'LogisticRegression C={c}: validation macro-F1={score:.4f}',flush=True)
        if best is None or score > best[0]: best=(score,c)
    best_c=best[1]
    classical = LogisticRegression(C=best_c, class_weight='balanced', max_iter=1200, solver='lbfgs', random_state=42)
    classical.fit(X_train, train['label'])
    classical_val = classical.predict(X_val)
    classical_pred = classical.predict(X_test)
    joblib.dump({'vectorizer':vectorizer,'classifier':classical,'labels':LABELS,'normalizer':'normalize_text'}, models/'tfidf_logistic_regression.joblib')
    comparison=[]
    val_summary={'model':'TF-IDF + Logistic Regression','split':'validation',**metric_block(val['label'],classical_val)}
    test_summary={'model':'TF-IDF + Logistic Regression','split':'test',**metric_block(test['label'],classical_pred)}
    comparison.extend([val_summary,test_summary])
    preds_df = test[['text','label','language','source','domain']].copy()
    preds_df['tfidf_prediction']=classical_pred
    preds_df['tfidf_correct']=(preds_df['label']==preds_df['tfidf_prediction'])

    transformer_details=None
    if not args.skip_transformer:
        combined=pd.concat([val,test],ignore_index=True)
        transformer_pred, transformer_probs, transformer_map, transformer_device = predict_transformer(combined['text'].tolist(), batch_size=args.batch_size)
        val_pred=transformer_pred[:len(val)]; test_pred=transformer_pred[len(val):]
        comparison.extend([
            {'model':'CardiffNLP XLM-R pretrained (no fine-tuning on this sample)','split':'validation',**metric_block(val['label'],val_pred)},
            {'model':'CardiffNLP XLM-R pretrained (no fine-tuning on this sample)','split':'test',**metric_block(test['label'],test_pred)},
        ])
        preds_df['xlmr_prediction']=test_pred
        preds_df['xlmr_correct']=(preds_df['label']==preds_df['xlmr_prediction'])
        preds_df['xlmr_confidence']=[max(p.values()) for p in transformer_probs[len(val):]]
        transformer_details={'model_id':MODEL_ID,'id2label':transformer_map,'evaluated_without_fine_tuning':True,
                             'max_length':160,'batch_size':args.batch_size,'runtime_device':transformer_device}
    preds_df.to_csv(reports/'test_predictions.csv',index=False)
    pd.DataFrame(comparison).to_csv(reports/'model_comparison.csv',index=False)
    pd.DataFrame(validation_grid).to_csv(reports/'baseline_validation_grid.csv',index=False)

    # Per-language metrics, keeping the same held-out examples for both methods.
    by_lang=[]
    for lang in ['ar','en']:
        mask=test['language'].to_numpy()==lang
        for name,pred in [('TF-IDF + Logistic Regression',classical_pred)]:
            by_lang.append({'model':name,'language':lang,**metric_block(test.loc[mask,'label'],pred[mask])})
        if not args.skip_transformer:
            by_lang.append({'model':'CardiffNLP XLM-R pretrained','language':lang,**metric_block(test.loc[mask,'label'],test_pred[mask])})
    pd.DataFrame(by_lang).to_csv(reports/'per_language_metrics.csv',index=False)

    test_rows=[row for row in comparison if row['split']=='test']
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    labels=['TF-IDF + Logistic\nRegression' if 'TF-IDF' in row['model'] else 'Pretrained XLM-R' for row in test_rows]
    bars=ax.bar(labels,[row['macro_f1'] for row in test_rows],color=['#238be6','#23a685'][:len(test_rows)])
    ax.set_ylim(0,1); ax.set_ylabel('Macro-F1'); ax.set_title('Held-out test comparison (same 900 rows)')
    ax.bar_label(bars,fmt='%.3f',padding=3); ax.grid(axis='y',alpha=.2); fig.tight_layout()
    fig.savefig(figures/'macro_f1_model_comparison.png',dpi=170); plt.close(fig)
    if len(by_lang)>2:
        lang_df=pd.DataFrame(by_lang)
        fig,ax=plt.subplots(figsize=(7.4,4.8)); x=np.arange(2); width=.34
        for j,name in enumerate(['TF-IDF + Logistic Regression','CardiffNLP XLM-R pretrained']):
            vals=[float(lang_df[(lang_df['language']==lang)&(lang_df['model']==name)]['macro_f1'].iloc[0]) for lang in ['ar','en']]
            bars=ax.bar(x+(j-.5)*width,vals,width,label='TF-IDF + Logistic Regression' if j==0 else 'Pretrained XLM-R',color=['#238be6','#23a685'][j])
            ax.bar_label(bars,fmt='%.3f',padding=2,fontsize=8)
        ax.set_xticks(x,['Arabic','English']); ax.set_ylim(0,1); ax.set_ylabel('Macro-F1'); ax.set_title('Held-out test macro-F1 by language')
        ax.legend(frameon=False); ax.grid(axis='y',alpha=.2); fig.tight_layout()
        fig.savefig(figures/'macro_f1_by_language.png',dpi=170); plt.close(fig)

    # Classification reports and confusion matrices.
    models_to_plot=[('TF-IDF + Logistic Regression',classical_pred)]
    if not args.skip_transformer: models_to_plot.append(('XLM-R pretrained',test_pred))
    report_text=[]
    for name,pred in models_to_plot:
        report_text.append(f'## {name}\n\n'+classification_report(test['label'],pred,labels=LABELS,zero_division=0,digits=4))
        cm=confusion_matrix(test['label'],pred,labels=LABELS)
        fig,ax=plt.subplots(figsize=(6.6,5.5))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=LABELS,yticklabels=LABELS,ax=ax)
        ax.set(title=f'{name} — held-out bilingual test',xlabel='Predicted label',ylabel='True label')
        fig.tight_layout(); fig.savefig(figures/(('confusion_tfidf.png' if 'TF-IDF' in name else 'confusion_xlmr.png')),dpi=160); plt.close(fig)
    (reports/'classification_reports.txt').write_text('\n\n'.join(report_text),encoding='utf-8')
    # Error analysis examples: incorrect rows only, limited for privacy/readability.
    err=preds_df[preds_df['label']!=preds_df['tfidf_prediction']].copy()
    if 'xlmr_prediction' in err:
        err['xlmr_correct']=(err['label']==err['xlmr_prediction'])
    err.head(50).to_csv(reports/'error_examples.csv',index=False)

    counts={f'{split}|{lang}|{label}':int(count) for (split,lang,label),count in df.groupby(['split','language','label']).size().items()}
    summary={'dataset':'clapAI/MultiLingualSentiment balanced Arabic-English subset',
             'labels':LABELS,'train_rows':int(len(train)),'validation_rows':int(len(val)),'test_rows':int(len(test)),
             'counts':counts,
             'selected_logistic_regression_C':best_c,
             'baseline_validation_grid':validation_grid,
             'comparison':comparison,'per_language_metrics':by_lang,
             'transformer':transformer_details,
             'limitations':['The dataset aggregates many sources and domains; all 17 test domains and all 27 test sources also occur in train, so this test does not measure unseen-domain/source generalization.',
                            'The pretrained model may have seen overlapping public material during pretraining/fine-tuning; no claim of contamination-free zero-shot evaluation is made.',
                            'This is a small balanced educational sample, not a production customer-feedback benchmark.']}
    (reports/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# Arabic / English Sentiment Classifier — Evaluation Report','',
            '## Dataset and protocol','',
            f"This experiment uses a balanced sample of {len(df):,} Arabic and English records from `clapAI/MultiLingualSentiment` (original labels: negative, neutral, positive). Sample sizes: train {len(train):,}, validation {len(val):,}, test {len(test):,}. Data are grouped by language and label for the sample; source/domain columns are retained. In this sample all 17 test domains and all 27 test sources also occur in train, so the split does not measure unseen-source or unseen-domain generalization. The dataset is not a domain-controlled Arabic-vs-English benchmark.",'',
            'The classical model regularization strength is selected on validation macro-F1. The XLM-R option is an existing multilingual Twitter sentiment checkpoint applied without fine-tuning to this sample. Both are evaluated on the same held-out test rows, but the tuning budgets are not identical. Results are descriptive; source overlap with upstream/pretraining data cannot be ruled out.','',
            '## Results','']
    for row in comparison:
        report.append(f"- {row['model']} — {row['split']}: accuracy {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}, weighted-F1 {row['weighted_f1']:.3f} (n={row['n']}).")
    report+=['','## Per-language test results','']
    for row in by_lang:
        report.append(f"- {row['model']} — {row['language']}: accuracy {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}, macro-recall {row['macro_recall']:.3f} (n={row['n']}).")
    neutral_note=''
    if not args.skip_transformer:
        tf_neutral=classification_report(test['label'],classical_pred,labels=LABELS,output_dict=True,zero_division=0)['neutral']['recall']
        xl_neutral=classification_report(test['label'],test_pred,labels=LABELS,output_dict=True,zero_division=0)['neutral']['recall']
        neutral_note=f' Neutral-class recall was {tf_neutral:.3f} for TF-IDF and {xl_neutral:.3f} for XLM-R.'
    report+=['','## Error analysis and limitations','',
             f'`error_examples.csv` contains up to 50 baseline mistakes with the source label and model predictions.{neutral_note} Inspect Arabic dialect, sarcasm, mixed-language text, negation, and domain vocabulary before using the model. The pretrained model is not fine-tuned on this sample. The balanced subset changes the class prevalence and must not be used to infer production prevalence or calibrated probabilities. Do not make consequential customer decisions solely from sentiment labels.','',
             '## References','',
             '[1]: https://huggingface.co/datasets/clapAI/MultiLingualSentiment "MultilingualSentiment dataset card"',
             '[2]: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment "Cardiff NLP XLM-R Twitter sentiment model card"',
             '[3]: https://arxiv.org/abs/2104.12250 "XLM-T multilingual Twitter language model paper"']
    (reports/'Evaluation_Report_EN.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'metrics':comparison,'per_language':by_lang,'output':str(root)},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()

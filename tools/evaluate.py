"""Evaluation tables for the article: NLG metrics overall, per class and per phototype,
plus the automatic correctness of the diagnosis / phototype stated in the report.
Several runs (seeds) are aggregated as mean +/- standard deviation.

Usage
-----
    python tools/evaluate.py \
        --ann data/brown_skin/annotation_lesion_split.json \
        --runs results/bs_vitl_rm_mcln_s0/test results/bs_vitl_rm_mcln_s1/test results/bs_vitl_rm_mcln_s2/test \
        --out results/tables/bs_vitl_rm_mcln

Each run directory must contain predictions.csv (id, gt, pred) written by main_test.py.
`--ann` supplies `label` and `phototype` per id (written by tools/make_splits.py).

Correctness metrics (no expert needed)
  * dx_acc:  the diagnosis mentioned in the generated report equals the true diagnosis.
  * dx_f1:   macro-F1 of that diagnosis over the 7 classes.
  * pt_acc:  the phototype phrase in the generated report equals the one in the reference
             (only when the references mention a phototype, e.g. Brown Skin).
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.metrics import compute_scores  # noqa: E402
from modules.tokenizers import Tokenizer  # noqa: E402


def clean(text):
    """Same normalisation R2Gen applies to training text (lower case, punctuation removed)."""
    return Tokenizer.clean_report_mimic_cxr(None, str(text))

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
SYNONYMS = {
    'akiec': ['akiec', 'actinic keratos', 'intraepithelial carcinoma', "bowen"],
    'bcc': ['bcc', 'basal cell carcinoma'],
    'bkl': ['bkl', 'benign keratosis', 'seborrheic keratosis', 'keratosis-like'],
    'df': ['df', 'dermatofibroma'],
    'mel': ['mel', 'melanoma'],
    'nv': ['nv', 'nevus', 'nevi', 'naevus'],
    'vasc': ['vasc', 'vascular lesion', 'hemangioma', 'angioma'],
}
# Phrases that state a skin phototype / tone. Edit to match the wording of your reports.
PHOTOTYPE_PATTERNS = [
    r'(sin indicios claros de piel morena|piel morena)',  # Spanish reports (morena / no morena)
    r'(?:phototype|fitzpatrick(?: type)?|skin type|type)\s*(vi|iv|v|iii|ii|i|[1-6])\b',
    r'\b(very light|light|medium|intermediate|tan|brown|dark|very dark|black)\s+skin(?:\s+tone)?',
]
NLG = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L']


def find_class(text):
    t = ' ' + str(text).lower() + ' '
    hits = []
    for c, words in SYNONYMS.items():
        for w in words:
            # abbreviations must be whole words ('mel' must not match 'melanocytic');
            # longer phrases may be prefixes ('actinic keratos' -> keratosis/keratoses)
            tail = '(?![a-z])' if (len(w) <= 5 and ' ' not in w) else ''
            for m in re.finditer(r'(?<![a-z]){}{}'.format(re.escape(w), tail), t):
                hits.append((m.start(), c))
    return min(hits)[1] if hits else None


def find_phototype(text):
    t = str(text).lower()
    for p in PHOTOTYPE_PATTERNS:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def nlg(df):
    if len(df) == 0:
        return {k: np.nan for k in NLG}
    gts = {i: [g] for i, g in enumerate(df['gt'])}
    res = {i: [r if str(r).strip() else '.' for r in [p]] for i, p in enumerate(df['pred'])}
    return compute_scores(gts, res)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    pairs = [(t, p if isinstance(p, str) else 'none') for t, p in zip(y_true, y_pred) if isinstance(t, str) and t]
    if not pairs:
        return np.nan  # no diagnostic labels (e.g. IU X-Ray)
    y_true, y_pred = zip(*pairs)
    labels = sorted(set(y_true))
    return f1_score(list(y_true), list(y_pred), labels=labels, average='macro', zero_division=0)


def evaluate_run(pred_csv, meta):
    df = pd.read_csv(pred_csv, dtype={'id': str}).fillna('')
    df = df.merge(meta, on='id', how='left')
    # Reference = the full annotated report (not the model's decoded, truncated, <unk>-ed copy),
    # so model outputs and baselines are scored against exactly the same text.
    df['gt'] = [clean(r) if isinstance(r, str) and r else clean(g) for r, g in zip(df['report'], df['gt'])]
    df['pred'] = df['pred'].map(clean)
    has_labels = meta['label'].notna().any()  # False for IU X-Ray: no diagnosis to check
    if has_labels:
        df['label'] = df['label'].fillna(df['gt'].map(find_class))
    df['dx_pred'] = df['pred'].map(find_class)
    df['dx_ok'] = np.where(df['label'].notna(), df['dx_pred'] == df['label'], np.nan)
    df['pt_gt'] = df['gt'].map(find_phototype)
    df['pt_pred'] = df['pred'].map(find_phototype)
    has_pt = df['pt_gt'].notna()
    df['pt_ok'] = np.where(has_pt, df['pt_pred'] == df['pt_gt'], np.nan)

    def block(sub):
        row = nlg(sub)
        row['n'] = len(sub)
        row['dx_acc'] = sub['dx_ok'].astype(float).mean() if sub['dx_ok'].notna().any() else np.nan
        row['pt_acc'] = sub['pt_ok'].astype(float).mean() if sub['pt_ok'].notna().any() else np.nan
        return row

    rows = []
    overall = block(df)
    overall['dx_f1'] = macro_f1(list(df['label']), list(df['dx_pred']))
    rows.append(dict(group='all', value='all', **overall))
    for c, sub in df.groupby('label'):
        rows.append(dict(group='class', value=c, **block(sub)))
    if 'phototype' in df and df['phototype'].notna().any():
        for p, sub in df.groupby('phototype'):
            rows.append(dict(group='phototype', value=p, **block(sub)))
    if 'tone' in df and df['tone'].notna().any():
        for t, sub in df.groupby('tone'):
            rows.append(dict(group='tone', value=t, **block(sub)))
    return pd.DataFrame(rows), df


def summarise(tables):
    cat = pd.concat([t.assign(run=i) for i, t in enumerate(tables)])
    metrics = [c for c in cat.columns if c not in ('group', 'value', 'run', 'n')]
    g = cat.groupby(['group', 'value'], sort=False)
    mean, std, n = g[metrics].mean(), g[metrics].std(ddof=1), g['n'].first()
    out = mean.copy()
    for m in metrics:
        out[m] = [('{:.4f}'.format(a) if len(tables) == 1 or np.isnan(b) else '{:.4f} ± {:.4f}'.format(a, b))
                  if not np.isnan(a) else '' for a, b in zip(mean[m], std[m])]
    out.insert(0, 'n', n)
    return out.reset_index(), mean.reset_index(), std.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='annotation JSON with label/phototype per id')
    ap.add_argument('--runs', nargs='+', required=True, help='directories (or files) with predictions.csv')
    ap.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()

    ann = json.load(open(args.ann, encoding='utf-8'))
    items = [e for v in ann.values() for e in v] if isinstance(ann, dict) else ann
    meta = pd.DataFrame([{'id': str(e['id']), 'label': e.get('label'), 'phototype': e.get('phototype'),
                          'tone': e.get('tone'), 'pair': e.get('pair'), 'report': e.get('report')}
                         for e in items]).drop_duplicates('id')

    tables = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    for i, r in enumerate(args.runs):
        path = os.path.join(r, 'predictions.csv') if os.path.isdir(r) else r
        table, detail = evaluate_run(path, meta)
        tables.append(table)
        detail.to_csv('{}_run{}_detail.csv'.format(args.out, i), index=False)
        print('run {}: {} examples, BLEU_4={:.4f}, dx_acc={:.4f}'.format(
            i, len(detail), table.iloc[0]['BLEU_4'], table.iloc[0]['dx_acc']))

    pretty, mean, std = summarise(tables)
    pretty.to_csv(args.out + '_summary.csv', index=False)
    mean.to_csv(args.out + '_mean.csv', index=False)
    std.to_csv(args.out + '_std.csv', index=False)
    with open(args.out + '_summary.md', 'w', encoding='utf-8') as f:
        f.write('Runs: {}\n\n'.format(len(tables)))
        f.write(pretty.to_markdown(index=False) if hasattr(pretty, 'to_markdown') else pretty.to_string(index=False))
        f.write('\n')
    print(pretty.to_string(index=False))


if __name__ == '__main__':
    main()

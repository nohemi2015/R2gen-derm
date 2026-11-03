"""Same lesion on light vs dark skin: paired comparison for the Brown Skin experiment.

    python tools/paired_tone.py --details results/tables/ham_vitl_rm_mcln_pairs_run0_detail.csv [more runs]

Input: the *_detail.csv files written by tools/evaluate.py on bs_paired_test.json
(they contain id, pair, tone, dx_ok, gt, pred). For every lesion with a light original and a dark
version, compares diagnosis correctness (exact McNemar test) and sentence-level BLEU-4.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycocoevalcap.bleu.bleu_scorer import BleuScorer  # noqa: E402


def sent_bleu4(gt, pred):
    s = BleuScorer(n=4)
    s += (pred if pred.strip() else '.', [gt])
    _, scores = s.compute_score(option='closest', verbose=0)
    return float(scores[3][0])


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyse(path):
    d = pd.read_csv(path, dtype={'id': str})
    d['bleu4'] = [sent_bleu4(g, p) for g, p in zip(d['gt'], d['pred'])]
    light = d[d['tone'] == 'light'].drop_duplicates('pair').set_index('pair')
    dark = d[d['tone'] == 'dark']
    rows = []
    for _, r in dark.iterrows():
        if r['pair'] in light.index:
            l = light.loc[r['pair']]
            rows.append(dict(pair=r['pair'], label=r['label'], light_ok=bool(l['dx_ok']), dark_ok=bool(r['dx_ok']),
                             light_bleu4=l['bleu4'], dark_bleu4=r['bleu4']))
    p = pd.DataFrame(rows)
    b = int((p.light_ok & ~p.dark_ok).sum())   # right on light, wrong on dark
    c = int((~p.light_ok & p.dark_ok).sum())   # wrong on light, right on dark
    diff = p.dark_bleu4 - p.light_bleu4
    return dict(pairs=len(p), dx_acc_light=p.light_ok.mean(), dx_acc_dark=p.dark_ok.mean(),
                gap_dx_acc=p.dark_ok.mean() - p.light_ok.mean(), right_light_wrong_dark=b,
                wrong_light_right_dark=c, mcnemar_p=mcnemar_exact(b, c),
                bleu4_light=p.light_bleu4.mean(), bleu4_dark=p.dark_bleu4.mean(), gap_bleu4=diff.mean()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--details', nargs='+', required=True)
    ap.add_argument('--out', default=None, help='CSV with one row per run (+ mean/std)')
    args = ap.parse_args()
    res = []
    for path in args.details:
        r, _ = analyse(path)
        r['run'] = os.path.basename(path)
        res.append(r)
    t = pd.DataFrame(res).set_index('run')
    if len(t) > 1:
        t.loc['mean'] = t.mean(numeric_only=True)
        t.loc['std'] = t.iloc[:-1].std(numeric_only=True, ddof=1)
    print(t.round(4).to_string())
    if args.out:
        t.to_csv(args.out)


if __name__ == '__main__':
    main()

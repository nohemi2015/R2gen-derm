"""Statistics added in the revision: lesion-clustered bootstrap CIs of paired differences between
models, corpus BLEU-4 recomputed inside every resample (same definition as pycocoevalcap: whitespace
tokens, clipped n-gram precision, closest-reference brevity penalty), diagnosis accuracy and
malignancy sensitivity/specificity, and exact McNemar contingency tables.

    python tools/revision_stats.py --details A_run{0,1,2}_detail.csv --details B_run{0,1,2}_detail.csv \
        --names proposed clf --groups test_groups.csv --out out.csv
"""
import argparse
import collections
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MAL = {'mel', 'bcc', 'akiec'}


def sent_stats(hyp, ref):
    h, r = str(hyp).split(), str(ref).split()
    out = [len(h), len(r)]
    for n in range(1, 5):
        hc = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        out += [sum(min(c, rc[g]) for g, c in hc.items()), max(len(h) - n + 1, 0)]
    return out


def corpus_bleu4(S):
    """S: array (n, 10) of sentence stats -> corpus BLEU-4."""
    t = S.sum(0)
    hl, rl = t[0], t[1]
    logp = sum(math.log((t[2 + 2 * k] + 1e-9) / (t[3 + 2 * k] + 1e-15)) for k in range(4)) / 4
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1e-15))
    return bp * math.exp(logp)


def load(path):
    d = pd.read_csv(path, dtype={'id': str}).fillna('')
    d['ok'] = (d['dx_pred'] == d['label']).astype(float)
    d['ymal'] = d['label'].isin(MAL)
    d['pmal'] = d['dx_pred'].isin(MAL)
    S = np.array([sent_stats(p, g) for p, g in zip(d['pred'], d['gt'])], dtype=float)
    return d.set_index('id'), S


def metrics(d, S, idx):
    dd, SS = d.iloc[idx], S[idx]
    y, p = dd['ymal'].values, dd['pmal'].values
    return dict(bleu4=corpus_bleu4(SS), dx_acc=dd['ok'].mean(),
                sens=(p & y).sum() / max(y.sum(), 1), spec=(~p & ~y).sum() / max((~y).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--details', nargs='+', action='append', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--groups', default=None, help='csv with id,group (lesion) for clustered resampling')
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = {n: [load(p) for p in paths] for n, paths in zip(args.names, args.details)}
    ids = runs[args.names[0]][0][0].index
    for n in args.names:
        for k, (d, S) in enumerate(runs[n]):
            order = d.index.get_indexer(ids)
            runs[n][k] = (d.iloc[order], S[order])
    if args.groups:
        g = pd.read_csv(args.groups, dtype=str).set_index('id')['group'].reindex(ids).fillna(pd.Series(ids, index=ids)).values
    else:
        g = np.asarray(ids)
    clusters = pd.Series(np.arange(len(ids))).groupby(g).apply(list).tolist()
    rng = np.random.default_rng(args.seed)

    def seed_avg(n, idx):
        ms = [metrics(d, S, idx) for d, S in runs[n]]
        return {k: np.mean([m[k] for m in ms]) for k in ms[0]}

    full = np.arange(len(ids))
    point = {n: seed_avg(n, full) for n in args.names}
    boots = {n: [] for n in args.names}
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        for n in args.names:
            boots[n].append(seed_avg(n, idx))
    rows = []
    for n in args.names:
        for k, v in point[n].items():
            b = np.array([x[k] for x in boots[n]])
            rows.append(dict(model=n, metric=k, value=v, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
    ref = args.names[0]
    for n in args.names[1:]:
        for k in point[ref]:
            diff = np.array([a[k] - b[k] for a, b in zip(boots[ref], boots[n])])
            rows.append(dict(model='{} - {}'.format(ref, n), metric=k, value=point[ref][k] - point[n][k],
                             lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
        for s in range(min(len(runs[ref]), len(runs[n]))):
            a, b = runs[ref][s][0]['ok'].values.astype(bool), runs[n][s][0]['ok'].values.astype(bool)
            n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
            rows.append(dict(model='{} vs {} seed {}'.format(ref, n, s), metric='mcnemar',
                             value=binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0,
                             lo=n10, hi=n01, both_right=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == '__main__':
    main()

"""Lesion-level (group-disjoint) splits for HAM10000 and Brown Skin, plus a leakage audit.

Every example of the same lesion goes to the same split:
  * HAM10000: several images share a `lesion_id` (HAM10000_metadata.csv).
  * Brown Skin: the 6 phototype versions of one source image share that image's group
    (and, when available, the HAM10000 lesion_id of the source image).

Splits are 70/10/20 (train/val/test), as in R2Gen for IU X-Ray, stratified by diagnosis.

Usage
-----
Audit an existing annotation file (reports groups that appear in more than one split):
    python tools/make_splits.py audit --ann data/ham10000/annotation.json \
        --ham_metadata data/HAM10000_metadata.csv

Create new splits:
    python tools/make_splits.py split --ann data/ham10000/annotation.json \
        --ham_metadata data/HAM10000_metadata.csv --out data/ham10000/annotation_lesion_split.json --seed 0

    python tools/make_splits.py split --ann data/brown_skin/annotation.json \
        --group_regex "(ISIC_\\d{7})" --ham_metadata data/HAM10000_metadata.csv \
        --phototype_regex "(?:PT|tipo|type)[ _-]?(VI|IV|V|III|II|I|[1-6])" \
        --out data/brown_skin/annotation_lesion_split.json --seed 0

The output keeps every original field (id, image_path, report) and adds
`label`, `group` and, if found, `phototype`, which tools/evaluate.py uses for the
per-class and per-phototype tables.
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
ROMAN = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V', '6': 'VI'}


def load_examples(path):
    ann = json.load(open(path, encoding='utf-8'))
    if isinstance(ann, list):
        return [dict(e, _old_split=e.get('split', 'none')) for e in ann]
    out = []
    for split, items in ann.items():
        for e in items:
            out.append(dict(e, _old_split=split))
    return out


def image_key(example, group_regex=None):
    """ISIC id of the (source) image, taken from the id or the image path."""
    text = ' '.join([str(example.get('id', ''))] + [str(p) for p in example.get('image_path', [])])
    m = re.search(group_regex or r'(ISIC_\d{7})', text)
    return m.group(1) if m else str(example.get('id'))


def label_from_text(text):
    t = text.lower()
    found = [(m.start(), c) for c in CLASSES for m in re.finditer(r'\b{}\b'.format(c), t)]
    return min(found)[1] if found else None


def annotate(examples, ham_metadata=None, group_regex=None, phototype_regex=None):
    meta = {}
    if ham_metadata:
        import pandas as pd
        df = pd.read_csv(ham_metadata)
        meta = {r.image_id: (r.lesion_id, r.dx) for r in df.itertuples()}
    missing_label = 0
    for e in examples:
        key = image_key(e, group_regex)
        lesion, dx = meta.get(key, (None, None))
        e['group'] = lesion or key
        e['label'] = e.get('label') or dx or label_from_text(str(e.get('report', '')))
        if e['label'] is None:
            missing_label += 1
        if phototype_regex and not e.get('phototype'):
            text = ' '.join([str(e.get('id', ''))] + [str(p) for p in e.get('image_path', [])])
            m = re.search(phototype_regex, text, flags=re.IGNORECASE)
            if m:
                e['phototype'] = ROMAN.get(m.group(1), m.group(1).upper())
    if missing_label:
        print('WARNING: {} examples without a diagnosis label'.format(missing_label), file=sys.stderr)
    if ham_metadata:
        hits = sum(1 for e in examples if image_key(e, group_regex) in meta)
        print('Matched {}/{} examples to HAM10000_metadata.csv'.format(hits, len(examples)))
    return examples


def audit(examples):
    by_group = collections.defaultdict(set)
    for e in examples:
        by_group[e['group']].add(e['_old_split'])
    leaking = {g: s for g, s in by_group.items() if len(s) > 1}
    counts = collections.Counter(e['_old_split'] for e in examples)
    print('Examples per split: {}'.format(dict(counts)))
    print('Groups (lesions): {}'.format(len(by_group)))
    print('Groups present in more than one split: {} ({:.1f}%)'.format(
        len(leaking), 100.0 * len(leaking) / max(1, len(by_group))))
    test_leak = [e for e in examples if e['_old_split'] == 'test' and len(by_group[e['group']]) > 1]
    print('Test examples whose lesion also appears in train/val: {} of {}'.format(
        len(test_leak), counts.get('test', 0)))
    for g, s in list(leaking.items())[:10]:
        print('  e.g. {} -> {}'.format(g, sorted(s)))
    return leaking


def split(examples, seed, n_folds=10, test_folds=2, val_folds=1):
    from sklearn.model_selection import StratifiedGroupKFold
    labels = [e['label'] or 'unknown' for e in examples]
    groups = [e['group'] for e in examples]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.zeros(len(examples), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(examples)), labels, groups)):
        fold_of[idx] = k
    out = {'train': [], 'val': [], 'test': []}
    for e, k in zip(examples, fold_of):
        name = 'test' if k < test_folds else ('val' if k < test_folds + val_folds else 'train')
        clean = {kk: vv for kk, vv in e.items() if not kk.startswith('_')}
        clean['split'] = name
        out[name].append(clean)
    # sanity: no group in two splits
    seen = {}
    for name, items in out.items():
        for e in items:
            assert seen.setdefault(e['group'], name) == name, 'group leak: {}'.format(e['group'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['audit', 'split'])
    ap.add_argument('--ann', required=True, help='existing R2Gen-style annotation JSON')
    ap.add_argument('--ham_metadata', default=None, help='HAM10000_metadata.csv (image_id, lesion_id, dx)')
    ap.add_argument('--group_regex', default=None, help='regex whose group(1) is the source image id')
    ap.add_argument('--phototype_regex', default=None, help='regex whose group(1) is the Fitzpatrick type')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    examples = annotate(load_examples(args.ann), args.ham_metadata, args.group_regex, args.phototype_regex)
    if args.mode == 'audit':
        audit(examples)
        return
    out = split(examples, args.seed)
    for name in ('train', 'val', 'test'):
        c = collections.Counter(e['label'] for e in out[name])
        p = collections.Counter(e.get('phototype') for e in out[name])
        print('{:5s}: {:5d} examples, {:5d} lesions | classes {} | phototypes {}'.format(
            name, len(out[name]), len({e['group'] for e in out[name]}), dict(sorted(c.items(), key=str)),
            dict(sorted(p.items(), key=str))))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('Saved', args.out)


if __name__ == '__main__':
    main()

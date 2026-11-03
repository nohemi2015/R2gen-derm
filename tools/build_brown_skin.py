"""Build the HAM10000 / Brown Skin annotation files for the article.

Brown Skin (BS) = dermoscopic images of HAM10000 and of the ISIC 2018 Task 3 test set re-rendered
on dark skin (folders "100 malignas" and "100 benignas", files ISIC_XXXXXXX_<dx>.jpg). Every BS image
therefore has a light-skin original, which gives a paired design: same lesion, two skin tones.

Outputs (in --out_dir):
  ham10000_lesion.json   HAM10000 train/val re-split by lesion_id (no lesion in both);
                         test = ISIC 2018 Task 3 test set (1,511 images), unchanged.
  ham10000_plus_bs.json  Same, plus the dark versions of BS lesions whose original is a HAM10000
                         training/validation image (added to that lesion's split). Test unchanged.
  bs_paired_test.json    test = every BS lesion whose original is in the ISIC 2018 test set, twice:
                         dark version and light original, with the same reference report.
                         These lesions are never seen in training by either model -> clean
                         comparison of the same lesions on light vs dark skin.
  brown_skin_summary.csv One row per BS image: source, diagnosis, lesion, where it was used.

Each entry carries label, group (lesion), tone ('light'/'dark') and pair (source ISIC id),
which tools/evaluate.py and tools/paired_tone.py use.

Usage (Windows example, run inside r2gen-derm):
    python tools/build_brown_skin.py ^
        --ham_ann "C:/1. HAM10000/data/ham10000/annotation_ham10000_full.json" ^
        --ham_metadata "C:/1. HAM10000/HAM10000_metadata.csv" ^
        --isic_gt "C:/1. HAM10000/data/ham10000/ISIC2018_Task3_Test_GroundTruth.csv" ^
        --image_root "C:/1. HAM10000/data/ham10000/images" ^
        --bs_dirs "100 malignas" "100 benignas" ^
        --out_dir "C:/1. HAM10000/data/brown_skin_v2"

--bs_dirs are relative to --image_root: copy the two BS folders into the HAM10000 images folder
(next to HAM10000_images_all and ISIC2018_Task3_Test_Images).
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import pandas as pd

DX = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ham_ann', required=True)
    ap.add_argument('--ham_metadata', required=True)
    ap.add_argument('--isic_gt', required=True, help='ISIC2018_Task3_Test_GroundTruth.csv (lesion_id, image_id, dx, ...)')
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--bs_dirs', nargs='+', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--val_fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ann = json.load(open(args.ham_ann, encoding='utf-8'))
    meta = pd.concat([pd.read_csv(args.ham_metadata).assign(source='HAM10000'),
                      pd.read_csv(args.isic_gt).assign(source='ISIC2018_test')], ignore_index=True)
    meta = meta.drop_duplicates('image_id').set_index('image_id')

    # ---- HAM10000 entries with label / group / tone --------------------------------------------
    entries = {}
    for split, items in ann.items():
        for e in items:
            iid = e['id']
            row = meta.loc[iid] if iid in meta.index else None
            entries[iid] = dict(id=iid, image_path=e['image_path'], report=e['report'],
                                label=(row['dx'] if row is not None else None),
                                group=(row['lesion_id'] if row is not None and isinstance(row['lesion_id'], str) else iid),
                                source=(row['source'] if row is not None else 'unknown'),
                                tone='light', pair=iid, old_split=split)

    # ---- re-split HAM10000 train/val by lesion; keep the ISIC 2018 test set -----------------------
    rng = np.random.RandomState(args.seed)
    dev = [e for e in entries.values() if e['old_split'] in ('train', 'val')]
    test = [e for e in entries.values() if e['old_split'] == 'test']
    by_dx = collections.defaultdict(set)
    for e in dev:
        by_dx[e['label']].add(e['group'])
    val_groups = set()
    for dx, groups in sorted(by_dx.items(), key=lambda x: str(x[0])):
        groups = sorted(groups)
        rng.shuffle(groups)
        val_groups.update(groups[:max(1, int(round(args.val_fraction * len(groups))))])
    for e in dev:
        e['split'] = 'val' if e['group'] in val_groups else 'train'
    for e in test:
        e['split'] = 'test'

    # ---- Brown Skin images --------------------------------------------------------------------
    bs_rows = []
    for d in args.bs_dirs:
        full = os.path.join(args.image_root, d)
        for f in sorted(os.listdir(full)):
            m = re.match(r'(ISIC_\d+)(_49)?(?:_([a-z]+))?\.(jpg|jpeg|png)$', f, flags=re.IGNORECASE)
            if not m:
                continue
            iid, v49, dx_name = m.group(1), m.group(2), (m.group(3) or '').lower() or None
            orig = entries.get(iid)
            row = dict(file=os.path.join(d, f).replace('\\', '/'), source_id=iid, version='49' if v49 else 'base',
                       dx_filename=dx_name, dx_true=orig['label'] if orig else None,
                       origin=orig['source'] if orig else 'not in HAM10000/ISIC2018', used_in='excluded')
            if orig is None:
                row['reason'] = 'no original image / no diagnosis'
            elif dx_name and dx_name != orig['label']:
                row['reason'] = 'diagnosis in file name ({}) differs from ground truth ({})'.format(dx_name, orig['label'])
            else:
                row['reason'] = ''
            bs_rows.append(row)

    dark_train, paired = [], []
    for r in bs_rows:
        if r['origin'] == 'not in HAM10000/ISIC2018' or r['reason'].startswith('diagnosis in file'):
            continue
        o = entries[r['source_id']]
        dark = dict(id='{}_dark{}'.format(r['source_id'], '_49' if r['version'] == '49' else ''),
                    image_path=[r['file']], report=o['report'], label=o['label'], group=o['group'],
                    source=o['source'], tone='dark', pair=r['source_id'], split=o['split'])
        if o['split'] == 'test':
            paired.append(dark)
            r['used_in'] = 'paired test (dark) + original as light'
        else:
            dark_train.append(dark)
            r['used_in'] = 'training ({})'.format(o['split'])

    def clean(e):
        return {k: v for k, v in e.items() if k != 'old_split'}

    def pack(items):
        out = {'train': [], 'val': [], 'test': []}
        for e in items:
            out[e['split']].append(clean(e))
        return out

    os.makedirs(args.out_dir, exist_ok=True)
    base = pack(list(entries.values()))
    plus = pack(list(entries.values()) + dark_train)
    light_of_pairs = [dict(entries[p['pair']]) for p in paired]
    seen = set()
    light_unique = [e for e in light_of_pairs if not (e['id'] in seen or seen.add(e['id']))]
    paired_ann = {'train': base['train'], 'val': base['val'],
                  'test': [clean(e) for e in light_unique] + [clean(e) for e in paired]}

    for name, obj in (('ham10000_lesion.json', base), ('ham10000_plus_bs.json', plus),
                      ('bs_paired_test.json', paired_ann)):
        with open(os.path.join(args.out_dir, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    pd.DataFrame(bs_rows).to_csv(os.path.join(args.out_dir, 'brown_skin_summary.csv'), index=False)

    # ---- report ----------------------------------------------------------------------------------
    def leak(obj):
        where = collections.defaultdict(set)
        for s, items in obj.items():
            for e in items:
                where[e['group']].add(s)
        return sum(1 for v in where.values() if len(v) > 1)

    s = pd.DataFrame(bs_rows)
    print('Brown Skin images found: {}'.format(len(s)))
    print(s['used_in'].value_counts().to_string())
    if (s['reason'] != '').any():
        print('\nExcluded / flagged:')
        print(s[s['reason'] != ''][['file', 'reason']].to_string(index=False))
    for name, obj in (('ham10000_lesion', base), ('ham10000_plus_bs', plus), ('bs_paired_test', paired_ann)):
        print('\n{}: {} | lesions in more than one split: {}'.format(
            name, {k: len(v) for k, v in obj.items()}, leak(obj)))
    print('\nPaired test: {} dark images of {} lesions, each with its light original.'.format(
        len(paired), len({p['pair'] for p in paired})))
    print('Paired lesions by diagnosis:', dict(collections.Counter(p['label'] for p in paired)))
    print('Saved in', args.out_dir)


if __name__ == '__main__':
    main()

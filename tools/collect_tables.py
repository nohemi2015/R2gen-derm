"""Collect tools/evaluate.py outputs into the article tables.

    python tools/collect_tables.py results/brown_skin/tables

Writes main_table.{csv,md} (one row per method, overall metrics, mean ± sd over seeds) and
by_phototype.{csv,md} / by_class.{csv,md} for the proposed model (ViT-L + RM + MCLN).
"""
import glob
import os
import sys

import pandas as pd

NAMES = {
    'baseline_retrieval': 'Retrieval (nearest training image)',
    'baseline_template': 'Classifier + template',
    'r101_base': 'ResNet-101 Transformer (BASE)',
    'r101_rm': 'ResNet-101 + RM',
    'r101_rm_mcln': 'ResNet-101 + RM + MCLN (R2Gen)',
    'vitb_rm_mcln': 'ViT-B + RM + MCLN',
    'vitl_base': 'ViT-L Transformer (BASE)',
    'vitl_rm': 'ViT-L + RM',
    'vitl_rm_mcln': 'ViT-L + RM + MCLN (proposed)',
}
COLS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L', 'dx_acc', 'dx_f1', 'pt_acc']


def md(df):
    return df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string(index=False)


def main(folder):
    rows = []
    for key, name in NAMES.items():
        path = os.path.join(folder, key + '_summary.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str).fillna('')
        row = df[(df['group'] == 'all')].iloc[0]
        rows.append({'Method': name, **{c: row.get(c, '') for c in COLS}})
    main_table = pd.DataFrame(rows)
    main_table = main_table.loc[:, (main_table != '').any(axis=0)]
    main_table.to_csv(os.path.join(folder, 'main_table.csv'), index=False)
    open(os.path.join(folder, 'main_table.md'), 'w', encoding='utf-8').write(md(main_table) + '\n')
    print(md(main_table))

    proposed = os.path.join(folder, 'vitl_rm_mcln_summary.csv')
    if os.path.exists(proposed):
        df = pd.read_csv(proposed, dtype=str).fillna('')
        for grp in ('phototype', 'class'):
            sub = df[df['group'] == grp].drop(columns=['group']).rename(columns={'value': grp})
            if len(sub):
                sub = sub.loc[:, (sub != '').any(axis=0)]
                sub.to_csv(os.path.join(folder, 'by_{}.csv'.format(grp)), index=False)
                open(os.path.join(folder, 'by_{}.md'.format(grp)), 'w', encoding='utf-8').write(md(sub) + '\n')
                print('\n' + md(sub))


if __name__ == '__main__':
    main(sys.argv[1])

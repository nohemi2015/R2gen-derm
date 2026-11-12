"""Evaluate every trained model on the tone-control set (no retraining).

    python scripts/run_tone_controls.py ham10000 --image_dir ../data/ham10000/images \
        --ann ../data/brown_skin_v2/bs_controls_test.json

For each configuration in results/<experiment>/ (all seeds with model_best.pth): runs main_test.py on
the control set, then tools/evaluate.py, and tools/paired_tone.py for light vs dark, light vs light_blur
and light vs light_color. Output: results/<experiment>/tables_tone_controls/<config>_{dark,light_blur,light_color}.csv
and a summary table tone_controls_summary.csv.
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', default='ham10000')
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    py = sys.executable
    res = os.path.join('results', args.experiment)
    out_dir = os.path.join(res, 'tables_tone_controls')
    os.makedirs(out_dir, exist_ok=True)

    def run(cmd):
        print('+ ' + ' '.join('"{}"'.format(c) if ' ' in c else c for c in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    configs = collections.defaultdict(list)
    for ck in sorted(glob.glob(os.path.join(res, '*_s*', 'model_best.pth'))):
        run_dir = os.path.dirname(ck)
        if not os.path.exists(os.path.join(run_dir, 'test', 'metrics.json')):
            print('skip (training not finished):', run_dir)
            continue
        tag = re.sub(r'_s\d+$', '', os.path.basename(run_dir))
        configs[tag].append(run_dir)

    rows = []
    for tag, runs in configs.items():
        details = []
        for run_dir in runs:
            a = json.load(open(os.path.join(run_dir, 'args.json'), encoding='utf-8'))
            if a.get('model') == 'classifier':  # tools/finetune_classifier.py
                run([py, 'tools/finetune_classifier.py', 'predict', '--ann_path', args.ann,
                     '--image_dir', args.image_dir, '--load', os.path.join(run_dir, 'model_best.pth'),
                     '--save_dir', os.path.join(run_dir, 'test_tone_controls')])
                details.append(os.path.join(run_dir, 'test_tone_controls'))
                continue
            run([py, 'main_test.py', '--image_dir', args.image_dir, '--ann_path', args.ann,
                 '--dataset_name', a['dataset_name'], '--visual_extractor', a['visual_extractor'],
                 '--variant', a.get('variant', 'rm_mcln'), '--visual_extractor_pretrained', 'false',  # weights come from the checkpoint
                 '--max_seq_length', str(a['max_seq_length']),
                 '--threshold', str(a['threshold']), '--batch_size', '16', '--beam_size', '3',
                 '--load', os.path.join(run_dir, 'model_best.pth'),
                 '--save_dir', os.path.join(run_dir, 'test_tone_controls'), '--seed', str(a['seed'])])
            details.append(os.path.join(run_dir, 'test_tone_controls'))
        prefix = os.path.join(out_dir, tag)
        run([py, 'tools/evaluate.py', '--ann', args.ann, '--runs'] + details + ['--out', prefix])
        files = ['{}_run{}_detail.csv'.format(prefix, i) for i in range(len(details))]
        for cond in ('dark', 'light_blur', 'light_color'):
            out = '{}_{}.csv'.format(prefix, cond)
            run([py, 'tools/paired_tone.py', '--details'] + files + ['--a', 'light', '--b', cond, '--out', out])
            if not args.dry_run:
                t = pd.read_csv(out, index_col=0)
                m = t.loc['mean'] if 'mean' in t.index else t.iloc[0]
                rows.append(dict(config=tag, condition=cond, seeds=len(details),
                                 dx_acc_light=m['dx_acc_light'], dx_acc_condition=m['dx_acc_dark'],
                                 gap_dx_acc=m['gap_dx_acc'], bleu4_light=m['bleu4_light'],
                                 bleu4_condition=m['bleu4_dark'], gap_bleu4=m['gap_bleu4'],
                                 mcnemar_p_max=t.drop(index=[i for i in ('mean', 'std') if i in t.index])['mcnemar_p'].max()))
    if rows:
        s = pd.DataFrame(rows)
        s.to_csv(os.path.join(out_dir, 'tone_controls_summary.csv'), index=False)
        print(s.round(4).to_string(index=False))


if __name__ == '__main__':
    main()

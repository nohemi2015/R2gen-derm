"""Full experiment grid for the article. Works on Windows, Linux and macOS.

Run inside the r2gen-derm folder, after tools/build_brown_skin.py:

    python scripts/run_experiments.py ham10000      --image_dir "C:/1. HAM10000/data/ham10000/images" --ann_dir "C:/1. HAM10000/data/brown_skin_v2"
    python scripts/run_experiments.py ham10000_bs   --image_dir "C:/1. HAM10000/data/ham10000/images" --ann_dir "C:/1. HAM10000/data/brown_skin_v2"
    python scripts/run_experiments.py iu_xray       --image_dir "C:/1. HAM10000/data/iu_xray/images"  --iu_ann  "C:/1. HAM10000/data/iu_xray/annotation.json"
    (add --dry_run to only print the commands)

ham10000     train on HAM10000 (lesion-level split). Test on the ISIC 2018 test set (main table)
             and on the Brown Skin paired set (same lesions, light vs dark skin).
             Grid: ResNet-101 and ViT-L x {BASE, +RM, +RM+MCLN}, ViT-B x +RM+MCLN, 3 seeds,
             plus retrieval and classifier+template baselines.
ham10000_bs  same, but Brown Skin dark images of training lesions are added to training
             (does adding dark-skin images reduce the light/dark gap?). R2Gen and proposed model only.
iu_xray      R2Gen (ResNet-101) vs proposed (ViT-L), 3 seeds.
"""
import argparse
import os
import subprocess
import sys

VITL = 'vit_large_patch16_224.augreg_in21k'
VITB = 'vit_base_patch16_224.augreg_in21k'
SHORT = {'resnet101': 'r101', VITL: 'vitl', VITB: 'vitb'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', choices=['ham10000', 'ham10000_bs', 'iu_xray'])
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--ann_dir', default=None, help='output folder of tools/build_brown_skin.py')
    ap.add_argument('--iu_ann', default=None)
    ap.add_argument('--seeds', default='0 1 2')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    py = sys.executable
    res = os.path.join('results', args.experiment)
    seeds = args.seeds.split()

    def run(cmd):
        print('+ ' + ' '.join('"{}"'.format(c) if ' ' in c else c for c in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if args.experiment == 'iu_xray':
        train_ann, dsname, paired = args.iu_ann, 'iu_xray', None
        epochs, step = args.epochs or 100, 50
        configs = [('resnet101', 'rm_mcln'), (VITL, 'rm_mcln')]
    else:
        name = 'ham10000_lesion.json' if args.experiment == 'ham10000' else 'ham10000_plus_bs.json'
        train_ann, dsname = os.path.join(args.ann_dir, name), 'ham10000'
        paired = os.path.join(args.ann_dir, 'bs_paired_test.json')
        epochs, step = args.epochs or 50, 25
        configs = [('resnet101', 'base'), ('resnet101', 'rm'), ('resnet101', 'rm_mcln'),
                   (VITL, 'base'), (VITL, 'rm'), (VITL, 'rm_mcln'), (VITB, 'rm_mcln')]
        if args.experiment == 'ham10000_bs':
            configs = [('resnet101', 'rm_mcln'), (VITL, 'rm_mcln')]

    common = ['--image_dir', args.image_dir, '--dataset_name', dsname,
              '--max_seq_length', '60', '--threshold', '3', '--batch_size', '16']
    for ve, var in configs:
        tag = '{}_{}'.format(SHORT[ve], var)
        main_runs, pair_details = [], []
        for seed in seeds:
            out = os.path.join(res, '{}_s{}'.format(tag, seed))
            if not os.path.exists(os.path.join(out, 'model_best.pth')):
                run([py, 'main_train.py'] + common + [
                    '--ann_path', train_ann, '--visual_extractor', ve, '--variant', var,
                    '--epochs', str(epochs), '--step_size', str(step), '--gamma', '0.1',
                    '--save_dir', out, '--record_dir', os.path.join(res, 'records'), '--seed', seed])
            test_args = ['--visual_extractor', ve, '--variant', var, '--beam_size', '3',
                         '--load', os.path.join(out, 'model_best.pth'), '--seed', seed]
            run([py, 'main_test.py'] + common + ['--ann_path', train_ann,
                                                '--save_dir', os.path.join(out, 'test')] + test_args)
            main_runs.append(os.path.join(out, 'test'))
            if paired:
                run([py, 'main_test.py'] + common + ['--ann_path', paired,
                                                    '--save_dir', os.path.join(out, 'test_bs_pairs')] + test_args)
        run([py, 'tools/evaluate.py', '--ann', train_ann, '--runs'] + main_runs +
            ['--out', os.path.join(res, 'tables', tag)])
        if paired:
            prefix = os.path.join(res, 'tables_bs_pairs', tag)
            run([py, 'tools/evaluate.py', '--ann', paired, '--runs'] +
                [os.path.join(res, '{}_s{}'.format(tag, s), 'test_bs_pairs') for s in seeds] + ['--out', prefix])
            run([py, 'tools/paired_tone.py', '--details'] +
                ['{}_run{}_detail.csv'.format(prefix, i) for i in range(len(seeds))] +
                ['--out', prefix + '_paired.csv'])

    if args.experiment == 'ham10000':
        run([py, 'tools/baselines.py', '--image_dir', args.image_dir, '--ann_path', train_ann,
             '--visual_extractor', VITL, '--out_dir', os.path.join(res, 'baselines')])
        for b in ('retrieval', 'template'):
            run([py, 'tools/evaluate.py', '--ann', train_ann, '--runs', os.path.join(res, 'baselines', b),
                 '--out', os.path.join(res, 'tables', 'baseline_' + b)])

    run([py, 'tools/collect_tables.py', os.path.join(res, 'tables')])


if __name__ == '__main__':
    main()

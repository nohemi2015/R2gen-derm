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
ham10000_clf     experiment A: strong baseline. ViT-L fine-tuned as a 3-head classifier (diagnosis,
                 confirmation type, localisation) + template, 3 seeds. Trained on HAM10000 only.
ham10000_toneaug experiment B: skin-tone augmentation balanced across the 7 classes (Reinhard colour
                 transfer to the colour of Brown Skin TRAINING images, p = 0.5). R2Gen, proposed model
                 and the fine-tuned classifier, 3 seeds. Trained on HAM10000 images only.
ham10000_desc    descriptive reports (tools/build_descriptive_reports.py, from data/lesion_descriptions_en.json)
                 instead of the short metadata template. R2Gen, ViT-L BASE and the proposed model, 3 seeds,
                 plus retrieval and classifier+template baselines. Same splits and test sets as ham10000.
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
    ap.add_argument('experiment', choices=['ham10000', 'ham10000_bs', 'iu_xray', 'ham10000_clf', 'ham10000_toneaug', 'ham10000_desc', 'ham10000_desc2'])
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--ann_dir', default=None, help='output folder of tools/build_brown_skin.py')
    ap.add_argument('--iu_ann', default=None)
    ap.add_argument('--seeds', default='0 1 2')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--tone_aug_p', type=float, default=0.5)
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
        name = 'ham10000_plus_bs.json' if args.experiment == 'ham10000_bs' else 'ham10000_lesion.json'
        train_ann, dsname = os.path.join(args.ann_dir, name), 'ham10000'
        paired = os.path.join(args.ann_dir, 'bs_paired_test.json')
        short_ann = train_ann
        if args.experiment in ('ham10000_desc', 'ham10000_desc2'):
            v2 = args.experiment == 'ham10000_desc2'
            desc = [os.path.splitext(f)[0] + ('_desc2.json' if v2 else '_desc.json') for f in (train_ann, paired)]
            if not all(os.path.exists(f) for f in desc):
                run([py, 'tools/build_descriptive_reports.py', '--descriptions',
                     os.path.join('data', 'lesion_descriptions_en.json'), '--ann', train_ann, paired]
                    + (['--v2'] if v2 else []))
            train_ann, paired = desc
        epochs, step = args.epochs or 50, 25
        configs = [('resnet101', 'base'), ('resnet101', 'rm'), ('resnet101', 'rm_mcln'),
                   (VITL, 'base'), (VITL, 'rm'), (VITL, 'rm_mcln'), (VITB, 'rm_mcln')]
        if args.experiment in ('ham10000_bs', 'ham10000_toneaug'):
            configs = [('resnet101', 'rm_mcln'), (VITL, 'rm_mcln')]
        if args.experiment == 'ham10000_clf':
            configs = []
        if args.experiment == 'ham10000_desc':
            configs = [('resnet101', 'rm_mcln'), (VITL, 'base'), (VITL, 'rm'), (VITL, 'rm_mcln')]
        if args.experiment == 'ham10000_desc2':
            # revision: references without the recommendation and without location when it is not recorded;
            # 'basewide' = BASE with d_ff 3584, which matches the parameters of the full model (capacity control)
            configs = [(VITL, 'rm_mcln'), (VITL, 'base'), (VITL, 'basewide'), (VITL, 'rm'), ('resnet101', 'rm_mcln')]

    tone = []
    if args.experiment == 'ham10000_toneaug':
        refs = os.path.join(args.ann_dir, 'tone_refs.json')
        if not os.path.exists(refs):
            run([py, 'tools/build_tone_refs.py', '--ann', os.path.join(args.ann_dir, 'ham10000_plus_bs.json'),
                 '--image_dir', args.image_dir, '--out', refs])
        tone = ['--tone_aug_refs', refs, '--tone_aug_p', str(args.tone_aug_p)]

    max_len = '80' if args.experiment in ('ham10000_desc', 'ham10000_desc2') else '60'
    common = ['--image_dir', args.image_dir, '--dataset_name', dsname,
              '--max_seq_length', max_len, '--threshold', '3', '--batch_size', '16']
    alt_sel = args.experiment == 'ham10000_desc2'
    for ve, var in configs:
        tag = '{}_{}'.format(SHORT[ve], var)
        extra = []
        if var == 'basewide':
            var, extra = 'base', ['--d_ff', '3584']
        main_runs, pair_details = [], []
        alt_runs = {'acc': [], 'youden': []}
        for seed in seeds:
            out = os.path.join(res, '{}_s{}'.format(tag, seed))
            if not os.path.exists(os.path.join(out, 'model_best.pth')):
                run([py, 'main_train.py'] + common + [
                    '--ann_path', train_ann, '--visual_extractor', ve, '--variant', var,
                    '--epochs', str(epochs), '--step_size', str(step), '--gamma', '0.1',
                    '--save_dir', out, '--record_dir', os.path.join(res, 'records'), '--seed', seed] + tone + extra)
            test_args = ['--visual_extractor', ve, '--variant', var, '--beam_size', '3',
                         '--load', os.path.join(out, 'model_best.pth'), '--seed', seed] + extra
            if alt_sel:
                for key in alt_runs:
                    ck = os.path.join(out, 'model_best_{}.pth'.format(key))
                    alt_args = ['--visual_extractor', ve, '--variant', var, '--beam_size', '3',
                                '--load', ck, '--seed', seed] + extra
                    run([py, 'main_test.py'] + common + ['--ann_path', train_ann,
                                                        '--save_dir', os.path.join(out, 'test_' + key)] + alt_args)
                    alt_runs[key].append(os.path.join(out, 'test_' + key))
            run([py, 'main_test.py'] + common + ['--ann_path', train_ann,
                                                '--save_dir', os.path.join(out, 'test')] + test_args)
            main_runs.append(os.path.join(out, 'test'))
            if paired:
                run([py, 'main_test.py'] + common + ['--ann_path', paired,
                                                    '--save_dir', os.path.join(out, 'test_bs_pairs')] + test_args)
        run([py, 'tools/evaluate.py', '--ann', train_ann, '--runs'] + main_runs +
            ['--out', os.path.join(res, 'tables', tag)])
        for key, runs in alt_runs.items():
            if runs:
                run([py, 'tools/evaluate.py', '--ann', train_ann, '--runs'] + runs +
                    ['--out', os.path.join(res, 'tables_' + key, tag)])
        if paired:
            prefix = os.path.join(res, 'tables_bs_pairs', tag)
            run([py, 'tools/evaluate.py', '--ann', paired, '--runs'] +
                [os.path.join(res, '{}_s{}'.format(tag, s), 'test_bs_pairs') for s in seeds] + ['--out', prefix])
            run([py, 'tools/paired_tone.py', '--details'] +
                ['{}_run{}_detail.csv'.format(prefix, i) for i in range(len(seeds))] +
                ['--out', prefix + '_paired.csv'])

    if args.experiment in ('ham10000_clf', 'ham10000_toneaug'):
        tag = 'clf_vitl'
        main_runs = []
        for seed in seeds:
            out = os.path.join(res, '{}_s{}'.format(tag, seed))
            if not os.path.exists(os.path.join(out, 'model_best.pth')):
                run([py, 'tools/finetune_classifier.py', 'train', '--ann_path', train_ann,
                     '--image_dir', args.image_dir, '--save_dir', out, '--seed', seed] + tone)
            ck = os.path.join(out, 'model_best.pth')
            for ann_file, sub_dir in ((train_ann, 'test'), (paired, 'test_bs_pairs')):
                run([py, 'tools/finetune_classifier.py', 'predict', '--ann_path', ann_file,
                     '--image_dir', args.image_dir, '--load', ck, '--save_dir', os.path.join(out, sub_dir)])
            main_runs.append(os.path.join(out, 'test'))
        run([py, 'tools/evaluate.py', '--ann', train_ann, '--runs'] + main_runs +
            ['--out', os.path.join(res, 'tables', tag)])
        prefix = os.path.join(res, 'tables_bs_pairs', tag)
        run([py, 'tools/evaluate.py', '--ann', paired, '--runs'] +
            [os.path.join(res, '{}_s{}'.format(tag, s), 'test_bs_pairs') for s in seeds] + ['--out', prefix])
        run([py, 'tools/paired_tone.py', '--details'] +
            ['{}_run{}_detail.csv'.format(prefix, i) for i in range(len(seeds))] + ['--out', prefix + '_paired.csv'])

    if args.experiment == 'ham10000_desc2':
        # diagnosis probabilities of the existing fine-tuned classifiers (no retraining), on validation and test,
        # to choose classifier operating points on validation
        for seed in seeds:
            ck = os.path.join('results', 'ham10000_clf', 'clf_vitl_s{}'.format(seed), 'model_best.pth')
            for split in ('val', 'test'):
                run([py, 'tools/finetune_classifier.py', 'predict', '--ann_path', short_ann, '--split', split,
                     '--image_dir', args.image_dir, '--load', ck, '--save_probs',
                     '--save_dir', os.path.join(res, 'clf_probs_s{}'.format(seed), split)])

    if args.experiment in ('ham10000', 'ham10000_desc', 'ham10000_desc2'):
        run([py, 'tools/baselines.py', '--image_dir', args.image_dir, '--ann_path', train_ann,
             '--visual_extractor', VITL, '--out_dir', os.path.join(res, 'baselines')])
        for b in ('retrieval', 'template'):
            run([py, 'tools/evaluate.py', '--ann', train_ann, '--runs', os.path.join(res, 'baselines', b),
                 '--out', os.path.join(res, 'tables', 'baseline_' + b)])

    run([py, 'tools/collect_tables.py', os.path.join(res, 'tables')])


if __name__ == '__main__':
    main()

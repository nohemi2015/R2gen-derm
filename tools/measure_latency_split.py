"""Latency split for the revision: visual extractor vs. report decoding, on real test images.

For each run directory (args.json, vocab.json, model_best.pth) the model is rebuilt and, for the first N
test images of its annotation file, the script times separately
  (1) the visual extractor (image tensor already on the GPU -> patch features), and
  (2) the encoder-decoder with beam search (features -> token sequence),
synchronising CUDA before and after each part. It also records the number of generated tokens.
Settings: model.eval(), torch.no_grad(), float32, batch size 1, beam size and maximum length from args.json,
image loading/resizing, GPU transfer and detokenisation are NOT included in the times.
Fine-tuned classifiers are timed as a single forward pass.

    python tools/measure_latency_split.py results/ham10000_desc/vitl_base_s0 results/ham10000_desc/vitl_rm_s0 ^
        results/ham10000_desc/vitl_rm_mcln_s0 results/ham10000_desc/r101_rm_mcln_s0 results/ham10000_clf/clf_vitl_s0 ^
        --image_dir "C:/1. HAM10000/data/ham10000/images" --n 200 --out results/latency_split.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.measure_cost import build  # noqa: E402
from modules.visual_extractor import is_vit  # noqa: E402



def make_tf(backbone):
    # same test-time preprocessing as modules/dataloaders.py (timm ViT augreg: mean=std=0.5)
    mean, std = ((0.5,) * 3, (0.5,) * 3) if is_vit(backbone) else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])


def test_images(run_dir, image_dir, ann_override, n):
    a = json.load(open(os.path.join(run_dir, 'args.json'), encoding='utf-8'))
    ann = ann_override or a.get('ann_path')
    d = json.load(open(ann, encoding='utf-8'))
    paths = []
    for ex in d['test']:
        p = ex['image_path'][0] if isinstance(ex['image_path'], list) else ex['image_path']
        paths.append(os.path.join(image_dir or a.get('image_dir'), p))
        if len(paths) == n:
            break
    return paths


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def run(model, is_clf, paths, device, tf, warmup=10):
    model.eval().to(device)
    xs = [tf(Image.open(p).convert('RGB')).unsqueeze(0) for p in paths]
    rows = []
    for k, x in enumerate(xs[:warmup] + xs):
        x = x.to(device)
        sync(device)
        t0 = time.perf_counter()
        if is_clf:
            model(x)
            sync(device)
            t1 = t2 = time.perf_counter()
            ntok = 0
        else:
            att, fc = model.visual_extractor(x)
            sync(device)
            t1 = time.perf_counter()
            seq, _ = model.encoder_decoder(fc, att, mode='sample')
            sync(device)
            t2 = time.perf_counter()
            ntok = int((seq[0] > 0).sum().item())
        if k >= warmup:
            rows.append(dict(extractor_ms=(t1 - t0) * 1000, decoding_ms=(t2 - t1) * 1000,
                             total_ms=(t2 - t0) * 1000, tokens=ntok))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--image_dir', default=None)
    ap.add_argument('--ann_path', default=None, help='override the annotation file stored in args.json')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--out', default='results/latency_split.csv')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary, per_image = [], []
    for r in args.runs:
        model, parts, fn, name = build(r)
        is_clf = name.startswith('classifier')
        paths = test_images(r, args.image_dir, args.ann_path, args.n)
        backbone = name.split()[1] if is_clf else model.args.visual_extractor
        df = run(model, is_clf, paths, device, make_tf(backbone))
        df.insert(0, 'run', r)
        per_image.append(df)
        row = dict(run=r, model=name, n_images=len(df), device=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                   precision='float32', beam_size=getattr(model, 'args', None) and model.args.beam_size,
                   max_len=getattr(model, 'args', None) and model.args.max_seq_length,
                   tokens_mean=round(df.tokens.mean(), 1))
        for c in ['extractor_ms', 'decoding_ms', 'total_ms']:
            row[c + '_mean'] = round(df[c].mean(), 1)
            row[c + '_median'] = round(df[c].median(), 1)
            row[c + '_p25'] = round(np.percentile(df[c], 25), 1)
            row[c + '_p75'] = round(np.percentile(df[c], 75), 1)
            row[c + '_sd'] = round(df[c].std(), 1)
        summary.append(row)
        print(row, flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(summary).to_csv(args.out, index=False)
    pd.concat(per_image).to_csv(args.out.replace('.csv', '_per_image.csv'), index=False)
    print('Saved', args.out)


if __name__ == '__main__':
    main()

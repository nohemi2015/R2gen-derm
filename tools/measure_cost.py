"""Computational cost of the trained models (revision): parameters, peak GPU memory and latency.

For every run directory given (each holds args.json, vocab.json and model_best.pth), the script rebuilds
the model, counts its parameters (total, visual extractor, Transformer + memory), and times report
generation for one 224x224 image at a time (beam search as at test time) after a warm-up.
Fine-tuned classifiers (tools/finetune_classifier.py) are also supported.

    python tools/measure_cost.py results/ham10000_desc/r101_rm_mcln_s0 results/ham10000_desc/vitl_base_s0 \
        results/ham10000_desc/vitl_rm_mcln_s0 results/ham10000_clf/clf_vitl_s0 --out results/cost.csv
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count(module):
    return sum(p.numel() for p in module.parameters())


def build(run_dir):
    a = json.load(open(os.path.join(run_dir, 'args.json'), encoding='utf-8'))
    ck = os.path.join(run_dir, 'model_best.pth')
    if a.get('model') == 'classifier':
        from tools.finetune_classifier import Classifier, FIELDS
        state = torch.load(ck, map_location='cpu', weights_only=False)
        model = Classifier(state['backbone'], [len(state['vocab'][f]) for f in FIELDS], pretrained=False)
        model.load_state_dict(state['state_dict'])
        parts = dict(visual=count(model.backbone), decoder=count(model.heads))
        return model, parts, lambda m, x: m(x), 'classifier ' + state['backbone']
    from models.r2gen import R2GenModel
    from modules.tokenizers import Tokenizer
    args = SimpleNamespace(**a)
    args.vocab_path = os.path.join(run_dir, 'vocab.json')
    args.visual_extractor_pretrained = False
    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer)
    state = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
    parts = dict(visual=count(model.visual_extractor), decoder=count(model.encoder_decoder))
    return model, parts, lambda m, x: m(x, mode='sample'), '{} {}'.format(a['visual_extractor'], a.get('variant'))


@torch.no_grad()
def time_model(model, fn, device, n=100, warmup=10):
    model.eval().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(warmup):
        fn(model, x)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(model, x)
    if device == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device == 'cuda' else float('nan')
    return ms, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default='results/cost.csv')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for r in args.runs:
        model, parts, fn, name = build(r)
        ms, mem = time_model(model, fn, device, n=args.n)
        rows.append(dict(run=r, model=name, params_total_M=round(count(model) / 1e6, 1),
                         params_visual_M=round(parts['visual'] / 1e6, 1), params_decoder_M=round(parts['decoder'] / 1e6, 1),
                         latency_ms_per_image=round(ms, 1), peak_gpu_MB=round(mem) if mem == mem else None, device=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'))
        print(rows[-1], flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('Saved', args.out)


if __name__ == '__main__':
    main()

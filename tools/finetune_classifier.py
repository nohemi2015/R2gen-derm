"""Strong non-generative baseline (experiment A of the article): fine-tuned classifier + template.

Every HAM10000 reference report has the form
    "The image shows possible evidence of {dx}, classified as {dx_type}, located on {localization}."
This script fine-tunes the whole backbone (default ViT-L/16, ImageNet-21k) with three classification
heads (diagnosis, confirmation type, localisation), selects the epoch on the validation split, and fills
the template with the three predicted phrases. If this baseline matches the report generator, the
generator is doing classification; if not, generation adds something. Either result is reportable.

Train + test on the ISIC 2018 test split and on the Brown Skin paired set:
    python tools/finetune_classifier.py train --ann_path ../data/brown_skin_v2/ham10000_lesion.json \
        --image_dir ../data/ham10000/images --save_dir results/ham10000_clf/clf_vitl_s0 --seed 0
    python tools/finetune_classifier.py predict --ann_path ../data/brown_skin_v2/bs_paired_test.json \
        --image_dir ../data/ham10000/images --load results/ham10000_clf/clf_vitl_s0/model_best.pth \
        --save_dir results/ham10000_clf/clf_vitl_s0/test_bs_pairs

`predict` writes predictions.csv (id, gt, pred) and metrics.json, so tools/evaluate.py and
tools/paired_tone.py score it exactly like the generated reports.
Optional --tone_aug_refs / --tone_aug_p: the same skin-tone augmentation as the generator (experiment B).
"""
import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.tokenizers import Tokenizer  # noqa: E402

PATTERN = re.compile(r'possible evidence of (.+?), classified as (.+?), located on (.+?)\.?\s*$', re.I)
TEMPLATE = 'The image shows possible evidence of {}, classified as {}, located on {}.'
FIELDS = ('dx', 'dx_type', 'localization')


def clean(text):
    return Tokenizer.clean_report_mimic_cxr(None, str(text))


def parse(report):
    m = PATTERN.search(report.strip())
    if not m:
        raise ValueError('report does not follow the template: ' + report)
    return [g.strip() for g in m.groups()]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Images(Dataset):
    def __init__(self, examples, image_dir, transform, vocab=None):
        self.ex, self.image_dir, self.transform, self.vocab = examples, image_dir, transform, vocab

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        img = self.transform(Image.open(os.path.join(self.image_dir, e['image_path'][0])).convert('RGB'))
        if self.vocab is None:
            return img, torch.zeros(3, dtype=torch.long)
        y = [self.vocab[f].get(v, -100) for f, v in zip(FIELDS, parse(e['report']))]
        return img, torch.tensor(y, dtype=torch.long)


class Classifier(nn.Module):
    def __init__(self, backbone, sizes, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(d, n) for n in sizes])

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return [h(f) for h in self.heads]


def make_transforms(backbone, train, args=None):
    vit = 'vit' in backbone
    mean, std = ((0.5,) * 3, (0.5,) * 3) if vit else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if not train:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    tone = []
    if args is not None and args.tone_aug_refs and args.tone_aug_p > 0:
        from modules.tone_aug import ToneAugment
        tone = [ToneAugment(args.tone_aug_refs, args.tone_aug_p)]
    return transforms.Compose([transforms.Resize(256)] + tone + [
        transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out = [[] for _ in FIELDS]
    for x, _ in loader:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = model(x.to(device, non_blocking=True))
        for k, l in enumerate(logits):
            out[k].append(l.float().argmax(1).cpu())
    return [torch.cat(o).numpy() for o in out]


def train(args):
    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ann = json.load(open(args.ann_path, encoding='utf-8'))
    values = list(zip(*[parse(e['report']) for e in ann['train']]))
    vocab = {f: {v: i for i, v in enumerate(sorted(set(vals)))} for f, vals in zip(FIELDS, values)}
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(dict(vars(args), model='classifier', vocab=vocab),
              open(os.path.join(args.save_dir, 'args.json'), 'w', encoding='utf-8'), indent=1)

    tr = DataLoader(Images(ann['train'], args.image_dir, make_transforms(args.backbone, True, args), vocab),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    va = DataLoader(Images(ann['val'], args.image_dir, make_transforms(args.backbone, False), vocab),
                    batch_size=args.batch_size, num_workers=args.num_workers)
    yv = np.array([[vocab[f].get(v, -1) for f, v in zip(FIELDS, parse(e['report']))] for e in ann['val']])

    model = Classifier(args.backbone, [len(vocab[f]) for f in FIELDS], pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': args.lr},
                             {'params': model.heads.parameters(), 'lr': args.lr * args.head_lr_mult}],
                            weight_decay=args.weight_decay)
    total = args.epochs * len(tr)
    warm = len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                logits = model(x)
                loss = sum(loss_fn(l.float(), y[:, k]) for k, l in enumerate(logits))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
        preds = predict_logits(model, va, device)
        accs = [float((p == yv[:, k]).mean()) for k, p in enumerate(preds)]
        score = float(np.mean(accs))  # selection: mean accuracy of the three fields on validation
        log.append(dict(epoch=epoch, loss=run_loss / len(tr), val_dx=accs[0], val_type=accs[1], val_loc=accs[2], val_mean=score))
        print('epoch {:2d} loss {:.4f} | val acc dx {:.4f} type {:.4f} loc {:.4f}'.format(epoch, run_loss / len(tr), *accs), flush=True)
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'backbone': args.backbone, 'epoch': epoch},
                       os.path.join(args.save_dir, 'model_best.pth'))
    pd.DataFrame(log).to_csv(os.path.join(args.save_dir, 'train_log.csv'), index=False)
    print('best validation mean accuracy {:.4f}'.format(best))


def predict(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.load, map_location='cpu', weights_only=False)
    vocab = ck['vocab']
    inv = {f: {i: v for v, i in vocab[f].items()} for f in FIELDS}
    model = Classifier(ck['backbone'], [len(vocab[f]) for f in FIELDS], pretrained=False)
    model.load_state_dict(ck['state_dict'])
    model.to(device)
    ann = json.load(open(args.ann_path, encoding='utf-8'))
    ex = ann[args.split]
    loader = DataLoader(Images(ex, args.image_dir, make_transforms(ck['backbone'], False)),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    p = predict_logits(model, loader, device)
    preds = [TEMPLATE.format(inv['dx'][a], inv['dx_type'][b], inv['localization'][c]) for a, b, c in zip(*p)]
    os.makedirs(args.save_dir, exist_ok=True)
    pd.DataFrame({'id': [e['id'] for e in ex], 'gt': [clean(e['report']) for e in ex],
                  'pred': [clean(t) for t in preds]}).to_csv(os.path.join(args.save_dir, 'predictions.csv'), index=False)
    truth = [parse(e['report']) for e in ex]
    acc = {f: float(np.mean([inv[f][int(q)] == t[k] for q, t in zip(p[k], truth)])) for k, f in enumerate(FIELDS)}
    json.dump(dict(field_accuracy=acc, n=len(ex)), open(os.path.join(args.save_dir, 'metrics.json'), 'w'), indent=1)
    print('saved {} predictions | field accuracy {}'.format(len(ex), {k: round(v, 4) for k, v in acc.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'predict'])
    ap.add_argument('--ann_path', required=True)
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--load', default=None)
    ap.add_argument('--backbone', default='vit_large_patch16_224.augreg_in21k')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--head_lr_mult', type=float, default=100.0)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tone_aug_refs', default=None)
    ap.add_argument('--tone_aug_p', type=float, default=0.0)
    ap.add_argument('--no_pretrained', action='store_true', help='random weights (pipeline test only)')
    args = ap.parse_args()
    train(args) if args.mode == 'train' else predict(args)


if __name__ == '__main__':
    main()

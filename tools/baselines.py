"""Non-generative baselines that every reviewer will ask for.

  * retrieval:  the report of the most similar training image (cosine similarity of frozen
                backbone features). Shows how much of the score is just "find a similar case".
  * template:   a linear classifier on frozen features predicts which training report applies
                (each distinct report is a class when there are <= --max_templates of them,
                otherwise the diagnosis, mapped to its most frequent report). Shows how much of
                the score is just "get the class right".

If the generative model does not clearly beat these, the article must say so.

Usage
-----
    python tools/baselines.py --image_dir data/brown_skin/images \
        --ann_path data/brown_skin/annotation_lesion_split.json \
        --visual_extractor vit_large_patch16_224.augreg_in21k --out_dir results/bs_baselines

Writes <out_dir>/retrieval/predictions.csv and <out_dir>/template/predictions.csv, which
tools/evaluate.py scores exactly like the model outputs.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.visual_extractor import VisualExtractor, is_vit  # noqa: E402


@torch.no_grad()
def extract(examples, image_dir, extractor, transform, device, batch_size=32):
    feats = []
    for i in range(0, len(examples), batch_size):
        batch = [transform(Image.open(os.path.join(image_dir, e['image_path'][0])).convert('RGB'))
                 for e in examples[i:i + batch_size]]
        _, avg = extractor(torch.stack(batch).to(device))
        feats.append(torch.nn.functional.normalize(avg.reshape(len(batch), -1), dim=-1).cpu())
    return torch.cat(feats).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--ann_path', required=True)
    ap.add_argument('--visual_extractor', default='vit_large_patch16_224.augreg_in21k')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_templates', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_pretrained', action='store_true', help='random weights (only for testing the pipeline)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ns = argparse.Namespace(visual_extractor=args.visual_extractor, visual_extractor_pretrained=not args.no_pretrained)
    extractor = VisualExtractor(ns).to(device).eval()
    mean, std = ((0.5,) * 3, (0.5,) * 3) if is_vit(args.visual_extractor) else \
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    ann = json.load(open(args.ann_path, encoding='utf-8'))
    train, test = ann['train'] + ann['val'], ann['test']
    f_train = extract(train, args.image_dir, extractor, transform, device)
    f_test = extract(test, args.image_dir, extractor, transform, device)
    reports = [e['report'] for e in train]

    # retrieval baseline
    nn_idx = (f_test @ f_train.T).argmax(1)
    retrieval = [reports[j] for j in nn_idx]

    # template baseline
    from sklearn.linear_model import LogisticRegression
    unique = sorted(set(reports))
    if len(unique) <= args.max_templates:
        y = reports
        to_report = {r: r for r in unique}
        mode = 'report'
    else:
        y = [e.get('label') or 'unknown' for e in train]
        by_label = collections.defaultdict(collections.Counter)
        for lab, r in zip(y, reports):
            by_label[lab][r] += 1
        to_report = {lab: c.most_common(1)[0][0] for lab, c in by_label.items()}
        mode = 'diagnosis'
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=args.seed).fit(f_train, y)
    template = [to_report[p] for p in clf.predict(f_test)]
    print('template baseline predicts one of {} {}s'.format(len(set(y)), mode))

    for name, preds in (('retrieval', retrieval), ('template', template)):
        d = os.path.join(args.out_dir, name)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({'id': [str(e['id']) for e in test], 'gt': [e['report'] for e in test],
                      'pred': preds}).to_csv(os.path.join(d, 'predictions.csv'), index=False)
        print('saved', os.path.join(d, 'predictions.csv'))


if __name__ == '__main__':
    main()

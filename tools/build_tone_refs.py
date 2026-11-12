"""Pre-compute the dark-skin colour references for the tone augmentation (modules/tone_aug.py).

Only Brown Skin images of the TRAINING split are used (tone == 'dark' in the train split of
ham10000_plus_bs.json), so the 83 paired test lesions are never seen.

    python tools/build_tone_refs.py --ann ../data/brown_skin_v2/ham10000_plus_bs.json \
        --image_dir ../data/ham10000/images --out ../data/brown_skin_v2/tone_refs.json
"""
import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.tone_aug import lab_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ann = json.load(open(args.ann, encoding='utf-8'))
    test_pairs = {e.get('pair') for e in ann.get('test', [])}
    refs = []
    for e in ann['train']:
        if e.get('tone') != 'dark':
            continue
        assert e.get('pair') not in test_pairs
        img = Image.open(os.path.join(args.image_dir, e['image_path'][0])).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        mean, std = lab_stats(img)
        refs.append(dict(id=e['id'], label=e.get('label'), mean=mean, std=std))
    json.dump(dict(source=os.path.basename(args.ann), refs=refs), open(args.out, 'w', encoding='utf-8'), indent=1)
    print('dark-skin references:', len(refs), '->', args.out)


if __name__ == '__main__':
    main()

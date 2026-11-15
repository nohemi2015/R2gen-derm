"""Descriptive reference reports built from the authors' lesion descriptions (data/lesion_descriptions_en.json).

The short metadata template ("The image shows possible evidence of {dx}, classified as {type}, located on
{site}.") is replaced by a multi-sentence descriptive report:

    The image shows a lesion suggestive of {dx}. {class description} The lesion is {symmetric|asymmetric},
    with {regular|irregular} borders and {uniform|variegated} colour, located on {site}.
    It is classified as {benign|malignant}. Recommended study: {recommended study}.

Diagnosis and site come from each image's metadata. The description, the ABC criteria, benign/malignant and
the recommended study come from the description of the image's class. Fields of the description file that
cannot be read from an image (diameter, evolution, example location, phototype) are NOT used, and sentences of
the description about growth or changes over time are dropped.

    python tools/build_descriptive_reports.py --descriptions data/lesion_descriptions_en.json \
        --ann ../data/brown_skin_v2/ham10000_lesion.json ../data/brown_skin_v2/bs_paired_test.json
writes ham10000_lesion_desc.json and bs_paired_test_desc.json next to the inputs.
"""
import argparse
import json
import os
import re

TEMPLATE = re.compile(r'possible evidence of (.+?), classified as (.+?), located on (.+?)\.?\s*$', re.I)
CODE = re.compile(r'\((AKIEC|BCC|BKL|DF|MEL|NV|VASC)\)')


EVOLUTION = re.compile(r'evolution|growth|changes?\b|recent', re.I)


def visual_only(text):
    """Keep only the sentences that describe what is visible; drop history (growth, changes, evolution)."""
    sents = [x.strip() for x in re.split(r'(?<=\.)\s+', text.strip()) if x.strip()]
    keep = [x for x in sents if not EVOLUTION.search(x)]
    return ' '.join(keep)


def load_descriptions(path):
    out = {}
    for d in json.load(open(path, encoding='utf-8')):
        code = CODE.search(d['diagnosis']).group(1).lower()
        f = d['features']
        out[code] = dict(
            description=visual_only(d['description']),
            shape='asymmetric' if f['asymmetry'] else 'symmetric',
            borders='irregular' if f['irregular_borders'] else 'regular',
            colour='variegated' if f['variegated_color'] else 'uniform',
            kind=d['lesion_type'].strip().lower(),
            study=d['recommended_study'].strip().rstrip('.'))
    assert len(out) == 7, sorted(out)
    return out


def describe(entry, desc):
    m = TEMPLATE.search(entry['report'].strip())
    if not m:
        raise ValueError('unexpected report: ' + entry['report'])
    dx, _, site = [g.strip() for g in m.groups()]
    site = 'an unrecorded site' if site.lower() == 'unknown' else site
    d = desc[entry['label']]
    return ('The image shows a lesion suggestive of {dx}. {description} The lesion is {shape}, with {borders} '
            'borders and {colour} colour, located on {site}. It is classified as {kind}. '
            'Recommended study: {study}.').format(dx=dx, site=site, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--descriptions', required=True)
    ap.add_argument('--ann', nargs='+', required=True)
    args = ap.parse_args()
    desc = load_descriptions(args.descriptions)
    for path in args.ann:
        ann = json.load(open(path, encoding='utf-8'))
        for split in ann.values():
            for e in split:
                e['report_template'] = e['report']
                e['report'] = describe(e, desc)
        out = os.path.splitext(path)[0] + '_desc.json'
        json.dump(ann, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        n = sum(len(v) for v in ann.values())
        distinct = len({e['report'] for v in ann.values() for e in v})
        words = sum(len(e['report'].split()) for v in ann.values() for e in v) / n
        print('{} -> {} | {} reports, {} distinct, {:.1f} words on average'.format(
            os.path.basename(path), os.path.basename(out), n, distinct, words))


if __name__ == '__main__':
    main()

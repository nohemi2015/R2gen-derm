"""Control experiment: is the light/dark gap caused by skin tone or by image degradation?

Brown Skin images are darker but also smoother (style transfer removes fine texture). For every
paired test lesion this script builds two extra versions of the LIGHT original:

  light_blur   the light original blurred with a Gaussian filter until its sharpness
               (variance of the Laplacian at 224x224) matches that of its dark version.
               Same colour as the original, same loss of detail as Brown Skin.
  light_color  the light original with the global colour statistics of its dark version
               (Reinhard colour transfer in CIELAB: mean and standard deviation of L*, a*, b*).
               Same detail as the original, same colour/tone as Brown Skin.

Comparing light, light_blur, light_color and dark on the same lesions separates the effect of
detail loss from the effect of tone. No model is retrained.

Usage (inside r2gen-derm):
    python tools/tone_controls.py --ann_path ../data/brown_skin_v2/bs_paired_test.json \
        --image_root ../data/ham10000/images --out_ann ../data/brown_skin_v2/bs_controls_test.json
Then:
    python scripts/run_tone_controls.py ham10000 --image_dir ../data/ham10000/images \
        --ann ../data/brown_skin_v2/bs_controls_test.json
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageFilter

SIZE = 224


# ---------------------------------------------------------------- colour space (sRGB <-> CIELAB, D65)
def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    xyz = _srgb_to_linear(rgb) @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    return np.clip(_linear_to_srgb(xyz @ np.linalg.inv(_M).T), 0, 1)


def reinhard(src, ref):
    """Give `src` the per-channel mean and std of `ref` in CIELAB (both float RGB in [0,1])."""
    a, b = rgb_to_lab(src), rgb_to_lab(ref)
    mu_a, sd_a = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
    mu_b, sd_b = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0)
    return lab_to_rgb((a - mu_a) / sd_a * sd_b + mu_b)


# ---------------------------------------------------------------- sharpness
def sharpness(img):
    """Variance of the discrete Laplacian of the grey image at 224x224 (what the model sees)."""
    g = np.asarray(img.convert('L').resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return lap.var()


def match_blur(light, target_sharpness, max_sigma=12.0, iters=25):
    """Smallest Gaussian sigma (in original pixels) that brings `light` down to the target sharpness."""
    if sharpness(light) <= target_sharpness:
        return light, 0.0
    lo, hi = 0.0, max_sigma
    for _ in range(iters):
        mid = (lo + hi) / 2
        if sharpness(light.filter(ImageFilter.GaussianBlur(mid))) > target_sharpness:
            lo = mid
        else:
            hi = mid
    return light.filter(ImageFilter.GaussianBlur(hi)), hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_path', required=True, help='bs_paired_test.json from tools/build_brown_skin.py')
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--out_ann', required=True)
    ap.add_argument('--out_subdir', default='bs_controls')
    args = ap.parse_args()

    ann = json.load(open(args.ann_path, encoding='utf-8'))
    test = ann['test']
    light = {e['pair']: e for e in test if e.get('tone') == 'light'}
    dark = [e for e in test if e.get('tone') == 'dark']
    os.makedirs(os.path.join(args.image_root, args.out_subdir, 'blur'), exist_ok=True)
    os.makedirs(os.path.join(args.image_root, args.out_subdir, 'color'), exist_ok=True)

    new, log = [], []
    for d in dark:
        l = light.get(d['pair'])
        if l is None:
            continue
        li = Image.open(os.path.join(args.image_root, l['image_path'][0])).convert('RGB')
        di = Image.open(os.path.join(args.image_root, d['image_path'][0])).convert('RGB')
        tag = d['id'].replace(d['pair'], '').strip('_') or 'dark'
        name = '{}_{}.png'.format(d['pair'], tag)

        blurred, sigma = match_blur(li, sharpness(di))
        rel_b = '/'.join([args.out_subdir, 'blur', name])
        blurred.save(os.path.join(args.image_root, rel_b))

        ref = np.asarray(di.resize(li.size, Image.BILINEAR), dtype=np.float64) / 255
        recol = reinhard(np.asarray(li, dtype=np.float64) / 255, ref)
        rel_c = '/'.join([args.out_subdir, 'color', name])
        Image.fromarray((recol * 255).round().astype(np.uint8)).save(os.path.join(args.image_root, rel_c))

        for cond, rel in (('light_blur', rel_b), ('light_color', rel_c)):
            e = dict(l)
            e.update(id='{}_{}_{}'.format(d['pair'], tag, cond), image_path=[rel], tone=cond)
            new.append(e)
        log.append(dict(pair=d['pair'], dark_id=d['id'], sigma=sigma, sharp_light=sharpness(li),
                        sharp_dark=sharpness(di), sharp_blur=sharpness(blurred)))

    out = dict(ann)
    out['test'] = test + new
    with open(args.out_ann, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    import pandas as pd
    lg = pd.DataFrame(log)
    lg.to_csv(os.path.splitext(args.out_ann)[0] + '_log.csv', index=False)
    print('pairs: {} | test entries: {} (light, dark, light_blur, light_color)'.format(len(log), len(out['test'])))
    print('sharpness  light {:.1f}  dark {:.1f}  light_blur {:.1f}  (median, variance of Laplacian)'.format(
        lg.sharp_light.median(), lg.sharp_dark.median(), lg.sharp_blur.median()))
    print('Gaussian sigma needed: median {:.2f} px (range {:.2f}-{:.2f})'.format(
        lg.sigma.median(), lg.sigma.min(), lg.sigma.max()))
    print('Saved', args.out_ann)


if __name__ == '__main__':
    main()

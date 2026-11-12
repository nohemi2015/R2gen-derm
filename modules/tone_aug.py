"""Skin-tone augmentation balanced across diagnostic classes (experiment B of the article).

With probability p, a training image is given the global colour statistics (mean and standard
deviation of L*, a*, b*) of a randomly chosen dark-skin reference image (Reinhard colour transfer).
The references are only the Brown Skin images whose originals are in the TRAINING split, so no test
lesion is ever seen. Because the transform is applied to every class with the same probability,
dark tone is no longer associated with any particular diagnosis.

Reference statistics are pre-computed once with tools/build_tone_refs.py (tone_refs.json).
"""
import json
import random

import numpy as np
from PIL import Image

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_MINV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    lin = xyz @ _MINV.T
    return np.clip(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055), 0, 1)


def lab_stats(img):
    lab = rgb_to_lab(np.asarray(img.convert('RGB'), dtype=np.float64) / 255).reshape(-1, 3)
    return lab.mean(0).tolist(), lab.std(0).tolist()


def reinhard_to_stats(img, mean, std):
    lab = rgb_to_lab(np.asarray(img, dtype=np.float64) / 255)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = lab_to_rgb((lab - mu) / sd * np.asarray(std) + np.asarray(mean))
    return Image.fromarray((out * 255).round().astype(np.uint8))


class ToneAugment:
    """PIL -> PIL. Apply after resizing (cheaper) and before cropping/normalisation."""

    def __init__(self, refs_path, p=0.5):
        self.refs = json.load(open(refs_path, encoding='utf-8'))['refs']
        self.p = p

    def __call__(self, img):
        if self.p <= 0 or random.random() >= self.p:
            return img
        r = random.choice(self.refs)
        return reinhard_to_stats(img.convert('RGB'), r['mean'], r['std'])

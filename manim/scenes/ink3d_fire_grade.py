"""ink_grade.py for frames with small lights in them. The grade darkens every edge as a drying wash does, which draws a black ring round anything small and bright.
Here that ring is taken away round the lights and left everywhere else, and the lights stay clear of the mist. Same arguments as ink_grade.py.
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

import ink_grade as ig


def grade_lights(rgba, z, night, rho, *rest):
    ringed = ig.grade(rgba, z, night, rho, *rest).astype(np.float32)
    keep = ig.gaussian_filter
    ig.gaussian_filter = lambda a, s: np.zeros_like(a)
    try:
        plain = ig.grade(rgba, z, night, rho, *rest).astype(np.float32)
        clear = ig.grade(rgba, z, night, 0.0, *rest).astype(np.float32)
    finally:
        ig.gaussian_filter = keep
    L = rgba[..., :3].mean(-1)
    lights = np.clip((L - gaussian_filter(L, 14)) / .05, 0, 1)
    m = np.clip(gaussian_filter(maximum_filter(lights, 7), 2.0) * 1.4, 0, 1)[..., None]
    out = ringed * (1 - m) + plain * m
    # a light has no surface and is not in the depth pass, so the mist would be judged by whatever stands behind it and fog it to a line across its middle. The render's haze has already dimmed it as much as it should be
    bright = np.clip((L - .85) / .08, 0, 1)
    m2 = np.clip(gaussian_filter(maximum_filter(bright, 5), 1.5), 0, 1)[..., None]
    return np.clip(out * (1 - m2) + clear * m2, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--night", action="store_true")
    p.add_argument("--mist", type=float, default=0.0)
    p.add_argument("--mist-h", type=float, default=14.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--sky", type=float, default=None)
    p.add_argument("--lift", type=float, default=None)
    p.add_argument("--mist-d", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rgba, z = ig.load(a.name + ".png"), ig.load(a.name + "_z.png")
    meta = json.load(open(a.name + ".json"))
    im = grade_lights(rgba, z, a.night, a.mist, a.mist_h, meta["cam_z"], a.gamma, a.contrast, a.sky, a.seed, meta["far"], meta["zmax"], a.lift, a.mist_d)
    Image.fromarray(im).save(a.out or a.name + "_ink.png")

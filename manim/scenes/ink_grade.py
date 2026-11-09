"""Turns the Blender scenes' plain renders into ink on paper, the same way for every shot, so the whole film is one world.

    python ink_grade.py NAME [--night] [--mist=RHO] [--mist-h=H] [--out=PATH]      # NAME.png, NAME_z.png, NAME.json -> NAME_ink.png

How light becomes ink: a painter does not paint light, they leave paper. So bright is bare paper, dark is thick ink, and the depth pass adds
the mist that separates near mountains from far ones. On top come the things a brush and wet paper do that a renderer does not: the ink
gathers at the edges of a wash, it creeps a little along the paper's fibres, and it settles in the paper's grain.
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

PAPER = np.array([241, 234, 219], np.float32)
INK_K = np.array([1.00, 1.03, 1.09], np.float32)
FLOOR = np.array([15, 13, 12], np.float32)
_cache = {}


def fft_noise(h, w, beta, seed):
    k = (h, w, beta, seed)
    if k not in _cache:
        r = np.random.default_rng(seed)
        fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
        f = np.sqrt(fx * fx + fy * fy)
        f[0, 0] = 1
        s = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
        s[0, 0] = 0
        v = np.fft.irfft2(s, s=(h, w))
        _cache[k] = ((v - v.mean()) / v.std()).astype(np.float32)
    return _cache[k]


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def read16(path):
    return np.asarray(Image.open(path), np.float32) / 65535.0 if Image.open(path).mode.startswith("I") else None


def load(name):
    """Pillow cannot read 16-bit RGBA PNGs, so they come through ffmpeg."""
    import subprocess
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", name]).decode().strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", name, "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"])
    return np.frombuffer(raw, np.uint16).reshape(h, w, 4).astype(np.float32) / 65535.0


def fog(s, hp, hc, rho, H):
    """How much of a point at distance s, height hp, still shows through mist that thins with height: the integral along the sight line."""
    dh = hp - hc
    small = np.abs(dh) < 1e-3
    d_safe = np.where(small, 1.0, dh)
    mean = np.where(small, np.exp(-hc / H), H * (np.exp(-hc / H) - np.exp(-hp / H)) / d_safe)
    return np.exp(-rho * s * mean)


def grade(rgba, z, night=False, rho=0.0, mist_h=14.0, cam_z=5.0, gamma=1.0, contrast=1.0, sky_d=None, seed=0, far=400.0, zmax=100.0, lift=None, mist_d=None):
    """rgba: the render in 0-1. z: the depth pass (red: distance / far, green: height / zmax). rho: how thick the valley mist is, 0 for none;
    mist_h: how high it reaches. lift: ink laid over everything, for night. mist_d: the mist's own darkness, 0 by day (bare paper), grey by night."""
    h, w = rgba.shape[:2]
    L = rgba[..., :3].mean(-1)
    a = rgba[..., 3]
    D = (np.clip(1 - L, 0, 1) ** gamma) * 3.1 * contrast + (.75 if night and lift is None else lift or 0)
    mist_d = (1.25 if night else 0.0) if mist_d is None else mist_d

    # mist: paper showing through. It pools in the valleys, so peaks rise out of it, and drifts in banks
    if rho > 0:
        s, hp = maximum_filter(z[..., 0], 3) * far, z[..., 1] * zmax     # a silhouette's edge pixels mix in the sky's zero: take the mountain's distance
        bank = fft_noise(h, w, 3.2, seed + 1)
        bank2 = fft_noise(h, w, 2.6, seed + 11)
        T = fog(s, hp + (6 * bank + 2 * bank2) * (mist_h / 14), cam_z, rho * np.clip(1 + .45 * bank, .25, 2.2), mist_h)
        D = D * T + mist_d * (1 - T)

    # sky
    sky = 1 - a
    if sky_d is None:
        sky_d = 2.6 if night else 0.0
    if sky_d:
        c = fft_noise(h, w, 3.3, seed + 2)
        ys = (np.arange(h, dtype=np.float32) / h)[:, None]
        sk = sky_d * (.72 + .5 * (1 - ys)) * np.clip(1 + .2 * c, .2, 2)
        D = D * a + sk * sky
    else:
        D = D * a

    # the ink creeps along the paper's fibres
    n1, n2 = fft_noise(h, w, 1.9, seed + 3), fft_noise(h, w, 1.9, seed + 4)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    D = map_coordinates(D, [yy + 1.1 * n1, xx + 1.1 * n2], order=1, mode="nearest")

    # a wash dries darker at its edge
    soft = gaussian_filter(D, 1.3)
    gy, gx = np.gradient(soft)
    edge = np.sqrt(gx * gx + gy * gy)
    D = D + np.clip(edge * 1.5, 0, .6)

    # it settles unevenly: dry-brush at the light end, flat at the dark end
    fine = fft_noise(h, w, .7, seed + 5)
    mid = fft_noise(h, w, 1.7, seed + 6)
    D = D * (1 + (.07 * fine + .09 * mid) * np.exp(-D * .55))

    # the paper
    tone = 1 + .028 * fft_noise(h, w, 2.6, seed + 7) + .012 * fft_noise(h, w, .4, seed + 8)
    D = np.maximum(D, 0)
    out = PAPER[None, None, :] * tone[..., None] * np.exp(-D[..., None] * INK_K[None, None, :])
    return np.clip(np.maximum(out, FLOOR[None, None, :]), 0, 255).astype(np.uint8)


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
    rgba = load(a.name + ".png")
    z = load(a.name + "_z.png")
    meta = json.load(open(a.name + ".json"))
    im = grade(rgba, z, a.night, a.mist, a.mist_h, meta["cam_z"], a.gamma, a.contrast, a.sky, a.seed, meta["far"], meta["zmax"], a.lift, a.mist_d)
    Image.fromarray(im).save(a.out or a.name + "_ink.png")

"""Colour for the ink film: a thin glaze of pigment over the finished ink, the last thing a painter does. It runs on any NAME_ink.png, so nothing has to be
rendered again, and it does not move the brightness of the picture, only its hue.

    python ink_color.py NAME PALETTE [--out=PATH]      # NAME_ink.png, and NAME_z.png if there is one -> NAME_col.png

How the old painters did it, and so how a palette is built: ink first, colour over it, bare paper left bare. Near rock takes warm 赭石, far mountains cool 花青,
because that is what distance looks like; a palette says what goes where. The one red, the seal's, is not in here on purpose.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from ink_grade import fft_noise, load, sm

LUM = np.array([0.30, 0.59, 0.11], np.float32)
PIGMENTS = dict(   # the wash on white paper
    zhe=(201, 150, 104),     # 赭石, warm earth
    teng=(236, 192, 92),     # 藤黄, gamboge
    hua=(112, 148, 172),     # 花青, indigo
    shilv=(88, 166, 130),    # 石绿, malachite
    shiqing=(94, 138, 186),  # 石青, azurite
    mo=(150, 150, 150),      # no hue at all
)
# near / far (by distance, or by height with by="height"; dark / light of the picture when a still has no depth), sky, strength, and where far begins and ends
PALETTES: dict[str, dict] = dict(
    wood=dict(near="zhe", far="zhe", sky="zhe", s=0.8),
    day=dict(near="zhe", far="hua", sky="mo", s=0.9, d0=30, d1=160),
    dawn=dict(near="zhe", far="teng", sky="teng", s=1.0, d0=20, d1=180, base=0.38),
    night=dict(near="hua", far="hua", sky="hua", s=0.9, d0=30, d1=160),
    moon=dict(near="zhe", far="hua", sky="hua", s=0.8, d0=8, d1=70),
    rain=dict(near="hua", far="mo", sky="mo", s=0.5, d0=20, d1=120),
    deep=dict(near="hua", far="shiqing", sky="hua", s=0.8, d0=10, d1=80),
    map=dict(near="zhe", far="shilv", high="shiqing", sky="mo", s=0.85, by="height"),
    grain=dict(near="zhe", far="teng", sky="hua", s=0.75, d0=20, d1=120),
    ghost=dict(near="hua", far="hua", sky="hua", s=0.9, d0=30, d1=160, glow="shilv", glow_s=0.45, glow_top=0.3),
    window=dict(near="hua", far="hua", sky="hua", s=0.62, d0=30, d1=160, glow="teng", glow_s=0.5, glow_fill=True),
    bronze=dict(near="mo", far="mo", sky="mo", s=0.0, d0=40, d1=70, patina=0.7),
    none=dict(near="mo", far="mo", sky="mo", s=0.0),
)


def absorb(name):
    """A pigment's absorption with its overall darkness taken out, so that a glaze changes hue and leaves brightness to the ink."""
    a = -np.log(np.array(PIGMENTS[name], np.float32) / 255)
    return a - (a * LUM).sum()


def blur(a, r):
    """A wide soft blur, by FFT, for telling what stands out from its surroundings."""
    h, w = a.shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None]
    return np.fft.irfft2(np.fft.rfft2(a) * np.exp(-2 * (np.pi * r) ** 2 * (fx * fx + fy * fy)), s=(h, w)).astype(np.float32)


def fit(z, h, w):
    """A depth pass rendered at another size, by nearest neighbour: depth must not be averaged across an edge."""
    if z.shape[:2] == (h, w):
        return z
    return z[(np.arange(h) * z.shape[0] // h)[:, None], np.arange(w) * z.shape[1] // w]


def depth_of(path):
    """Where a still's depth pass is: beside it, or in the raw or work folder its shot was rendered in."""
    head, tail = os.path.split(path)
    for p in (path + "_z.png", os.path.join(head, "raw", tail + "_z.png"), os.path.join(head, "work", tail + "_f_z.png")):
        if os.path.exists(p):
            return p
    return None


def glaze(ink, z=None, palette="day", seed=0, far=400.0, white=None):
    """ink: the finished picture, uint8. z: its depth pass in 0-1, if it has one. white: how bright bare paper is, for a picture with no bare paper in it. Returns the coloured picture, uint8."""
    p = PALETTES[palette]
    f = ink.astype(np.float32) / 255
    h, w = f.shape[:2]
    L = (f * LUM).sum(-1)
    paper = white or np.percentile(L, 99.5)
    D = -np.log(np.clip(L / max(paper, 1e-3), 1e-3, 1))
    cover = 1 - np.exp(-(D / 0.55) ** 2)                     # bare paper takes none at all, and the lightest wash hardly any
    cover = np.maximum(cover, p.get("base", 0.0) * sm(D / 0.05))     # unless the whole sky is meant to be washed; the sun itself stays paper
    if z is not None:
        z = fit(z, h, w)
    if z is not None and p.get("by") == "height":
        land = z[..., 1] > 0
        lo, hi = np.percentile(z[..., 1][land], [5, 97]) if land.any() else (0, 1)
        t = np.clip((z[..., 1] - lo) / max(hi - lo, 1e-4), 0, 1) ** 0.55     # most land is low; give each band of colour its share of it
        sky = ~land
    elif z is not None:
        t = sm((z[..., 0] * far - p.get("d0", 30)) / (p.get("d1", 160) - p.get("d0", 30)))
        sky = z[..., 0] <= 0
    else:
        t = sm((0.8 - np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((h, w), np.float32)) / 0.6)     # a still with no depth: its top is the far part
        sky = np.zeros((h, w), bool)
    m = (sm((t - 0.08) / 0.3) if "high" in p else t)[..., None]
    A = absorb(p["near"])[None, None] * (1 - m) + absorb(p["far"])[None, None] * m
    if "high" in p:                                          # the order of 青绿山水: 赭 at the foot, 石绿 on the slopes, 石青 on the peaks
        u = sm((t - 0.62) / 0.25)[..., None]
        A = A * (1 - u) + absorb(p["high"])[None, None] * u
    A[sky] = absorb(p["sky"])
    uneven = 1 + 0.28 * np.clip(fft_noise(h, w, 3.0, seed + 21), -2, 2) + 0.08 * fft_noise(h, w, 1.2, seed + 22)
    C = p["s"] * cover * np.clip(uneven, 0.2, 1.8)
    out = f * np.exp(-C[..., None] * A)
    if "glow" in p:                                          # a light in the dark takes its own colour: a flame, a window
        g = sm((L - blur(L, 40) - 0.12) / 0.3)
        if p.get("glow_fill"):                               # a big window is lit all over, not just where it meets the dark; the moon stays pale
            fill = sm((L - 0.58) / 0.2) * (1 - sky)
            g = np.maximum(g, np.maximum(fill, sm(blur(fill, 7) / 0.45)))      # and what is written on it stays ink, not blue
        g = g * sm((np.arange(h, dtype=np.float32)[:, None] / h - p.get("glow_top", 0.0)) / 0.05 + 1) * (1 - sky)
        out *= np.exp(-(p["glow_s"] * g)[..., None] * absorb(p["glow"])[None, None])
    if "patina" in p:                                        # verdigris is mineral: it sits on the metal instead of staining it, so it can be lighter than what is under it
        o = p["patina"] * sm((fft_noise(h, w, 2.4, seed + 23) + 0.1) / 1.0) * cover * (1 - t)
        crust = np.array(PIGMENTS["shilv"], np.float32) / 255 * (0.3 + L[..., None])
        out = out * (1 - o[..., None]) + crust * o[..., None]
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("palette", choices=sorted(PALETTES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ink = np.asarray(Image.open(a.name + "_ink.png").convert("RGB"))
    found = depth_of(a.name)
    z = load(found) if found else None
    meta = json.load(open(a.name + ".json")) if os.path.exists(a.name + ".json") else {}
    Image.fromarray(glaze(ink, z, a.palette, a.seed, meta.get("far", 400.0))).save(a.out or a.name + "_col.png")

"""ink_grade.grade with the fibre creep tied to distance, for shots that have something man-made close to the lens: a bronze rim or a cut stone edge should not turn
furry, while mountains in the distance keep the same bleed as the rest of the film. Everything else is the same recipe.

    python ink3d_gns_grade.py NAME [--mist=RHO] [--mist-h=H] [--near-creep=.35] [--gamma] [--contrast] [--lift] [--mist-d] [--sky] [--night] [--seed] [--out]
"""
import argparse
import math
import json

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, zoom

from ink_grade import FLOOR, INK_K, PAPER, fft_noise, fog, load


def glow(rgba, k, rays=0.0, centre=(.5, .2)):
    """Light through water: what is brightest blooms over its surroundings, and streams away from a point in the frame. Only ever brightens, so ink held in front of
    the light stays dark where the bright part behind it is cut off."""
    L = rgba[..., :3].mean(-1)
    h, w = L.shape
    bright = np.clip((L - .55) / .45, 0, 1) ** 2
    add = k * (gaussian_filter(bright, 18) + .7 * gaussian_filter(bright, 70))
    if rays:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        cx, cy = centre[0] * w, centre[1] * h
        acc = bright.copy()
        for i in range(6):
            sc = 1 / (1 + .045 * 2 ** i)
            acc = .5 * acc + .5 * map_coordinates(acc, [cy + (yy - cy) * sc, cx + (xx - cx) * sc], order=1, mode="constant")
        add = add + rays * gaussian_filter(acc, 2)
    out = rgba.copy()
    out[..., :3] = np.clip(rgba[..., :3] + add[..., None], 0, 1)
    return out


def through_water(rgba, s, rays=.32):
    """Light as it looks through water: nothing has an edge. The pattern on the surface is melted to a soft mottling; what is brightest keeps its shape, loses its outline, blooms, and
    streams down and away from where the bright pieces are. s: how far the pattern melts, in pixels of the full frame."""
    L = rgba[..., 0]
    h, w = L.shape
    k = w / 1920
    core = (L > .985).astype(np.float32)
    m = gaussian_filter(maximum_filter(core, 13), 1.0)
    fill = gaussian_filter(L * (1 - m), 40 * k) / np.maximum(gaussian_filter(1 - m, 40 * k), .01)
    bg = L * (1 - m) + fill * m
    bg = .55 * gaussian_filter(bg, s * k) + .45 * gaussian_filter(bg, 3.2 * s * k)
    n1, n2 = fft_noise(h, w, 2.8, 21), fft_noise(h, w, 2.8, 22)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # the outline goes three ways: bent by the water, melted, and thin bright lines fade into a glow rather than stay drawn
    bent = map_coordinates(core, [yy + 5.5 * k * n1, xx + 5.5 * k * n2], order=1, mode="nearest")
    shard = .5 * gaussian_filter(bent, .62 * s * k) + .5 * gaussian_filter(bent, 1.5 * s * k)
    shard = 1 - np.exp(-3.1 * shard)
    halo = .20 * gaussian_filter(core, 22 * k) + .55 * gaussian_filter(core, 70 * k) + .7 * gaussian_filter(core, 170 * k)
    cy, cx = (yy * core).sum() / core.sum(), (xx * core).sum() / core.sum()
    acc = core.copy()
    for i in range(6):
        sc = 1 / (1 + .06 * 2 ** i)
        acc = .5 * acc + .5 * map_coordinates(acc, [cy + (yy - cy) * sc, cx + (xx - cx) * sc], order=1, mode="constant")
    light = np.clip(shard + halo + rays * gaussian_filter(acc, 6 * k) * 6, 0, 1)
    out = rgba.copy()
    out[..., :3] = (bg + (1 - bg) * light)[..., None]
    return out


def grade2(rgba, z, night=False, rho=0.0, mist_h=14.0, cam_z=5.0, gamma=1.0, contrast=1.0, sky_d=None, seed=0, far=400.0, zmax=100.0, lift=None, mist_d=None, near_creep=.35, edge_k=1.5, calm=None):
    h, w = rgba.shape[:2]
    L = rgba[..., :3].mean(-1)
    a = rgba[..., 3]
    D = (np.clip(1 - L, 0, 1) ** gamma) * 3.1 * contrast + (.75 if night and lift is None else lift or 0)
    mist_d = (1.25 if night else 0.0) if mist_d is None else mist_d
    s = maximum_filter(z[..., 0], 3) * far
    if rho > 0:
        hp = z[..., 1] * zmax
        bank = fft_noise(h, w, 3.2, seed + 1)
        bank2 = fft_noise(h, w, 2.6, seed + 11)
        T = fog(s, hp + (6 * bank + 2 * bank2) * (mist_h / 14), cam_z, rho * np.clip(1 + .45 * bank, .25, 2.2), mist_h)
        D = D * T + mist_d * (1 - T)
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
    # the creep, less on what is near. Sky counts as far
    n1, n2 = fft_noise(h, w, 1.9, seed + 3), fft_noise(h, w, 1.9, seed + 4)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    amt = gaussian_filter(np.where(a > .5, np.clip(s / 90, near_creep, 1.1), 1.1), 2)
    # calm: a part of the frame the ink leaves alone (small type on a screen): it neither creeps nor pools
    still = 1.0
    if calm:
        x0, y0, x1, y1 = [int(v) * w // 1920 for v in calm.split(",")]
        m = np.zeros((h, w), np.float32)
        m[y0:y1, x0:x1] = 1
        still = 1 - gaussian_filter(m, 4)
    D = map_coordinates(D, [yy + amt * still * n1, xx + amt * still * n2], order=1, mode="nearest")
    soft = gaussian_filter(D, 1.3)
    gy, gx = np.gradient(soft)
    edge = np.sqrt(gx * gx + gy * gy)
    D = D + np.clip(edge * edge_k, 0, .6) * still
    fine = fft_noise(h, w, .7, seed + 5)
    mid = fft_noise(h, w, 1.7, seed + 6)
    D = D * (1 + (.07 * fine + .09 * mid) * np.exp(-D * .55))
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
    p.add_argument("--near-creep", type=float, default=.35)
    p.add_argument("--calm", default=None)
    p.add_argument("--patch", default=None)
    p.add_argument("--over-local", action="store_true")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--sky", type=float, default=None)
    p.add_argument("--lift", type=float, default=None)
    p.add_argument("--mist-d", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--glow", type=float, default=0.0)
    p.add_argument("--rays", type=float, default=0.0)
    p.add_argument("--rays-at", default=".5,.2")
    p.add_argument("--stretch", default=None, help="a,b,c,d: maps render lightness a..b to c..d before anything else, for a plate lit too low")
    p.add_argument("--through-water", type=float, default=0.0, help="melts the surface's pattern by this many pixels: light as seen through water, no outlines")
    p.add_argument("--edge", type=float, default=1.5, help="how much ink pools at edges")
    p.add_argument("--whole-moon", action="store_true", help="what is brightest closes into one disc of the same area")
    p.add_argument("--mend", default=None, help="x0,y0,x1,y1: fills a box at the frame's edge from the frame beside it, mirrored")
    p.add_argument("--over", default=None, help="a transparent plate laid over the render before the ink goes on, by its alpha")
    p.add_argument("--keep", type=float, default=1.0)
    p.add_argument("--over-at", default=None, help="dx,dy,scale of the plate in pixels of the full frame, about its middle")
    p.add_argument("--floor", type=float, default=0.0, help="lifts the darkest light: grey wash on paper, not black")
    a = p.parse_args()
    meta = json.load(open(a.name + ".json"))
    rgba, z = load(a.name + ".png"), load(a.name + "_z.png")
    if a.glow or a.rays:
        rgba = glow(rgba, a.glow, a.rays, tuple(float(v) for v in a.rays_at.split(",")))
    if a.mend:
        x0, y0, x1, y1 = [int(v) * rgba.shape[1] // 1920 for v in a.mend.split(",")]
        rgba = rgba.copy()
        wx = np.clip(np.minimum(np.arange(x1 - x0) + 40, x1 - x0 - np.arange(x1 - x0)) / 40, 0, 1)[None, :]
        wy = np.clip(np.minimum(np.arange(y1 - y0), y1 - y0 - np.arange(y1 - y0)) / 30, 0, 1)[:, None]
        w_ = (wx * wy)[..., None]
        for im_ in (rgba, z):
            im_[y0:y1, x0:x1] = im_[y0:y1, x0:x1] * (1 - w_) + im_[y0:y1, x1:x1 + (x1 - x0)][:, ::-1] * w_
    if a.patch:
        # a small thing taken out: the same picture from further along the flow laid over it, feathered all round
        x0, y0, x1, y1, dx, dy = [int(v) for v in a.patch.split(",")]
        rgba = rgba.copy()
        wx = np.clip(np.minimum(np.arange(x1 - x0), x1 - x0 - 1 - np.arange(x1 - x0)) / 12, 0, 1)[None, :]
        wy = np.clip(np.minimum(np.arange(y1 - y0), y1 - y0 - 1 - np.arange(y1 - y0)) / 12, 0, 1)[:, None]
        w_ = (wx * wy)[..., None]
        for im_ in (rgba, z):
            im_[y0:y1, x0:x1] = im_[y0:y1, x0:x1] * (1 - w_) + im_[y0 + dy:y1 + dy, x0 + dx:x1 + dx] * w_
    if a.stretch:
        lo, hi, to_lo, to_hi = [float(v) for v in a.stretch.split(",")]
        rgba = rgba.copy()
        rgba[..., :3] = np.clip(to_lo + (rgba[..., :3] - lo) * (to_hi - to_lo) / (hi - lo), 0, 1)
    if a.whole_moon:
        rgba = rgba.copy()
        L = rgba[..., 0]
        core = (L > .985).astype(np.float32)
        m = gaussian_filter(maximum_filter(core, 13), 1.0)       # with the fringe the waves left round it
        yy, xx = np.mgrid[:L.shape[0], :L.shape[1]].astype(np.float32)
        cy, cx, r = (yy * core).sum() / core.sum(), (xx * core).sum() / core.sum(), math.sqrt(core.sum() / math.pi)
        fill = gaussian_filter(L * (1 - m), 40) / np.maximum(gaussian_filter(1 - m, 40), .01)
        disc = np.clip(r + .8 * fft_noise(*L.shape, 2.4, 3) - np.hypot(xx - cx, yy - cy), 0, 1)
        L = L * (1 - m) + fill * m
        rgba[..., :3] = (L * (1 - disc) + disc)[..., None]
    if a.through_water:
        rgba = through_water(rgba, a.through_water)
    if a.over:
        c = load(a.over + ".png")
        if c.shape[:2] != rgba.shape[:2]:
            c = zoom(c, (rgba.shape[0] / c.shape[0], rgba.shape[1] / c.shape[1], 1), order=1)
        if a.over_at:
            dx, dy, k = [float(v) for v in a.over_at.split(",")]
            h, w = c.shape[:2]
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            at = [(yy - h / 2 - dy * h / 1080) / k + h / 2, (xx - w / 2 - dx * w / 1920) / k + w / 2]
            c = np.stack([map_coordinates(c[..., i], at, order=1, mode="constant") for i in range(4)], -1)
        cover = c[..., 3:4] * a.keep
        rgba = rgba.copy()
        rgba[..., :3] = rgba[..., :3] * (1 - cover) + c[..., :3] * cover
        rgba[..., 3] = np.maximum(rgba[..., 3], cover[..., 0]) if a.over_local else 1
    if a.floor:
        rgba = rgba.copy()
        rgba[..., :3] = a.floor + (1 - a.floor) * rgba[..., :3]
    im = grade2(rgba, z, a.night, a.mist, a.mist_h, meta["cam_z"], a.gamma, a.contrast, a.sky, a.seed, meta["far"], meta["zmax"], a.lift, a.mist_d, a.near_creep, a.edge, a.calm)
    Image.fromarray(im).save(a.out or a.name + "_ink.png")
    lum = (im[..., :3] @ np.array([.2126, .7152, .0722])) / 255
    print(f"{a.name.split('/')[-1]}: mean {lum.mean():.3f}  under.08 {(lum < .08).mean():.1%}  over.5 {(lum >= .5).mean():.1%}")

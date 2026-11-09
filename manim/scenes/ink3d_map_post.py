"""The ink for the high shots of ink3d_map.py and ink3d_bed.py: ink_grade.py plus the two things only a view from high up needs.

    python ink3d_map_post.py NAME [--mist=.05 --mist-h=6 --air=.006 --air-from=260 --clouds=.7 --cloud-seed=1 --drift=0 --gamma=1 --contrast=1 --seed=0 --vignette=0 --sun=X,Y,R]

--air is aerial perspective by distance: ink_grade's mist thins with height, so from above it cannot pale a far mountain top. It is applied to the render before the ink, so the far land is simply painted paler.
--clouds are near clouds drifting under the camera, gathered in two opposite corners, which is what makes a high view feel high. --drift slides them; give it the shot's t to animate.
--vignette dissolves the edges of the picture into bare paper, as a painter stops painting before the paper ends. --sun=X,Y,R lifts the ink around a sun at that pixel, radius R.
Night shots: --night --lift --sky --mist-d pass through to ink_grade, and --cloud-dark is the clouds' ink.
--river=LEVEL,X,HORIZON,FLOOR is for the river from the air (ink3d_bed.py): the water, which is whatever lies at LEVEL in the height pass, takes the far bank's reflection, ripples and the sun's path down X, and is at least FLOOR bright. HORIZON is its row at full size.
--spray=ROW,WIDTH,AMOUNT is for the flood: a low bank of spray across the picture.
--net=LEVEL,X is for bed_low (ink3d_bed_low.py): it calms the sun's net on the near floor, in the water at LEVEL, leaving the sun's own reflection at X alone.
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy.ndimage import grey_opening, map_coordinates

from ink_grade import PAPER, fft_noise, grade, load, sm


def aerial(rgba, z, far, k, start):
    s = z[..., 0] * far
    T = np.exp(-k * np.maximum(s - start, 0))[..., None]
    out = rgba.copy()
    out[..., :3] = 1 - (1 - rgba[..., :3]) * T
    return out


def river_water(rgba, z, zmax, level, sun_x, horizon, floor=.9, seed=31):
    """What makes a pale ribbon seen from the air read as water and not salt. It is done on the render because it follows the picture: every channel, in each column of pixels, starts under its far bank, and the bank lies in the water there."""
    h, w = rgba.shape[:2]
    s = h / 1080
    L = rgba[..., :3].mean(-1)
    water = (z[..., 1] * zmax < level + .08) & (rgba[..., 3] > .5)
    rows, cols = np.mgrid[:h, :w].astype(np.float32)
    top = np.maximum.accumulate(np.where(water, -1, rows), axis=0)
    d = rows - top
    hz = horizon * s
    near = sm((rows - hz - 70 * s) / (620 * s))             # 0 far off, 1 under the camera
    n = fft_noise(h, w, 2.8, seed)
    mirrored = map_coordinates(L, [np.maximum(top - d, 0), cols + 2.5 * s * fft_noise(h, w, 2.4, seed + 1)], order=1, mode="nearest").astype(np.float32)
    band = np.exp(-d / ((3 + 9 * near) * s * np.clip(1 + .5 * n, .35, 2)))
    lit = np.maximum(L, floor)
    out = lit + (mirrored - lit) * .6 * band
    # ripples: a noise lying on the water, drawn out across the stream, so they crowd together going away
    tile = fft_noise(1024, 1024, 2.0, seed + 2)
    t = np.maximum(rows - hz, 8 * s)
    gy, gx = 60 * 1280 * s / t * 1.1, (cols - w / 2) * 60 / t * .45
    rip = map_coordinates(tile, [gy % 1024, gx % 1024], order=1, mode="wrap").astype(np.float32)
    fade = sm((t - 50 * s) / (240 * s))
    out = out * (1 - .06 * sm((rip - .25) / .8) * fade)
    # the sun's path: everywhere else the water is a little less bright, so there is something for it to be brighter than
    path = np.exp(-((cols - sun_x * s) / ((34 + 90 * near) * s)) ** 2)
    sliver = sm((rip + .55) / .5)
    out = out + (1 - out) * .95 * path * sliver
    # the sand beside it stays a step darker than the water, and darkest where it is still wet
    up_ = z[..., 1] * zmax - level
    sand = sm((3.2 - up_) / 1.2)
    dry = L * (1 - sand * (.13 + .24 * np.exp(-np.maximum(up_, 0) / .7)))
    res = rgba.copy()
    for i in range(3):
        res[..., i] = np.where(water, out, dry)
    return res


def calm_net(rgba, z, zmax, level, sun_x):
    """Thin bright lines on the near floor give way to their surroundings, by most of what they stand out; the sun's reflection, with its slivers, stays."""
    h, w = rgba.shape[:2]
    s = h / 1080
    L = rgba[..., :3].mean(-1)
    water = np.abs(z[..., 1] * zmax - level) < .03
    rows, cols = np.mgrid[:h, :w].astype(np.float32)
    k = max(int(9 * s), 3)
    where = water * sm((rows - 760 * s) / (70 * s)) * np.maximum(sm((np.abs(cols - sun_x * s) - 120 * s) / (40 * s)), sm((rows - 960 * s) / (40 * s)))
    out = L - .72 * (L - grey_opening(L, size=(k, k))) * where
    res = rgba.copy()
    for i in range(3):
        res[..., i] = out
    return res


def clouds(im, seed=1, amount=.7, dark=0.0, drift=0.0):
    h, w = im.shape[:2]
    n1 = np.roll(fft_noise(h, w, 3.3, seed), int(drift * 200), 1)
    n2 = np.roll(fft_noise(h, w, 2.4, seed + 1), int(drift * 320), 1)
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    ys, xs = ys / h, xs / w
    where = np.maximum(sm((ys - .62 + .25 * (xs - .3)) / .4) * sm((.72 - xs) / .5), sm((.28 - ys + .25 * (.8 - xs)) / .3) * sm((xs - .35) / .4))
    c = sm((n1 + .35 * n2 + 1.5 * where - 1.1) / .8) * amount
    target = PAPER * np.exp(-dark)
    return np.clip(im * (1 - c[..., None]) + target[None, None, :] * c[..., None], 0, 255).astype(np.uint8)


def spray(im, row, width, amount, seed=41):
    """A low bank of spray across the picture at that row: far water and near water are painted differently, and this is what lies between them."""
    h, w = im.shape[:2]
    ys = np.arange(h, dtype=np.float32)[:, None]
    n = fft_noise(h, w, 3.0, seed) * .55 + fft_noise(h, w, 2.2, seed + 1) * .2
    c = np.exp(-((ys - row + 26 * n) / (width * np.clip(1 + .35 * n, .4, 2))) ** 2)[..., None] * amount
    return im * (1 - c) + PAPER[None, None, :] * c


def vignette(im, amount, seed=21):
    h, w = im.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    rr = np.hypot((xs / w - .5) * 1.12, (ys / h - .5) * 1.0)
    n = fft_noise(h, w, 3.4, seed)
    c = sm((rr - .37 + .07 * n) / .30)[..., None] * amount
    return im * (1 - c) + PAPER[None, None, :] * c


def sun_glow(im, x, y, r):
    h, w = im.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    g = np.exp(-np.hypot(xs - x, ys - y) / r)[..., None]
    return im + (PAPER[None, None, :] - im) * np.clip(g * 1.25, 0, 1) ** 1.5


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--night", action="store_true")
    for k, d in dict(mist=.05, mist_h=6.0, air=.006, air_from=260.0, clouds=.7, cloud_seed=1, cloud_dark=0.0, drift=0.0, gamma=1.0, contrast=1.0, seed=0).items():
        p.add_argument("--" + k.replace("_", "-"), type=type(d), default=d)
    for k in ("sky", "lift", "mist_d"):
        p.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    p.add_argument("--vignette", type=float, default=0.0)
    p.add_argument("--sun", default=None)
    p.add_argument("--river", default=None)
    p.add_argument("--spray", default=None)
    p.add_argument("--net", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    meta = json.load(open(a.name + ".json"))
    rgba, z = load(a.name + ".png"), load(a.name + "_z.png")
    if a.river:
        rgba = river_water(rgba, z, meta["zmax"], *map(float, a.river.split(",")))
    if a.net:
        rgba = calm_net(rgba, z, meta["zmax"], *map(float, a.net.split(",")))
    if a.air:
        rgba = aerial(rgba, z, meta["far"], a.air, a.air_from)
    im = grade(rgba, z, a.night, a.mist, a.mist_h, meta["cam_z"], a.gamma, a.contrast, a.sky, a.seed, meta["far"], meta["zmax"], a.lift, a.mist_d)
    if a.sun:
        im = sun_glow(im.astype(np.float32), *map(float, a.sun.split(",")))
    if a.spray:
        im = spray(im.astype(np.float32), *map(float, a.spray.split(",")))
    if a.clouds:
        im = clouds(im.astype(np.float32), a.cloud_seed, a.clouds, a.cloud_dark, a.drift)
    if a.vignette:
        im = vignette(im.astype(np.float32), a.vignette)
    im = np.clip(im, 0, 255).astype(np.uint8)
    Image.fromarray(im).save(a.out or a.name + "_ink.png")

"""Chapter 13's «世界到了地图的边», finished on the paper. The last mountains come from the render of `ink3d_new.py --shot=edge`; what lies ahead of them is ink on bare paper, so it is
painted here, where the same paper and the same creep as every other still are at hand: the mountains give out, the river goes on up the sheet with no banks, thinning into strands.

    python ink3d_new_post.py [../../ink/film_out/3d/new/edge]

Reads NAME.png, NAME_z.png, NAME.json and NAME_base_ink.png (the plain grade), writes NAME_ink.png.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from ink_grade import INK_K, PAPER, fft_noise, load, sm

TIP_Y = 170.0
SEED = 91


def river(h, w):
    """The ink's density: as wide as the gap between the mountains at the foot of the frame, one winding thread at the tip."""
    y, x = np.mgrid[:h, :w].astype(np.float32)
    t = np.clip((h - y) / (h - TIP_Y * h / 1080), 0, 1.25)
    k = w / 1920
    big = gaussian_filter(fft_noise(h, w, 3.0, SEED), 14 * k)
    big = big / big.std()
    # the line it takes, and how wide it is along it: a long thin tail, winding, as a river seen going away
    c = w * .5 + k * ((60 + 60 * t) * np.sin(t * 6.8 + .15) * sm(t * 1.6) + 18 * big * sm(t * 2.5))
    left = np.clip(1 - t, 0, 1)
    wid = k * (300 * left ** 2.4 + 10 * left + 1.6) * (1 + .16 * big * sm(t * 3))
    rough = fft_noise(h, w, 2.2, SEED + 1)
    d = np.abs(x - c) + k * rough * (1.0 + 2.5 * t)
    band = sm((wid - d) / (1.5 * k + .5 * wid))
    # darker threads running up it, following its line; they only shade it
    n = gaussian_filter(fft_noise(h, w, 1.3, SEED + 2), (30 * k, 2.2 * k))
    n = n / n.std()
    threads = map_coordinates(n, [y, np.clip(x - c + w * .5, 0, w - 1)], order=1, mode="nearest")
    th = np.clip(.5 + .36 * threads, 0, 1)
    # the brush running dry: toward the tip the line breaks, and stops
    dry = sm((th - (t - .72) * 2.0) / .35)
    tip = 1 - sm((t - .90 + .06 * big) / .10)
    # where the wet ink stops, it dries a little darker
    tide = np.exp(-((wid - d) / (2.0 * k + .10 * wid)) ** 2) * sm(t * 3) * .9
    wave = 1 + .10 * gaussian_filter(fft_noise(h, w, 1.2, SEED + 3), (1.4 * k, 34 * k)) / .25 * sm((.30 - t) / .30)
    return np.clip(band * (.20 + .20 * th * sm(t * 2) + .06 * sm(.4 - t)) * wave + .34 * tide * band * sm(t * 1.5), 0, 1.2) * dry * tip


def main(name):
    rgba, z = load(name + ".png"), load(name + "_z.png")
    meta = json.load(open(name + ".json"))
    h, w = rgba.shape[:2]
    k = w / 1920
    base = np.asarray(Image.open(name + "_base_ink.png"), np.float32)
    tone = 1 + .028 * fft_noise(h, w, 2.6, SEED + 7) + .012 * fft_noise(h, w, .4, SEED + 8)
    D_m = -np.log(np.clip(base.mean(-1) / (PAPER.mean() * tone), .012, 1))
    # the mountains: what stands above the water. Farther, they stop being finished: strokes, then nothing
    high = gaussian_filter((z[..., 1] * meta["zmax"] > .75).astype(np.float32) * rgba[..., 3], 1.2)
    s = z[..., 0] * meta["far"]
    strokes = fft_noise(h, w, 1.5, SEED + 4)
    wash = gaussian_filter(fft_noise(h, w, 3.0, SEED + 5), 10 * k)
    gone = sm((s - 92 + 26 * wash / wash.std() + 3 * strokes) / 86)
    D = D_m * .64 * high * (1 - gone)
    # the paper, nearly bare
    paper = .035 + .018 * gaussian_filter(fft_noise(h, w, 3.3, SEED + 6), 6 * k) / .5
    r = river(h, w)
    n1, n2 = fft_noise(h, w, 1.9, SEED + 9), fft_noise(h, w, 1.9, SEED + 10)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = map_coordinates(r, [yy + 1.1 * n1, xx + 1.1 * n2], order=1, mode="nearest")
    gy, gx = np.gradient(gaussian_filter(r, 1.3))
    r = r + np.clip(np.hypot(gx, gy) * 1.2, 0, .3)
    r = r * (1 + .10 * fft_noise(h, w, .7, SEED + 11) * np.exp(-r))
    total = np.maximum(paper, 0) + D + r * (1 - .9 * high * (1 - gone))
    out = PAPER[None, None, :] * tone[..., None] * np.exp(-total[..., None] * INK_K[None, None, :])
    y_ = out.mean(-1) / 255
    print(f"{name.split('/')[-1]}: mean {y_.mean():.3f}  under.08 {(y_ < .08).mean() * 100:.1f}%  over.5 {(y_ >= .5).mean() * 100:.1f}%")
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(name + "_ink.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../ink/film_out/3d/new/edge")

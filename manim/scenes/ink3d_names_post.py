"""The rift across 申 in worn_shen. On the slab the crack goes in deep where it crosses the character on a diagonal, its lit wall facing the camera, and reads as a few specks there. So the
stretch through the box is painted over the plate as what it should look like from here, before the grade: a jagged dark line, chipped along its lips, a lit edge down its near side.
    python ink3d_names_post.py        writes names/worn_shen_rift.png, for ink3d_gns_grade.py --over"""
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from ink_grade import fft_noise

OUT = "../../ink/film_out/3d/names"
W, H = 1920, 1080
# through the frame: down from where the crack is already plain, across the box of 申 from upper right to lower left, out to where it is plain again
PATH = np.array([(838, 586), (836, 612), (829, 630), (813, 650), (806, 672), (797, 690), (788, 707), (778, 722), (769, 738), (756, 758), (750, 772)], np.float32)
r = np.random.default_rng(83)
seg = np.hypot(*np.diff(PATH, axis=0).T)
t = np.concatenate([[0], np.cumsum(seg)])
s = np.arange(0, t[-1], .5, dtype=np.float32)
x, y = np.interp(s, t, PATH[:, 0]), np.interp(s, t, PATH[:, 1])
tx, ty = np.gradient(gaussian_filter1d(x, 4)), np.gradient(gaussian_filter1d(y, 4))
n_ = np.hypot(tx, ty)
nx, ny = -ty / n_, tx / n_
# rock breaks in short straight runs: piecewise-linear wander, not a smooth wave
knots = np.arange(0, s[-1] + 9, 9.0)
off = np.interp(s, knots, r.uniform(-2.6, 2.6, len(knots)))
wid = 2.1 + .9 * np.interp(s, knots[::2], r.uniform(0, 1, len(knots[::2])))
wid = wid * np.clip(np.minimum(s, s[-1] - s) / 14.0, .35, 1)
x, y = x + off * nx, y + off * ny

X0, X1, Y0, Y1 = 730, 860, 570, 790
Y_, X_ = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
d, w_, side = np.full(X_.shape, 1e9, np.float32), np.zeros_like(X_), np.zeros_like(X_)
for i in range(len(s)):
    q = np.hypot(X_ - x[i], Y_ - y[i])
    better = q < d
    d[better], w_[better] = q[better], wid[i]
    side[better] = ((X_ - x[i]) * nx[i] + (Y_ - y[i]) * ny[i])[better]
sm = lambda v: np.clip(v, 0, 1) ** 2 * (3 - 2 * np.clip(v, 0, 1))
n1 = fft_noise(H, W, 1.2, 831)[Y0:Y1, X0:X1]
n2 = fft_noise(H, W, .6, 832)[Y0:Y1, X0:X1]
core = sm((w_ * (1 + .25 * n1) - d) / 1.1)
chips = sm((n2 - .9) / .4) * np.exp(-(d / (w_ + 3.5)) ** 2) * .85
shade = .30 * np.exp(-(d / (w_ + 5.0)) ** 2)
ink = np.clip(np.maximum.reduce([core, chips, shade]), 0, 1)
lip = sm((1.6 - np.abs(np.abs(side) - w_ - 1.4)) / 1.2) * (side * nx.mean() < 0) * (.42 + .25 * n1).clip(0, .7) * (1 - ink)
L = np.full((H, W), .03, np.float32)
A = np.zeros((H, W), np.float32)
L[Y0:Y1, X0:X1] = (.03 * ink + .88 * lip) / np.maximum(ink + lip, 1e-3)
A[Y0:Y1, X0:X1] = np.clip(ink + lip, 0, 1)
im = np.zeros((H, W, 4), np.uint8)
im[..., :3] = (np.clip(L, 0, 1)[..., None] * 255).astype(np.uint8)
im[..., 3] = (gaussian_filter(A, .5) * 255).astype(np.uint8)
Image.fromarray(im, "RGBA").save(os.path.join(OUT, "worn_shen_rift.png"))

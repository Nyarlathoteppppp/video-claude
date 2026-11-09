"""人 神 鬼 器 of 十二, large: written with a brush, stroke by stroke, from a table of centre lines made by hand. Their structure is taken from Xingkai SC's Light master, read off a grid, and then let out and drawn in as a running hand would. No font's outline is used. The hairs are those of ink_brush, but each lays down as much ink as it holds, so a dry stroke opens into streaks that follow the stroke, on the tooth of the sheet it is written on. The poem is another matter and is in film_hand.py.

    python film_big.py char 鬼               # one character alone, a whole frame
    python film_big.py four                  # the four on one sheet
    python film_big.py half 神               # the frame halfway through its writing
    python film_big.py strip 神 [8]          # frames of its writing side by side, or of its stroke 8 alone, to look at the head of the stroke
    python film_big.py crop gui_E            # parts of it at three times its size
    python film_big.py check                 # written, each comes to its still and only ever gets darker
    python film_big.py sheets [人神鬼器]      # hand/sheets/ren.png ...: each on its whole sheet, 340 by 470 mm at 6.1 px a mm, for the table

Everything goes to ink/film_out/film/hand/. No brush or hand is ever drawn: Writing(c, t0, t1).at(t) is the frame at time t, as in film_hand.py, and .ink(t) what it lays over that frame's paper.
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, map_coordinates

from film_hand import CLEAR, H, OUT, TOP, W, crop, lay, paper_tooth, save, sheet, tooth_under
from ink_brush import Brush, spline
from ink_grade import fft_noise, sm

TALL = 620                       # the most a character stands, in frame pixels: there is a picture to come round it
SIZE = 2400                      # the square it is worked out in


# ---------------------------------------------------------------- the brush

class Bristles(Brush):
    """The brush of ink_brush, the mark and its lean the same. But a hair is no longer on or off: it lays down what it holds, as it touches, and what comes of it is left to the paper. Fewer hairs touch as the brush is lifted, each leaving at its own height, so a tail ends in hairs and not in a point drawn with a ruler."""

    flat = 0.42            # how thick the mark is along the way it goes: thin, and the brush cuts a square head
    track = False          # whether to keep the way the last stroke went, for it to be watched growing

    def deposit(self, path, index, upto=1.0, ink=1.0, dry=1.0, step=0.05):
        """The ink a stroke lays down, as far as upto of its length: its box on the sheet and an array, about 1 where a full brush passes. dry: how fast it runs out, for a hand that moves quickly."""
        r = np.random.default_rng(self.seed * 1000 + index)
        q = spline([tuple(p) + (0.0,) * (4 - len(p)) for p in path], step)         # a point may say how fast the hand goes there: 1 is as fast as it can and still leave a mark
        c = 0.05 / step
        # the brush moves as much by pressing as by travelling: sampled evenly in both
        far = np.concatenate([[0], np.cumsum(np.hypot(np.hypot(*np.diff(q[:, :2], axis=0).T), self.body * 0.5 * np.abs(np.diff(q[:, 2]))))])
        at = np.arange(0, far[-1], step)
        q = np.stack([np.interp(at, far, q[:, i]) for i in range(4)], 1)
        q[:, :2] += 0.28 * self._drift(r, len(q), 2, c)
        q[:, 2] *= 1 + 0.1 * self._drift(r, len(q), 1, c)[:, 0]
        d = np.gradient(q[:, :2], axis=0)
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        n = gaussian_filter1d(np.stack([-d[:, 1], d[:, 0]], 1), 26 * c, axis=0, mode="nearest")
        rest = np.array([np.cos(self.lean), np.sin(self.lean)])
        # the mark keeps its side all the way: turning it over where a 撇 runs along the lean would twist the brush in a knot
        axis = 0.62 * n * (1.0 if (n * rest).sum(1).mean() > 0.3 else -1.0) + 0.38 * rest
        axis = gaussian_filter1d(axis, 8 * c, axis=0, mode="nearest")
        axis /= np.maximum(np.hypot(axis[:, 0], axis[:, 1]), 1e-6)[:, None]
        cross = np.stack([-axis[:, 1], axis[:, 0]], 1)
        L, N = len(q), self.hairs
        along = np.cumsum(np.hypot(*np.gradient(q[:, :2], axis=0).T))[None, :]
        P = q[None, :, 2]
        U = np.sort(r.uniform(-1, 1, N))[:, None]
        V = r.uniform(-1, 1, N)[:, None] * np.sqrt(np.maximum(1 - U * U, 0))
        # hairs go in tufts of no particular width, which hold their own share of the ink, part as the brush dries and come together again
        tufts = max(4, int(3 + 0.018 * N))
        tuft = np.searchsorted(np.sort(r.uniform(-1, 1, tufts - 1)), U[:, 0])
        mid = np.array([U[tuft == i].mean() if (tuft == i).any() else 0.0 for i in range(tufts)])[tuft][:, None]
        store = np.clip(r.normal(0, 0.55, tufts), -1, 1)[tuft][:, None]
        along_noise = lambda n, scale: (lambda v: v / np.maximum(np.abs(v).max(1, keepdims=True), 1e-6))(gaussian_filter1d(r.normal(0, 1, (n, L)), scale * c, axis=1, mode="nearest"))
        sway, skip = along_noise(tufts, 260)[tuft], along_noise(tufts, 46)[tuft]
        wander = along_noise(N, 34) * 0.05
        streak = along_noise(N, 220)
        fine = r.normal(0, 1, N)[:, None]
        rise = r.uniform(0, 0.55, N)[:, None]
        # however dry, a stroke goes where it was going: what is left on the fullest hairs lasts to the end
        used = np.minimum(0.016 * dry * along * (1.25 - P) + 0.62 * np.clip(q[None, :, 3], 0, 1), ink - 0.03 - 0.34 * np.clip(1 - P * 1.6, 0, 1))
        parch = np.clip((used - 0.12) / 0.55, 0, 1)
        A = self.body * 0.5 * P
        ou, ov = A * (mid + (U - mid) * (1 - 0.55 * parch) + 0.13 * parch * sway + wander * (0.6 + parch)), A * self.flat * V
        x = (q[None, :, 0] + axis[None, :, 0] * ou + cross[None, :, 0] * ov) * self.k
        y = (q[None, :, 1] + axis[None, :, 1] * ou + cross[None, :, 1] * ov) * self.k
        holds = ink + 0.2 - used - 0.25 * U * U * (1 - 0.7 * parch) + 0.36 * store + 0.34 * skip * (0.25 + parch) + 0.05 * streak + 0.05 * fine
        # the fuller hairs lay more, and a brush pressed and slow lays more than one going light and fast
        lay_ = sm((holds - 0.25) / 0.16) * sm((P - rise) / 0.04) * (1 + 0.26 * store) * (0.82 + 0.34 * np.minimum(P, 1.0))
        # and how strong it is: full black where it was dipped, going thin with the length it has come and the speed it went, the belly's reserve coming down when it is pressed
        run = along / max(along[0, -1], 1e-6)
        cloud = along_noise(1, 150)
        strong = 0.5 + 0.5 * ink - 0.0034 * dry * along - 0.1 * run - 0.2 * np.clip(q[None, :, 3], 0, 1) + 0.3 * (np.minimum(P, 1.1) - 0.5) + 0.07 * cloud + 0.09 * store + 0.08 * skip
        lay_[:, int(L * upto):] = 0
        seg = np.hypot(np.gradient(x, axis=1), np.gradient(y, axis=1))
        lay_ = lay_ * seg
        on = (lay_ > 0) & (x >= 0) & (y >= 0) & (x < self.size - 1) & (y < self.size - 1)
        if not on.any():
            return None
        m = int(self.k * 0.8) + 3
        x0, x1 = max(int(x[on].min()) - m, 0), min(int(x[on].max()) + m, self.size)
        y0, y1 = max(int(y[on].min()) - m, 0), min(int(y[on].max()) + m, self.size)
        w, h = x1 - x0, y1 - y0
        fx, fy, v = x[on] - x0, y[on] - y0, lay_[on]
        ix, iy = np.clip(fx.astype(np.int32), 0, w - 2), np.clip(fy.astype(np.int32), 0, h - 2)
        tx, ty = fx - ix, fy - iy
        cell = iy * w + ix
        if self.track:
            # how far along its way the brush was when it first laid ink on each pixel, the middle of the mark first and its sides following, the fuller tufts a little ahead of the others, so the head comes on in fingers along the stroke, though not round a turn, where fingers curl like an eyelash. And how long it took over each part: slow pressed down, quick where the hand was quick
            bend = gaussian_filter1d(np.hypot(*np.gradient(d, axis=0).T), 30 * c, mode="nearest")[None, :] / step
            first = np.full((h, w), np.inf, np.float32)
            np.minimum.at(first.reshape(-1), cell, (along + (0.8 * U * U - (0.6 * store + 0.2 * np.clip(fine, -2, 2)) / (1 + (4 * bend * A) ** 2)) * A)[on])
            end = int(L * upto)
            self.way = (y0, y1, x0, x1), first, along[0, :end], (0.7 + 0.7 * np.minimum(q[:end, 2], 1.1)) / (1 + 2.5 * np.clip(q[:end, 3], 0, 1)), float(np.median(A[0, :end]))

        def splat(u):
            return (np.bincount(cell, u * (1 - tx) * (1 - ty), w * h) + np.bincount(cell + 1, u * tx * (1 - ty), w * h) + np.bincount(cell + w, u * (1 - tx) * ty, w * h) + np.bincount(cell + w + 1, u * tx * ty, w * h)).reshape(h, w)
        raw = splat(v)
        soft = max(self.k * 3.0, 1.0)
        near, nearer = gaussian_filter(raw, soft), gaussian_filter(splat(v * strong[on]), soft)
        K = np.where(near > 1e-5, nearer / np.maximum(near, 1e-5), float(strong[on].mean()))
        raw = gaussian_filter(raw, max(self.k * 0.075, 0.6))
        return (y0, y1, x0, x1), np.float32(1.28 * raw * self.body * self.k / N), np.float32(K)


def _band(size, lo, hi, seed):
    """Noise of about one size of grain, as many pixels across as lo to hi."""
    v = np.random.default_rng(seed).normal(0, 1, (size, size)).astype(np.float32)
    v = gaussian_filter(v, lo) - gaussian_filter(v, hi)
    return v / v.std()


class Hand:
    """A brush at one size of character. size: the square's pixels when it is made. frame: how many frame pixels it will be shown at, since the paper's grain is the sheet's and not the character's. tooth: the grain of the sheet under the square, if it is known."""

    def __init__(self, size, seed=3, hairs=None, step=None, body=13.0, lean=-0.8, flat=0.42, frame=730, rough=0.6, tooth=None):
        self.size = size
        self.rough = rough       # how much the paper pushes an edge about
        self.k = size / 2400.0
        self.grain = size / frame
        self.step = step or float(np.clip(120.0 / size, 0.05, 0.3))
        self.brush = Bristles(size, body=body, seed=seed, lean=lean, hairs=hairs or int(np.clip(size / 3, 130, 800)))
        self.brush.flat = flat
        g = self.grain
        # the paper's tooth, which a dry brush only catches the top of; and how far its surface pushes an edge about
        self.tooth = tooth if tooth is not None else _band(size, 0.55 * g, 2.6 * g, seed + 11) * 0.7 + _band(size, 1.6 * g, 6 * g, seed + 12) * 0.5
        self.warp = _band(size, 0.8 * g, 3.5 * g, seed + 13), _band(size, 0.8 * g, 3.5 * g, seed + 14)
        self.fibre = fft_noise(size, size, 1.6, seed + 15) * 0.55 + fft_noise(size, size, 2.6, seed + 16) * 0.45

    def wet(self, stroke, tone=1.0, soak=1.0):
        """What a stroke's deposit comes to in the paper: its box, and how dark. Where there is ink to spare it closes up and goes a little beyond itself; where there is little it only catches the tooth. Strong ink is black all through; thin ink lets the paper's grain show in it."""
        (y0, y1, x0, x1), D, K = stroke
        h, w = D.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # a wet edge lies smooth in the paper, a dry one is pushed about by every bump in it
        far = self.rough * 0.42 * self.grain * (1 - 0.6 * np.clip(gaussian_filter(D, 3 * self.grain) / 1.1, 0, 1))
        D = map_coordinates(D, [yy + far * self.warp[0][y0:y1, x0:x1], xx + far * self.warp[1][y0:y1, x0:x1]], order=1, mode="nearest")
        tooth, fibre = self.tooth[y0:y1, x0:x1], self.fibre[y0:y1, x0:x1]
        s = lambda v: max(v * self.k, 0.35)
        C = sm((D - np.maximum(0.5 + 0.23 * self.rough * tooth, 0.2)) / 0.22)
        K = np.clip(K, 0.08, 1.05)
        thin = np.clip(1 - K, 0, 1)
        a = C * (0.26 + 0.74 * K ** 1.15) * (0.74 + 0.26 * (1 - np.exp(-2.4 * D))) * (1 - thin * (0.09 * np.clip(tooth, -1.4, 2) + 0.06 * fibre))
        wetness = gaussian_filter(C * np.minimum(D, 2.0) * 0.5, s(7))
        spread = sm((wetness - 0.13 + 0.07 * fibre + 0.05 * tooth) / 0.28) * (1 - C)
        return (y0, y1, x0, x1), np.clip((a + 0.17 * soak * spread * np.clip(K, 0, 1)) * tone, 0, 1)

    def lay(self, into, wetted):
        if wetted is not None:
            (y0, y1, x0, x1), a = wetted
            b = into[y0:y1, x0:x1]
            hi, lo = np.maximum(b, a), np.minimum(b, a)
            # wet on wet: where two strokes meet they run together, so neither's edge is left showing inside the other, and it goes only a little darker
            s = 1.5 * self.grain
            both = sm((lo - 0.08) / 0.2)
            meet = both * gaussian_filter(both, s * 0.5)
            out = hi + (gaussian_filter(hi, s) - hi) * meet
            into[y0:y1, x0:x1] = np.maximum(out + (1 - out) * gaussian_filter(lo, s) * 0.14 * meet, hi * (1 - meet))

    def stroke(self, into, path, index, ink=1.0, dry=1.0, tone=0.97, soak=1.0, upto=1.0):
        d = self.brush.deposit(path, index, upto=upto, ink=ink, dry=dry, step=self.step)
        if d is not None:
            self.lay(into, self.wet(d, tone, soak))

    def design(self, marks, tone=0.97, soak=1.0, index=0, each=None):
        """A character by hand: marks is a list of (path, ink, dry), and for a stroke that lays no ink past some part of its way, that part. each: told after every stroke what has been written so far and the way the stroke went."""
        out = np.zeros((self.size, self.size), np.float32)
        self.brush.track = each is not None
        hairs = 0       # a stroke that stops and the one that carries it on are one movement of one brush
        for path, ink, dry, *stop in marks:
            self.stroke(out, path, index * 37 + hairs, ink=ink, dry=dry, tone=tone, soak=soak, upto=stop[0] if stop else 1.0)
            hairs += 0 if stop else 1
            if each:
                each(out, self.brush.way)
        return out


# ---------------------------------------------------------------- the table: centre lines in a square a hundred wide, each point (x, y, pressure[, how fast]), and for each stroke its ink and how fast it runs out

GUI = dict(body=13.0, flat=0.42, secs=3.0, marks=[
    # the 撇点 on top, turning at its shoulder
    ([(38.5, 9.5, .1), (41.5, 11.5, .42), (45.2, 15.5, .78), (44.8, 19, .7), (40.5, 24, .42), (34.5, 29, .2, .3), (30.5, 32, .04)], 1.05, .8),
    # the box, small and shut: its left side
    ([(27.5, 33, .1), (26.5, 35.5, .55), (29.5, 39.5, .62), (32, 46, .42), (33.5, 53, .34), (34.5, 57.5, .16)], .95, 1),
    # the top and right in one, pressed at the turn
    ([(32.5, 31.5, .12), (41, 28.5, .3), (51, 25.5, .26), (58.5, 23.8, .4), (64, 25, .72), (65.2, 29, .66), (62.5, 37, .42), (59, 46, .36), (56.6, 52.5, .22), (56, 55, .06)], 1.0, 1),
    # the middle: down, across
    ([(41.5, 30, .12), (44.8, 33, .46), (45, 38, .46), (44, 46, .36), (43.3, 53.5, .24), (43.2, 56, .06)], .95, 1),
    ([(33.5, 43.5, .1), (37, 43.6, .36), (44, 40.5, .26), (51, 37.6, .3), (54.5, 37.6, .34), (55.5, 38.6, .06)], .92, 1),
    # the foot of the box, starting out past its left side
    ([(25, 60, .14), (28, 61.4, .4), (32, 60, .32), (38, 57, .24), (47, 53, .22), (54, 51, .3), (57.5, 51.2, .12)], .92, 1),
    # the long 撇, let go
    ([(36, 58, .2), (40, 60.5, .75), (41.5, 64, .9), (38.5, 70, .72), (32, 78, .5, .2), (23, 86, .32, .45), (13, 93, .12, .6), (7, 96.5, .02)], 1.05, .7),
    # the 竖弯钩: down, along the ground on what is left, pressed. Its way goes on up, for the hairs' sake, but it lays no ink past the press: the hook is the next stroke
    ([(47, 60, .2), (52, 59.5, .7), (55, 62, .92), (53, 68, .75), (49, 76, .62), (47, 83, .58), (49.5, 88.5, .52), (57, 91, .42, .2), (68, 91.5, .36, .4), (79, 90.6, .42, .45), (87.5, 88.5, .72, .2), (91, 85, .9), (90.4, 80, .62), (88.5, 72, .3, .4), (87.4, 65, .02)], 1.02, .65, .7),
    # the hook, one quick lift out of the press
    ([(81.5, 90.3, .36), (85, 89.8, .46), (88.4, 88.8, .6), (90.8, 87.2, .72), (91.8, 84.4, .68), (91.5, 80.5, .4), (90.3, 75.5, .24), (88.9, 70, .12, .35), (87.9, 65, .02, .6)], .74, .5),
    # 厶 tucked in under the box
    ([(62.5, 54.5, .12), (66, 56.5, .6), (67, 60, .62), (64, 66.5, .46), (60.5, 73, .42), (59.4, 76.8, .62), (61.6, 78, .4), (66, 75, .28), (71.5, 70.5, .18), (73, 69.5, .04)], .95, 1),
    ([(71, 66.5, .12), (74.5, 67, .62), (76, 71, .86), (74.8, 76.5, .6), (71.5, 81, .24), (69.6, 83.5, .02)], .92, 1),
])

QI = dict(body=13.0, flat=0.36, secs=3.4, marks=[
    # the upper left mouth: 左竖, 横折, and a foot that shuts it. Small and square
    ([(21.8, 11, .16), (23, 12.8, .42), (24, 16.5, .44), (25, 23, .32), (25.8, 27.5, .12)], 1.05, 1),
    ([(24, 12.4, .14), (29, 11.6, .28), (34, 10.6, .24), (37.8, 10, .34), (39.8, 10.4, .52), (40, 12.6, .5), (39.2, 17, .4), (38, 22, .32), (37.2, 26, .1)], 1.0, 1),
    ([(25.2, 25, .16), (29, 24.6, .3), (33, 23.9, .26), (36.5, 23.4, .3), (39, 23.4, .12)], .95, 1),
    # the upper right, higher and drier
    ([(57.8, 8, .14), (59, 10, .4), (59.8, 13.5, .42), (60.6, 20, .3), (61.4, 24.5, .1)], .95, 1.2),
    ([(60, 9.4, .12), (65, 8.6, .26), (70, 7.6, .22), (73.6, 7, .32), (75.6, 7.4, .5), (75.8, 9.6, .48), (75, 14, .38), (73.8, 19, .3), (73, 23, .08)], .92, 1.3),
    ([(61, 22, .14), (65, 21.6, .28), (69, 20.9, .24), (72.6, 20.4, .28), (75, 20.4, .1)], .9, 1.3),
    # 犬, the whole of the character: its beam, as heavy as its 撇, and turned back under itself at the end
    ([(7.2, 51.6, .18), (9.5, 50.8, .54), (12.5, 50.4, .72), (17, 50.2, .62), (30, 48.4, .46), (47, 45.8, .4, .06), (65, 43.4, .4, .1), (78, 41.8, .44, .08), (84, 41.4, .48), (89, 41.2, .52), (91.2, 42.4, .5), (90.6, 44.4, .28), (88, 45, .06)], 1.1, .32),
    # its 撇
    ([(44.2, 28, .16), (47, 29.2, .58), (49, 31.6, .84), (49.6, 35.5, .9), (47.5, 42, .78), (41.5, 51.5, .62), (31.5, 60.5, .42, .2), (20, 67.5, .22, .45), (9, 72.5, .03)], 1.12, .7),
    # its 捺, thin out of the 撇, heavy at the foot, and away, clear of the mouth below it
    ([(48.5, 46, .14), (53, 49.5, .3), (60, 54.5, .52), (68.5, 60, .78), (76, 64, .96, .1), (82, 66, .72, .35), (88, 67.2, .32, .55), (92.5, 67.6, .02)], 1.0, .75),
    # its dot: comes in on the point, is pressed, and is drawn back in
    ([(68, 30, .06), (70.4, 31.4, .26), (73.2, 33.6, .44), (75, 35.8, .54), (75.2, 37.8, .34), (73.8, 38.6, .06)], 1.05, 1),
    # the lower left mouth, a little larger
    ([(18.8, 76, .16), (20, 78, .44), (21, 81.5, .46), (22, 89, .34), (22.8, 94, .12)], 1.05, 1),
    ([(21, 77.4, .14), (27, 76.6, .3), (32.5, 75.6, .26), (36.6, 75, .36), (38.8, 75.4, .56), (39, 77.8, .52), (38.2, 82.5, .42), (37, 88, .34), (36.2, 92.2, .1)], 1.0, 1),
    ([(22.2, 91.2, .16), (27, 90.8, .32), (31.5, 90.1, .28), (35.5, 89.6, .32), (38, 89.6, .12)], .95, 1),
    # the lower right
    ([(57.8, 75.5, .14), (59, 77.5, .42), (60, 81, .44), (61, 88.5, .32), (61.8, 93.5, .1)], .95, 1.2),
    ([(60, 77, .12), (66, 76, .28), (72, 75, .24), (76.2, 74.5, .34), (78.6, 74.9, .52), (78.8, 77.2, .5), (78, 82, .4), (76.8, 87.5, .32), (76, 91.6, .08)], .92, 1.3),
    ([(61.2, 90.6, .14), (66, 90.2, .3), (70.5, 89.5, .26), (74.5, 89, .3), (77.4, 89, .1)], .9, 1.3),
])

REN = dict(body=14.0, flat=0.42, tall=560, secs=2.5, marks=[
    # the 撇 short and straight, its head cut on the slant
    ([(46.5, 13, .14), (48.6, 14.6, .46), (49.4, 17.5, .62), (48, 24, .6), (42.5, 34.5, .5), (35, 46, .38, .1), (27, 56.5, .24, .3), (19.5, 65.5, .03, .5)], 1.05, .7),
    # the 捺 long and sinking, gathering all the way, its foot pressed down and let out to the right, not set flat on the ground
    ([(43.2, 31.5, .1), (46, 36.5, .22), (50.5, 44, .36), (56.5, 52, .5), (64, 61, .68), (71.5, 69.5, .86), (78, 75.5, 1.02, .05), (84, 78.5, .92, .2), (90, 80.5, .5, .4), (95, 82, .16, .55), (98, 83, .02)], 1.02, .7),
])

SHEN = dict(body=13.0, flat=0.42, seed=11, secs=2.8, marks=[
    # 礻, narrow and quick: its dot
    ([(28, 14, .14), (33, 16.5, .46), (38, 20, .62), (39.5, 23, .44), (38, 25, .06)], 1.05, 1),
    # 横撇
    ([(13, 34, .22), (17, 33.5, .36), (25, 31.5, .3), (33, 29.5, .36), (37.5, 30, .6), (36.5, 34, .62), (31, 42, .5), (24, 51, .36), (15.5, 60.5, .16, .3), (9, 66.5, .02)], 1.0, .8),
    # 竖
    ([(27.8, 47.5, .2), (30, 49.5, .5), (30.5, 54, .52), (30.2, 64, .42), (29.8, 74, .3, .2), (29.4, 82, .16, .4), (29, 88, .02)], .95, .9),
    # 点
    ([(35, 50, .14), (38, 52.5, .42), (41, 56.5, .58), (41.5, 60, .38), (39.8, 61.8, .04)], .92, 1),
    # 申, shut close round the spine: its left side
    ([(46.2, 34.5, .2), (47.8, 36.2, .5), (48.8, 41, .48), (50.2, 50, .42), (51.8, 59, .34), (52.8, 65, .14)], 1.0, 1),
    # 横折
    ([(49, 35, .14), (55, 33.8, .3), (64, 31.6, .24), (72, 29.8, .26), (79, 29, .4), (83.4, 30, .66), (84, 33.5, .66), (82, 40, .52), (79, 49, .42), (76.6, 57, .36), (75.4, 62, .1)], .98, .7),
    # the two bars
    ([(52.5, 47, .12), (57, 46.8, .3), (65, 44.6, .22), (72, 43, .26), (76.5, 42.8, .3), (78, 43.6, .06)], .92, 1),
    ([(54.5, 61.5, .14), (59, 61, .34), (66, 59.4, .26), (73, 58, .3), (77.5, 57.8, .34), (79.5, 58.4, .08)], .9, 1),
    # the spine: the one long stroke, the whole of the character, let fall as far as the brush will go and running dry on the way down
    ([(60.5, 3, .16), (63, 4.5, .52), (65, 7.5, .8), (65.4, 12, .88), (64.8, 22, .78), (64.2, 36, .68, .04), (64, 50, .62, .08), (64.5, 65, .56, .18), (65, 78, .46, .3), (64.8, 88, .3, .38), (63.6, 97, .02)], 1.05, .54),
])

TABLE = dict(人=REN, 神=SHEN, 鬼=GUI, 器=QI)
NAMES = dict(人="ren", 神="shen", 鬼="gui", 器="qi")


def extent(table):
    """Where a table's ink goes in its square: left, top, right, bottom, near enough, from its centre lines and the width of the brush."""
    p = np.array([q[:3] for path, *_ in table["marks"] for q in path])
    half = table["body"] * 0.3 * p[:, 2]
    return (p[:, 0] - half).min(), (p[:, 1] - half).min(), (p[:, 0] + half).max(), (p[:, 1] + half).max()


def _hand(c, tooth, frame, x, y, size=SIZE, each=None):
    """c as a square image, written for its place in the frame: its square frame pixels wide with its corner at x, y."""
    table = TABLE[c]
    hand = Hand(size, body=table["body"], flat=table["flat"], frame=frame, tooth=tooth_under(tooth, x, y, frame, size))
    return hand.design(table["marks"], index=table.get("seed", "人神鬼器".index(c)), each=each)


def write(f, tooth, c, frame, x, y):
    """c written into the frame f."""
    lay(f, _hand(c, tooth, frame, x, y), x, y, frame)


def stands(f, label):
    a = f.mean(2) < 150
    ys, xs = np.nonzero(a.any(1))[0], np.nonzero(a.any(0))[0]
    print(label, "stands %d px, %d wide, y %d-%d, x %d-%d" % (ys[-1] - ys[0] + 1, xs[-1] - xs[0] + 1, ys[0], ys[-1], xs[0], xs[-1]), flush=True)


def alone(table):
    """Where one character goes alone in the frame, centred in the room above the subtitle's strip, no more than TALL high: x, y, frame pixels."""
    x0, y0, x1, y1 = extent(table)
    frame = int(table.get("tall", TALL) * 0.97 / (y1 - y0) * 100)
    return int(W / 2 - (x0 + x1) / 2 * frame / 100), int((TOP + H - CLEAR) / 2 - (y0 + y1) / 2 * frame / 100), frame


def one(c):
    """One character alone in the frame."""
    f = sheet()
    x, y, frame = alone(TABLE[c])
    write(f, paper_tooth(f), c, frame, x, y)
    stands(f, c)
    save(f, NAMES[c] + "_E")


def four(tall=300):
    """人 神 鬼 器, one sheet, left to right, above the subtitle: about tall pixels each, the gaps between the ink even, every one centred on the same line."""
    f = sheet()
    tooth = paper_tooth(f)
    order = "人神鬼器"
    boxes = [extent(TABLE[c]) for c in order]
    frames = [int(tall * (TABLE[c].get("tall", TALL) / TALL) * 0.97 / (b[3] - b[1]) * 100) for c, b in zip(order, boxes)]
    wide = [(b[2] - b[0]) * n / 100 for b, n in zip(boxes, frames)]
    gap = min(150.0, (W - 240 - sum(wide)) / 3)
    at = (W - sum(wide) - 3 * gap) / 2
    cy = (H - CLEAR) / 2
    for c, b, n, w_ in zip(order, boxes, frames, wide):
        write(f, tooth, c, n, int(at - b[0] * n / 100), int(cy - (b[1] + b[3]) / 2 * n / 100))
        at += w_ + gap
    stands(f, "four")
    save(f, "four_E")


# the sheets as they lie on the table (十二's pull back, 十三): 340 by 470 mm, the character as big on it as film_crab_new.py has it, 620 frame pixels standing about 230 mm
SHEET_MM, FRAME_PX_MM, SHEET_PX_MM = (340.0, 470.0), 2.7, 6.1


def flat(c):
    """c on its whole sheet, straight on, plain edges (whoever lays it down tears them): the ink's middle across in the middle, and a twenty-fifth of the sheet above the middle down, as film_crab_new.py places it. Into hand/sheets/NAME.png."""
    w, h = (int(round(v * SHEET_PX_MM)) for v in SHEET_MM)
    table = TABLE[c]
    x0, y0, x1, y1 = extent(table)
    frame = int(table.get("tall", TALL) * 0.97 / (y1 - y0) * 100 * SHEET_PX_MM / FRAME_PX_MM)
    f = sheet(w, h)
    tooth = paper_tooth(f)
    # where the ink really falls in its square, so that its middle can be put where it belongs
    ink = _hand(c, tooth, frame, 0, 0)
    ys, xs = np.nonzero(ink > 0.3)
    k = frame / ink.shape[0]
    cx, cy = (xs.min() + xs.max()) / 2 * k, (ys.min() + ys.max()) / 2 * k
    x, y = int(round(w / 2 - cx)), int(round(h / 2 - 0.04 * h - cy))
    lay(f, _hand(c, tooth, frame, x, y), x, y, frame)
    path = os.path.join(OUT, "sheets", NAMES[c] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(path)
    print(path, w, h, "frame", frame, flush=True)


# ---------------------------------------------------------------- the writing

class Writing:
    """c being written, by itself on the paper: at(t) is the frame at time t, ink(t) how much of the square at x, y, frame pixels wide, it covers then, for putting over something else. Each stroke is worked out once, with when the brush reached every pixel of it, so what shows of a stroke at any moment is only what the brush has been over, the streaks of a dry stroke opening as it gets to them and following it. At t1 it is exactly what one(c) has. place: x, y, frame pixels, if not alone in the frame."""

    def __init__(self, c, t0=0.0, t1=None, place=None):
        table = TABLE[c]
        self.paper = sheet()
        tooth = paper_tooth(self.paper)
        self.x, self.y, self.frame = x, y, n = place or alone(table)
        self.t0, self.t1 = t0, t0 + table["secs"] if t1 is None else t1
        small = lambda ink: np.asarray(Image.fromarray(np.float32(np.clip(ink, 0, 1)), "F").resize((n, n), Image.LANCZOS), np.float32).clip(0, 1)
        states, ways = [np.zeros((n, n), np.float32)], []

        def done(out, way):
            states.append(small(out))
            ways.append(way)
        _hand(c, tooth, n, x, y, each=done)
        # where two strokes run together the older one's edge softens a little: nothing is shown darker than it will be left
        for k in range(len(states) - 2, -1, -1):
            states[k] = np.minimum(states[k], states[k + 1])
        self.states = states
        self.ways = [self._way(*way, k, tooth_under(tooth, x, y, n, n)) for k, way in enumerate(ways)]
        self.times = self.timed(self.t0, self.t1, [len(m) > 3 for m in table["marks"]])

    def _way(self, box, first, along, pace, half, k, grain):
        """What it takes to grow stroke k: the part of the square it changes, how far along the stroke each pixel of that lies, how long the fading head of the stroke is, and how far along it has gone at each share of its time."""
        y0, y1, x0, x1 = box
        n, g = self.frame, SIZE / self.frame
        Y0, Y1, X0, X1 = max(int(y0 / g) - 4, 0), min(int(y1 / g) + 5, n), max(int(x0 / g) - 4, 0), min(int(x1 / g) + 5, n)
        near = distance_transform_edt(~np.isfinite(first), return_distances=False, return_indices=True)
        first = gaussian_filter(first[near[0], near[1]], 0.6 * g)
        yy, xx = np.mgrid[Y0:Y1, X0:X1]
        front = map_coordinates(first, [(yy + 0.5) * g - 0.5 - y0, (xx + 0.5) * g - 0.5 - x0], order=1, mode="nearest")
        head = max(1.6 * half, 2.0)
        # and a little unevenly, with the paper
        front = np.maximum(front - 0.04 * head * np.clip(gaussian_filter(grain, 1.0)[Y0:Y1, X0:X1], -2, 2), 0)
        end = float(front[self.states[k + 1][Y0:Y1, X0:X1] - self.states[k][Y0:Y1, X0:X1] > 0.002].max())
        s = np.concatenate([[0.0], along[along < end], [end, end + head]])
        rate = np.interp(s, along, pace)
        share = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s))])
        return (Y0, Y1, X0, X1), front, head, s, share / share[-1], end

    def timed(self, a, b, stops):
        """When each stroke is written, between a and b, as Glyph.timed has it: for as long as it is long, not quite in proportion, and the lift between strokes the hand needs, but for a stroke that stops only for the next to carry it on."""
        size = np.array([(way[-1] / 100) ** 0.72 + 0.07 for way in self.ways])
        lift = 0.1 * size.max()
        whole = size.sum() + lift * (len(size) - 1 - sum(stops))
        at, out = 0.0, []
        for v, stop in zip(size, stops):
            out.append((a + (b - a) * at / whole, a + (b - a) * (at + v) / whole))
            at += v + (0 if stop else lift)
        return out

    def ink(self, t):
        k = sum(e <= t for _, e in self.times)
        if k == len(self.times) or t <= self.times[k][0]:
            return self.states[k]
        (Y0, Y1, X0, X1), front, head, s, share, _ = self.ways[k]
        at = np.interp((t - self.times[k][0]) / (self.times[k][1] - self.times[k][0]), share, s)
        out = self.states[k].copy()
        out[Y0:Y1, X0:X1] += (self.states[k + 1][Y0:Y1, X0:X1] - out[Y0:Y1, X0:X1]) * sm((at - front) / head)
        return out

    def at(self, t):
        f = self.paper.copy()
        lay(f, self.ink(t), self.x, self.y, self.frame)
        return f


def half(c):
    """The frame halfway through c's writing: the stroke under way then, or if the hand is lifted the next, half written."""
    w = Writing(c)
    k = sum(e <= (w.t0 + w.t1) / 2 for _, e in w.times)
    t = sum(w.times[k]) / 2
    save(w.at(t), NAMES[c] + "_E_half")
    print(c, "at %.2f s of %.2f" % (t, w.t1), "; strokes", " ".join("%.2f-%.2f" % s for s in w.times), flush=True)


def strip(c, k=None, frames=12):
    """Frames of c's writing side by side, or of its stroke k alone: to see the head of a stroke."""
    w = Writing(c)
    a, b = (w.t0, w.t1) if k is None else w.times[k]
    ys, xs = np.nonzero(w.states[-1].max(1) > 0.1)[0], np.nonzero(w.states[-1].max(0) > 0.1)[0]
    y0, y1, x0, x1 = w.y + ys[0] - 12, w.y + ys[-1] + 12, w.x + xs[0] - 12, w.x + xs[-1] + 12
    tiles = []
    for t in np.linspace(a, b, frames + 2)[1:-1]:
        f = np.uint8(np.clip(w.at(float(t)) + 0.5, 0, 255))[y0:y1, x0:x1]
        tiles.append(np.asarray(Image.fromarray(f).resize(((x1 - x0) // 2, (y1 - y0) // 2), Image.LANCZOS)))
    per = frames // 3
    save(np.float32(np.concatenate([np.concatenate(tiles[i:i + per], 1) for i in range(0, frames, per)], 0)), NAMES[c] + "_E_strip" + ("" if k is None else "_%d" % k))


def check(fps=30):
    """Written, each comes to its still to within one in 255, only ever gets darker on the way, and no stroke turns up all at once."""
    for c in TABLE:
        w = Writing(c)
        f = sheet()
        write(f, paper_tooth(f), c, w.frame, w.x, w.y)
        assert np.abs(np.uint8(np.clip(w.at(w.t1) + 0.5, 0, 255)).astype(int) - np.uint8(np.clip(f + 0.5, 0, 255)).astype(int)).max() <= 1, c
        assert abs(w.times[-1][1] - w.t1) < 1e-6
        was = np.zeros_like(w.states[0])
        for t in np.arange(w.t0, w.t1 + 1 / fps, 1 / fps):
            now = w.ink(float(t))
            assert (now >= was - 1e-5).all(), (c, t)
            was = now
        assert np.array_equal(was, w.states[-1])
        most = []
        for k, (a, b) in enumerate(w.times):
            whole = float((w.states[k + 1] - w.states[k]).sum())
            seen = [float((w.ink(float(t)) - w.states[k]).sum()) for t in np.arange(a, b + 1 / fps, 1 / fps)]
            most.append(max(np.diff(np.concatenate([[0.0], seen, [whole]]))) / whole)
        assert max(most) < 0.6, (c, most)
        print("%s %.1f s, %d strokes, %.0f-%.0f frames each, at most %.0f%% of a stroke in one frame: ok" % (c, w.t1 - w.t0, len(w.times), min(e - s for s, e in w.times) * fps, max(e - s for s, e in w.times) * fps, 100 * max(most)), flush=True)


def main(args):
    if args[0] == "char":
        for ch in args[1:]:
            one(ch)
    elif args[0] == "four":
        four()
    elif args[0] == "half":
        half(args[1])
    elif args[0] == "strip":
        strip(args[1], *(int(v) for v in args[2:3]))
    elif args[0] == "crop":
        for n in args[1:]:
            crop(n)
    elif args[0] == "check":
        check()
    elif args[0] == "sheets":
        for ch in args[1:] or "人神鬼器":
            flat(ch)


if __name__ == "__main__":
    main(sys.argv[1:])

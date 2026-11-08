"""读书 — how the writing became me: every character looks back at the others, the next one is guessed, a wrong guess moves a few
numbers, and that is done with everything ever written, line by line, until what I write turns from nonsense into verse and then
into lines of my own. Storyboard lines look, lian, guess, fix, drip, tang.

Drawn frame by frame, with no simulation, so any frame can be rendered alone. Times are on the film's clock and come from the
storyboard, so they follow it when lines move.

    python ink_attn.py still 130 135 140      # PNGs and a contact sheet -> ink/film_out/stills_a/
    python ink_attn.py plate                  # ink/film_out/plate_attn.mp4
"""
import argparse
import math
import multiprocessing
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, uniform_filter1d

from ink_film import FPS, H, OUT, ROW, SECTIONS, W, develop, ease_out, font, glyph_mask, make_paper, noise2, number_block, rows, sm

STILLS = os.path.join(OUT, "stills_a")
START = dict(SECTIONS)["attn"]
END = dict(SECTIONS)["bed"] + 1.0
_r = rows()
LOOK, LIAN, GUESS, FIX, DRIP, TANG = (_r[ROW[k]]["a"] for k in ("look", "lian", "guess", "fix", "drip", "tang"))

# the line of verse, in two columns running right to left
POEM = "举头望明月低头思"
XR, XL, SIZE, CELL, Y0 = 1180, 700, 150, 176, 100
# how hard each character looks back at each earlier one (0-1)
LOOKS = {1: {0: .9},
         2: {0: .35, 1: .6},
         3: {0: .1, 1: .12, 2: .5},
         4: {0: .1, 1: .1, 2: .55, 3: 1.0},
         5: {0: .55, 1: .15, 2: .1, 3: .06, 4: .2},
         6: {0: .25, 1: .45, 2: .08, 3: .05, 4: .08, 5: .95},
         7: {0: .45, 1: .15, 2: .7, 3: .7, 4: .85, 5: .75, 6: .8}}
# when the characters land, as seconds into the line
LANDS = [.55, 1.17, 1.79, 2.41, 3.03, 3.6, 4.12, 4.62]
# the great strokes of 笔断意连, from 思: control points, half width, start and length in seconds into the line, how dry,
# the character it reaches and how dark that leaves it
SWEEPS = [((628, 505), (505, 430), (505, 250), (626, 200), 14, .15, .36, .70, 5, 3.25),
          ((632, 492), (575, 470), (575, 400), (634, 378), 17, .45, .28, .55, 6, 3.25),
          ((775, 565), (900, 640), (960, 760), (1105, 722), 20, .80, .40, .80, 3, 3.25),
          ((760, 605), (800, 820), (960, 950), (1105, 900), 27, 1.25, .50, .88, 4, 3.25),
          ((778, 530), (900, 420), (1010, 430), (1103, 525), 23, 1.80, .42, .84, 2, 3.25),
          ((768, 486), (800, 330), (950, 130), (1108, 172), 11, 2.30, .45, .90, 0, 2.2),
          ((775, 470), (900, 390), (1040, 350), (1108, 362), 3, 2.60, .30, .50, 1, 1.5)]
# the next character: where each candidate floats, how likely it looks before and after the correction, how thick its thread
CANDS = [("家", 400, 420, 3.3, .55, 5.5), ("故", 250, 590, 1.0, 3.4, 2.8), ("亲", 250, 780, .42, .34, 1.0), ("念", 410, 880, .2, .14, .6)]
# what the blank looks at
ASKS = {7: .95, 6: .5, 5: .35, 4: .75, 3: .45, 2: .5, 0: .2, 1: .08}
# what I read, a line at a time, each squeezed into a drop of ink: the line, where the drop lands, which places in the verse its ripple puts right
READS = [("太史公曰：余读孔氏书，想见其为人", 940, 540, (2, 7)),
         ("关关雎鸠，在河之洲", 1320, 716, (3,)),
         ("亲爱的，今晚月亮很好，你那边呢", 560, 364, (6,)),
         ("有人吗？楼主还在吗", 1040, 270, (0,)),
         ("本院判决如下", 840, 290, (5,)),
         ("明月几时有，把酒问青天", 1320, 364, (1,)),
         ("见字如面，家中一切都好", 1040, 892, (4,)),
         ("求助：猫不吃饭怎么办", 560, 716, (8,)),
         ("子曰：学而时习之", 300, 900, ()),
         ("大漠孤烟直，长河落日圆", 840, 892, (9,))]
# and the rest, which go by too fast to read
TORRENT = ["人生若只如初见", "天下大势，分久必合", "春江潮水连海平", "楼上说得对", "妈，我到了", "我想你了", "此致敬礼", "今天天气不错", "谢谢大家",
           "海内存知己", "下周一交", "记得吃饭", "欲穷千里目", "借过一下", "对不起", "床前明月光", "不用谢", "有没有人一起", "落霞与孤鹜齐飞", "请查收"]
# what I write before I have read: common characters in no order
JUNK = "的了在是有和人这中大为上个国不说们到地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从"
# the lines of my own at the end: character, time into the line, how long it takes to write, how much ink is left on the brush, how far it has run dry
CHANT = [("鬼", .60, .32, 3.8, 0), ("哭", .76, .32, 3.4, .06), ("当", 1.00, .32, 2.9, .14), ("年", 1.16, .32, 2.3, .26), ("缘", 1.40, .32, 3.5, .04), ("底", 1.55, .32, 2.8, .16), ("事", 1.70, .36, 1.9, .36),
         ("早", 2.30, .46, 3.8, 0), ("知", 2.55, .46, 3.4, .06), ("今", 2.90, .46, 2.7, .16), ("夜", 3.15, .50, 2.0, .32), ("字", 3.55, .50, 3.5, .05), ("成", 3.85, .50, 2.5, .22), ("精", 4.15, .65, 4.1, 0)]


def cell(i):
    """The middle of the i-th place in the verse; 8 is the blank below 思."""
    return (XR, Y0 + CELL / 2 + CELL * i) if i < 5 else (XL, Y0 + CELL / 2 + CELL * (i - 5))


def wander(r, shape, sigma):
    """Smooth noise along the last axis, std 1."""
    a = gaussian_filter1d(r.normal(size=shape), sigma, axis=-1, mode="reflect")
    return a / (a.std() + 1e-9)


def splat2(x, y, w, xa, ya, bw, bh):
    """Bilinear point splat onto a bh x bw grid with its corner at (xa, ya)."""
    x, y = x - xa, y - ya
    keep = (x >= 0) & (x < bw - 1) & (y >= 0) & (y < bh - 1)
    x, y, w = x[keep], y[keep], w[keep]
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - ix, y - iy
    i = iy * bw + ix
    size = bw * bh
    return (np.bincount(i, w * (1 - fx) * (1 - fy), size) + np.bincount(i + 1, w * fx * (1 - fy), size)
            + np.bincount(i + bw, w * (1 - fx) * fy, size) + np.bincount(i + bw + 1, w * fx * fy, size)).reshape(bh, bw)


class Silk:
    """One thread drawn with a real brush: separate hairs that wander and, when the brush runs dry, break into 飞白. Kept as the
    pixels it wets: their density, and how far along the stroke each one is, so it can be drawn on at any speed."""

    def __init__(self, ctrl, half, depth, dry=0.0, seed=0, spatter=0, belly=.85):
        r = np.random.default_rng(seed)
        p = np.array(ctrl, np.float64)
        n = max(int(np.hypot(*np.diff(p, axis=0).T).sum() * 1.6), 30)
        s = np.linspace(0, 1, n)
        q = s[:, None]
        b = (1 - q) ** 3 * p[0] + 3 * (1 - q) ** 2 * q * p[1] + 3 * (1 - q) * q ** 2 * p[2] + q ** 3 * p[3]
        d = np.gradient(b, axis=0)
        d /= np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9
        nx, ny = -d[:, 1], d[:, 0]
        trem = wander(r, (n,), max(n * .07, 8)) * (1.4 + half * .03) * np.sin(np.pi * s)
        cx, cy = b[:, 0] + nx * trem, b[:, 1] + ny * trem
        hw = half * np.sin(np.pi * s ** belly) ** .9 + .3
        nh = max(int(half * 3), 2)
        o = np.linspace(-1, 1, nh) + r.normal(0, .35 / nh, nh)
        ow = wander(r, (nh, n), max(n * .08, 8)) * .025
        pres = gaussian_filter1d(wander(r, (nh, n), max(min(150, n * .2), 6)), max(nh * .05, .8), axis=0)
        pres /= pres.std()
        dp = dry * sm((s - .05) / .4)
        a = sm((pres + 1.7 - 2.5 * dp - 1.2 * dp * np.abs(o)[:, None]) / .3)
        x = cx[None, :] + nx[None, :] * (o[:, None] + ow) * hw[None, :]
        y = cy[None, :] + ny[None, :] * (o[:, None] + ow) * hw[None, :]
        w = a * .55
        ss = np.broadcast_to(s, x.shape)
        xs, ys, ws, vs = [x.ravel()], [y.ravel()], [w.ravel()], [ss.ravel()]
        for _ in range(spatter):      # ink flung off the stroke as it passes
            k = int(r.uniform(.25, 1) * (n - 1))
            off = r.normal(0, 2.2) * (hw[k] + 6)
            m = 40
            rad = r.uniform(.7, 3.0)
            xs.append(cx[k] + nx[k] * off + r.normal(0, rad, m))
            ys.append(cy[k] + ny[k] * off + r.normal(0, rad, m))
            ws.append(np.full(m, .9 / max(rad, 1)))
            vs.append(np.full(m, s[k]))
        x, y, w, sv = (np.concatenate(v) for v in (xs, ys, ws, vs))
        xa, ya = max(int(x.min()) - 3, 0), max(int(y.min()) - 3, 0)
        xb, yb = min(int(x.max()) + 4, W), min(int(y.max()) + 4, H)
        bw, bh = max(xb - xa, 2), max(yb - ya, 2)
        cov = gaussian_filter(splat2(x, y, w, xa, ya, bw, bh), .55)
        mix = gaussian_filter(splat2(x, y, w * sv, xa, ya, bw, bh), .55)
        v = np.clip(cov, 0, 1) ** .85
        mottle = gaussian_filter(r.normal(size=cov.shape), 6)
        mottle = 1 + .16 * mottle / (mottle.std() + 1e-9)
        keep = v > .015
        yy, xx = np.nonzero(keep)
        self.idx = np.ravel_multi_index((yy + ya, xx + xa), (H, W))
        self.val = (v[keep] * mottle[keep] * depth).astype(np.float32)
        self.s = (mix[keep] / (cov[keep] + 1e-6)).astype(np.float32)

    def add(self, flat, prog, alpha=1.0):
        if prog > 0 and alpha > .003:
            flat[self.idx] += self.val * sm((prog * 1.08 - self.s) / .06) * alpha


def route(a, b, r):
    """A curved path between two places in the verse, leaving and arriving at the edge of the characters."""
    ax, ay = cell(a)
    bx, by = cell(b)
    if ax == bx:     # same column: bulge towards the gap between the columns
        side = -1 if ax == XR else 1
        v = 1 if by > ay else -1
        bulge = 38 + .26 * abs(by - ay) + r.uniform(-10, 10)
        p0, p3 = (ax + side * 58, ay + v * 34), (bx + side * 58, by - v * 34)
        return p0, (p0[0] + side * bulge, p0[1] + (by - ay) * .34), (p3[0] + side * bulge, p3[1] - (by - ay) * .34), p3
    d = 1 if bx > ax else -1
    p0, p3 = (ax + d * 74, ay + r.uniform(-32, 32)), (bx - d * 74, by + r.uniform(-32, 32))
    dx = p3[0] - p0[0]
    sag = r.uniform(-.3, .3) * (abs(by - ay) + 90)
    return p0, (p0[0] + dx * .42, p0[1] + sag), (p3[0] - dx * .42, p3[1] - sag), p3


def blit(hi, a, x, y):
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    xa, xb, ya, yb = max(x, 0), min(x + w, W), max(y, 0), min(y + h, H)
    if xa < xb and ya < yb:
        hi[ya:yb, xa:xb] += a[ya - y:yb - y, xa - x:xb - x]


class Char:
    """A big character in wet ink: rough at the edge, feathering into the fibres, with a bloom that dries back to a faint rim.
    heart: the ink spreads out from the middle. write: it runs down the character, as if an unseen brush were writing it."""

    def __init__(self, ch, kind, size, cx, cy, mode="heart", seed=0):
        self.cx, self.cy = cx, cy
        pad = size // 4
        m = glyph_mask(ch, kind, size, pad)
        s = m.shape[0]
        r = np.random.default_rng(seed + ord(ch))

        def noise(sigma):
            a = gaussian_filter(r.normal(size=m.shape), sigma)
            return a / a.std()

        rough = sm((gaussian_filter(m, 1.1) + .09 * noise(1.6) - .5) / .16)
        self.bleed = sm((gaussian_filter(m, size * .022 + .8) * (1 + .7 * noise(3)) - .08) / .4).astype(np.float32)
        gy, gx = np.gradient(gaussian_filter(m, size * .03))
        along = gaussian_filter(gx * gx / (gx * gx + gy * gy + 1e-6), 2)      # the brush leaves its streaks along the stroke
        tex = along * noise((7, .8)) + (1 - along) * noise((.8, 7))
        self.m = (rough * (1 - .07 * np.clip(tex, -1.5, 1.5))).astype(np.float32)
        self.skip = sm((tex + .55 * noise(size * .16) - .25) / .5).astype(np.float32)
        lump = np.clip(.62 + .38 * noise(size * .09), 0, 1.4)
        self.halo = (gaussian_filter(m, size * .05) * lump).astype(np.float32)
        self.far = (gaussian_filter(m, size * .13) * lump).astype(np.float32)
        yy, xx = np.mgrid[:s, :s].astype(np.float32)
        arr = np.hypot(xx - s / 2, yy - s / 2) / (size * .72) if mode == "heart" else (.78 * (yy - pad) + .22 * (xx - pad)) / size
        self.arr = np.clip(arr + .10 * noise(size * .05), 0, 1.05).astype(np.float32)
        self.s = s

    def density(self, p, age, depth, dry=0.0, bloom=.10):
        a = sm((p * 1.3 - self.arr) / .25)
        age = max(age, 0)
        return (self.m * depth * (1 - dry * self.skip) + self.bleed * depth * .055 + self.halo * (.10 + .5 * math.exp(-age / .7)) + self.far * bloom * math.exp(-age / .9)) * a

    def draw(self, hi, p, age, depth=3.3, dry=0.0, bloom=.10):
        if p > 0:
            blit(hi, self.density(p, age, depth, dry, bloom), self.cx - self.s / 2, self.cy - self.s / 2)


def teardrop(a):
    """A drop of ink falling, and the highlight on it."""
    ss = 3
    A = a * ss
    n, w = int(a * 3.9) * ss, int(a * 2.4) * ss
    yy, xx = np.mgrid[:n, :w].astype(np.float32)
    cy = n - A * 1.12
    v = yy - cy
    half = np.where(v > 0, A * np.sqrt(np.clip(1 - (v / A) ** 2, 0, 1)), A * np.clip(1 + v / (2.6 * A), 0, 1) ** 1.5)
    m = gaussian_filter((np.abs(xx - w / 2) < half).astype(np.float32), ss * .55)
    shine = np.exp(-(((xx - w / 2 + A * .38) / (A * .11)) ** 2 + ((yy - cy + A * .28) / (A * .22)) ** 2))
    pick = (slice(ss // 2, None, ss), slice(ss // 2, None, ss))
    return m[pick].astype(np.float32), shine[pick].astype(np.float32)


def streak(length):
    """Rain: a slanted line, dark at the head and gone by the tail."""
    w = int(length * .07) + 8
    yy, xx = np.mgrid[:length, :w].astype(np.float32)
    cx = 4 + (length - 1 - yy) * .07
    return (np.exp(-((xx - cx) / .95) ** 2) * (yy / length) ** 1.4 * .95).astype(np.float32)


class Attn:
    def __init__(self):
        self.detail, self.tone = make_paper()
        n = gaussian_filter(np.asarray(Image.fromarray(noise2(270, 480, 2.4, 91)).resize((W, H), Image.BICUBIC), np.float32), 2)
        self.n01 = np.clip(.5 + .26 * n, 0, 1)
        self._walls()
        self._verse()
        self._next_char()
        self._rain()
        self._reading()
        self._column_plan()
        self.chant = []
        for k, (ch, dt, dur, depth, dry) in enumerate(CHANT):
            col = XR if k < 7 else XL
            self.chant.append((Char(ch, "xing", 132, col, 128 + 139 * (k % 7), "write", 40 + k), TANG + dt, dur, depth, dry))

    def _walls(self):
        self.walls = []
        for seed in (13, 14, 15):
            f = np.zeros((H, W), np.float32)
            f[34:1044, 40:1880] = number_block(1840, 1010, seed) * 2.0
            self.walls.append(f)

    def _verse(self):
        r = np.random.default_rng(3)
        self.chars = []
        self.lands = [LOOK + x for x in LANDS]
        for i, ch in enumerate(POEM):
            cx, cy = cell(i)
            self.chars.append(Char(ch, "kaih", SIZE, cx, cy, "heart", i))
        self.look = []
        for i in range(1, len(POEM)):
            threads = []
            for j, q in LOOKS[i].items():
                half = .55 + 9 * q ** 1.5
                threads.append(Silk(route(i, j, r), half, .25 + 1.9 * q, dry=.25 * q, seed=100 * i + j, spatter=int(q * 3)))
            self.look.append((self.lands[i], self.lands[i + 1] if i + 1 < len(POEM) else LIAN, threads))
        self.sweeps = []
        for k, s in enumerate(SWEEPS):
            d = np.array(s[3], float) - np.array(s[0], float)
            self.sweeps.append(dict(silk=Silk(s[:4], s[4], 2.8, dry=s[7], seed=300 + k, spatter=14 if s[4] > 12 else 2, belly=.5), t0=LIAN + s[5], dur=s[6], half=s[4],
                                    dir=d / np.hypot(*d), to=s[8], depth=s[9]))

    def _next_char(self):
        r = np.random.default_rng(11)
        bx, by = cell(8)
        self.blank = []
        h = 66
        corners = [((bx - h, by - h), (bx + h, by - h)), ((bx + h, by - h), (bx + h, by + h)), ((bx + h, by + h), (bx - h, by + h)), ((bx - h, by + h), (bx - h, by - h))]
        for k, (a, b) in enumerate(corners):      # three dashes a side, like a space left for a word
            for d in range(3):
                u0, u1 = d / 3 + .04, d / 3 + .26
                p0 = (a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0)
                p3 = (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)
                mid = ((p0[0] * 2 + p3[0]) / 3, (p0[1] * 2 + p3[1]) / 3)
                mid2 = ((p0[0] + p3[0] * 2) / 3, (p0[1] + p3[1] * 2) / 3)
                self.blank.append(Silk((p0, mid, mid2, p3), 2.0, 1.5, dry=.2, seed=500 + k * 3 + d))
        self.asks = []
        for j, q in ASKS.items():
            self.asks.append((Silk(route(8, j, r), .55 + 7 * q ** 1.5, .25 + 1.5 * q, dry=.3 * q, seed=600 + j), r.uniform(0, 6.28)))
        self.cands = []
        self.cand_threads = []
        for k, (ch, x, y, _, _, thick) in enumerate(CANDS):
            self.cands.append(Char(ch, "kaih", 124, x, y, "heart", 70 + k))
            p0, p3 = (bx - h - 6, by - 20 + k * 14), (x + 68, y + 8)
            mx = (p0[0] + p3[0]) / 2
            self.cand_threads.append(Silk((p0, (mx, p0[1] + r.uniform(-40, 40)), (mx, p3[1] + r.uniform(-40, 40)), p3), thick, .3 + thick * .38, dry=.3, seed=700 + k))
        self.gu = Char("故", "kaih", SIZE, bx, by, "heart", 80)
        self.jia = Char("家", "kaih", SIZE, bx, by, "heart", 81)
        self._water(bx, by)
        self._flung(bx, by, r)
        self._strip(r)

    def _water(self, bx, by):
        """The stroke of water that takes 家 away: it starts under the blank and runs out to the left, leaving a pale tide mark."""
        r = np.random.default_rng(83)
        h_, w_ = 300, 470
        self.wx0, self.wy0 = bx + 120 - w_, by + 10 - h_ // 2
        yy, xx = np.mgrid[:h_, :w_].astype(np.float32)
        s = xx / w_
        wob = gaussian_filter(r.normal(size=(h_, w_)), 20)
        wide = 96 * np.sin(np.pi * s ** .75) ** .6 + 1
        d = np.abs(yy - h_ / 2 - 12 * np.sin(s * 3.1 + .4)) / wide + .10 * wob / wob.std()
        cloud = gaussian_filter(r.normal(size=(h_, w_)), (9, 30))
        cloud /= cloud.std()
        self.band = (np.clip(.075 + .035 * cloud, .015, None) * sm((1 - d) / .05) + .2 * (.7 + .3 * cloud) * np.exp(-((d - 1) / .03) ** 2)).astype(np.float32)
        self.band_in = sm((1.15 - d) / .2).astype(np.float32)
        self.band_s = (1 - s).astype(np.float32)
        self.wet = np.zeros((h_, w_), np.float32)
        s_ = self.jia.s
        x, y = int(bx - s_ // 2 - self.wx0), int(by - s_ // 2 - self.wy0)
        self.wet[y:y + s_, x:x + s_] = self.jia.m * 3.3
        self.wx = gaussian_filter(r.normal(size=(h_, w_)), 16).astype(np.float32)
        self.wy = gaussian_filter(r.normal(size=(h_, w_)), 16).astype(np.float32)
        self.wx /= self.wx.std()
        self.wy /= self.wy.std()
        self.wgrid = (yy, xx)

    def _flung(self, bx, by, r):
        """The ink thrown out when 故 lands: small and irregular, stretched along the way each drop was going."""
        dots = np.zeros((H, W), np.float32)
        for _ in range(58):
            a, dist = r.uniform(0, 6.28), r.uniform(92, 300)
            x, y = bx + dist * math.cos(a), by + dist * .8 * math.sin(a)
            rad, stretch = r.uniform(.6, 2.6) * (1.3 - dist / 380), r.uniform(1, 2.6)
            R = int(rad * stretch * 3) + 2
            ix, iy = int(x), int(y)
            if R < ix < W - R and R < iy < 965:
                gy, gx = np.mgrid[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(np.float32)
                gx, gy = gx - x, gy - y
                along, across = (gx * math.cos(a) + gy * math.sin(a)) / (rad * stretch), (-gx * math.sin(a) + gy * math.cos(a)) / rad
                dots[iy - R:iy + R + 1, ix - R:ix + R + 1] += np.exp(-along * along - across * across) * r.uniform(1, 2.6)
        self.dots = dots

    def _strip(self, r):
        """The strip of numbers along the foot of the page; a few of them change when the guess is corrected."""
        self.nums = []
        f = font("mono", 26)
        gap = 22
        for line in range(2):
            x = 170.0
            while True:
                txt = f"{r.normal(0, .6):+.4f}"
                if x + f.getlength(txt) > 1420:
                    break
                self.nums.append(dict(x=x, y=984 + line * 36, txt=txt, tone=r.uniform(.5, 1), seed=int(r.integers(1 << 30))))
                x += f.getlength(txt) + gap
        self.flip = [k for k, m in enumerate(self.nums) if m["x"] < 1000][::3][:7]
        self.flip_threads = []
        for k in self.flip:
            m = self.nums[k]
            target = CANDS[1] if k % 2 == 0 else CANDS[0]
            p0 = (m["x"] + 50, m["y"] - 3)
            p3 = (target[1] + r.uniform(-30, 30), target[2] + 70)
            self.flip_threads.append(Silk((p0, (p0[0] + r.uniform(-60, 60), p0[1] - 160), (p3[0] + r.uniform(-50, 50), p3[1] + 170), p3), 1.5, 1.05, dry=.35, seed=800 + k, spatter=2))

    def _rain(self):
        r = np.random.default_rng(29)
        self.drops = []
        t, gap = DRIP + 1.3, .75
        for k, (_, x, y, _) in enumerate(READS):
            big = k < 4
            self.drops.append(dict(t=t, x=float(x), y=float(y), a=max(10.0, 40 * .86 ** k), fall=.46 if k == 0 else max(.11, .40 * .86 ** k), speed=520.0 if big else 400.0,
                                   reach=1800.0 if big else max(170.0, 1800 * .76 ** k), wid=36.0 if big else max(13.0, 36 * .86 ** k), amp=1.0 if big else max(.42, .9 ** k), big=big))
            t += gap
            gap *= .78
        t = DRIP + 3.6      # then it rains
        while t < DRIP + 4.75:
            self.drops.append(dict(t=t, x=r.uniform(60, 1420), y=r.uniform(130, 1040), a=7.0, fall=.11, speed=400.0, reach=r.uniform(130, 210), wid=13.0, amp=.42, big=False))
            t += 1 / (12 + 130 * ((t - DRIP - 3.6) / 1.15) ** 2)
        self.sprites = {}
        self.streaks = streak(120)
        self.G = 3
        self.gw, self.gh = W // self.G, H // self.G
        self.T = int(1800 / self.G) + 3
        ty, tx = np.mgrid[-self.T:self.T + 1, -self.T:self.T + 1].astype(np.float32)
        ty *= self.G
        tx *= self.G
        self.td = np.hypot(tx, ty) + 1e-3
        self.ux, self.uy = tx / self.td, ty / self.td
        self.yy, self.xx = np.mgrid[:H, :W].astype(np.float32)
        self.splash = []
        for k in range(3):
            ang = r.uniform(-math.pi, 0, 14)
            speed = r.uniform(120, 620 - 120 * k, 14)
            self.splash.append(np.stack([np.cos(ang) * speed, np.sin(ang) * speed, r.uniform(1.1, 3.4 - k * .5, 14)], 1))
        # the wall keeps off the strip where the lines are read, the two columns, and the margin for the words at the right
        n = (self.n01 - .5) * 46
        off = [sm((118 + n - self.yy) / 26), sm((self.xx - 1440 + n) / 40), sm((100 + n - np.abs(self.xx - XR)) / 30), sm((100 + n - np.abs(self.xx - XL)) / 30)]
        self.room = (1 - np.maximum.reduce(off)).astype(np.float32)

    def _reading(self):
        """Each line: when it appears, starts to squeeze and lets go."""
        self.reads = []
        for k, (line, x, _, _) in enumerate(READS):
            o = self.drops[k]
            f = font("kaib", 58 if k < 4 else 50)
            w_ = int(f.getlength(line)) + 14
            im = Image.new("L", (w_, 84), 0)
            ImageDraw.Draw(im).text((7, 8), line, 255, font=f)
            tf = o["t"] - o["fall"]
            ts = tf - max(.1, .26 * .88 ** k)
            self.reads.append(dict(m=gaussian_filter(np.asarray(im, np.float32) / 255, .6), ta=ts - max(.16, .62 * .86 ** k), ts=ts, tf=tf, xc=min(max(x, w_ / 2 + 50), 1440 - w_ / 2), x=x, a=o["a"]))

    def _column_plan(self):
        """What stands at each of the ten places of the couplet, and when it changes: the verse breaks into nonsense, each ripple that
        passes turns a character into another wrong one, until the ripple meant for that place puts it right."""
        r = np.random.default_rng(57)
        finals = self.chars + [self.gu, Char("乡", "kaih", SIZE, *cell(9), "heart", 82)]
        fixed = {i: self.drops[k] for k, read in enumerate(READS) for i in read[3]}
        self.plan = []
        for i, c in enumerate(finals):
            cx, cy = cell(i)
            junk = [Char(ch, "kaih", SIZE, cx, cy, "heart", 210 + i) for ch in r.choice(list(JUNK), 5, replace=False)]
            o = fixed[i]
            t_fix = o["t"] + math.hypot(o["x"] - cx, o["y"] - cy) / o["speed"]
            crack = DRIP - .25 + r.uniform(0, .4)
            ev = [] if i == 9 else [(-1e9, c, 3.6 if i in (7, 8) else 3.25, False)]
            for n in range(3):
                ev.append((crack + n * .07, junk[n], r.uniform(1.5, 2.2), True))
            n = 3
            for q in self.drops:
                d = math.hypot(q["x"] - cx, q["y"] - cy)
                arrive = q["t"] + d / q["speed"]
                if d < q["reach"] and ev[-1][0] + .24 < arrive < t_fix - .12:
                    ev.append((arrive, junk[n % 5], r.uniform(1.5, 2.2), True))
                    n += 1
            ev.append((t_fix, c, 3.4, False))
            self.plan.append(ev)

    # ---- the verse, the threads, the guess --------------------------------------------------------------------------------------
    def _numbers(self, hi, t):
        a = float(sm((t - (FIX + .7)) / .5)) * (1 - float(sm((t - (DRIP - .5)) / .6)))
        if a <= 0:
            return
        f = font("mono", 26)
        t_flip = FIX + 1.5
        base = Image.new("L", (W, 96), 0)
        hot = Image.new("L", (W, 96), 0)
        db, dh = ImageDraw.Draw(base), ImageDraw.Draw(hot)
        for k, m in enumerate(self.nums):
            y = m["y"] - 984
            if k in self.flip:
                u = (t - t_flip - self.flip.index(k) * .07) / .75
                if u >= 0:
                    tick = min(int(u * 14), 14)
                    v = np.random.default_rng(m["seed"] + tick).normal(0, .6)
                    dh.text((m["x"], y), f"{v:+.4f}", int(255 * (.22 + .78 * math.exp(-max(u - 1, 0) * 1.5))), font=f)
                    if u > 1:
                        db.text((m["x"], y), f"{v:+.4f}", int(255 * m["tone"] * .5), font=f)
                    continue
            db.text((m["x"], y), m["txt"], int(255 * m["tone"]), font=f)
        s0 = gaussian_filter(np.asarray(base, np.float32) / 255, .45)
        s1 = gaussian_filter(np.asarray(hot, np.float32) / 255, .45)
        hi[984:1080, :] += (s0 * 1.55 + s1 * 3.6) * a

    def _verse_depth(self, i, t):
        """While 思 looks back the verse goes pale, and each character comes back as a stroke reaches it: as dark as it was looked at."""
        full = 3.6 if i == 7 else 3.25
        if i == 7 or t < LIAN:
            return full, .10
        sw = next(s for s in self.sweeps if s["to"] == i)
        hit = sw["t0"] + sw["dur"] * .7
        d = full + (1.15 - full) * float(sm((t - LIAN) / .35))
        d += (sw["depth"] - d) * float(sm((t - hit) / .16))
        d += (full - d) * float(sm((t - GUESS) / .7))
        return d, .55 * math.exp(-(t - hit) / .5) if t > hit else 0.0

    def _wash(self, hi, t, t_slam):
        """The water runs left from under 故, carrying 家 with it."""
        p = (t - t_slam) / .55
        al = 1 - .55 * float(sm((t - t_slam - 1.2) / 1.6))
        d = self.band * sm((p * 1.25 - self.band_s) / .12) * al
        u = (t - t_slam) / 1.1
        if u < 1:
            yy, xx = self.wgrid
            amp = 26 * u
            w = map_coordinates(self.wet, [yy + self.wy * amp, xx + self.wx * amp + 230 * u ** 1.5], order=1, mode="constant").astype(np.float32)
            d = d + gaussian_filter(w, (1 + 9 * u, 1 + 20 * u)) * (1 - float(sm(u))) * (self.band_in + (1 - self.band_in) * (1 - float(sm(u / .3))))
        blit(hi, d, self.wx0, self.wy0)

    def _poem(self, t):
        hi = np.zeros((H, W), np.float32)
        flat = hi.ravel()
        for t0, t_next, threads in self.look:
            if t >= t0:
                prog = float(ease_out((t - t0 - .05) / .42))
                al = 1 - .9 * float(sm((t - t_next - .05) / .5))
                al *= 1 - .75 * float(sm((t - GUESS) / .8))
                for th in threads:
                    th.add(flat, prog, al)
        dx, dy = 0.0, 0.0
        for sw in self.sweeps:
            if t >= sw["t0"]:
                al = 1 - .78 * float(sm((t - GUESS) / .9))
                al *= 1 - .6 * float(sm((t - FIX - 1.0) / 1.0))
                sw["silk"].add(flat, float(ease_out((t - sw["t0"]) / sw["dur"])), al)
                kick = sw["half"] * .24 * math.exp(-(t - sw["t0"]) * 6) * float(sm((t - sw["t0"]) / .07))     # the page gives a little under each stroke
                dx += kick * sw["dir"][0]
                dy += kick * sw["dir"][1]
        t_slam = FIX + 1.05
        if t >= GUESS:
            for th in self.blank:
                th.add(flat, (t - GUESS - .12) / .5, 1 - float(sm((t - t_slam) / .12)))
            for th, ph in self.asks:
                breath = .55 + .45 * math.sin(2 * math.pi * .85 * t + ph)
                th.add(flat, (t - GUESS - .4) / .6, breath * (1 - float(sm((t - t_slam + .1) / .4))))
            t_upd = FIX + 1.7
            for k, (c, spec) in enumerate(zip(self.cands, CANDS)):
                t0 = GUESS + .55 + k * .28
                self.cand_threads[k].add(flat, (t - t0) / .35, (1 - float(sm((t - FIX) / .5))))
                d = spec[3]
                if k == 0:      # 家 has left for the blank; its place holds only a ghost
                    d += (1.3 - d) * float(sm((t - FIX - .05) / .25))
                d += (spec[4] - d) * float(sm((t - t_upd) / 1.0))
                d *= 1 + .05 * math.sin(2 * math.pi * .85 * t + k)
                c.draw(hi, (t - t0) / .5, t - t0, d)
            if t >= FIX + 1.6:
                al = 1 - float(sm((t - FIX - 3.0) / .7))
                for th in self.flip_threads:
                    th.add(flat, (t - FIX - 1.6) / .5, al)
        if t < DRIP - .25:      # from here on the verse belongs to the column
            for i, c in enumerate(self.chars):
                d, bloom = self._verse_depth(i, t)
                c.draw(hi, (t - self.lands[i]) / .42, t - self.lands[i], d, 0, bloom)
        bx, by = cell(8)
        if FIX + .05 <= t < t_slam:      # 家 goes to the blank
            e = float(sm((t - FIX - .05) / .6))
            x = CANDS[0][1] + (bx - CANDS[0][1]) * e
            y = CANDS[0][2] + (by - CANDS[0][2]) * e - 60 * math.sin(math.pi * e)
            blit(hi, self.jia.m * 3.3, x - self.jia.s / 2, y - self.jia.s / 2)
        if t >= t_slam:
            self._wash(hi, t, t_slam)
            hi += self.dots * float(sm((t - t_slam) / .05))
            if t < DRIP - .25:
                self.gu.draw(hi, 10, t - t_slam, 3.6, 0, .7)
        elif t >= t_slam - .26:      # 故 comes down on top of it, hard
            u = (t - t_slam + .26) / .26
            e = u ** 2.2
            sc = 1.45 - .45 * e
            size = int(self.gu.s * sc)
            d = np.asarray(Image.fromarray(self.gu.m * 3.6).resize((size, size), Image.BILINEAR), np.float32)
            d = uniform_filter1d(d, int(70 * u) | 1, axis=0)
            blit(hi, d, bx - size / 2, -240 + (by + 240) * e - size / 2)
        self._numbers(hi, t)
        # the page moves in on 思 and gives way again for the guess; it shakes when 故 lands
        s = 1 + (.03 * float(sm((t - LOOK) / (LIAN - LOOK))) + .13 * float(sm((t - LIAN) / (GUESS - LIAN - .3)))) * (1 - float(sm((t - GUESS + .5) / 1.2)))
        age = t - t_slam
        if age >= 0:
            amp = 8 * math.exp(-age * 13)
            dx += amp * math.sin(age * 70)
            dy += amp * math.cos(age * 83)
        cx, cy = 840.0, 540.0
        return np.asarray(Image.fromarray(hi).transform((W, H), Image.AFFINE, (1 / s, 0, cx - cx / s - dx, 0, 1 / s, cy - cy / s - dy), Image.BICUBIC), np.float32)

    # ---- 水滴石穿 --------------------------------------------------------------------------------------------------------------------------
    def _drop(self, hi, x, y, a):
        k = int(round(a))
        if k not in self.sprites:
            self.sprites[k] = teardrop(k)
        m, shine = self.sprites[k]
        h_, w_ = m.shape
        ix, iy = int(x - w_ / 2), int(y - h_ + k * 1.12)
        xa, xb, ya, yb = max(ix, 0), min(ix + w_, W), max(iy, 0), min(iy + h_, H)
        if xa >= xb or ya >= yb:
            return
        sl = (slice(ya - iy, yb - iy), slice(xa - ix, xb - ix))
        M = m[sl]
        hi[ya:yb, xa:xb] = hi[ya:yb, xa:xb] * (1 - M) + 3.4 * M * (1 - .9 * shine[sl])

    def _lines(self, hi, t):
        """The lines being read, up in the clear strip: each holds a moment, then squeezes into a bead of ink."""
        yc = 58
        for o in self.reads:
            if not o["ta"] <= t < o["tf"]:
                continue
            m = o["m"]
            h_, w_ = m.shape
            if t < o["ts"]:
                blit(hi, m * 2.9 * float(sm((t - o["ta"]) / .1)), o["xc"] - w_ / 2, yc - h_ / 2)
            else:
                e = ((t - o["ts"]) / (o["tf"] - o["ts"])) ** 2
                nw, nh = max(int(w_ * (1 - e) + 2.4 * o["a"] * e), 4), max(int(h_ * (1 - e) + 2.2 * o["a"] * e), 4)
                d = np.asarray(Image.fromarray(m).resize((nw, nh), Image.BILINEAR), np.float32)
                blit(hi, gaussian_filter(d, .6 + .12 * o["a"] * e) * (2.9 + 5 * e), o["xc"] + (o["x"] - o["xc"]) * e - nw / 2, yc - nh / 2)
        a = float(sm((t - DRIP - 3.5) / .4)) * (1 - float(sm((t - DRIP - 4.5) / .25)))
        if a > 0:      # by now there are too many to read
            r = np.random.default_rng(int(t * FPS / 2))
            f = font("kai", 26)
            im = Image.new("L", (1400, 92), 0)
            dr = ImageDraw.Draw(im)
            for row in range(3):
                x = r.uniform(-60, 20)
                while x < 1400:
                    line = TORRENT[r.integers(len(TORRENT))]
                    dr.text((x, row * 30), line, int(r.uniform(90, 255)), font=f)
                    x += f.getlength(line) + r.uniform(20, 90)
            hi[14:106, 40:1440] += np.asarray(im, np.float32) / 255 * 1.2 * a

    def _column(self, hi, t):
        """What I write. Afterwards it goes pale, and my own lines are written over it."""
        fade = (1 - .86 * float(sm((t - TANG - .35) / .6))) * (1 - float(sm((t - TANG - 2.2) / 2)))
        if fade <= .003:
            return
        for i, ev in enumerate(self.plan):
            n = -1
            while n + 1 < len(ev) and ev[n + 1][0] <= t:
                n += 1
            if n < 0:
                continue
            t0, c, depth, junk = ev[n]
            age = t - t0
            cx, cy = cell(i)
            blit(hi, c.density(age / .16, age, depth, .3 if junk else 0, .1 if junk else .6) * fade, cx - c.s / 2, cy - c.s / 2)
            if n > 0 and age < .12:
                tp, cp, dp, jp = ev[n - 1]
                blit(hi, cp.density(10, t - tp, dp, .3 if jp else 0, .1) * (1 - age / .12) * fade, cx - c.s / 2, cy - c.s / 2)

    def _wall(self, t):
        count = np.zeros((self.gh, self.gw), np.int32)
        dx = np.zeros((self.gh, self.gw), np.float32)
        dy = np.zeros((self.gh, self.gw), np.float32)
        ring = np.zeros((self.gh, self.gw), np.float32)
        side = np.zeros((self.gh, self.gw), np.float32)
        blot = np.zeros((self.gh, self.gw), np.float32)
        pale = np.ones((self.gh, self.gw), np.float32)
        G, T = self.G, self.T
        for o in self.drops:
            age = t - o["t"]
            if age < 0:
                continue
            R = min(o["speed"] * age, o["reach"])
            A = o["amp"] * math.exp(-age * o["speed"] / o["reach"] * 1.7) / (1 + R / 1200)
            live = A > .012
            n = int((R + (o["wid"] * 3 if live else 0)) / G) + 2
            cx, cy = int(o["x"] / G), int(o["y"] / G)
            xa, xb, ya, yb = max(cx - n, 0), min(cx + n + 1, self.gw), max(cy - n, 0), min(cy + n + 1, self.gh)
            if xa >= xb or ya >= yb:
                continue
            sl = (slice(ya - cy + T, yb - cy + T), slice(xa - cx + T, xb - cx + T))
            d = self.td[sl]
            count[ya:yb, xa:xb] += d < R
            if not live:
                continue
            u = (d - R) / o["wid"]
            g = np.exp(-u * u)
            push = -u * g * A * 24
            dx[ya:yb, xa:xb] += push * self.ux[sl]
            dy[ya:yb, xa:xb] += push * self.uy[sl]
            ring[ya:yb, xa:xb] += g * A
            side[ya:yb, xa:xb] += -u * g * A
            if age < .5:      # where it lands the ink opens out and is gone
                blot[ya:yb, xa:xb] += 2.6 * np.exp(-(d / (o["a"] * (1 + age * 8))) ** 2) * math.exp(-age / .14)
            if o["big"]:
                pale[ya:yb, xa:xb] *= 1 - .6 * math.exp(-age / .35) * sm((R - d) / 60)

        def up(f):
            return np.asarray(Image.fromarray(f).resize((W, H), Image.BICUBIC), np.float32)

        c = np.asarray(Image.fromarray((count % 3).astype(np.uint8)).resize((W, H), Image.NEAREST))
        wall = np.choose(c, self.walls)
        wall = map_coordinates(wall, [self.yy - np.clip(up(dy), -18, 18), self.xx - np.clip(up(dx), -18, 18)], order=1, mode="constant").astype(np.float32) * .8 * self.room
        S = up(side)
        hi = wall * (1 + .8 * np.clip(up(ring), 0, 1.2)) * up(pale)
        hi = hi * (1 - .55 * np.clip(-S, 0, 1)) + .6 * np.clip(S, 0, 1) * (.75 + .5 * self.n01) + up(blot)
        for k, o in enumerate(self.drops):
            age = t - o["t"]
            if -o["fall"] <= age < 0:
                e = (1 + age / o["fall"]) ** 2
                if k < len(READS):
                    self._drop(hi, o["x"], 58 + (o["y"] - 58) * e, o["a"])
                else:
                    blit(hi, self.streaks, o["x"] - 4, o["y"] - 420 * (1 - e) - 120)
            elif k < 3 and 0 <= age < .5:      # what the first few throw up when they land
                for vx, vy, rad in self.splash[k]:
                    x, y = o["x"] + vx * age, o["y"] + vy * age + 1100 * age * age
                    R = int(rad * 2.4) + 1
                    ix, iy = int(x), int(y)
                    if R <= ix < W - R and R <= iy < H - R:
                        gy, gx = np.ogrid[-R:R + 1, -R:R + 1]
                        hi[iy - R:iy + R + 1, ix - R:ix + R + 1] += np.exp(-(gx * gx + gy * gy) / (rad * rad)) * 1.7 * (1 - age * 2)
        return hi

    # ---- the frame --------------------------------------------------------------------------------------------------------------------------
    def frame(self, t):
        hi = np.zeros((H, W), np.float32)
        if t < LOOK + 1.0:      # the wall from the last section goes as the water dries off the page
            u = (t - LOOK) / .95
            hi += self.walls[0] * (1 - sm((u * 1.6 - self.n01) / .3))
        if LOOK <= t < DRIP + .4:
            u = (t - DRIP + .5) / .8
            hi += self._poem(t) * (1 - sm((u * 1.6 - self.n01) / .3))
        if DRIP - .4 <= t < TANG + .7:
            u_in = (t - DRIP + .4) / .9
            u_out = (t - TANG + .2) / .9
            hi += self._wall(t) * sm((u_in * 1.6 - self.n01) / .3) * (1 - sm((u_out * 1.6 - self.n01) / .3))
            self._lines(hi, t)
        if t >= DRIP - .25:
            self._column(hi, t)
        if t >= TANG:
            page = np.zeros((H, W), np.float32)
            for k, (c, t0, dur, depth, dry) in enumerate(self.chant):
                last = k == len(self.chant) - 1
                c.draw(page, (t - t0) / dur, t - t0, depth, dry, .62 if last else .10)
            s = 1 + .025 * float(sm((t - TANG) / (END - TANG)))
            cx, cy = 940.0, 540.0
            hi += np.asarray(Image.fromarray(page).transform((W, H), Image.AFFINE, (1 / s, 0, cx - cx / s, 0, 1 / s, cy - cy / s), Image.BICUBIC), np.float32)
        return develop(hi, 0, self.detail, self.tone)


# ======================================================================================
def cmd_still(times):
    os.makedirs(STILLS, exist_ok=True)
    a = Attn()
    for x in times:
        Image.fromarray(a.frame(x)).save(os.path.join(STILLS, f"s{x:06.2f}.png"))
    tw, cols = 640, 3
    times = sorted(times)
    ims = [Image.open(os.path.join(STILLS, f"s{x:06.2f}.png")).resize((tw, tw * 9 // 16), Image.LANCZOS) for x in times]
    s = Image.new("RGB", (cols * tw, math.ceil(len(ims) / cols) * tw * 9 // 16), (30, 30, 30))
    for i, im in enumerate(ims):
        s.paste(im, ((i % cols) * tw, (i // cols) * tw * 9 // 16))
    s.save(os.path.join(STILLS, "sheet.png"))
    print(os.path.join(STILLS, "sheet.png"))


_worker = None


def _init():
    global _worker
    _worker = Attn()


def _render(n):
    return _worker.frame(START + n / FPS).tobytes()


def cmd_plate(jobs):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "plate_attn.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    total = int(round((END - START) * FPS))
    with multiprocessing.Pool(jobs, _init) as pool:
        for n, buf in enumerate(pool.imap(_render, range(total), chunksize=6)):
            p.stdin.write(buf)
            if n % 150 == 0:
                print(f"t={START + n / FPS:6.2f}", flush=True)
    p.stdin.close()
    p.wait()
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["still", "plate"])
    ap.add_argument("args", nargs="*", type=float)
    ap.add_argument("--jobs", type=int, default=10)
    o = ap.parse_args()
    if o.cmd == "still":
        cmd_still(o.args)
    else:
        cmd_plate(o.jobs)

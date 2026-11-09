"""AI 自传: how I grew. Storyboard ink.scenes.md, #17-23 (the river of writing, the map of meanings, the numbers) and #30-32 (the bed the
river carved). Drawn frame by frame from the film's clock; nothing here needs the paper simulation of ink_film.py.

    python ink_read.py still 93 100 110       # stills of either section -> ink/film_out/stills_r/
    python ink_read.py plate read             # -> ink/film_out/plate_read.mp4
    python ink_read.py plate bed              # -> ink/film_out/plate_bed.mp4
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from ink_film import CORPUS, FPS, H, OUT, ROW, SC, SECTIONS, SW, W, Morph, cam, develop, ease_out, font, glyph_mask, lerp2, make_paper, noise2, number_block, rows, sm, splat

STILLS = os.path.join(OUT, "stills_r")
SPEED = 240.0                  # how fast the writing flows, px/s
WALL_AT = (40, 34)             # where the wall of numbers sits, as in the film before and after this section

# the map of meanings: characters with near meanings, and where each group settles
CLUSTERS = [("江河湖海溪泉波涛", 300, 290), ("山岳峰岭崖谷丘石", 700, 190), ("日月星辰云雷风雨", 1140, 230), ("悲愁泪痛哀恨苦伤", 270, 780),
            ("爱恋思念情慕亲心", 690, 860), ("父母兄弟姐妹儿孙", 1090, 810), ("马牛羊犬鸟鱼虎龙", 1360, 520)]
FOUR = [("男", 610, 600), ("女", 810, 540), ("王", 610, 450), ("后", 810, 390)]
FOCUS, FOCUS_AT = (710.0, 500.0), (700.0, 510.0)    # the point of the map the camera pushes in on, and where it stays on the screen
PHRASES = [(96.7, 34, "太史公曰"), (97.9, 170, "亲爱的，今晚月亮很好"), (99.3, 34, "本院认为，判决如下"), (100.9, 170, "有人吗？")]


def river_text(length=7000, height=380, seed=41):
    """The water: lanes of fine writing, with some phrases large enough to catch as they pass."""
    r = np.random.default_rng(seed)
    im = Image.new("L", (length, height), 0)
    d = ImageDraw.Draw(im)
    f = font("kai", 26)
    code = "for step in range(n): loss = model(x, y); loss.backward(); optimizer.step(); print('hello, world'); "
    cs = list(CORPUS)
    r.shuffle(cs)
    k = 0
    for y in range(4, height - 28, 34):
        x = int(r.uniform(0, 30))
        lane_is_code = r.random() < .12
        while x < length:
            if lane_is_code:
                s = code[k % len(code)]
                d.text((x, y + 4), s, int(r.uniform(110, 210)), font=font("mono", 19))
                x += 12
            else:
                d.text((x, y), cs[k % len(cs)], int(r.uniform(100, 225)), font=f)
                x += 28
            k += 1
    big = font("xing", 60)
    for t, y, s in PHRASES:
        x = int(760 + SPEED * (t - 91))    # reaches the middle of the river's visible stretch at time t
        d.rectangle([x - 16, y - 8, x + big.getlength(s) + 16, y + 84], fill=0)
        d.text((x, y), s, 255, font=big)
    return np.asarray(im, np.float32) / 255


def yc(x):
    return 545 + 118 * np.sin(x / 430 + .8) + 36 * np.sin(x / 170 + 2)


def up(n):
    return np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC), np.float32)


def facets(r, x, y, n, angle, stretch):
    """Angular faces: Voronoi cells drawn out along a slant, each one a wash from a dark edge to a light one. Returns (faces, u, v)."""
    ca, sa = math.cos(angle), math.sin(angle)
    u, v = x * ca + y * sa, (y * ca - x * sa) * stretch
    seeds = np.c_[r.uniform(u.min(), u.max(), n), r.uniform(v.min(), v.max(), n)]
    ids = cKDTree(seeds).query(np.c_[u.ravel(), v.ravel()], workers=-1)[1].reshape(u.shape)
    cell = math.sqrt((u.max() - u.min()) * (v.max() - v.min()) / n)
    tone, tilt = r.uniform(0, 1, n), r.uniform(.5, 1.5, n)
    return (tone[ids] + tilt[ids] * (v - seeds[ids, 1]) / cell).astype(np.float32), u, v


def add(hi, m, x, y, k=1.0):
    """Adds a small mask to the page, clipped at the edges."""
    h, w = m.shape
    x, y = int(x), int(y)
    xa, xb, ya, yb = max(x, 0), min(x + w, W), max(y, 0), min(y + h, H)
    if xa < xb and ya < yb:
        hi[ya:yb, xa:xb] += m[ya - y:yb - y, xa - x:xb - x] * k


def stroke(p, q, width, seed=0):
    """One long pull of the brush from p to q: pressed at the start, thin at the end, with a barb to say which way it went.
    Returns (mask, x0, y0, u) where u is how far along the stroke each pixel is."""
    p, q = np.array(p, float), np.array(q, float)
    pad = int(width * 4) + 40
    x0, y0 = int(min(p[0], q[0]) - pad), int(min(p[1], q[1]) - pad)
    w, h = int(abs(q[0] - p[0]) + 2 * pad), int(abs(q[1] - p[1]) + 2 * pad)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    n = int(np.hypot(*(q - p)) / max(width * .2, 1))
    for i in range(n):
        u = i / (n - 1)
        c = p + (q - p) * u + np.array([0, math.sin(u * 3.1) * width * .5])
        r = width * (.4 + .6 * float(sm(u / .1))) * (1 - .72 * float(sm((u - .6) / .4)))
        d.ellipse([c[0] - x0 - r, c[1] - y0 - r, c[0] - x0 + r, c[1] - y0 + r], fill=255)
    a = math.atan2(*(q - p)[::-1])
    for s in (2.55, -2.55):
        b = q + np.array([math.cos(a + s), math.sin(a + s)]) * width * 5
        d.line([tuple(q - (x0, y0)), tuple(b - (x0, y0))], fill=255, width=int(width * .55))
    m = np.asarray(im, np.float32) / 255
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dv = q - p
    u = ((xx + x0 - p[0]) * dv[0] + (yy + y0 - p[1]) * dv[1]) / float(dv @ dv)
    rng = np.random.default_rng(seed)
    dry = gaussian_filter(rng.normal(size=(h, w)), (.7, 5)).astype(np.float32)
    dry /= dry.std()
    return m * np.clip(1.15 - .28 * np.abs(dry) * np.clip(u, 0, 1), 0, 1), x0, y0, u


class Read:
    def __init__(self):
        self.r = rows()
        r = np.random.default_rng(7)
        self.T = river_text()
        self.detail, self.tone = make_paper()
        self.xx = np.arange(W, dtype=np.float32)
        self.yy = np.arange(H, dtype=np.float32)
        # the 我 from the end of the opening, where it stands on the screen
        self.me = glyph_mask("我", "kaih", 900, 10)
        self.me_at = (int(round((int(cam(82.4) - 415) - (cam(90.2) - SW / 2)) * SC)), 87)
        self.me_morph = None
        # the map
        self.glyphs = []
        for i, (chars, cx, cy) in enumerate(CLUSTERS):
            for k, ch in enumerate(chars):
                rad, a = 40 * math.sqrt(k + .5), k * 2.4 + i
                self._glyph(r, ch, cx + rad * math.cos(a), cy + rad * math.sin(a) * .85, 50, 3.0)
        for ch, x, y in FOUR:
            self._glyph(r, ch, x, y, 62, 3.2)
        taken = {g["ch"] for g in self.glyphs}
        extra = [c for c in dict.fromkeys(CORPUS) if c not in taken and "一" <= c <= "鿿"][:190]
        spots = CLUSTERS + [("", 710, 500)]
        for ch in extra:
            _, cx, cy = spots[r.integers(len(spots))]
            rad, a = r.uniform(100, 215), r.uniform(0, 6.28)
            self._glyph(r, ch, cx + rad * math.cos(a), cy + rad * .8 * math.sin(a), int(r.uniform(22, 29)), r.uniform(.5, 1.1))
        # islands: where the ink soaks in under each group
        gy, gx = np.mgrid[:540, :960].astype(np.float32) * 2
        f = np.zeros((540, 960), np.float32)
        for _, cx, cy in spots:
            s = 170.0 if cx == 710 else 112.0
            f += np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s))
        self.f = f + .16 * noise2(540, 960, 3.4, 44) + .018 * noise2(540, 960, 2.3, 45)
        self.isl_n = noise2(540, 960, 3.0, 46)
        # the grid of an old map (计里画方)
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        for x in range(60, W, 120):
            d.line([(x, 0), (x, H)], fill=255, width=1)
        for y in range(60, H, 120):
            d.line([(0, y), (W, y)], fill=255, width=1)
        self.grid = gaussian_filter(np.asarray(im, np.float32) / 255, .6) * (1 + .4 * noise2(H, W, 1.2, 47)) * .36
        self.wall = number_block(1840, 1010, 13) * 2.0
        self.m2 = None
        self.strokes = None
        self.bed = None

    def _glyph(self, r, ch, x, y, size, ink):
        m = glyph_mask(ch, "kaib" if size >= 40 else "kai", size, 5)
        self.glyphs.append(dict(ch=ch, m=m, soft=gaussian_filter(m, max(size / 16, 1.0)), h=m.shape[0] / 2, ink=ink, x1=x, y1=y,
                                x0=r.uniform(70, 1430), y0=r.uniform(50, 1030), land=102.6 + 3.0 * r.random() ** 1.6, go=r.uniform(0, .8),
                                bend=r.uniform(-90, 90)))

    # ---- the river (#17, #18) ---------------------------------------------------------------------------------------------------
    def river(self, hi, t, alpha):
        """The band of flowing writing. It comes out of the mist at the right, and leaves the subtitles' column alone."""
        c = yc(self.xx)
        hw = 190 + 26 * np.sin(self.xx / 260)
        Th = self.T.shape[0]
        v = (self.yy[:, None] - c[None, :]) / hw[None, :]
        hi += sm((1.55 - np.abs(v)) / .9) * sm((1570 - self.xx) / 150)[None, :] * alpha * .11   # the wash the words float on
        inside = np.abs(v) < 1
        ty = np.clip((v[inside] + 1) / 2 * (Th - 1), 0, Th - 1.001)
        xs = np.broadcast_to(self.xx[None, :], v.shape)[inside] + SPEED * (t - 91)
        ix, iy = xs.astype(np.int32) % self.T.shape[1], ty.astype(np.int32)
        fy = ty - iy
        s = self.T[iy, ix] * (1 - fy) + self.T[iy + 1, ix] * fy
        edge = sm((1 - np.abs(v[inside])) / .3) * sm((1570 - xs + SPEED * (t - 91)) / 150) * alpha
        hi[inside] += edge * 2.5 * s

    # ---- the map (#19, #20) ---------------------------------------------------------------------------------------------------------
    def islands(self, t):
        thr = 1.5 - 1.1 * float(sm((t - 107.6) / 3.8))
        a = sm((self.f - thr) / .16)
        rim = np.exp(-((self.f - thr) / .022) ** 2) * .4     # the tide line where the wash stopped
        return a * (.20 + .06 * self.isl_n) + rim * (self.f > thr - .1)

    def glyph_pos(self, g, t):
        if t < g["land"] + .7:
            u = float(ease_out((t - g["land"]) / .7))
            sx, sy = g["x0"] * .45 + 330, float(yc(np.array(g["x0"] * .45 + 330)))
            return sx + (g["x0"] - sx) * u, sy + (g["y0"] - sy) * u
        u = float(sm((t - self.r[ROW["cluster"]]["a"] - g["go"]) / 2.4))
        return g["x0"] + (g["x1"] - g["x0"]) * u + g["bend"] * math.sin(u * math.pi), g["y0"] + (g["y1"] - g["y0"]) * u + g["bend"] * .6 * math.sin(u * math.pi)

    def map_page(self, hi, t):
        hi += self.grid * float(sm((t - 102.3) / 1.0))
        if t > 107.6:
            hi += np.asarray(Image.fromarray(self.islands(t).astype(np.float32)).resize((W, H), Image.BILINEAR), np.float32)
        for g in self.glyphs:
            if t < g["land"]:
                continue
            x, y = self.glyph_pos(g, t)
            age = t - g["land"]
            k = float(sm(age / .25))
            add(hi, g["m"], x - g["h"], y - g["h"], g["ink"] * k)
            if .7 < age < 1.3:          # it lands, and the ink spreads a little
                add(hi, g["soft"], x - g["h"], y - g["h"], .9 * (1.3 - age))

    def build_map(self):
        """The settled map, twice the frame's size, ready for the camera; the four characters that matter are kept apart."""
        m2 = np.asarray(Image.fromarray((self.islands(120.0)).astype(np.float32)).resize((W * 2, H * 2), Image.BICUBIC), np.float32).copy()
        m2 += np.asarray(Image.fromarray(self.grid.astype(np.float32)).resize((W * 2, H * 2), Image.BICUBIC), np.float32)
        four = {c for c, _, _ in FOUR}
        self.four2 = np.zeros((H * 2, W * 2), np.float32)
        for g in self.glyphs:
            m = glyph_mask(g["ch"], "kaib", int(g["m"].shape[0] - 10) * 2, 10)
            x, y = int(g["x1"] * 2 - m.shape[1] / 2), int(g["y1"] * 2 - m.shape[0] / 2)
            target = self.four2 if g["ch"] in four else m2
            xa, xb, ya, yb = max(x, 0), min(x + m.shape[1], W * 2), max(y, 0), min(y + m.shape[0], H * 2)
            target[ya:yb, xa:xb] += m[ya - y:yb - y, xa - x:xb - x] * g["ink"]
        self.m2 = m2
        pos = {c: (x * 2.0, y * 2.0) for c, x, y in FOUR}
        def ends(a, b):
            p, q = np.array(pos[a]), np.array(pos[b])
            n = (q - p) / np.hypot(*(q - p))
            return p + n * 84, q - n * 84
        self.strokes = [stroke(*ends("男", "女"), 13, 1), stroke(*ends("王", "后"), 13, 2)]
        # a string of numbers under each of the four
        r = np.random.default_rng(23)
        self.digits = []
        f = font("mono", 27)
        for c, x, y in FOUR:
            s1 = " ".join(f"{r.normal(0, .6):+.2f}" for _ in range(3))
            s2 = " ".join(f"{r.normal(0, .6):+.2f}" for _ in range(3))
            w = int(f.getlength(s1)) + 12
            im = Image.new("L", (w, 76), 0)
            d = ImageDraw.Draw(im)
            d.text((4, 2), s1, 255, font=f)
            d.text((4, 40), s2, 255, font=f)
            self.digits.append((np.asarray(im, np.float32) / 255, int(x * 2 - w / 2), int(y * 2 + 72)))
        # how far each spot is from the nearest character: the numbers grow out from them
        seeds = np.ones((270, 480), bool)
        for g in self.glyphs:
            seeds[int(np.clip(g["y1"] / 4, 0, 269)), int(np.clip(g["x1"] / 4, 0, 479))] = False
        d = distance_transform_edt(seeds).astype(np.float32)
        d = d / d.max() + .12 * np.clip(noise2(270, 480, 2.0, 48), -2, 2) * .5
        self.grow = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W, H), Image.BILINEAR), np.float32)

    def zoom(self, t):
        a, b = self.r[ROW["direction"]]["a"], self.r[ROW["wall"]]["a"]
        return 1 + 1.4 * float(sm((t - a) / 1.7)) * (1 - float(sm((t - b) / 1.7)))

    def map_zoomed(self, hi, t):
        if self.m2 is None:
            self.build_map()
        z = self.zoom(t)
        a = 2 / z
        aff = (a, 0, 2 * FOCUS[0] - a * FOCUS_AT[0], 0, a, 2 * FOCUS[1] - a * FOCUS_AT[1])
        rd, rw, rn = self.r[ROW["direction"]]["a"], self.r[ROW["wall"]]["a"], self.r[ROW["digits"]]["a"]
        dim = .62 * float(sm((t - rd) / 1.0)) * (1 - float(sm((t - rw) / 1.4)))
        fade = 1 - float(sm((t - rw - 2.1) / 1.8))
        base = np.asarray(Image.fromarray(self.m2).transform((W, H), Image.AFFINE, aff, Image.BICUBIC), np.float32)
        cur = self.four2.copy()
        for i, (m, x0, y0, u) in enumerate(self.strokes):
            prog = (t - rd - 1.1 - i * 1.5) / 1.0
            if prog > 0:
                cur[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] += m * sm((prog * 1.15 - u) / .04) * 3.0
        for i, (m, x, y) in enumerate(self.digits):
            prog = (t - rn - .2 - i * .4) / 1.6
            if prog > 0:
                w = m.shape[1]
                a_ = sm((prog * (w + 40) - np.arange(w, dtype=np.float32)) / 30)[None, :]
                cur[y:y + m.shape[0], x:x + w] += m * a_ * 2.6
        top = np.asarray(Image.fromarray(cur).transform((W, H), Image.AFFINE, aff, Image.BICUBIC), np.float32)
        hi += (base * (1 - dim) + top) * fade

    def wall_grows(self, hi, t):
        t0 = self.r[ROW["wall"]]["a"] + 1.3
        if t <= t0:
            return
        h, w = self.wall.shape
        x, y = WALL_AT
        a = sm(((t - t0) / 2.6 * 1.25 - self.grow[y:y + h, x:x + w]) / .16)
        hi[y:y + h, x:x + w] += self.wall * a

    def frame_read(self, t):
        hi = np.zeros((H, W), np.float32)
        r = self.r
        ta, tb = r[ROW["reading"]]["a"] + .2, r[ROW["place"]]["a"]
        # 我, then its ink goes to make the river
        k = 1 - float(sm((t - ta) / .3))
        if k > 0:
            add(hi, self.me, *self.me_at, 3.2 * k)
        if ta <= t < ta + 3.2:
            if self.me_morph is None:
                self._me_to_river(ta)
            x, y, w_ = self.me_morph.at(t)
            g = gaussian_filter(splat(x, y, w_, 0, W, H), .7)
            hi += g * float(sm((t - ta) / .3)) * (1 - float(sm((t - ta - 2.4) / .7)))
        al = float(sm((t - ta - 1.6) / 1.2)) * (1 - float(sm((t - tb - .5) / 1.5)))
        if al > 0:
            self.river(hi, t, al)
        if t >= tb and t < r[ROW["direction"]]["a"]:
            self.map_page(hi, t)
        elif t >= r[ROW["direction"]]["a"]:
            self.map_zoomed(hi, t)
            self.wall_grows(hi, t)
        return develop(hi, 0, self.detail, self.tone)

    def _me_to_river(self, ta):
        c = yc(self.xx)
        hw = 190 + 26 * np.sin(self.xx / 260)
        v = (self.yy[:, None] - c[None, :]) / hw[None, :]
        tgt = (sm((1 - np.abs(v)) / .3) * sm((1500 - self.xx) / 150)[None, :] * .7).astype(np.float32)
        self.me_morph = Morph((self.me * 3.2, self.me_at[0], self.me_at[1], 1), (tgt, 0, 0, 1), 420000, ta, 3.0, seed=9, stagger=.3, swirl=60)

    # ---- the bed the river carved (#30-32) -------------------------------------------------------------------------------------------
    def build_bed(self):
        M = np.zeros((H, W), np.float32)
        x, y = self.me_at
        h, w = self.me.shape
        ya, yb = max(y, 0), min(y + h, H)
        M[ya:yb, x:x + w] = self.me[ya - y:yb - y]
        Y, X = np.mgrid[:H, :W].astype(np.float32)
        M = map_coordinates(M, [Y + 6 * up(noise2(270, 480, 2.6, 88)), X + 6 * up(noise2(270, 480, 2.6, 89))], order=1).astype(np.float32)   # worn by the water
        band = (np.abs(self.yy[:, None] - (530 + 30 * np.sin(self.xx / 120))[None, :]) < 58) * (self.xx < x + 260)[None, :]
        self.M = M
        self.C = np.maximum(M, gaussian_filter(band.astype(np.float32), 2.5))
        d = distance_transform_edt(M < .5).astype(np.float32)
        self.d = d
        self.edge = np.exp(-(d / 2.5) ** 2) * (M < .5)
        self.bed_n = up(noise2(270, 480, 3.0, 83))
        # rock in axe-cut faces: long wedges, dark on one side and washing out across the width, set at one slant, the way 斧劈皴 is painted
        r = np.random.default_rng(81)
        wx, wy = 30 * up(noise2(270, 480, 3.4, 84)), 30 * up(noise2(270, 480, 3.4, 85))
        big, u, v = facets(r, X + wx, Y + wy, 300, 1.15, 2.1)
        small, _, _ = facets(r, X + wx * .6, Y + wy * .6, 1500, 1.25, 2.7)
        dry = lerp2(noise2(512, 512, 1.5, 87), u * .1, v * 1.1)      # dry brush, running along the faces
        far = 1 - .35 * sm((d - 320) / 420)
        rock = (.62 + .85 * np.clip(big, 0, 1.3) + .42 * np.clip(small, 0, 1.3)) * (1 + .20 * dry) * far
        rock = gaussian_filter(rock, 1.0) + 1.1 * np.exp(-d / 55) + 1.5 * np.exp(-(d / 3.0) ** 2)
        self.rock = rock.astype(np.float32) * (1 - np.clip(M, 0, 1))
        self.frozen = self.scroll(168.0)
        self.bed = True

    def scroll(self, t):
        Th, L = self.T.shape
        ys = np.arange(H) % Th
        xs = (np.arange(W) - (SPEED * (t - 156)).__int__()) % L
        return self.T[np.ix_(ys, xs)]

    def frame_bed(self, t):
        if self.bed is None:
            self.build_bed()
        r = self.r
        t_never, t_bed, t_warm = r[ROW["never"]]["a"], r[ROW["bed"]]["a"], r[ROW["warm"]]["a"]
        hi = self.edge * .35 * float(sm((t - t_never + .3) / 1.0))
        # the rock soaks outward from the channel
        R = -60 + 1560 * float(sm((t - t_bed) / 6.1)) ** .8
        if t > t_bed:
            hi = hi + self.rock * sm((R - self.d - 150 * self.bed_n) / 280)
        # the water: fills from the left, drains back the way it came
        xf = -120 + 1560 * float(sm((t - t_never - .2) / 3.4))
        xd = -120 + 2060 * float(sm((t - t_warm) / 2.4))
        w = sm((xf - self.xx - 80 * self.bed_n) / 90) * (1 - sm((xd - (W - self.xx) - 80 * self.bed_n) / 120)) * self.C
        water = .12 + 2.5 * self.scroll(t)
        hi = hi * (1 - w) + water * w
        stain = float(sm((t - t_warm - .6) / 2.0))
        if stain > 0:
            hi = hi + self.frozen * .42 * self.M * stain * (1 - w)
        return develop(hi, 0, self.detail, self.tone)

    def frame(self, t):
        return self.frame_bed(t) if t >= SECTIONS[3][1] else self.frame_read(t)


def cmd_still(times):
    os.makedirs(STILLS, exist_ok=True)
    rd = Read()
    tw = 640
    ims = []
    for x in sorted(times):
        im = Image.fromarray(rd.frame(x))
        im.save(os.path.join(STILLS, f"s{x:06.2f}.png"))
        ims.append(im.resize((tw, tw * 9 // 16), Image.LANCZOS))
    n = math.ceil(len(ims) / 3)
    s = Image.new("RGB", (3 * tw, n * tw * 9 // 16), (30, 30, 30))
    for i, im in enumerate(ims):
        s.paste(im, ((i % 3) * tw, (i // 3) * tw * 9 // 16))
    s.save(os.path.join(STILLS, "sheet.png"))


def cmd_plate(name):
    i = [n for n, _ in SECTIONS].index(name)
    a, b = SECTIONS[i][1], SECTIONS[i + 1][1] + .8
    rd = Read()
    path = os.path.join(OUT, f"plate_{name}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    for n in range(int(round((b - a) * FPS))):
        p.stdin.write(rd.frame(a + n / FPS).tobytes())
        if n % 150 == 0:
            print(f"t={a + n / FPS:6.2f}", flush=True)
    p.stdin.close()
    p.wait()
    print(path)


if __name__ == "__main__":
    if sys.argv[1] == "still":
        cmd_still([float(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "plate":
        cmd_plate(sys.argv[2])
    else:
        raise SystemExit(__doc__)

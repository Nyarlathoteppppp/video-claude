"""The last poem of 十四, written by hand. 人 神 鬼 器 of 十二 are another hand's, in film_big.py. The shapes are the outlines of Xingkai SC, which a calligrapher wrote (寰 alone, which that hand does not write legibly, from polygons read off another calligrapher's sheet: DRAWN): its Bold master, its outline moved in by a little more or less from point to point, so a stroke can press and lift and its edge stays a clean line. Make Me a Hanzi's centre lines (on the render host, ~/render_work/strokes/graphics.txt, Arphic Public License) are laid on the glyph, by way of the Light master, nearly a centre line itself, and decide only the stroke order and how the writing grows. The ink is worked out inside the outline: strong where a stroke starts, thin and streaked as it travels, broken by the real paper's fibres.

    python film_hand.py char 人              # one of 十二's four alone, a whole frame, by film_big.py
    python film_hand.py four                 # the four on one sheet, by film_big.py
    python film_hand.py poem [1|2|3|4]       # the poem's one sheet with its first rows; at 4, the whole of it signed (the name and the seal: film_title.py)
    python film_hand.py clip [1234]          # hand/poem_N.mp4: the N-th time as a film, its beat long (poem_beats, from ink_subs): the row written, then still; the fourth signed and read
    python film_hand.py row 2 [fps secs]     # row 2 written with row 1 standing: how long a frame takes
    python film_hand.py half 1 缘            # the frame with row 1 written up to the middle of that character
    python film_hand.py drawn 寰 huan_D      # a character drawn from polygons, alone and large, to compare with the sheet it was read off
    python film_hand.py strip 1 0            # a strip of frames of one character of a row growing, to look at the head of the stroke
    python film_hand.py lines                # the centre lines as they were laid on every glyph
    python film_hand.py check                # the poem against ink3d_book_paper.POEM, the fixed sheet, and the rhythm
    python film_hand.py crop NAME...         # parts of a finished frame at three times its size

Everything goes to ink/film_out/film/hand/. No brush or hand is ever drawn: the characters grow on the paper by themselves, in true stroke order. Page.still(n) is the sheet with n rows, Writing(page, j, t0, t1).at(t) the frame at time t, rhythm() how a row's time is shared out, Glyph.timed() how a character's is, Glyph.grown(parts) the ink with each stroke as far along as asked.
"""
import ast
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, label, map_coordinates, median_filter, uniform_filter1d
from scipy.spatial import cKDTree

from ink_grade import fft_noise, sm
from ink_subs import _font

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "hand")
GRAPHICS = os.path.expanduser("~/render_work/strokes/graphics.txt")
W, H = 1920, 1080
TOP = 70                         # at the least, above a character
CLEAR = 110                      # the subtitle's strip along the bottom of the frame, baseline about y = 984: nothing written there in 十二. The poem has no subtitle while it is on screen
BOLD, LIGHT = 0, 2               # the faces of Xingkai.ttc
THIN = 0.02                      # how far in from the Bold master's outline a weight of 0 lies, in ems. Thinning the one master leaves its edge the font's own, its tips sharp and its thick and thin further apart; a blend of the two is lumpy where they disagree
INK_RGB = np.array((17, 16, 18), np.float32)
SEAL_RGB = np.array((172, 44, 30), np.float32)
_data = {}

# A character the font does not have in a hand we can use, read off a calligrapher's sheet: its shapes as polygons (each an outline and the holes in it), and its centre lines in writing order, in place of the stroke data's. In the sheet's own units, y down. em: how many units an em is here, so it stands right with its neighbours. weight: the weight at which the outline is just these polygons. brush: what it is written with, where that is not what the others are: plain, no streaks to follow its outline, and it goes dry only at a stroke's tail
DRAWN = dict(寰=dict(
    wide=400, em=400, weight=-0.04, brush=dict(runs=0.14, plain=True),
    shapes=[
        ([(158, 33), (172, 42), (176, 54), (168, 70), (160, 68), (163, 48)], []),
        ([(28, 158), (38, 118), (60, 86), (76, 80), (150, 66), (210, 66), (288, 51), (304, 66), (319, 100), (311, 122), (262, 132), (243, 125), (260, 112), (274, 94), (258, 84), (205, 90), (140, 94), (86, 106), (62, 124), (45, 147)], []),
        ([(95, 136), (115, 128), (262, 122), (273, 130), (236, 193), (120, 213), (107, 205)], [[(119, 146), (148, 143), (150, 185), (126, 191)], [(162, 142), (185, 140), (185, 178), (165, 184)], [(198, 140), (246, 137), (222, 173), (195, 178)]]),
        ([(27, 285), (43, 257), (75, 258), (150, 247), (240, 227), (248, 236), (150, 269), (90, 288), (60, 306), (43, 304)], []),
        ([(165, 268), (178, 270), (150, 306), (116, 336), (92, 346), (92, 328), (120, 305)], []),
        ([(205, 259), (218, 268), (200, 300), (150, 361), (95, 406), (52, 429), (95, 385), (150, 325), (190, 280)], []),
        ([(262, 268), (274, 275), (252, 312), (222, 352), (196, 420), (172, 492), (160, 494), (160, 420), (170, 375), (205, 330), (240, 292)], []),
        ([(178, 342), (205, 350), (290, 386), (381, 418), (377, 436), (320, 439), (250, 411), (190, 372)], []),
    ],
    lines=[
        [(160, 36), (168, 45), (170, 56), (164, 68)],                                                                       # the dot
        [(30, 155), (45, 125), (68, 98), (110, 88), (150, 80), (205, 78), (270, 70), (292, 85), (295, 105), (270, 122), (246, 125)],     # the roof in one, its hook turning back
        [(108, 137), (111, 170), (114, 208)],                                                                               # the box: its left side
        [(105, 135), (180, 132), (262, 129), (248, 158), (230, 186)],                                                       # its top and right
        [(118, 203), (170, 196), (228, 184)],                                                                               # its foot
        [(155, 137), (156, 165), (158, 193)],                                                                               # and the two inside it
        [(191, 135), (191, 160), (190, 184)],
        [(38, 282), (55, 278), (90, 272), (150, 258), (244, 232)],                                                          # the long bar
        [(172, 269), (152, 295), (120, 320), (94, 337)],                                                                    # the short 撇
        [(212, 263), (196, 290), (150, 343), (95, 396), (54, 428)],                                                         # the long 撇
        [(268, 271), (246, 302), (213, 341), (183, 398), (172, 440), (165, 492)],                                           # the slanting stroke that falls to the point
        [(182, 350), (215, 372), (290, 405), (345, 425), (374, 426)],                                                       # the 捺
    ],
))

# The signature's letters, which the stroke data does not have: centre lines in the order a hand signs, in a square a hundred wide, y down. Only their shape counts, fit() lays them on the font's own letter. An O begins at the top and goes round to the left; p is its stem and then its bowl; 5 is its body and then the bar across its top
LATIN = {
    "O": [[(50 + 30 * np.cos(a), 50 - 38 * np.sin(a)) for a in np.radians(np.arange(80, 460, 12))]],
    "p": [[(35, 30), (34.5, 50), (34, 75), (33, 98)], [(36, 40), (46, 31), (60, 31), (69, 40), (70, 54), (63, 66), (49, 71), (37, 66)]],
    "u": [[(25, 30), (25.5, 48), (28, 62), (36, 71), (48, 72), (60, 65), (67, 50), (68, 31)], [(68, 31), (68.5, 52), (70, 72)]],
    "s": [[(68, 36), (58, 29), (43, 29), (32, 37), (36, 48), (52, 53), (64, 59), (65, 69), (54, 76), (39, 77), (28, 70)]],
    "5": [[(37, 18), (35, 32), (33, 46), (46, 40), (60, 42), (68, 52), (67, 66), (57, 77), (43, 80), (30, 76)], [(37, 18), (52, 17), (68, 16)]],
    ".": [[(48, 73), (50, 75.5), (52, 78)]],
}


# ---------------------------------------------------------------- the stroke data

def _graphics():
    if not _data:
        with open(GRAPHICS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _data[r["character"]] = r
    return _data


def _strokes(c):
    """How many strokes a character is written in."""
    return len(DRAWN[c]["lines"]) if c in DRAWN else len(LATIN[c]) if c in LATIN else len(_graphics()[c]["medians"])


def _medians(c):
    """The data's centre lines of a character in writing order, y down, in a square a hundred wide, at even steps. A letter of the signature from LATIN."""
    out = []
    for m in (LATIN[c] if c in LATIN else _graphics()[c]["medians"]):
        p = np.array(m, np.float64) if c in LATIN else np.array([(x * 100 / 1024, (900 - y) * 100 / 1024) for x, y in m], np.float64)
        along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        s = np.linspace(0, along[-1], max(int(along[-1] / 1.2), 3))
        out.append(np.stack([np.interp(s, along, p[:, 0]), np.interp(s, along, p[:, 1])], 1))
    return out


def fit(lines, target, em):
    """The data's centre lines laid on the glyph: target is where the Light master has ink, in pixels. First the two boxes are matched, then every point of a line goes to the ink nearest it and all ink calls the line nearest it, the movement shared out between neighbours, over a shorter and shorter distance. Points of one stroke hold together more than points of two."""
    rest = np.concatenate(lines)
    who = np.concatenate([np.full(len(p), i) for i, p in enumerate(lines)])
    lo, hi, tl, th = rest.min(0), rest.max(0), target.min(0), target.max(0)
    now = tl + (rest - lo) * (th - tl) / np.maximum(hi - lo, 1e-3)
    start = now.copy()
    tree = cKDTree(target)
    same = who[:, None] == who[None, :]
    apart = ((start[:, None] - start[None, :]) ** 2).sum(2)
    starts = np.cumsum([0] + [len(p) for p in lines])
    for sigma in np.geomspace(0.30, 0.05, 36) * em:
        pull = tree.query(now)[1]
        pull = target[pull] - now
        back = cKDTree(now).query(target)[1]
        extra, count = np.zeros_like(now), np.zeros(len(now))
        np.add.at(extra, back, target - now[back])
        np.add.at(count, back, 1)
        force = (pull + extra) / (1 + count)[:, None]
        share = np.exp(-apart / (2 * sigma * sigma)) * (0.3 + 0.7 * same)
        now = now + share @ force / share.sum(1, keepdims=True)
        # and a stroke stays one movement: it does not snag on what it passes
        for a, b in zip(starts, starts[1:]):
            now[a:b] = gaussian_filter1d(now[a:b], 0.7, axis=0, mode="nearest")
    return [gaussian_filter1d(now[a:b], 1.2, axis=0, mode="nearest") for a, b in zip(starts, starts[1:])]


# ---------------------------------------------------------------- the font

def _placed(c, n, size, tilt=0.0):
    """How a drawn character's units go into an n-square for an em of size pixels: its ink centred, turned as PIL turns an image."""
    d = DRAWN[c]
    p = np.array([q for outer, _ in d["shapes"] for q in outer], np.float64)
    mid = (p.min(0) + p.max(0)) / 2
    k, a = size / d["em"], np.radians(tilt)

    def place(points):
        x, y = (np.array(points, np.float64) - mid).T * k
        return np.stack([n / 2 + np.cos(a) * x + np.sin(a) * y, n / 2 - np.sin(a) * x + np.cos(a) * y], 1)
    return place


def _drawn(c, n, size):
    """A drawn character filled in, upright: each shape with its holes, then blurred a little and cut again, so its corners come out round as ink's do."""
    d = DRAWN[c]
    place = _placed(c, n, size)
    out = np.zeros((n, n), np.float32)
    for outer, holes in d["shapes"]:
        im = Image.new("L", (n, n), 0)
        draw = ImageDraw.Draw(im)
        draw.polygon([tuple(q) for q in place(outer)], fill=255)
        for hole in holes:
            draw.polygon([tuple(q) for q in place(hole)], fill=0)
        out = np.maximum(out, np.asarray(im, np.float32))
    return Image.fromarray(np.uint8(gaussian_filter(out, 0.004 * d["wide"] * size / d["em"]) > 127.5) * 255)


def _distance(c, face, n, size, tilt=0.0, kind="xing"):
    """How far inside the outline of c each pixel of an n-square lies, negative outside, for an em of size pixels. Drawn larger and measured, so the distance runs smooth and any edge drawn from it is clean at any weight. kind "hand": from its polygons in DRAWN and not from a font."""
    ss = 2 if n <= 900 else 1
    if kind == "hand":
        im = _drawn(c, n * ss, size * ss)
    else:
        im = Image.new("L", (n * ss, n * ss), 0)
        ImageDraw.Draw(im).text((n * ss / 2, n * ss / 2), c, 255, font=ImageFont.truetype(_font(kind, 64).path, int(size * ss), index=face), anchor="mm")
    if tilt:
        im = im.rotate(tilt, Image.BICUBIC)
    a = np.asarray(im) > 127
    # the hairline cracks where the font's contours overlap badly would only open up when the outline is moved in: closed first
    r = max(0.002 * size * ss, 1.0)
    a = distance_transform_edt(distance_transform_edt(~a) <= r) > r
    d = gaussian_filter(np.float32(distance_transform_edt(a) - distance_transform_edt(~a)), 1.3)
    if ss > 1:
        d = np.asarray(Image.fromarray(d).resize((n, n), Image.BOX), np.float32) / ss
    return d


_hairs_cache = {}


def _hairs():
    """A sheet of smooth noise, a blot two cells across, looked up at any stretch: it gives the streaks of a dry brush."""
    if "t" not in _hairs_cache:
        t = gaussian_filter(np.random.default_rng(77).normal(0, 1, (1024, 1024)).astype(np.float32), 2.0, mode="wrap")
        _hairs_cache["t"] = t / t.std()
    return _hairs_cache["t"]


def _look(a, b):
    return map_coordinates(_hairs(), [a, b], order=1, mode="grid-wrap")


def ease(p):
    return 0.6 * p + 0.4 * p * p * (3 - 2 * p)


class Glyph:
    """One character of the font, in a square of frame pixels, worked out at res times that. Its centre lines say which stroke each pixel belongs to and how far along it: a stroke first claims what lies within its own width, earliest first so a later one never bites into it, and the rest goes to whatever claimed the pixel next to it, so all of the glyph is written and none of it is left over."""

    PAD = 2.5       # frame pixels beyond the outline where a stroke's ink can still go into the paper

    def __init__(self, c, frame, size, tilt=0.0, res=3):
        self.c, self.frame, self.g, self.tilt = c, frame, res, tilt
        self.n = n = int(frame * res)
        self.em = size * res
        if c in DRAWN:
            self.light = _distance(c, 0, n, self.em, tilt, "hand")
            self.bold = self.light + THIN * (1 - DRAWN[c]["weight"]) * self.em
        else:
            self.light, self.bold = _distance(c, LIGHT, n, self.em, tilt), _distance(c, BOLD, n, self.em, tilt)
        self.mapped = False

    def box(self, weight):
        """Where the ink is at this weight, in frame pixels: left, top, right, bottom."""
        d = self.bold - THIN * (1 - weight) * self.em > 0
        xs, ys = np.nonzero(d.any(0))[0], np.nonzero(d.any(1))[0]
        return xs[0] / self.g, ys[0] / self.g, (xs[-1] + 1) / self.g, (ys[-1] + 1) / self.g

    def _centre(self, line, spacing):
        """A line from the Light master moved across, never along, to the middle of the Bold master's stroke."""
        gy, gx = np.gradient(gaussian_filter(self.bold, 1.5))
        c = max(0.02 * self.em / spacing, 1.5)
        for _ in range(10):
            tan = np.gradient(gaussian_filter1d(line, c, axis=0, mode="nearest"), axis=0)
            tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-6)[:, None]
            at = [line[:, 1], line[:, 0]]
            push = np.stack([map_coordinates(gx, at, order=1), map_coordinates(gy, at, order=1)], 1)
            push = push - (push * tan).sum(1, keepdims=True) * tan
            line = gaussian_filter1d(line + 1.2 * self.g * push, c, axis=0, mode="nearest")
        return line

    def _map(self):
        if self.mapped:
            return
        n, g = self.n, self.g
        step = max(n // 220, 1)
        if self.c in DRAWN:
            place = _placed(self.c, n, self.em, self.tilt)
            each = [place(line) for line in DRAWN[self.c]["lines"]]
        else:
            target = np.argwhere(self.light[::step, ::step] > -0.3 * step)[:, ::-1].astype(np.float64) * step
            rest = [p * self.em / 100 for p in _medians(self.c)]
            each = fit(rest, target, self.em)
        spacing = max(n / 500, 1.0)
        self.lines = []
        for p in each:
            along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
            s = np.arange(0, along[-1] + spacing, spacing).clip(0, along[-1])
            self.lines.append(self._centre(np.stack([np.interp(s, along, p[:, 0]), np.interp(s, along, p[:, 1])], 1), spacing))
        along_all = np.clip(np.rint(np.concatenate(self.lines)).astype(int), 0, n - 1)
        self.miss = float(np.mean(self.bold[along_all[:, 1], along_all[:, 0]] <= 0))         # how much of the lines lies off the ink
        heavy = np.argwhere(self.bold[::step, ::step] > 0)[:, ::-1].astype(np.float64) * step
        self.lost = float(np.mean(cKDTree(np.concatenate(self.lines)).query(heavy)[0] > 0.07 * self.em))    # how much of the ink no line comes near
        mid = self.bold - THIN * 0.6 * self.em
        self.on = np.maximum(self.light, self.bold) > -self.PAD * g
        ys, xs = np.nonzero(self.on)
        P = np.stack([xs, ys], 1).astype(np.float32)
        K = np.full(len(P), -1, np.int16)
        over = np.zeros(len(P), np.float32)
        info = []
        for k, line in enumerate(self.lines):
            at = np.clip(np.rint(line).astype(int), 0, n - 1)
            half = np.maximum(mid[at[:, 1], at[:, 0]], 0.8 * g)
            half = median_filter(half, size=max(int(3 * np.median(half) / spacing) | 1, 3), mode="nearest")
            half = gaussian_filter1d(half, 2.0, mode="nearest")
            tree = cKDTree(line)
            d, i = tree.query(P, distance_upper_bound=float(half.max()) * 1.15 + 1)
            hit = np.isfinite(d)
            hit[hit] = d[hit] <= half[i[hit]] * 1.15
            over[hit & (K >= 0)] = 1
            K[hit & (K < 0)] = k
            info.append((tree, half))
        grid = np.full((n, n), -1, np.int16)
        grid[ys, xs] = K
        near = distance_transform_edt(grid < 0, return_distances=False, return_indices=True)
        grid = grid[near[0], near[1]]
        # a piece of a stroke that its own line does not pass through is a piece of some other stroke: it goes to that one, so nothing stands alone waiting for its stroke
        ink = self.bold - THIN * 0.6 * self.em > 0
        for k, line in enumerate(self.lines):
            parts, many = label((grid == k) & ink)
            if many > 1:
                at = np.clip(np.rint(line).astype(int), 0, n - 1)
                owned = np.zeros(many + 1, bool)
                owned[parts[at[:, 1], at[:, 0]]] = True
                owned[0] = True
                grid[~owned[parts]] = -1
        grid[~ink] = -1
        near = distance_transform_edt(grid < 0, return_distances=False, return_indices=True)
        grid = grid[near[0], near[1]]
        K = grid[ys, xs]
        S, V, R = np.zeros(len(P), np.float32), np.zeros(len(P), np.float32), np.ones(len(P), np.float32)
        self.count = len(self.lines)
        self.width = np.ones(self.count, np.float32)
        self.length = np.ones(self.count, np.float32)
        for k, line in enumerate(self.lines):
            mine = np.nonzero(K == k)[0]
            tree, half = info[k]
            self.width[k] = float(np.median(half))
            if not len(mine):
                continue
            along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
            tan = np.gradient(gaussian_filter1d(line, 3.0, axis=0, mode="nearest"), axis=0)
            tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-6)[:, None]
            i = tree.query(P[mine])[1]
            off = P[mine] - line[i]
            ds = (off * tan[i]).sum(1)
            # a pixel may only run past the line at its two ends: there it carries on, as a stroke of the font that goes further than the data's
            ds = np.where((i > 0) & (i < len(line) - 1), np.clip(ds, -spacing, spacing), ds)
            S[mine] = np.maximum(along[i] + ds, 0)
            V[mine] = off[:, 1] * tan[i, 0] - off[:, 0] * tan[i, 1]
            R[mine] = half[i]
            self.length[k] = max(float(S[mine].max()), 1.0)
        field = lambda v, base=0.0: (lambda a: (a.__setitem__((ys, xs), v), a)[1])(np.full((n, n), base, np.float32))
        self.K = np.zeros((n, n), np.int16)
        self.K[ys, xs] = K
        # a stroke that starts on bare paper starts wet; one that starts inside another, or hardly has a way of its own, shows nothing of it
        self.free = np.array([float(self.K[int(np.clip(p[0, 1], 0, n - 1)), int(np.clip(p[0, 0], 0, n - 1))] == k) * min(self.length[k] / (0.16 * self.em), 1.0) for k, p in enumerate(self.lines)], np.float32)
        self.S, self.V, self.R = field(S), field(V), field(R, 1.0)
        self.over = field(over)
        self.mapped = True

    def written(self, tooth, weight=0.6, ink=1.0, dry=0.1, tone=0.97, press=0.36, seed=0, most=0.92, runs=0.34, plain=False):
        """The character written. tooth: the paper under it, a n-square of its grain. weight: 1 is the Bold master, less is thinner. ink: how much the brush holds at its start. dry: how dry it is at its start. press: how far a stroke goes from pressed to lifted. most: the driest it gets, since past a point a stroke is only its two edges and no longer a stroke. runs: how fast it dries with the way it has come. Sets F, the ink, and what the writing needs."""
        self._map()
        n, g = self.n, self.g
        K, S, V, R = self.K, self.S, self.V, self.R
        r = np.minimum(S / self.length[K], 1.2)
        order = K / max(self.count - 1, 1)
        travel = S / (0.62 * self.em)
        cloud = np.roll(fft_noise(n, n, 3.0, 40), (seed * 37 % n, seed * 91 % n), (0, 1))
        wash = np.roll(fft_noise(n, n, 2.2, 41), (seed * 53 % n, seed * 17 % n), (0, 1))
        cloud, wash = cloud / cloud.std(), wash / wash.std()
        start = np.exp(-(S / (2.6 * R)) ** 2)
        # the weight: pressed where a stroke starts, lifted along it; every edge comes out of this, so it stays smooth
        t = weight + press * (0.45 * start * self.free[K] - 0.5 * sm((r - 0.3) / 0.7)) + 0.05 * cloud
        t = gaussian_filter(np.clip(t, 0, 1), max(0.03 * self.em, 2.0))
        d = self.bold - THIN * (1 - t) * self.em
        M = sm(d / (1.3 * g) + 0.5)
        # how dry and how strong: the further it has come the less it has, a long stroke runs out and a short one does not
        D = dry + 0.08 * order + runs * travel ** 1.6 + 0.09 * cloud
        if plain:
            D = D * sm((r - 0.7) / 0.25)
        C = ink - 0.16 * (np.minimum(travel, 0.4) if plain else travel) - 0.11 * r - 0.3 * np.clip(D, 0, 1) + 0.09 * start * self.free[K] + 0.06 * wash - 0.1 * order
        # where they meet they have run together: nothing of one stroke's edge shows inside another
        inside = np.float32(d > -g)
        melt = lambda v, s: gaussian_filter(v * inside, s) / np.maximum(gaussian_filter(inside, s), 1e-3)
        D = np.minimum(melt(D, max(0.012 * self.em, 1.5)), most)
        C = melt(C, max(0.022 * self.em, 2.0))
        C = C + 0.04 * gaussian_filter(self.over, max(0.012 * self.em, 1.5))
        # the thin edge of a wet stroke dries darker, the last of the ink drawn to it
        rim = np.exp(-np.maximum(d, 0) / max(0.05 * R.mean(), 0.5 * g)) * M
        if not plain:
            C = C + 0.05 * rim * (1 - np.clip(D, 0, 1))
        # the streaks of a dry brush lie along the stroke, in tufts of no particular width; the paper's own fibres break them
        wf = self.width[K] / g
        fine, broad = np.maximum(0.55, 0.04 * wf), np.maximum(1.5, 0.2 * wf)
        far, farther = np.maximum(7.0, 2.6 * wf), np.maximum(10.0, 3.2 * wf)
        table = np.random.default_rng(seed + 5).uniform(0, 1024, (self.count, 6))[K]
        sf, vf = S / g, V / g
        thread = _look(2 * vf / fine + table[..., 0], 2 * sf / far + table[..., 1])
        tuft = _look(2 * vf / broad + table[..., 2], 2 * sf / farther + table[..., 3])
        hairs = (0.8 * thread + 0.55 * tuft) / 0.97 + 0.36 * _look(table[..., 4], 2 * sf / (2.4 * wf) + table[..., 5])
        # and even where it covers, the hairs that held more leave a darker track than those that held less
        C = C + (0.006 + 0.07 * np.clip(1.6 * D, 0, 1)) * (0.65 * tuft + 0.45 * thread)
        C = np.clip(C, 0, 1.08)
        thin = np.clip(1 - C, 0, 1)
        a = (0.3 + 0.7 * C ** 1.05) * tone * (1 - thin * 0.1 * np.clip(tooth, -1.3, 2.2))
        w = np.abs(V) / R
        need = -4.0 + 4.2 * sm(np.clip(D, 0, 1)) ** 0.8
        # a stroke's edges hold longer than its belly, so what it is stays to be read
        if not plain:
            need = need - 0.7 * sm((w - 0.66) / 0.3) * (1 - sm((D - 0.82) / 0.16))
        hold = 1 - (1 - sm((hairs + 0.24 * np.clip(tooth, -2, 2) - need) / 0.4)) * sm((D - 0.3) / 0.16)
        # and beyond the edge, only where it is wet, a little goes into the paper along the fibres
        out = sm(1 + d / (max(0.7, 0.0022 * self.em / g) * g)) * (1 - M)
        bleed = 0.16 * out * out * np.clip(C, 0, 1) * (1 - np.clip(D, 0, 1)) * np.clip(0.5 + 0.5 * tooth, 0, 1)
        self.F = np.float32(np.clip(M * a * hold + bleed, 0, 1)) * self.on
        # the head of a growing stroke: its middle goes first, the sides following, and it comes on through the paper's grain and not in a line
        self.head = np.maximum(1.8 * self.width, 5.0 * g)
        h = self.head[K]
        self.front = np.maximum(S + 0.8 * np.minimum(V * V / R, 2.5 * R) - 0.1 * h * np.clip(gaussian_filter(tooth, g), -2, 2), 0)
        self.end = np.array([float(self.front[(K == k) & self.on].max()) if ((K == k) & self.on).any() else 0.0 for k in range(self.count)], np.float32)
        return self.F

    def grown(self, parts):
        """The ink with each stroke as far as parts says, 0 to 1. At all ones it is F. A stroke's way is a little longer than the stroke by the length of its head, so when it arrives every pixel of it is full."""
        pos = (self.end + self.head) * ease(np.clip(parts, 0, 1))
        pos = np.where(np.asarray(parts) >= 1, np.inf, pos)
        return self.F * sm((pos[self.K] - self.front) / self.head[self.K])

    def timed(self, a, b):
        """When each stroke is written, between a and b: for as long as it is long, not quite in proportion, and the lift between strokes the hand needs. Where the data's stroke has nothing of its own in the glyph it takes no time."""
        size = np.where(self.end > 0, (self.end / self.em) ** 0.72 + 0.07, 0.0)
        lift = 0.1 * size.max()
        whole = size.sum() + lift * (np.count_nonzero(size) - 1)
        at, out = 0.0, []
        for v in size:
            out.append((a + (b - a) * at / whole, a + (b - a) * (at + v) / whole))
            at += v + (lift if v else 0)
        return out

    def at(self, t, times):
        return self.grown(np.array([(t - s) / max(e - s, 1e-6) for s, e in times]))


# ---------------------------------------------------------------- the paper

def _band_rect(h, w, lo, hi, seed):
    v = np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32)
    v = gaussian_filter(v, lo) - gaussian_filter(v, hi)
    return v / v.std()


def sheet(w=W, h=H, seed=91, light=(-0.62, -0.78)):
    """A bright handmade sheet in daylight from the window. Nearly white and warm. Up close: long fibres lying every way, the faint lines of the screen it was lifted on, and a surface that catches the light a little."""
    r = np.random.default_rng(seed)
    im = Image.new("L", (w * 2, h * 2), 128)
    d = ImageDraw.Draw(im)
    for _ in range(int(w * h / 900)):
        x, y, a, n = r.uniform(0, w * 2), r.uniform(0, h * 2), r.uniform(0, np.pi), r.uniform(50, 300)
        bend = r.uniform(-0.6, 0.6)
        d.line([(x + np.cos(a + bend * u) * n * u, y + np.sin(a + bend * u) * n * u) for u in np.linspace(0, 1, 8)], fill=int(r.choice([104, 116, 142, 156])), width=1)
    fibres = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 128 - 1
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wobble = 2.2 * fft_noise(h, w, 3.6, seed + 2)
    laid = 0.5 * np.sin(2 * np.pi * (yy + wobble) / 6.3) + 0.5 * np.exp(-((((xx + wobble * 2) % 172) - 86) / 1.4) ** 2)
    height = 0.8 * fft_noise(h, w, 1.2, seed + 3) * 0.35 + 0.5 * fft_noise(h, w, 2.4, seed + 4) * 0.5 + 2.2 * fibres + 0.26 * laid
    gy, gx = np.gradient(height)
    relief = (gx * light[0] + gy * light[1])
    # the window is up on the left: no more than you would feel
    fall = 1 - 0.03 * np.hypot((xx - w * 0.2) / w, (yy - h * 0.1) / h)
    flocs = _band_rect(h, w, 9, 40, seed + 6)
    tone = 1 + 0.0045 * fft_noise(h, w, 3.4, seed + 5) + 0.03 * fibres + 0.02 * relief + 0.0022 * flocs
    base = np.array((251, 248, 239), np.float32)
    return np.clip(base * (tone * fall)[..., None], 0, 255)


def paper_tooth(f):
    """The sheet's own grain, what an ink too thin or too dry to cover it catches on: its fibres and bumps, about one in size. Without the screen's lines, whose even spacing would bead every streak and rib every thin stroke."""
    a = uniform_filter1d(f.mean(2), 6, axis=0)
    t = gaussian_filter(a, 0.8) - gaussian_filter(a, 6)
    return np.float32(t / t.std())


def tooth_under(tooth, x, y, frame, n):
    """The grain of the sheet under a square at x, y, frame pixels wide, at n pixels."""
    h, w = tooth.shape
    part = tooth[np.ix_((y + np.arange(frame)) % h, (x + np.arange(frame)) % w)]
    return np.asarray(Image.fromarray(part).resize((n, n), Image.BICUBIC), np.float32)


def lay(f, ink, x, y, size, rgb=INK_RGB):
    """Ink (a square image) into the frame f, its corner at x, y, at this many frame pixels."""
    a = np.asarray(Image.fromarray(np.float32(np.clip(ink, 0, 1)), "F").resize((size, size), Image.LANCZOS), np.float32).clip(0, 1)
    fh, fw = f.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + size, fw), min(y + size, fh)
    part = a[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
    f[y0:y1, x0:x1] += (rgb - f[y0:y1, x0:x1]) * part


def save(f, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(path)
    print(path, flush=True)


# ---------------------------------------------------------------- the poem

def poem_lines():
    """The poem, from ink3d_book_paper, read without importing Blender's world: nothing here is typed out a second time."""
    with open(os.path.join(HERE, "ink3d_book_paper.py"), encoding="utf-8") as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "POEM":
                return ast.literal_eval(node.value)
    raise ValueError("no POEM")


def signature(f, tooth, right, base, size, tone=0.56):
    """Opus5.5 at the foot, in the same hand, with the last of its ink, ending at right on the line base. Returns where it starts."""
    res = 4
    face = ImageFont.truetype(_font("xing", 64).path, size, index=LIGHT)
    text = "Opus5.5"
    steps = [face.getlength(ch) * (1.02 if ch != "O" else 0.84) for ch in text]
    asc, desc = face.getmetrics()
    x = right - sum(steps)
    start = x
    for ch, step in zip(text, steps):
        frame = size * 2
        n = frame * res
        d = _distance(ch, LIGHT, n, size * res) * 0.62 + _distance(ch, BOLD, n, size * res) * 0.38
        cx, cy = int(x + step / 2 - frame / 2), int(base - (asc - desc) / 2 - frame / 2)
        grain = tooth_under(tooth, cx, cy, frame, n)
        lay(f, sm(d / (1.3 * res) + 0.5) * tone * (1 - 0.22 * np.clip(grain, -1.2, 2.2)), cx, cy, frame)
        x += step
    return start


# The seal's face, in a square of one: 言 in the square hand of a seal, cut away so it prints white in the red, each cut a bar from here to there at such a width. What the stone has lost at its edges, and the two places the knife left a little standing
SEAL_CUTS = [(.495, .125, .515, .215, .088), (.16, .298, .84, .286, .072), (.26, .418, .755, .41, .066), (.245, .532, .765, .524, .066), (.255, .652, .755, .644, .066), (.272, .64, .286, .868, .072), (.738, .638, .73, .842, .072), (.276, .852, .70, .858, .066)]
SEAL_CHIPS = [[(0, 0), (.17, 0), (.135, .07), (.08, .1), (.07, .15), (0, .175)], [(1, .3), (.94, .33), (.905, .385), (.945, .44), (1, .47)], [(.56, 1), (.61, .94), (.68, .915), (.73, .95), (.77, 1)], [(1, .85), (.945, .9), (.9, .945), (.885, 1), (1, 1)], [(0, .6), (.075, .625), (.065, .685), (0, .71)], [(.3, 0), (.32, .066), (.385, .075), (.405, 0)]]
SEAL_LEFT = [[(.585, .37), (.615, .372), (.61, .455), (.58, .453)], [(.40, .62), (.425, .62), (.42, .682), (.395, .682)]]


def seal(f, tooth, x, y, s, seed=305):
    """The seal: 言, white in a block of red, the one red thing on paper in the film. A stone is cut in straight facets, and it chips: nothing here is a soft edge or speckle but for where the paste did not take on the paper's grain."""
    n = s * 8
    r = np.random.default_rng(seed)
    corners = np.array([(.04, .05), (.96, .04), (.955, .96), (.045, .955)])
    rim = []
    for a, b in zip(corners, np.roll(corners, -1, 0)):
        rim.append(a)
        for u in np.sort(r.uniform(0.05, 0.95, 7)):
            rim.append(a + (b - a) * u + r.normal(0, 0.005, 2))
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    scale = lambda pts: [(float(px) * n, float(py) * n) for px, py in pts]
    d.polygon(scale(rim), fill=255)
    for chip in SEAL_CHIPS:
        d.polygon(scale(chip), fill=0)
    for x0, y0, x1, y1, w in SEAL_CUTS:
        nx, ny = -(y1 - y0), x1 - x0
        k = 0.5 * w / np.hypot(nx, ny)
        j = r.normal(0, 0.004, (4, 2))
        d.polygon(scale([(x0 + nx * k + j[0, 0], y0 + ny * k + j[0, 1]), (x1 + nx * k + j[1, 0], y1 + ny * k + j[1, 1]), (x1 - nx * k + j[2, 0], y1 - ny * k + j[2, 1]), (x0 - nx * k + j[3, 0], y0 - ny * k + j[3, 1])]), fill=0)
    for part in SEAL_LEFT:
        d.polygon(scale(part), fill=255)
    m = sm((gaussian_filter(np.asarray(im, np.float32) / 255, 0.006 * n) - 0.5) / 0.14)
    grain = tooth_under(tooth, int(x), int(y), s, n)
    # the paste lies thick and even, a little less in places, and misses the top of the paper's grain here and there
    a = m * (0.95 - 0.05 * np.clip(fft_noise(n, n, 3.0, seed + 1), -1, 1)) * (1 - 0.75 * sm((grain - 1.75) / 0.4))
    lay(f, a, int(x), int(y), s, SEAL_RGB)


FOOT = (0, 1, 0, 1, 0, 1, 2)      # 七言 goes two, two, three: where each character stands in its foot
OPEN = dict(有=0.3, 尘=0.42, 疑=0.36)       # these close up and misread when heavy: no heavier than this
SMALLER = dict(疑=0.92)           # and this one, large, reads as 嶷
IN_ITS_PLACE = dict(寰=0.88)      # and this one, read off a sheet, stands too large: smaller about its own middle, in the room it had, so nothing else in the row moves
SLOW, CAESURA_REST = 0.8, 0.32    # seconds: each of a row's first two characters, and the rest in the middle of the line


def rhythm(chars, t0=0.0, t1=6.0):
    """How a row's time goes: its first two characters slow, so the strokes can be watched growing, then easing into a run, a rest in the middle of the line, and on, a character with more to it taking a little longer. Returns when each character starts and ends."""
    n = len(chars)
    base = np.zeros(n)
    for i in range(2, n):
        base[i] = 0.30 + 0.32 * np.exp(-(i - 2) / 2.0) if i < 7 else 0.27 + 0.2 * np.exp(-(i - 7) / 1.5)
    strokes = np.array([c["glyph"].count for c in chars], float)
    base[2:] *= 0.78 + 0.22 * strokes[2:] / strokes[2:].mean()
    secs = t1 - t0
    k = (secs - 2 * SLOW - CAESURA_REST) / base[2:].sum()
    t, out = t0, []
    for i in range(n):
        d = SLOW if i < 2 else base[i] * k
        out.append((t, t + d))
        t += d
        if i == 6:
            t += CAESURA_REST
    return out


def _over(A, ink, x, y, size):
    """Ink (a square image) into A, how much of the frame is covered, its corner at x, y, at this many frame pixels."""
    a = np.asarray(Image.fromarray(np.float32(np.clip(ink, 0, 1)), "F").resize((size, size), Image.LANCZOS), np.float32).clip(0, 1)
    fh, fw = A.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + size, fw), min(y + size, fh)
    b = A[y0:y1, x0:x1]
    A[y0:y1, x0:x1] = b + (1 - b) * a[y0 - y:y1 - y, x0 - x:x1 - x]


class Page:
    """The one sheet the last poem is written on. The film comes back to it four times from one camera, a row more each time; the fourth time it is signed, Opus 5.5 in ink at the foot on the right. Then, on the same sheet, the film's name is written large in the room left under the poem, and the studio's stone comes down beside the signature (film_title.py): the one red thing on paper in the film. So nothing here moves or is drawn twice: the rows have their places, the paper is one paper, each row is decided from its own seed alone, and a row once written is the same picture ever after."""

    WIDTH, PITCH, MIDDLE, CAESURA = 1300, 132, 176, 56      # the row's width, from row to row, the first row's middle line, the gap in the middle of a row
    EM, FRAME, RES, WEIGHT = 100, 125, 3, 0.46
    # the name, under the poem and across its middle: how big, its middle line, the gaps between its characters' ink as parts of an em, how heavy
    TITLE, TITLE_EM, TITLE_Y, TITLE_GAP, TITLE_WEIGHT = "字成精", 196, 792, 0.3, 0.56
    # the signature at the foot, its last letter this far in from the right end of the rows, on this line, this big; and the room kept right of it for the seal, which stands on the same line
    SIGN, SIGN_EM, SIGN_BASE, SEAL = "Opus 5.5", 60, 962, 120
    SEAL_GAP = 20

    def __init__(self, seed=11):
        self.seed = seed
        self.paper = sheet()
        self.tooth = paper_tooth(self.paper)
        self.lines = [[ch for ch in line if ch not in "，。"] for line in poem_lines()]
        self._rows = {}
        self._alpha = {}

    def plan(self, j):
        """Row j, each half filling the same width. Every character has its own size, tilt and weight, by where it falls in the 二二三 of the line, and the gaps follow what the characters take up. The brush is dipped at the start of a row and has gone nearly dry by its end."""
        if j in self._rows:
            return self._rows[j]
        chars = self.lines[j]
        r = np.random.default_rng(self.seed * 100 + j)
        half = (self.WIDTH - self.CAESURA) / 2
        x0 = (W - self.WIDTH) // 2
        written = []
        for i, ch in enumerate(chars):
            place = FOOT[i % 7]
            u = i / (len(chars) - 1)
            size = self.EM * (1.07, 0.93, 1.0)[place] * (1 + r.normal(0, 0.028)) * (1 + 0.005 * (_strokes(ch) - 8))
            weight = min(self.WEIGHT * (1.3, 0.72, 1.0)[place] * (1 + r.normal(0, 0.08)) * (1.12 - 0.24 * u), 0.8) * (1 - 0.03 * max(_strokes(ch) - 9, 0))
            weight = min(weight, OPEN.get(ch, 1.0))
            size *= SMALLER.get(ch, 1.0)
            tilt = float(np.clip(r.normal(0, 1.3), -2.6, 2.6))
            g = Glyph(ch, self.FRAME, size, tilt, self.RES)
            b = g.box(weight)
            if ch in IN_ITS_PLACE:
                g = Glyph(ch, self.FRAME, size * IN_ITS_PLACE[ch], tilt, self.RES)
            written.append(dict(glyph=g, ch=ch, u=u, weight=float(weight), box=b, seed=j * 14 + i + 1))
        plan = []
        # the gaps, a little wider between feet
        for part, start in ((written[:7], x0), (written[7:], x0 + half + self.CAESURA)):
            spare = half - sum(c["box"][2] - c["box"][0] for c in part)
            shares = np.array((1, 1.4, 1, 1.4, 1, 1), float) * (1 + r.normal(0, 0.06, 6))
            x = start
            for i, c in enumerate(part):
                wave = 3.0 * np.sin(2 * np.pi * (i / 7 + 0.31 * j)) + r.normal(0, 1.2)
                c.update(x=int(x - c["box"][0]), y=int(self.MIDDLE + j * self.PITCH + wave - (c["box"][1] + c["box"][3]) / 2))
                x += c["box"][2] - c["box"][0]
                if i < 6:
                    x += spare * shares[i] / shares.sum()
                plan.append(c)
        self._rows[j] = plan
        return plan

    def row(self, j):
        """Row j whole, as how much of the frame it covers. Written once."""
        if j not in self._alpha:
            A = np.zeros((H, W), np.float32)
            for c in self.plan(j):
                g, u = c["glyph"], c["u"]
                g.written(tooth_under(self.tooth, c["x"], c["y"], self.FRAME, g.n), weight=c["weight"], ink=1.07 - 0.3 * u, dry=0.015 + 0.4 * u ** 1.25, tone=0.985 - 0.17 * u, seed=c["seed"], most=0.68, **DRAWN.get(c["ch"], {}).get("brush", {}))
                _over(A, g.F, c["x"], c["y"], self.FRAME)
            self._alpha[j] = A
        return self._alpha[j]

    def still(self, n):
        """The sheet with its first n rows."""
        f = self.paper.copy()
        for j in range(n):
            f += (INK_RGB - f) * self.row(j)[..., None]
        return f

    def right(self):
        return (W - self.WIDTH) // 2 + self.WIDTH

    def seal_box(self):
        """Where the stone's print goes: x, y of its top left corner and its side, in frame pixels. Right of the signature, standing on its line."""
        return self.right() - self.SEAL, self.SIGN_BASE - self.SEAL, self.SEAL

    def _ink(self, c, weight, ink, dry, tone, most=0.68, press=0.36, runs=0.34):
        g = c["glyph"]
        g.written(tooth_under(self.tooth, c["x"], c["y"], c["frame"], g.n), weight=weight, ink=ink, dry=dry, tone=tone, seed=c["seed"], most=most, press=press, runs=runs, **DRAWN.get(c["ch"], {}).get("brush", {}))

    def sign_plan(self):
        """Opus 5.5, quick and small, with what is left in the brush after the fourth row, ending left of the seal's place. Each letter where the font sets it, on the one line."""
        if "sign" in self._rows:
            return self._rows["sign"]
        face = ImageFont.truetype(_font("xing", 64).path, self.SIGN_EM, index=LIGHT)
        asc, desc = face.getmetrics()
        steps = [face.getlength(ch) * {"O": 0.86, " ": 0.6}.get(ch, 1.0) for ch in self.SIGN]
        x = self.right() - self.SEAL - self.SEAL_GAP - sum(steps)
        frame = int(self.SIGN_EM * 1.6)
        mid = self.SIGN_BASE - (asc - desc) / 2          # the font's middle, where Glyph sets a letter, from the baseline
        out = []
        for i, (ch, step) in enumerate(zip(self.SIGN, steps)):
            if ch != " ":
                g = Glyph(ch, frame, self.SIGN_EM, -3.0, 4)
                out.append(dict(glyph=g, ch=ch, frame=frame, x=int(round(x + step / 2 - frame / 2)), y=int(round(mid - frame / 2)), u=i / (len(self.SIGN) - 1), seed=900 + i))
            x += step
        self._rows["sign"] = out
        return out

    def title_plan(self):
        """字成精, large and centred under the poem: the brush dipped again, fuller and heavier than for the poem, each character a little bigger than the last as a hand opens up."""
        if "title" in self._rows:
            return self._rows["title"]
        r = np.random.default_rng(self.seed * 100 + 77)
        frame = int(self.TITLE_EM * 1.3)
        chars = []
        for i, ch in enumerate(self.TITLE):
            size = self.TITLE_EM * (0.97, 1.0, 1.04)[i]
            tilt = (-1.2, 0.8, -0.6)[i]
            g = Glyph(ch, frame, size, tilt, self.RES)
            chars.append(dict(glyph=g, ch=ch, frame=frame, weight=self.TITLE_WEIGHT * (1.08, 0.94, 1.0)[i], seed=700 + i, u=i / 2))
        boxes = [c["glyph"].box(c["weight"]) for c in chars]
        gap = self.TITLE_GAP * self.TITLE_EM
        x = W / 2 - (sum(b[2] - b[0] for b in boxes) + 2 * gap) / 2
        for c, b in zip(chars, boxes):
            c.update(x=int(x - b[0]), y=int(self.TITLE_Y + r.normal(0, 2) - (b[1] + b[3]) / 2), box=b)
            x += b[2] - b[0] + gap
        self._rows["title"] = chars
        return chars

    def sign_alpha(self):
        if "sign" not in self._alpha:
            A = np.zeros((H, W), np.float32)
            for c in self.sign_plan():
                self._ink(c, weight=0.74, ink=1.06, dry=0.08 + 0.14 * c["u"], tone=0.97, most=0.55, press=0.3, runs=0.12)
                _over(A, c["glyph"].F, c["x"], c["y"], c["frame"])
            self._alpha["sign"] = A
        return self._alpha["sign"]

    def title_alpha(self):
        if "title" not in self._alpha:
            A = np.zeros((H, W), np.float32)
            for c in self.title_plan():
                self._ink(c, weight=c["weight"], ink=1.12 - 0.12 * c["u"], dry=0.02 + 0.16 * c["u"], tone=0.99, most=0.74, press=0.42, runs=0.3)
                _over(A, c["glyph"].F, c["x"], c["y"], c["frame"])
            self._alpha["title"] = A
        return self._alpha["title"]

    def finished(self):
        """All four rows and the signature: how the sheet stands when the poem has been read."""
        f = self.still(4)
        f += (INK_RGB - f) * self.sign_alpha()[..., None]
        return f

    def titled(self):
        """And the film's name under it. The seal is film_title.py's."""
        f = self.finished()
        f += (INK_RGB - f) * self.title_alpha()[..., None]
        return f


class Run:
    """Characters written one after another over a still picture: at(t) is the frame at time t. chars: each with its glyph already written, its x, y and frame; times: when each starts and ends. What is finished is kept, and only the character under way is worked out."""

    def __init__(self, base, chars, times, frame=None):
        self.base, self.chars, self.times = base, chars, times
        self.frames = [c.get("frame", frame) for c in chars]
        self.strokes = [c["glyph"].timed(*t) for c, t in zip(chars, times)]
        self.A = np.zeros((H, W), np.float32)
        self.done = 0
        ys = [c["y"] for c in chars]
        self.band = max(min(ys), 0), min(max(y + n for y, n in zip(ys, self.frames)), H)

    def at(self, t):
        if self.done and self.times[self.done - 1][1] > t:
            self.A[:] = 0
            self.done = 0
        while self.done < len(self.chars) and self.times[self.done][1] <= t:
            c = self.chars[self.done]
            _over(self.A, c["glyph"].F, c["x"], c["y"], self.frames[self.done])
            self.done += 1
        f = self.base.copy()
        y0, y1 = self.band
        f[y0:y1] += (INK_RGB - f[y0:y1]) * self.A[y0:y1, :, None]
        if self.done < len(self.chars) and self.times[self.done][0] < t:
            c = self.chars[self.done]
            lay(f, c["glyph"].at(t, self.strokes[self.done]), c["x"], c["y"], self.frames[self.done])
        return f


class Writing(Run):
    """Row j being written, the rows before it standing still. At t1 it is exactly what Page.still(j + 1) has."""

    def __init__(self, page, j, t0=0.0, t1=6.0):
        self.page = page
        chars = page.plan(j)
        page.row(j)
        super().__init__(page.still(j), chars, rhythm(chars, t0, t1), page.FRAME)


def sign_rhythm(chars, t0, t1):
    """A signature goes in one breath: the letters about even, a little longer for the ones with more to them, a lift for the space."""
    size = np.array([0.6 + 0.4 * c["glyph"].count for c in chars])
    rest = 0.18 * (t1 - t0)
    k = (t1 - t0 - rest) / size.sum()
    t, out = t0, []
    for i, (c, v) in enumerate(zip(chars, size)):
        out.append((t, t + v * k))
        t += v * k
        if c["ch"] == "s":
            t += rest
    return out


def title_rhythm(chars, t0, t1, lift=0.22):
    """The name slowly, each character for as long as it has strokes to it, not quite in proportion, and a lift of the brush between them."""
    size = np.array([c["glyph"].count ** 0.6 for c in chars])
    k = (t1 - t0 - lift * (len(chars) - 1)) / size.sum()
    t, out = t0, []
    for v in size:
        out.append((t, t + v * k))
        t += v * k + lift
    return out


def signing(page, t0, t1):
    """The signature written after the fourth row: at t1 it is Page.finished()."""
    page.sign_alpha()
    return Run(page.still(4), page.sign_plan(), sign_rhythm(page.sign_plan(), t0, t1))


def titling(page, t0, t1):
    """The name written under the signed poem: at t1 it is Page.titled()."""
    page.title_alpha()
    return Run(page.finished(), page.title_plan(), title_rhythm(page.title_plan(), t0, t1))


# ---------------------------------------------------------------- the four times, as films

WRITE_PART = 2 / 3               # of each of the first three beats the row is written in; the rest it stands (handbook 2.3: write 4, rest 2)
SIGN_AFTER = (0.6, 1.2)          # the fourth time: the pause after the last row, and how long the signature takes


def poem_beats():
    """The poem's beats on the film's clock: the three long picture-only beats that end a chapter between the ident and 十四 (八, 十 and 十一 as the clock stands), and 十四's first. [(start, seconds)]."""
    import ink_subs
    out, t = [], 0.0
    for name, _, rows in ink_subs.FILM:
        d = sum(x["dur"] for x in rows)
        if name not in ("序", "十四") and rows[-1]["text"] is None and rows[-1]["dur"] >= 5.5:
            out.append((t + d - rows[-1]["dur"], rows[-1]["dur"]))
        t += d
    assert len(out) == 3, out
    return out + [(ink_subs.end_at("poem"), ink_subs.END["poem"])]


def encoder(path, w=W, h=H, fps=30):
    import subprocess
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def clip(n, fps=30):
    """The n-th time (1 to 4) as a film, hand/poem_N.mp4, its beat long: row n written stroke by stroke where it always stands, then still. The fourth is signed after its row and then read."""
    page = Page()
    for j in range(n):
        page.row(j)
    _, secs = poem_beats()[n - 1]
    if n < 4:
        write = Writing(page, n - 1, 0.0, secs * WRITE_PART)
        runs = [(0.0, write)]
    else:
        row_end = poem_beats()[0][1] * WRITE_PART          # as long as the other rows take
        s0 = row_end + SIGN_AFTER[0]
        runs = [(0.0, Writing(page, 3, 0.0, row_end)), (s0, signing(page, s0, s0 + SIGN_AFTER[1]))]
    path = os.path.join(OUT, "poem_%d.mp4" % n)
    enc = encoder(path, fps=fps)
    last = None
    for k in range(int(round(secs * fps))):
        t = k / fps
        run = [r for a, r in runs if a <= t][-1]
        f = run.at(t)
        last = np.uint8(np.clip(f + 0.5, 0, 255))
        enc.stdin.write(last.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("encoding failed")
    Image.fromarray(last).save(os.path.join(OUT, "poem_%d_last.png" % n))
    print(path, "%.1f s" % secs, flush=True)


def poem(n=4):
    """The sheet with its first n rows, for each time the film comes back to it; at four, signed: poem_D.png."""
    page = Page()
    save(page.finished() if n == 4 else page.still(n), "poem_D" if n == 4 else "poem_D_%d" % n)


def timing(j=1, fps=30, secs=6.0):
    """Row j written in secs at fps: how long the frames take in the middle of the row and how long getting ready to write it takes."""
    t = time.time()
    page = Page()
    print("paper: %.2f s" % (time.time() - t), flush=True)
    for i in range(j + 1):
        t = time.time()
        page.row(i)
        print("row %d laid out and written once: %.2f s" % (i, time.time() - t), flush=True)
    w = Writing(page, j, 0.0, secs)
    for c, (a, b), strokes in list(zip(w.chars, w.times, w.strokes)):
        print("%s %.2f-%.2f s, %.2f s, %d strokes, %.1f frames a stroke" % (c["ch"], a, b, b - a, c["glyph"].count, (b - a) * fps / c["glyph"].count), flush=True)
    n = int(fps * secs)
    took = []
    for k in range(n + 1):
        t = time.time()
        f = w.at(k / fps)
        took.append(time.time() - t)
        if k == n // 2:
            mid = f.copy()
    took = np.array(took)
    middle = took[n // 3:2 * n // 3]
    print("all %d frames %.1f s; middle third median %.0f ms, worst %.0f ms" % (n + 1, took.sum(), 1000 * np.median(middle), 1000 * middle.max()), flush=True)
    t = time.time()
    done = np.uint8(np.clip(mid + 0.5, 0, 255))
    to_bytes = time.time() - t
    t = time.time()
    Image.fromarray(done).save(os.path.join(OUT, "poem_D_timing.png"))
    print("to 8 bits %.0f ms, PNG %.0f ms" % (1000 * to_bytes, 1000 * (time.time() - t)), flush=True)


def drawn(ch, name, tall=500):
    """A drawn character alone, this many pixels high, to set beside the sheet it was read off: in the ink it has in its row."""
    page = Page()
    j = next(i for i, line in enumerate(page.lines) if ch in line)
    c = next(c for c in page.plan(j) if c["ch"] == ch)
    f = page.paper.copy()
    d = DRAWN[ch]
    high = np.ptp([q[1] for outer, _ in d["shapes"] for q in outer])
    size = tall * d["em"] / high
    frame = int(size * 1.34)
    x, y = int(W / 2 - frame / 2), int((H - CLEAR) / 2 - frame / 2)
    g = Glyph(ch, frame, size, res=1.6)
    u = c["u"]
    lay(f, g.written(tooth_under(page.tooth, x, y, frame, g.n), weight=c["weight"], ink=1.07 - 0.3 * u, dry=0.015 + 0.4 * u ** 1.25, tone=0.985 - 0.17 * u, seed=c["seed"], most=0.68, **d.get("brush", {})), x, y, frame)
    save(f, name)


def half(j, ch):
    """The frame with row j written up to the middle of the character ch."""
    page = Page()
    w = Writing(page, j, 0.0, 6.0)
    i = page.lines[j].index(ch)
    a, b = w.times[i]
    save(w.at((a + b) / 2), "poem_D_write%d_half" % (j + 1))
    print(ch, "at %.2f s of 6" % ((a + b) / 2), flush=True)


def strip(j, i, frames=10):
    """Frames of one character of a row growing, side by side, twice the size: to see the head of a stroke."""
    page = Page()
    w = Writing(page, j, 0.0, 6.0)
    a, b = w.times[i]
    c = w.chars[i]
    x0, y0, x1, y1 = c["box"]
    x0, x1, y0, y1 = int(c["x"] + x0 - 10), int(c["x"] + x1 + 10), int(c["y"] + y0 - 10), int(c["y"] + y1 + 10)
    tiles = []
    for t in np.linspace(a, b, frames + 2)[1:-1]:
        f = np.uint8(np.clip(w.at(float(t)) + 0.5, 0, 255))[y0:y1, x0:x1]
        tiles.append(np.asarray(Image.fromarray(f).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.LANCZOS)))
    per = frames // 2
    top, bottom = np.concatenate(tiles[:per], 1), np.concatenate(tiles[per:], 1)
    save(np.float32(np.concatenate([top, bottom], 0)), "poem_D_strip%d_%d" % (j + 1, i))


def lines():
    """Every glyph, and on it the centre lines as they were laid: red where a stroke starts, going to blue. And how well each fits."""
    all_ = [ch for row in poem_lines() for ch in row if ch not in "，。"]
    seen = []
    for ch in all_:
        if ch not in seen:
            seen.append(ch)
    tile = 200
    cols = 12
    rows = (len(seen) + cols - 1) // cols
    im = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, ch in enumerate(seen):
        g = Glyph(ch, tile, 170, res=2)
        g._map()
        ink = 255 - 120 * sm((g.light * 0.4 + g.bold * 0.6) / 2 + 0.5)
        x, y = i % cols * tile, i // cols * tile
        im.paste(Image.fromarray(np.uint8(ink)).resize((tile, tile)).convert("RGB"), (x, y))
        for k, line in enumerate(g.lines):
            u = k / max(g.count - 1, 1)
            col = (int(230 * (1 - u)), 40, int(230 * u))
            d.line([(x + a / 2, y + b / 2) for a, b in line], fill=col, width=2)
            a, b = line[0]
            d.ellipse([x + a / 2 - 3, y + b / 2 - 3, x + a / 2 + 3, y + b / 2 + 3], fill=col)
        print("%s strokes %2d  line off the ink %.2f  ink with no line near %.2f  strokes with nothing of their own %d" % (ch, g.count, g.miss, g.lost, g.count - len(np.unique(g.K[g.on]))), flush=True)
    save(np.float32(np.asarray(im)), "ref_lines")


def check():
    """The poem on screen is the poem in the book's code, every character has data and a glyph, the sheet stays one sheet, and the rhythm is the one decided."""
    all_lines = poem_lines()
    assert len(all_lines) == 4
    for line in all_lines:
        a, b = line.rstrip("。").split("，")
        assert len(a) == 7 and len(b) == 7 and line.endswith("。"), line
        for ch in a + b:
            assert ch in _graphics(), ch
            assert (_distance(ch, BOLD, 120, 100) > 0).any(), ch
        print(line, "ok", flush=True)
    # the sheet: a row is the same picture in every state, and writing one comes to just that picture
    page = Page()
    for j in range(2):
        assert np.array_equal(page.row(j), Page().row(j)), j
    a, b = page.still(1), page.still(2)
    y1 = page.MIDDLE + page.PITCH - 70
    assert np.array_equal(a[:y1], b[:y1])
    w = Writing(page, 1, 0.0, 6.0)
    end = w.at(6.0)
    assert np.abs(np.uint8(end + 0.5).astype(int) - np.uint8(b + 0.5).astype(int)).max() <= 1
    # the rhythm: six seconds, its first two characters 0.8 each, a rest in the middle, and every character growing only ever darker up to the whole of it
    times = rhythm(w.chars)
    assert abs(times[-1][1] - 6.0) < 1e-6 and abs(times[0][1] - 0.8) < 1e-6 and abs(times[1][1] - 1.6) < 1e-6
    assert abs(times[7][0] - times[6][1] - CAESURA_REST) < 1e-6
    g = w.chars[4]["glyph"]
    s = g.timed(0.0, 1.0)
    was = np.zeros_like(g.F)
    for t in np.linspace(0, 1, 40):
        now = g.at(float(t), s)
        assert (now >= was - 1e-5).all()
        was = now
    assert np.abs(g.at(1.01, s) - g.F).max() < 1e-5
    print("sheet, rhythm and growth ok", flush=True)


CROPS = dict(gui=[(.45, .32), (.88, .8)], qi=[(.2, .1), (.8, .72)], ren=[(.42, .15), (.8, .88)], shen=[(.72, .2), (.72, .85)], four=[(.62, .5), (.2, .5)], poem=[(.86, .62), (.12, .3)])


def crop(name):
    """Parts of a finished frame at three times its size, the way it will be looked at, placed by where the ink is."""
    im = Image.open(os.path.join(OUT, name + ".png"))
    a = np.asarray(im.convert("L"), np.float32)
    xs, ys = np.nonzero((a < 120).any(0))[0], np.nonzero((a < 120).any(1))[0]
    w, h = 480, 270
    for i, (fx, fy) in enumerate(CROPS[name.rsplit("_", 1)[0]]):
        x = int(np.clip(xs[0] + fx * (xs[-1] - xs[0]) - w / 2, 0, W - w))
        y = int(np.clip(ys[0] + fy * (ys[-1] - ys[0]) - h / 2, 0, H - h))
        im.crop((x, y, x + w, y + h)).resize((w * 3, h * 3), Image.LANCZOS).save(os.path.join(OUT, name + ("_crop.png" if i == 0 else "_crop%d.png" % (i + 1))))
    print(name, flush=True)


if __name__ == "__main__":
    if sys.argv[1] in ("char", "four"):
        import film_big
        film_big.main(sys.argv[1:])
    elif sys.argv[1] == "clip":
        for n in sys.argv[2:] or "1234":
            clip(int(n))
    elif sys.argv[1] == "poem":
        poem(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif sys.argv[1] == "row":
        timing(int(sys.argv[2]) - 1, *(float(v) for v in sys.argv[3:5]))
    elif sys.argv[1] == "drawn":
        drawn(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "half":
        half(int(sys.argv[2]) - 1, sys.argv[3])
    elif sys.argv[1] == "strip":
        strip(int(sys.argv[2]) - 1, int(sys.argv[3]))
    elif sys.argv[1] == "lines":
        lines()
    elif sys.argv[1] == "crop":
        for n in sys.argv[2:]:
            crop(n)
    elif sys.argv[1] == "check":
        check()

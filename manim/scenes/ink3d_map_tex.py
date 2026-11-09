"""The ground and the paint of the map of meanings (ink3d_map.py). Runs in the render environment, not Blender, because it needs fonts and scipy.

    python ink3d_map_tex.py SET [--n=2400] [--tex=8192]      # SET: delta, map, route, coords or numbers -> ink/film_out/3d/map/tex/SET_{h,a}.npy

The geography says what the words mean: the island of 山 is all peaks, the island of 水 is riddled with lagoons, the island of 悲 lies low and dark,
the island of 父母 is soft hills with terraced fields, and the sky's words stand on bare pinnacles.
"""
import argparse
import math
import os
import sys
import types

sys.modules.setdefault("bpy", types.ModuleType("bpy"))       # ink3d's terrain maths does not touch Blender
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink3d import OUT3D, erode, ink_albedo, ridged, sm, spectral
from ink_film import CORPUS, font, glyph_mask

TEXDIR = os.path.join(OUT3D, "map", "tex")
SIZE = 380.0

ISLES = dict(main=(120, 222, 230, 76), shan=(-62, 64, 66, 52), shui=(76, 62, 48, 40), ren=(0, -14, 58, 34), ri=(-122, -28, 34, 40),
             bei=(-56, -92, 40, 26), fu=(64, -82, 46, 30))
CHANNELS = [([(215, 178), (150, 138), (98, 108), (40, 96)], 7.0, 13.0), ([(40, 96), (-3, 44), (-56, 8), (-92, -16)], 8.0, 8.0),
            ([(40, 96), (20, 40), (66, 22), (108, -18), (126, -60)], 7.0, 8.0), ([(-56, 8), (-62, -30), (-14, -62), (12, -116)], 4.5, 6.0)]
SHAN_PEAKS = [(-84, 88, 42, 25), (-50, 102, 30, 20), (-24, 78, 25, 16), (-110, 66, 27, 18), (-60, 68, 19, 12), (-120, 94, 21, 14)]
RI_SPIRES = [(-140, -6, 30, 8.5), (-122, 0, 25, 7.0), (-106, 4, 20, 6.0), (-146, -22, 18, 6.5), (-132, 12, 16, 5.5)]
FU_HILLS = [(44, -68, 11, 20), (84, -66, 12, 22), (62, -104, 8, 16), (100, -90, 7, 14)]
MAIN_PEAKS = [(70, 186, 36, 24), (28, 172, 26, 18), (118, 198, 42, 26), (170, 190, 36, 24), (-4, 194, 32, 24), (-60, 186, 24, 22), (150, 166, 17, 12)]
FOUR = [("男", -22, -32), ("女", 20, -20), ("王", -22, -6), ("后", 20, 6)]
NOTES = [(c, x, y - 13, 9) for c, x, y in FOUR]      # the ground under the four names' coordinates

# every place name: (char, x, y, size in world units)
LABELS = [("山", -94, 30, 24), ("岳", -68, 24, 13), ("峰", -46, 26, 13), ("岭", -80, 48, 10), ("崖", -58, 42, 10), ("谷", -34, 40, 10),
          ("江", 56, 60, 19), ("河", 80, 66, 12), ("湖", 96, 54, 12), ("海", 68, 42, 15), ("溪", 88, 36, 10), ("泉", 42, 72, 10),
          ("日", -126, -30, 20), ("月", -106, -34, 15), ("星", -134, -50, 10), ("辰", -114, -52, 10), ("风", -96, -50, 10), ("雨", -124, -66, 10),
          ("悲", -80, -90, 22), ("愁", -58, -82, 13), ("泪", -38, -88, 13), ("痛", -70, -106, 11), ("哀", -50, -106, 11), ("恨", -30, -106, 11),
          ("父", 42, -86, 19), ("母", 64, -80, 19), ("兄", 86, -86, 11), ("弟", 48, -104, 10), ("姐", 68, -102, 10), ("妹", 88, -104, 10)]
NUM_RNG = np.random.default_rng(7)


def up(a, T, method=Image.BICUBIC):
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((T, T), method), np.float32)


def seg_dist(X, Y, pts):
    d = np.full_like(X, 1e9)
    u = np.zeros_like(X)
    total = 0.0
    lens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    L = sum(lens)
    for i in range(len(pts) - 1):
        ax, ay, bx, by = *pts[i], *pts[i + 1]
        vx, vy = bx - ax, by - ay
        t = np.clip(((X - ax) * vx + (Y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        s = np.hypot(X - ax - t * vx, Y - ay - t * vy)
        better = s < d
        d = np.where(better, s, d)
        u = np.where(better, (total + t * lens[i]) / L, u)
        total += lens[i]
    return d, u


def grid(N):
    y, x = np.mgrid[:N, :N].astype(np.float32)
    return (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE


def footprint(N):
    """Where the land will be, and which island each spot belongs to. The silt of the delta shot and the islands of the map lie on the same ground."""
    X, Y = grid(N)
    n1, n2, n3 = spectral(N, 2.7, 301), spectral(N, 2.0, 302), spectral(N, 1.6, 303)
    wx, wy = spectral(N, 3.0, 304) * 13, spectral(N, 3.0, 305) * 13
    isle = {}
    base = np.full_like(X, -3.0)
    for k, (cx, cy, rx, ry) in ISLES.items():
        isle[k] = 1 - (((X + wx - cx) / rx) ** 2 + ((Y + wy - cy) / ry) ** 2)
        base = np.maximum(base, isle[k])
    coast = base + .30 * n1 + .08 * n2
    for pts, w0, grow in CHANNELS:
        d, u = seg_dist(X + wx * .6, Y + wy * .6, pts)
        w = w0 + grow * u
        coast = coast - 1.6 * (1 - sm((d - w) / 7))
    return X, Y, n1, n2, n3, wx, wy, isle, coast


def dome(X, Y, wx, wy, peaks, power, wobble):
    out = np.zeros_like(X)
    for cx, cy, top, rad in peaks:
        a_ = np.arctan2(Y - cy, X - cx)
        lobes = 1 + .22 * np.sin(a_ * 3 + cx) + .12 * np.sin(a_ * 5 + cy)
        d = np.hypot(X + wx * wobble - cx, Y + wy * wobble - cy) / (rad * lobes)
        out = np.maximum(out, top * np.exp(-d ** power))
    return out


def ground(N):
    """Heights and which island each spot belongs to."""
    X, Y, n1, n2, n3, wx, wy, isle, coast = footprint(N)
    cell = SIZE / N
    # no name may fall in the sea
    for ch, x, y, s in LABELS + [(c, px, py, 15) for c, px, py in FOUR]:
        coast = np.maximum(coast, .62 * np.exp(-(np.hypot(X - x, Y - y) / (s * 1.25)) ** 4))
    land = sm(coast / .22)
    # the land comes down to the water as a low shore, not as a cut edge
    h = (.32 * land + 2.5 * sm(coast / 1.0)) + (.55 * n2 + .25 * n3) * land * sm(coast / .5)

    # each island's character
    rid = ridged(N, 313, base=1.5, octaves=6, warp=.05)
    h = np.maximum(h, dome(X, Y, wx, wy, SHAN_PEAKS, 1.5, .9) * (.62 + .62 * rid) * sm(isle["shan"] / .25) * sm(coast / .7))
    h = np.maximum(h, dome(X, Y, wx, wy, MAIN_PEAKS, 1.5, .9) * (.62 + .62 * rid) * sm(isle["main"] / .35) * sm(coast / .7))
    h = np.maximum(h, dome(X, Y, wx, wy, RI_SPIRES, 5.0, .35) * (.86 + .22 * rid) * land)
    for cx, cy, top, rad in FU_HILLS:
        d = np.hypot(X + wx * .5 - cx, Y + wy * .5 - cy) / rad
        h = np.maximum(h, top * np.exp(-d ** 2) * land)
    h = h + .9 * sm(isle["ren"] / .4) * (1 + .6 * n1) * land
    h = erode(h, cell, 41, wavelength=22, octaves=6, gain=.5, strength=.19)

    # soft hills stay soft: the erosion is undone on the island of 父母, and on the plain where the four names lie
    keep = sm(isle["fu"] / .15) * .78 + sm(isle["ren"] / .1) * .94
    soft = gaussian_filter(h, 1.4 / cell)
    h = h * (1 - keep) + (soft + .10 * n3 + .08 * n2) * keep

    # lagoons on the island of 水, tear-shaped ponds on the island of 悲
    r = np.random.default_rng(320)
    ponds = np.zeros_like(h)
    for k, count, rmin, rmax in (("shui", 22, 2.4, 6.5), ("bei", 12, 1.6, 3.6)):
        cx, cy, rx, ry = ISLES[k]
        for _ in range(count):
            a, q = r.uniform(0, 6.283), math.sqrt(r.uniform(0, .8))
            px, py = cx + rx * q * math.cos(a), cy + ry * q * math.sin(a)
            s = r.uniform(rmin, rmax)
            if any(math.hypot(px - x, py - y) < s * 1.9 + t * .7 for _, x, y, t in LABELS):       # a lagoon must not drown a name
                continue
            stretch = 1.9 if k == "bei" else r.uniform(1.0, 1.6)
            d = np.hypot((X + wx * .2 - px) / s, (Y + wy * .2 - py) / (s * stretch))
            ponds = np.maximum(ponds, 1 - d)
    h = h - 7 * sm(ponds / .35) * (sm(isle["shui"] / .1) + sm(isle["bei"] / .1))
    h = h * (1 - .55 * sm(isle["bei"] / .25))                      # the island of 悲 lies low
    # far away the land sinks and is lost in the paper
    h = h * sm((SIZE / 2 - np.abs(X)) / 34) * sm((SIZE / 2 - Y) / 30)

    # the ground is flattened under each name, or the steep slopes would smear it
    flat = np.zeros_like(h)
    for ch, x, y, s in LABELS + [(c, px, py, 15) for c, px, py in FOUR] + NOTES:
        d = np.hypot(X - x, Y - y) / (s * 1.05)
        flat = np.maximum(flat, np.exp(-d ** 4))
    for (_, ax, ay), (_, bx, by) in ((FOUR[0], FOUR[1]), (FOUR[2], FOUR[3])):
        flat = np.maximum(flat, np.exp(-(seg_dist(X, Y, [(ax, ay), (bx, by)])[0] / 7) ** 4))
    flat = gaussian_filter(flat, 1.0 / cell)
    target = np.minimum(gaussian_filter(h, 6 / cell), 4.5)
    target = np.maximum(target, 1.6) * (h > .1)
    h = h * (1 - .92 * flat) + target * .92 * flat
    return np.maximum(h, 0).astype(np.float32), isle, X, Y


class Painter:
    """Ink laid into a texture by world position. The texture's rows run with +y, so glyphs get flipped on the way in."""

    def __init__(self, T, size=SIZE, cx=0.0, cy=0.0):
        self.T = T
        self.size = size
        self.cx, self.cy = cx, cy
        self.scale = T / size
        self.ink = np.zeros((T, T), np.float32)
        self.halo = np.zeros((T, T), np.float32)
        self.cache = {}

    def _xy(self, x, y):
        return (x - self.cx + self.size / 2) * self.scale, (y - self.cy + self.size / 2) * self.scale

    def _add(self, layer, m, x, y, k=1.0):
        h, w = m.shape
        px, py = self._xy(x, y)
        x0, y0 = int(px - w / 2), int(py - h / 2)
        xa, xb, ya, yb = max(x0, 0), min(x0 + w, self.T), max(y0, 0), min(y0 + h, self.T)
        if xa >= xb or ya >= yb:
            return
        layer[ya:yb, xa:xb] = np.maximum(layer[ya:yb, xa:xb], m[ya - y0:yb - y0, xa - x0:xb - x0] * k)

    def glyph(self, ch, kind, size, angle=0.0):
        px = max(int(size * self.scale), 6)
        ang = round(angle / 15) * 15 % 360
        k = (ch, kind, px, ang)
        if k not in self.cache:
            m = glyph_mask(ch, kind, px, max(px // 5, 2))
            if ang:
                m = np.asarray(Image.fromarray(m).rotate(ang, Image.BILINEAR, expand=True), np.float32)
            self.cache[k] = np.flipud(m).copy()
        return self.cache[k]

    def char(self, ch, x, y, size, kind="kaib", k=1.0, angle=0.0, halo=0.0):
        m = self.glyph(ch, kind, size, angle)
        if halo:
            s = max(m.shape[0] * .05, 2)
            pad = int(s * 4)
            g = gaussian_filter(np.pad(m, pad), s)
            self._add(self.halo, np.clip(g * 4.5, 0, 1), x, y, halo)
        self._add(self.ink, m, x, y, k)

    def text(self, s, x, y, size, kind="song", k=1.0):
        """A short horizontal note, its left end at x."""
        px = max(int(size * self.scale), 6)
        f = font(kind, px)
        w = int(f.getlength(s)) + 6
        im = Image.new("L", (w, int(px * 1.3)), 0)
        ImageDraw.Draw(im).text((3, 0), s, 255, font=f)
        m = np.flipud(np.asarray(im, np.float32) / 255).copy()
        self._add(self.ink, m, x + w / 2 / self.scale, y, k)

    def brush(self, pts, width, k=1.0, seed=0, dry=.6):
        """One pull of a small brush along pts, width being its half-width when pressed: down hard, lifting all the way, and made of hairs, so each one runs dry at its own place and the end of the stroke goes to streaks."""
        r = np.random.default_rng(seed)
        pts = np.asarray(pts, np.float32)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        n = int(s[-1] * self.scale * 2) + 2
        u = np.linspace(0, 1, n)
        x, y = np.interp(u * s[-1], s, pts[:, 0]), np.interp(u * s[-1], s, pts[:, 1])
        tx, ty = np.gradient(gaussian_filter(x, n / 12)), np.gradient(gaussian_filter(y, n / 12))
        m = np.hypot(tx, ty) + 1e-6
        nx, ny = -ty / m, tx / m
        w = width * (.5 + .5 * sm(u / .12)) * (1 - .86 * sm((u - .34) / .66))
        for b in np.linspace(-1, 1, max(int(width * self.scale * 4.5), 9)):
            off = b + r.normal(0, .05)
            wander = np.interp(u, np.linspace(0, 1, 14), r.normal(0, 1, 14))
            ink = sm((1 - dry * (u + abs(b) * .45) * 1.2 + .24 * wander) / .1)
            px, py = self._xy(x + nx * off * w, y + ny * off * w)
            ix, iy = np.clip(px.astype(int), 0, self.T - 1), np.clip(py.astype(int), 0, self.T - 1)
            np.maximum.at(self.ink, (iy, ix), (ink * k).astype(np.float32))

    def stroke(self, p, q, width, k=1.0, seed=0):
        """A pull of the brush, pressed at the start, lifting toward the end, with dry streaks."""
        r = np.random.default_rng(seed)
        n = int(math.hypot(q[0] - p[0], q[1] - p[1]) * self.scale / max(width * self.scale * .2, 1)) + 2
        wob = r.uniform(0, 6.28)
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        nx, ny = -(q[1] - p[1]) / L, (q[0] - p[0]) / L
        for i in range(n):
            u = i / (n - 1)
            b = math.sin(u * math.pi) * width * .35 * math.sin(wob)
            x, y = p[0] + (q[0] - p[0]) * u + nx * b, p[1] + (q[1] - p[1]) * u + ny * b
            rad = width * (.55 + .45 * sm(u / .15)) * (1 - .55 * sm((u - .55) / .45)) * self.scale
            R = int(rad) + 2
            yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
            m = np.clip(rad - np.sqrt(xx * xx + yy * yy) + .5, 0, 1).astype(np.float32)
            self._add(self.ink, m, x, y, k)


def dotted(h, isle, N, T):
    """Where the trees are: patches of small round dots on the land, thickest at the foot of the hills, none on bare rock or on the names."""
    M = 2048
    r = np.random.default_rng(370)
    H = up(h, M, Image.BILINEAR)
    gy, gx = np.gradient(gaussian_filter(H, 2), SIZE / M)
    slope = np.hypot(gx, gy)
    patch = sm((up(spectral(1024, 2.6, 371), M, Image.BILINEAR) + .15) / .9)
    where = patch * sm((H - .5) / .6) * (1 - sm((slope - .5) / .9)) * (1 - sm((H - 14) / 8))
    xs, ys = np.meshgrid((np.arange(M) + .5) / M * SIZE - SIZE / 2, (np.arange(M) + .5) / M * SIZE - SIZE / 2)
    where = where * (1 - sm((ys - 112) / 50))                       # far off, trees are too small to see
    for ch, x, y, s in LABELS + [(c, px, py, 15) for c, px, py in FOUR] + NOTES:
        where = where * (1 - np.exp(-(np.hypot(xs - x, ys - y) / (s * .95)) ** 4))
    for (_, ax, ay), (_, bx, by) in ((FOUR[0], FOUR[1]), (FOUR[2], FOUR[3])):
        where = where * (1 - np.exp(-(seg_dist(xs, ys, [(ax, ay), (bx, by)])[0] / 6) ** 4))
    dots = gaussian_filter((r.random((M, M)) < where * .085).astype(np.float32), 1.05)
    big = gaussian_filter((r.random((M, M)) < where * .012).astype(np.float32), 1.9)
    d = np.clip(dots * 5.5, 0, 1) * .34 + np.clip(big * 17, 0, 1) * .26
    return up(np.clip(d, 0, .62), T, Image.BILINEAR)


def sea_wash(h, N, T):
    D = distance_transform_edt(h < .12).astype(np.float32) * (SIZE / N)
    n = spectral(N, 2.3, 372)
    w = .15 * np.exp(-D / 6.5) * (1 + .55 * n) + .05 * np.exp(-D / 26)
    X, Y = grid(N)
    w = w * sm((SIZE / 2 - np.abs(X)) / 30) * sm((SIZE / 2 - np.abs(Y)) / 30)     # or the mesh's edge shows on the bare paper around it
    return up(.985 - np.clip(w, 0, .28), T, Image.BILINEAR)


def shore_rim(h, N, T):
    I = distance_transform_edt(h >= .12).astype(np.float32) * (SIZE / N)
    return up(np.exp(-I / 1.8), T, Image.BILINEAR)


def wave_lines(T, h, painter, X, Y):
    """The sea as old maps draw it: fine lines following the shore, the nearest ones made of very small words lapping at the land."""
    N = h.shape[0]
    d = distance_transform_edt(h < .12).astype(np.float32) * (SIZE / N)
    D = up(d, T, Image.BILINEAR)
    n = up(spectral(1024, 2.6, 330), T, Image.BILINEAR)
    D2 = D + 1.1 * n
    spacing = 2.4 + .06 * D2
    phase = np.cumsum(np.ones(1)) * 0
    lines = np.abs(np.sin(D2 / spacing * math.pi))
    thin = sm((.10 - lines) / .08)
    breaks = sm((up(spectral(1024, 1.4, 331), T, Image.BILINEAR) + .7) / .8)
    fade = (1 - sm((D - 8) / 26)) * sm((D - 7) / 1.5)
    painter.ink = np.maximum(painter.ink, thin * breaks * fade * .42)
    # small words along the shoreline: walk along the level lines of the distance
    r = np.random.default_rng(332)
    cs = list(CORPUS)
    r.shuffle(cs)
    gy, gx = np.gradient(gaussian_filter(d, 4), SIZE / N)
    cell = SIZE / N
    k = 0
    for level, count in ((2.6, 160), (5.0, 170)):
        for _ in range(count):
            for _try in range(12):
                x, y = r.uniform(-SIZE / 2 + 2, SIZE / 2 - 2, 2)
                ix, iy = int((x + SIZE / 2) / cell), int((y + SIZE / 2) / cell)
                if abs(d[iy, ix] - level) < 1.0:
                    break
            else:
                continue
            for _s in range(10):
                ix, iy = int((x + SIZE / 2) / cell), int((y + SIZE / 2) / cell)
                if not (1 < ix < N - 2 and 1 < iy < N - 2):
                    break
                g = math.hypot(gx[iy, ix], gy[iy, ix]) + 1e-6
                nx, ny = gx[iy, ix] / g, gy[iy, ix] / g
                x += nx * (level - d[iy, ix]) * .8
                y += ny * (level - d[iy, ix]) * .8
                ch = cs[k % len(cs)]
                k += 1
                if "一" <= ch <= "鿿":
                    painter.char(ch, x, y, 1.45, "kai", .62, math.degrees(math.atan2(-nx, ny)))
                x += -ny * 1.75
                y += nx * 1.75


def river_words(painter, r, lanes=5, step=1.9, size=1.5, k=.72):
    """The current in the channels: rows of very small words."""
    cs = list(CORPUS)
    r.shuffle(cs)
    c = 0
    for pts, w0, grow in CHANNELS:
        total = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
        for lane in range(-lanes, lanes + 1):
            s = r.uniform(0, step)
            for i in range(len(pts) - 1):
                ax, ay, bx, by = *pts[i], *pts[i + 1]
                L = math.hypot(bx - ax, by - ay)
                vx, vy = (bx - ax) / L, (by - ay) / L
                while s < L:
                    u = s / L
                    w = w0 + grow * (i + u) / (len(pts) - 1)
                    off = lane / lanes * w * .62
                    x = ax + vx * s - vy * off + r.normal(0, .2)
                    y = ay + vy * s + vx * off + r.normal(0, .2)
                    ch = cs[c % len(cs)]
                    c += 1
                    if "一" <= ch <= "鿿" and r.random() > .12:
                        painter.char(ch, x, y, size, "kai", k * r.uniform(.7, 1), math.degrees(math.atan2(vy, vx)) - 90)
                    s += step
                s -= L


# 五's last frame lies on the pool of 三, on the pool's own scale. See ink3d_map_numbers.py, which has the same camera and square
POOL = dict(size=110.0, cy=38.0, eye=(-.6, -10.0, 7.0), aim=(-.6, 14.0, 1.0), lens=28.0)


def pool_screen(x, y):
    """Where a point on the water falls in the picture."""
    e, t = POOL["eye"], POOL["aim"]
    p = math.atan2(e[2] - t[2], t[1] - e[1])
    F = 1920 * POOL["lens"] / 36
    D = y - e[1]
    zc = D * math.cos(p) + e[2] * math.sin(p)
    return 960 + F * (x - e[0]) / zc, 540 - F * (D * math.sin(p) - e[2] * math.cos(p)) / zc


def build_numbers(h, N, T):
    """The delta as a stain on the pool's water, in the pool's square. land: where the islands were; relief: their high ground; names: the place names. The map is not laid flat at its true proportions, or the far names would be slits
    and the near ones collide: it is laid out in the picture, as a painter would keep it readable, with a little perspective left, and then carried back onto the water."""
    size, cy = POOL["size"], POOL["cy"]
    p = Painter(8192)
    for ch, x, y, s in [*LABELS, *((c, a, b, 13) for c, a, b in FOUR)]:
        p.char(ch, x, y, s, "kaib", 1.0, 0)
    y_, x_ = np.mgrid[:T, :T].astype(np.float32)
    px, py = (x_ + .5) / T * size - size / 2, (y_ + .5) / T * size - size / 2 + cy
    U, V = pool_screen(px, py)
    U, V = np.where(py > POOL["eye"][1] + 1, U, -1e5), np.where(py > POOL["eye"][1] + 1, V, 1e5)
    f = (905 - V) / 570
    t = np.sign(f) * np.abs(f) ** (1 / .92)
    mx, my = -20 + (U - 960) / (5.8 - 1.7 * np.clip(t, 0, 1)), -110 + 182 * t
    def sample(a, order=1):
        n = a.shape[0]
        return map_coordinates(a, [(my + SIZE / 2) / SIZE * (n - 1), (mx + SIZE / 2) / SIZE * (n - 1)], order=order, mode="constant", cval=0).astype(np.float32)
    Hp = sample(h)
    land, relief = sm((Hp - .15) / .35), sm((Hp - 1.5) / 16)
    names = np.clip(sample(p.ink), 0, 1)
    np.save(os.path.join(TEXDIR, "numbers_land.npy"), land)
    np.save(os.path.join(TEXDIR, "numbers_relief.npy"), relief)
    np.save(os.path.join(TEXDIR, "numbers_names.npy"), names)
    # how it will lie in the frame, to check it without rendering
    sc = np.zeros((540, 960), np.float32)
    iy, ix = (V[::2, ::2] / 2).astype(int), (U[::2, ::2] / 2).astype(int)
    ok = (iy >= 0) & (iy < 540) & (ix >= 0) & (ix < 960)
    np.maximum.at(sc, (iy[ok], ix[ok]), (land[::2, ::2] * .35 + names[::2, ::2] * .65)[ok])
    Image.fromarray(((1 - sc) * 255).astype(np.uint8)).save(os.path.join(TEXDIR, "numbers_preview.png"))
    print("numbers", land.shape, float(land.mean()))


def build(which, N, T):
    os.makedirs(TEXDIR, exist_ok=True)
    if which == "delta":
        build_delta(N, T)
        return
    h, isle, X, Y = ground(N)
    if which == "numbers":
        build_numbers(h, N, T)
        return
    cell = SIZE / N
    a = ink_albedo(h, 21, cell)
    # the islands' colours
    a = a * .96 * (1 - .30 * sm(isle["bei"] / .2)) * (1 + .10 * sm(isle["fu"] / .2))
    a = a + (.92 - a) * .55 * sm((Y - 115) / 60)                    # the air between us and the far land
    A = up(a, T)
    H = up(h, T, Image.BILINEAR)
    # terraced fields on the soft hills
    Fu = up(sm(isle["fu"] / .15), T, Image.BILINEAR)
    ter = sm((.12 - np.abs(np.sin(H * math.pi / 1.05))) / .1) * Fu * sm((H - 1.8) / 1.5)
    A = A * (1 - .34 * ter)
    # Mi Fu's dots: trees and moss, gathered in patches
    A = A * (1 - dotted(h, isle, N, T))
    # water is bare paper, washed a little darker where it meets the land, which is what makes the land stand up on it
    sea = sm((.22 - H) / .14)
    A = A * (1 - sea) + sea_wash(h, N, T) * sea
    # and the wash dries darker along the shore
    A = A * (1 - .34 * shore_rim(h, N, T) * (1 - sea))
    p = Painter(T)
    r = np.random.default_rng(360)

    wave_lines(T, h, p, X, Y)
    river_words(p, r)
    for ch, x, y, s in LABELS:
        p.char(ch, x, y, s, "kaib", r.uniform(.9, 1), r.normal(0, 2.5), halo=.42)
    for ch, x, y in FOUR:
        p.char(ch, x, y, 15, "kaib", 1.0, 0, halo=.5)
    if which in ("route", "coords"):
        route(p, FOUR[0], FOUR[1], 11)
        route(p, FOUR[2], FOUR[3], 12)
    if which == "coords":
        rr = np.random.default_rng(23)
        for ch, x, y in FOUR:
            for i in range(2):
                s = " ".join(f"{rr.normal(0, .6):+.2f}" for _ in range(3))
                p.text(s, x - 9.5, y - 10.5 - i * 2.6, 2.2, "song", .95)
        for ch, x, y, s0 in LABELS:
            st = " ".join(f"{rr.normal(0, .6):+.2f}" for _ in range(3))
            p.text(st, x - 7.0, y - s0 * .66, 1.9, "song", .85)
    halo = gaussian_filter(p.halo, 2)
    edge = up(spectral(1024, 1.5, 361), T, Image.BILINEAR)
    halo = np.clip(halo * (1 + .35 * edge), 0, 1)
    A = A + (.93 - A) * halo * (1 - sea * .0)
    A = A * (1 - .95 * np.clip(p.ink, 0, 1))
    save(which, h, np.clip(A, .02, 1))


def distributaries(N, coast, seed=500):
    """The river breaking up as it reaches the sea: branches off the main channels, each branching again, thinner, until they give out. Returns the distance to the nearest water."""
    r = np.random.default_rng(seed)
    im = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(im)
    k = N / SIZE

    def grow(x, y, ang, w, depth):
        pts = [((x + SIZE / 2) * k, (y + SIZE / 2) * k)]
        for _ in range(int(r.uniform(14, 34) * .86 ** depth / 2.6) + 2):
            ang += r.normal(0, .24)
            x, y = x + math.cos(ang) * 2.6, y + math.sin(ang) * 2.6
            pts.append(((x + SIZE / 2) * k, (y + SIZE / 2) * k))
        draw.line(pts, fill=255, width=max(int(w * k), 1))
        ix, iy = int((x + SIZE / 2) * k), int((y + SIZE / 2) * k)
        if depth < 7 and 0 <= ix < N and 0 <= iy < N and coast[iy, ix] > -.1:
            grow(x, y, ang + r.uniform(.25, .75), w * .74, depth + 1)
            grow(x, y, ang - r.uniform(.25, .75), w * .74, depth + 1)

    for pts, w0, _ in CHANNELS:
        for i in range(len(pts) - 1):
            ax, ay, bx, by = *pts[i], *pts[i + 1]
            L = math.hypot(bx - ax, by - ay)
            s = r.uniform(4, 14)
            while s < L:
                x, y = ax + (bx - ax) * s / L, ay + (by - ay) * s / L
                grow(x, y, math.atan2(by - ay, bx - ax) + r.choice([-1, 1]) * r.uniform(.45, 1.0), r.uniform(1.8, 3.2), 0)
                s += r.uniform(9, 20)
    return distance_transform_edt(np.asarray(im) < 128).astype(np.float32) / k


def build_delta(N, T):
    """Before the islands: the river's mouth, the silt just laid down, low and cut by a thousand little channels, and all of it made of words. Mountains only far upriver."""
    X, Y, n1, n2, n3, wx, wy, isle, coast = footprint(N)
    cell = SIZE / N
    water = distributaries(N, coast)
    land = sm((coast - 1.5 * (1 - sm((water - .2) / 1.5))) / .18)
    rid = ridged(N, 313, base=1.5, octaves=6, warp=.05)
    h = land * (.30 + 1.3 * sm(coast / 1.0) * sm(water / 7) + .22 * np.exp(-((water - 1.6) / 1.0) ** 2) + .16 * n2 * sm(coast / .5))
    h = h + dome(X, Y, wx, wy, SHAN_PEAKS, 1.5, .9) * .14 * sm(isle["shan"] / .25) * land * sm(water / 5)
    far = dome(X, Y, wx, wy, MAIN_PEAKS, 1.5, .9) * (.62 + .62 * rid) * sm(isle["main"] / .35) * sm(coast / .7)
    h = erode(np.maximum(h, far), cell, 41, wavelength=22, octaves=6, gain=.5, strength=.19)
    h = np.maximum(h, 0) * sm((SIZE / 2 - np.abs(X)) / 34) * sm((SIZE / 2 - Y) / 30)
    a = ink_albedo(h, 22, cell) * .94
    a = a + (.92 - a) * .55 * sm((Y - 115) / 60)
    A, H = up(a, T), up(h, T, Image.BILINEAR)
    sea = sm((.22 - H) / .14)
    A = A * (1 - sea) + sea_wash(h, N, T) * sea
    A = A * (1 - .30 * shore_rim(h, N, T) * (1 - sea))
    # the silt: the words lie thick on the land, thin out into the water beyond each shore, and are never in the channels themselves
    p = Painter(T)
    r = np.random.default_rng(360)
    cs = [c for c in CORPUS if "一" <= c <= "鿿"]
    Dsea = distance_transform_edt(h < .12).astype(np.float32) * cell
    plume_n = spectral(N, 2.4, 501)
    spacing = 2.3
    c = 0
    for yy in np.arange(-SIZE / 2 + 2, SIZE / 2 - 2, spacing):
        for xx in np.arange(-SIZE / 2 + 2, SIZE / 2 - 2, spacing):
            x, y = xx + r.uniform(-.9, .9), yy + r.uniform(-.9, .9)
            ix, iy = int((x + SIZE / 2) / cell), int((y + SIZE / 2) / cell)
            onland = h[iy, ix] > .22
            if h[iy, ix] > 4.5:
                continue
            chance = .86 if onland else float(np.exp(-Dsea[iy, ix] / 7) * (.62 + .5 * plume_n[iy, ix]))
            if r.random() < chance:
                size = r.uniform(1.7, 3.3) if onland else r.uniform(1.5, 3.0)
                p.char(cs[c % len(cs)], x, y, size, "kai", r.uniform(.5, 1) * (1 if onland else .8), r.uniform(0, 360))
                c += 1
    river_words(p, r, lanes=2, step=2.3, size=1.7, k=.6)
    A = A * (1 - .9 * np.clip(p.ink, 0, 1))
    save("delta", h, np.clip(A, .02, 1))


def route(p, a, b, seed):
    """The march, as a brush would set it down on an old map: not a line with a head on it but a file of small tracks, each two quick sweeps pressed at the point and flicked off behind, like the bird tracks 仓颉 read.
    They get a little bigger toward where they are going. Every route bows the same way by the same amount, so two marches are the same shape as well as the same direction."""
    (_, ax, ay), (_, bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -uy, ux
    s0, s1 = 12.5, L - 10.5

    def at(s, off=0.0):
        bow = 1.5 * math.sin(math.pi * s / L) + off
        return ax + ux * s + nx * bow, ay + uy * s + ny * bow

    r = np.random.default_rng(seed)
    n = 5
    for i in range(n):
        s = s0 + (s1 - s0) * i / (n - 1) + r.uniform(-.2, .2)
        k = .82 + .28 * i / (n - 1)
        for sgn in (1, -1):
            spread, back = k * r.uniform(1.5, 1.85), k * r.uniform(3.3, 3.9)
            lean = r.uniform(-.14, .14)
            p.brush([at(s, lean), at(s - back * .38, lean + sgn * spread * .28), at(s - back * .72, lean + sgn * spread * .66), at(s - back, lean + sgn * spread)], k * .34, 1.0, seed * 11 + i * 3 + sgn, dry=.62)


def save(which, h, a):
    np.save(os.path.join(TEXDIR, f"{which}_h.npy"), h.astype(np.float32))
    np.save(os.path.join(TEXDIR, f"{which}_a.npy"), (a * 255).astype(np.uint8))
    print(which, h.shape, a.shape, "h max", float(h.max()))


if __name__ == "__main__":
    q = argparse.ArgumentParser()
    q.add_argument("which")
    q.add_argument("--n", type=int, default=2400)
    q.add_argument("--tex", type=int, default=8192)
    args = q.parse_args()
    build(args.which, args.n, args.tex)

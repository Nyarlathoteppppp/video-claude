"""千窗 — the chat window pulls back into one lit window among thousands on a night mountain; then they go out one by one.

Drawn frame by frame, with no simulation, so any frame can be rendered alone. The mountains are worked out at whatever zoom
the camera is at (ridges from tables, ink texture whose octaves fade in as the camera moves closer), so they stay ink at 70x.
Times are on the film's clock; the section starts at ink_film.QT0.

    python ink_qian.py still 95 105 120      # PNGs and a contact sheet -> ink/film_out/stills_q/
    python ink_qian.py plate 130.5           # ink/film_out/plate_qian.mp4
"""
import argparse
import math
import multiprocessing
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from ink_film import FPS, H, OUT, QT0, W, develop, font, make_paper, noise2, sm

QC = (860.0, 560.0)             # the first window's place on the screen; it never moves
QZ0 = 70.0                      # how far in the pull-back starts (the window is 16 px wide at the end)
CARD = (1120, 680)
QUESTION, ANSWER = "你好", "你好！有什么可以帮你的吗？"
Q_TYPED = (.25, .55)            # the question is typed into the box, one character at each of these times
Q_SEND = .85                    # and sent
A_TYPE = (1.4, 2.5)             # the answer streams in: when it starts, how long it takes (done before the night comes at 4.4)
Q_OFF_FIRST = 38.9              # the last window to go out is the one we started in
Q_OFF_SINGLES = [24.9, 25.7, 26.4, 27.0, 27.5, 27.9, 28.2, 28.4]
Q_OFF_LAST = [35.8, 36.5, 37.1, 37.7, 38.3]


# ======================================================================================
# the chat window
# ======================================================================================
class Card:
    """The chat window, with its lines drawn by a brush: they waver, swell and thin, and the ink goes dark and pale along them."""

    def __init__(self, seed=61):
        w, h = CARD
        r = np.random.default_rng(seed)
        def field(sigma):
            n = gaussian_filter(r.normal(size=(h, w)), sigma, mode="wrap")
            return (n / n.std()).astype(np.float32)
        self.dx, self.dy = field(46) * 2.2, field(46) * 2.2
        self.thick, self.tone = field(30), field(18)
        self.yy, self.xx = np.mgrid[:h, :w].astype(np.float32)
        self.cache = {}

    def _brush(self, m, sharp=True):
        warped = map_coordinates(m, [self.yy + self.dy, self.xx + self.dx], order=1)
        if not sharp:
            return warped * np.clip(.84 + .16 * self.tone, .5, 1)
        soft = gaussian_filter(warped, 1.3)
        return sm((soft - (.42 + .10 * self.thick)) / .16) * np.clip(.86 + .16 * self.tone, .55, 1)

    def warp(self, m):
        return map_coordinates(m, [self.yy + self.dy, self.xx + self.dx], order=1).astype(np.float32)

    def state(self, t):
        typed = sum(t >= x for x in Q_TYPED)
        sent = t >= Q_SEND
        na = int(np.clip((t - A_TYPE[0]) / A_TYPE[1], 0, 1) * len(ANSWER) + (t >= A_TYPE[0]))
        na = min(na, len(ANSWER))
        caret = not sent and t % .8 < .4
        return typed, sent, na, caret

    def image(self, t):
        k = self.state(t)
        if k not in self.cache:
            self.cache[k] = Image.fromarray(self._draw(*k))
        return self.cache[k]

    def _draw(self, typed, sent, na, caret):
        w, h = CARD
        line, fill, text = [Image.new("L", (w, h), 0) for _ in range(3)]
        dl, df, dt = ImageDraw.Draw(line), ImageDraw.Draw(fill), ImageDraw.Draw(text)
        dl.rounded_rectangle([8, 8, w - 9, h - 9], 34, outline=255, width=7)
        dl.line([(8, 88), (w - 9, 88)], fill=190, width=4)
        for i in range(3):
            dl.ellipse([40 + i * 34, 36, 60 + i * 34, 56], outline=230, width=4)
        dl.rounded_rectangle([60, h - 104, w - 60, h - 40], 22, outline=190, width=4)
        f = font("kai", 58)
        if sent:
            tw = f.getlength(QUESTION)
            dl.rounded_rectangle([w - 70 - tw - 72, 136, w - 70, 244], 28, outline=255, width=5)
            dt.text((w - 70 - tw - 36, 190), QUESTION, 255, font=f, anchor="lm")
        else:
            fi = font("kai", 42)
            dt.text((88, h - 72), QUESTION[:typed], 255, font=fi, anchor="lm")
            if caret:
                x = 90 + fi.getlength(QUESTION[:typed])
                dt.rectangle([x, h - 94, x + 3, h - 50], fill=255)
        if na:
            tw = f.getlength(ANSWER[:na])
            df.rounded_rectangle([70, 300, 70 + tw + 72, 408], 28, fill=255)
            dt.text((106, 354), ANSWER[:na], 255, font=f, anchor="lm")
        to = lambda im: np.asarray(im, np.float32) / 255
        return (self._brush(to(line)) * 3.0 + self._brush(to(fill), sharp=False) * .30 + to(text) * 3.0).astype(np.float32)


def lattice():
    """The same rectangle as a window frame with a cross in it; the bars are thick so they survive being tiny."""
    w, h = CARD
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=255, width=64)
    d.rectangle([w // 2 - 24, 0, w // 2 + 24, h], fill=255)
    d.rectangle([0, h // 2 - 24, w, h // 2 + 24], fill=255)
    return np.asarray(im, np.float32) / 255 * 3.0


def card_alpha():
    w, h = CARD
    im = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(im).rounded_rectangle([22, 22, w * 2 - 23, h * 2 - 23], 62, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255


# ======================================================================================
# the mountains
# ======================================================================================
def bilerp(tile, v, u):
    """Wrapping bilinear lookup; u, v in tile pixels."""
    n = tile.shape[0]
    iu, iv = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    fu, fv = u - iu, v - iv
    iu, iv = iu % n, iv % n
    ju, jv = (iu + 1) % n, (iv + 1) % n
    return (tile[iv, iu] * (1 - fu) + tile[iv, ju] * fu) * (1 - fv) + (tile[jv, iu] * (1 - fu) + tile[jv, ju] * fu) * fv


def znoise(tile, x, y, z, top, lo=24.0, hi=600.0):
    """Noise that looks the same at any zoom: octaves come in as the camera moves closer and go out as they grow too large."""
    out, ws = 0.0, 0.0
    k0 = max(math.ceil(math.log2(top * z / hi)), 0)
    k1 = math.floor(math.log2(top * z / lo))
    for k in range(k0, k1 + 1):
        lam = top / 2 ** k
        on = lam * z
        w = math.sin(math.pi * math.log(on / lo) / math.log(hi / lo)) * (on / hi) ** .3
        out = out + w * bilerp(tile, y / lam * 8, x / lam * 8 + k * 37.0)
        ws += w * w
    return out / math.sqrt(max(ws, 1e-6))


class Range:
    """Five ranges, far to near. The fourth rises to a peak above the first window, so that window is always on a slope."""
    #         base  amp  top   fall
    LAYERS = [(300, 110, 1.75, 150), (395, 120, 2.05, 160), (500, 120, 2.30, 170), (650, 120, 2.60, 190), (850, 100, 2.95, 190)]
    HERO = 3
    MIST = 1.2

    def __init__(self, seed=23):
        self.xs = np.arange(-400.0, 2400.0, .125)
        self.ridges, self.lines = [], []
        for i, (base, amp, _, _) in enumerate(self.LAYERS):
            r = np.random.default_rng(seed + 2 + i)
            big = np.zeros_like(self.xs)
            for L, a in ((1000, 1.0), (430, .5), (170, .18)):
                ph = r.uniform(0, 6.28)
                xs = self.xs / L * 6.28 + ph
                big += a * (2 * np.abs(np.sin(xs / 2 + .6 * np.sin(xs * .37 + ph))) - 1)
            ridge = base + amp * .55 * big + self._rough(r, 1.15) * amp * .10
            if i == self.HERO:
                ridge += (470 - np.interp(QC[0], self.xs, ridge)) * np.exp(-((self.xs - QC[0]) / 300) ** 2)
            self.ridges.append(ridge.astype(np.float32))
            self.lines.append(np.clip(.65 + .3 * self._rough(r, 1.0), .2, 1).astype(np.float32))
        t = gaussian_filter(np.random.default_rng(seed).normal(size=(1024, 1024)), 2.4, mode="wrap")
        self.t_blot = (t / t.std()).astype(np.float32)
        t = gaussian_filter(np.random.default_rng(seed + 1).normal(size=(1024, 1024)), (2.6, .9), mode="wrap")
        self.t_cun = (t / t.std()).astype(np.float32)
        self.sx = (np.arange(W // 2, dtype=np.float32) + .5) * 2
        self.sy = (np.arange(H // 2, dtype=np.float32) + .5) * 2

    def _rough(self, r, beta):
        """1-D noise along x, from hundreds of pixels down to half a pixel."""
        n = len(self.xs)
        f = np.fft.rfftfreq(n, .125)
        spec = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / np.maximum(f, 1 / 400) ** beta
        x = np.fft.irfft(spec, n)
        return x / x.std()

    def ridge(self, i, x):
        return np.interp(x, self.xs, self.ridges[i])

    @staticmethod
    def wash(w):
        """Noise -> layers of wash, each with a soft edge and a dark tide line where it stopped, at any zoom."""
        out = 0.0
        for c, a, b in ((-.75, .15, .07), (.05, .15, .09), (.8, .13, .07)):
            e = w - c
            out = out + a * sm(e / .12) + b * np.exp(-(e / .05) ** 2)
        return out.astype(np.float32) - .24

    def paint(self, z):
        """The range and the sky as ink density at frame size, at zoom z about QC."""
        xw = QC[0] + (self.sx - QC[0]) / z
        yw = QC[1] + (self.sy - QC[1]) / z
        gx, gy = np.meshgrid(xw, yw)
        blot = self.wash(znoise(self.t_blot, gx, gy, z, 2048.0, lo=18.0, hi=700.0))
        cun = znoise(self.t_cun, gx, gy, z, 1024.0, lo=10.0, hi=170.0)     # short strokes only: long ones read as streaks of light
        y = yw[:, None]
        out = 1.72 - .62 * y / H + .55 * blot
        dry, wet = np.clip(cun - .55, 0, None), np.clip(-cun - .45, 0, None)
        strokes = .35 + .65 / (1 + z / 10)
        for i, (_, _, top, fall) in enumerate(self.LAYERS):
            r = np.interp(xw, self.xs, self.ridges[i]).astype(np.float32)
            ln = np.interp(xw, self.xs, self.lines[i]).astype(np.float32)
            d = y - r[None, :]
            pos = np.maximum(d, 0)
            fade = np.exp(-pos / fall)
            body = self.MIST * (1 - fade) + top * fade
            body = body + (.55 + .55 * fade) * blot - strokes * .34 * dry * fade + strokes * .30 * wet * fade   # washes, and dry strokes down the slope
            body = body + .55 * ln[None, :] * np.exp(-pos / (2.6 + 1.2 / z))                # the dark line along the ridge
            mask = np.clip(d * z / 3.0 + .5, 0, 1)
            out = out * (1 - mask) + body * mask
        return np.asarray(Image.fromarray(np.maximum(out, .3).astype(np.float32)).resize((W, H), Image.BICUBIC), np.float32)


def scatter(x, y, gap):
    """Darts: one stays only if nothing kept is within gap, so none touch and no rows or grid show.
    Windows are wider than tall, so y counts for a little more."""
    grid, keep = {}, []
    g2 = gap * gap
    for k, (a, b) in enumerate(zip(x.tolist(), (y * 1.25).tolist())):
        i, j = int(a // gap), int(b // gap)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for u, v in grid.get((i + di, j + dj), ()):
                    if (a - u) ** 2 + (b - v) ** 2 < g2:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            grid.setdefault((i, j), []).append((a, b))
            keep.append(k)
    return np.array(keep, int)


def window_field(rg, seed=29):
    """The windows, thrown at the hills: none touch, villages come from where the darts are allowed to land.
    Each lies on the visible part of its own range: below its ridge and above the ridge of the range in front."""
    r = np.random.default_rng(seed)
    #        gap  size        brightness
    kinds = [(4.4, 2.0, 2.6, .58, .78), (5.6, 2.2, 3.0, .66, .86), (7.4, 2.8, 3.8, .76, .94), (11.5, 4.5, 6.0, .86, 1.0), (19, 8.0, 11.0, .92, 1.0)]
    parts = []
    for i, (gap, s0, s1, b0, b1) in enumerate(kinds):
        n = int(3 * W * (H - 180) / gap ** 2)
        x, y = r.uniform(0, W, n), r.uniform(180, H, n)
        s = r.uniform(s0, s1, n)
        top = rg.ridge(i, x)
        near = rg.ridge(i + 1, x) if i + 1 < len(kinds) else np.full(n, 1e9)
        fall = rg.LAYERS[i][3]
        d = y - top
        on_slope = (d > 4 + s) & (y < near - 3 - s) & (d < fall * 2.2)
        clump = noise2(270, 480, 2.5 + .3 * i, seed + 7 + i)
        c = clump[np.clip((y / 4).astype(int), 0, 269), np.clip((x / 4).astype(int), 0, 479)]
        p = .05 + .9 * sm((c + .25) / .9)
        corner = (x > 1590) & (y < 790)     # the subtitles stand here
        p = np.where(corner, p * .3, p)
        keep = np.nonzero(on_slope & (r.random(n) < p) & (np.hypot(x - QC[0], y - QC[1]) > 46))[0]
        keep = keep[scatter(x[keep], y[keep], gap)]
        x, y, s, d, corner = x[keep], y[keep], s[keep], d[keep], corner[keep]
        b = r.uniform(b0, b1, x.size) * (.70 + .30 * np.exp(-d / fall)) * np.where(corner, .6, 1)     # mist dims the lower ones
        parts.append((x, y, s, b, np.full(x.shape, i)))
    x, y, s, b, layer = [np.concatenate(a) for a in zip(*parts)]
    # the ones that go out on their own: big, bright, clear of the subtitles and of each other, with dark ground around them
    can = np.nonzero((layer >= 3) & (x > 260) & (x < 1480) & (y > 560) & (y < 1010))[0]
    can = can[np.argsort(-(s[can] + r.uniform(0, 2, can.size)))]
    picked = []
    for k in can:
        if all(math.hypot(x[k] - x[j], y[k] - y[j]) > 170 for j in picked):
            picked.append(int(k))
        if len(picked) == len(Q_OFF_SINGLES) + len(Q_OFF_LAST):
            break
    picked = np.array(picked)
    clear = np.ones(x.size, bool)
    for k in picked:
        clear &= np.hypot(x - x[k], y - y[k]) > 42
    clear[picked] = True
    s[picked], b[picked] = 11.5, 1.0
    n_kept = np.cumsum(clear) - 1
    x, y, s, b = x[clear], y[clear], s[clear], b[clear]
    picked = n_kept[picked]
    n = x.size
    d = np.hypot(x - QC[0], y - QC[1])
    on = 11.2 + 8.3 * (d / d.max()) ** .75 + r.uniform(0, 1.2, n)
    first, last = picked[:len(Q_OFF_SINGLES)], picked[len(Q_OFF_SINGLES):]
    rest = r.permutation(np.setdiff1d(np.arange(n), picked))
    off = np.zeros(n)
    off[first] = Q_OFF_SINGLES
    off[last] = Q_OFF_LAST
    off[rest] = off_times(rest.size)
    return dict(x=x, y=y, s=s, h=s * .72, on=on, off=off, b=b)


def off_times(m):
    """The crowd: slowly, then a long steady fall while the hills go dark, then a thin tail. What is left goes out on Q_OFF_LAST."""
    a, b = Q_OFF_SINGLES[-1], Q_OFF_LAST[0] - .4
    return a + (b - a) * np.sqrt(-np.log(1 - (np.arange(m) + 1) / (m + 1)) / math.log(m + 1))


# ======================================================================================
# the section
# ======================================================================================
class Qian:
    def __init__(self):
        self.rg = Range()
        self.win = window_field(self.rg)
        self.card = Card()
        self.alpha = Image.fromarray(self.card.warp(card_alpha()))
        self.lat = Image.fromarray(lattice())
        yy, xx = np.ogrid[:H, :W]
        self.xx, self.yy = xx.astype(np.float32), yy.astype(np.float32)
        ax, ay = np.maximum(np.abs(self.xx - QC[0]) - 560, 0), np.maximum(np.abs(self.yy - QC[1]) - 340, 0)
        # how far from the card: big soft lobes as ink spreads in wet paper, with fine feathering only along the edge
        self.dn = np.hypot(ax / 860, ay / 400) + (.13 * noise2(H, W, 4.2, 37) + .012 * noise2(H, W, 2.0, 38)).astype(np.float32)

    @staticmethod
    def zoom(t):
        if t < 6:
            return QZ0 * (1.05 - .05 * t / 6)
        if t < 16.5:     # an even pace, about 1.6x a second, easing in and out
            x, r = (t - 6) / 10.5, .12
            e = x * x / (2 * r) / (1 - r) if x < r else 1 - (1 - x) ** 2 / (2 * r) / (1 - r) if x > 1 - r else (x - r / 2) / (1 - r)
            return math.exp(math.log(QZ0) * (1 - e) + math.log(1.06) * e)
        return 1.06 - .06 * min((t - 16.5) / 23, 1)

    def windows(self, t, z):
        """The lit windows as a mask at frame size, or None before any comes on.
        Each is drawn with its exact coverage of every pixel, so the small ones glide and do not shimmer as the camera drifts."""
        w = self.win
        a = np.minimum(np.clip((t - w["on"]) / .35, 0, 1), 1 - np.clip((t - w["off"]) / .25, 0, 1)) * w["b"]
        X, Y = QC[0] + (w["x"] - QC[0]) * z, QC[1] + (w["y"] - QC[1]) * z
        hw, hh = w["s"] * z / 2, w["h"] * z / 2
        vis = (a > .01) & (X + hw > 0) & (X - hw < W) & (Y + hh > 0) & (Y - hh < H)
        if not vis.any():
            return None
        a, X, Y, hw, hh = a[vis], X[vis], Y[vis], hw[vis], hh[vis]
        nx, ny = int(2 * hw.max()) + 3, int(2 * hh.max()) + 3
        ix, iy = np.floor(X - hw).astype(int)[:, None] + np.arange(nx), np.floor(Y - hh).astype(int)[:, None] + np.arange(ny)
        span = lambda i, c, h: np.clip(np.minimum(i + 1, (c + h)[:, None]) - np.maximum(i, (c - h)[:, None]), 0, 1)
        bar = np.maximum(hw / 11, .3)
        bx, by = span(ix, X, bar)[:, None, :], span(iy, Y, bar)[:, :, None]
        cross = np.clip((2 * hw - 5) / 2.5, 0, 1)[:, None, None]    # close enough to see the cross in the window; it comes in gradually
        m = a[:, None, None] * span(iy, Y, hh)[:, :, None] * span(ix, X, hw)[:, None, :] * (1 - cross * (bx + by - bx * by))
        R, C = np.repeat(iy[:, :, None], nx, 2), np.repeat(ix[:, None, :], ny, 1)
        ok = (m > 0) & (R >= 0) & (R < H) & (C >= 0) & (C < W)
        out = np.zeros((H, W), np.float32)
        np.add.at(out, (R[ok], C[ok]), m[ok].astype(np.float32))
        return np.minimum(out, 1)

    def frame(self, t, detail, tone):
        """The plate at t seconds into the section."""
        z = self.zoom(t)
        sc = z / QZ0
        u = (t - 4.4) / 2.2
        if u <= 0:
            hi = np.zeros((H, W), np.float32)
        else:
            hi = self.rg.paint(z)
            if u < 1:    # the night soaks in from the edges of the paper: a crisp wet edge, pale behind the front and deepening, closing on the window
                e = self.dn - (1.3 - 1.85 * float(sm(u)))
                hi = sm(e / .022) * (hi * (.42 + .58 * sm(e / .5)) + .22 * np.exp(-e / .035))
        lit = self.windows(t, z)
        if lit is not None:     # the windows, and the light they throw on the slopes around them
            g = gaussian_filter(lit[::4, ::4], 2.6)
            g = np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR), np.float32)
            hi = hi * (1 - np.clip(lit * .98 + g * .85, 0, .985))
        off = 1 - float(sm((t - Q_OFF_FIRST) / .3))
        hw, hh = 560 * sc, 340 * sc
        if off > 0:
            if u > 0:   # the light of this one
                dist = np.hypot(np.maximum(np.abs(self.xx - QC[0]) - hw, 0), np.maximum(np.abs(self.yy - QC[1]) - hh, 0))
                hi = hi * (1 - .36 * off * float(sm((.5 - sc) / .4)) * np.exp(-dist / (22 + 30 * sc)))    # none while it is still a big card, or it reads as a halo
            x0, y0 = QC[0] - hw, QC[1] - hh
            ia, ja, ib, jb = max(int(x0), 0), max(int(y0), 0), min(int(QC[0] + hw) + 2, W), min(int(QC[1] + hh) + 2, H)
            box = ((ia - x0) / sc, (ja - y0) / sc, (ib - x0) / sc, (jb - y0) / sc)
            box = (max(box[0], 0), max(box[1], 0), min(box[2], CARD[0]), min(box[3], CARD[1]))
            size = (ib - ia, jb - ja)
            A = np.clip(np.asarray(self.alpha.resize(size, Image.LANCZOS, box=box), np.float32), 0, 1) * off
            D = np.asarray(self.card.image(t).resize(size, Image.LANCZOS, box=box), np.float32)
            la = float(sm((.25 - sc) / .12))
            if la > 0:
                D = D * (1 - la) + np.asarray(self.lat.resize(size, Image.LANCZOS, box=box), np.float32) * la
            hi[ja:jb, ia:ib] = hi[ja:jb, ia:ib] * (1 - A) + np.maximum(D, 0) * A
        return develop(hi, 0, detail, tone)


# ======================================================================================
# passes
# ======================================================================================
STILLS = os.path.join(OUT, "stills_q")


def sheet(times, cols=3, tw=640):
    ims = [Image.open(os.path.join(STILLS, f"s{x:06.2f}.png")).resize((tw, tw * 9 // 16), Image.LANCZOS) for x in times]
    rows_ = math.ceil(len(ims) / cols)
    s = Image.new("RGB", (cols * tw, rows_ * tw * 9 // 16), (30, 30, 30))
    for i, im in enumerate(ims):
        s.paste(im, ((i % cols) * tw, (i // cols) * tw * 9 // 16))
    s.save(os.path.join(STILLS, "sheet.png"))


def cmd_still(times):
    os.makedirs(STILLS, exist_ok=True)
    q = Qian()
    print(len(q.win["x"]), "windows")
    detail, tone = make_paper()
    for x in times:
        Image.fromarray(q.frame(x - QT0, detail, tone)).save(os.path.join(STILLS, f"s{x:06.2f}.png"))
    sheet(sorted(times))


_w = {}


def _init():
    _w["q"] = Qian()
    _w["paper"] = make_paper()


def _render(n):
    return _w["q"].frame(n / FPS, *_w["paper"]).tobytes()


def cmd_plate(b, jobs):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "plate_qian.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    total = int(round((b - QT0) * FPS))
    with multiprocessing.Pool(jobs, _init) as pool:
        for n, buf in enumerate(pool.imap(_render, range(total), chunksize=6)):
            p.stdin.write(buf)
            if n % 150 == 0:
                print(f"t={QT0 + n / FPS:6.2f}", flush=True)
    p.stdin.close()
    p.wait()
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["still", "plate"])
    ap.add_argument("args", nargs="+", type=float)
    ap.add_argument("--jobs", type=int, default=10)
    o = ap.parse_args()
    if o.cmd == "still":
        cmd_still(o.args)
    else:
        cmd_plate(o.args[0], o.jobs)

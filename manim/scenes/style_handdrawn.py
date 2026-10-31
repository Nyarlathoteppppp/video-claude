"""逐帧手绘：紺紙金泥的红白锦鲤。Gold line and gouache on indigo paper, one drawing every 1/12 s.

The fish is a chain of curvature along its length, limited by its own width, so it cannot fold.
Fins follow the body's older directions, which gives real follow-through. Every line is a brush
stroke redrawn with a small wobble on every drawing, and the paint sits a little off the line.
"""
import math
from functools import lru_cache

import cairo
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from styles_common import H, W, alpha_of, hx, poly, rgb_of, surface

RATE = 12
L = 980.0
AX = np.linspace(0, 1, 240) ** 1.25
WIDTH = PchipInterpolator(
    (0, .0008, .003, .007, .012, .02, .035, .07, .12, .2, .3, .42, .55, .68, .8, .9, 1),
    (0, .0065, .0138, .0225, .031, .0415, .0558, .0774, .090, .106, .118, .120, .106, .084, .062, .044, .032))(AX) * L
TG = np.arange(-1.0, 5.75, 1 / 120)

INDIGO = np.array(hx("#13233C"), np.float32)
GOLD = hx("#E5B65E")
GOLD_DIM = hx("#B48F4C")
UMBER = hx("#8D6A34")
SHELL = hx("#F4ECDD")
RED = hx("#DE3A1B")
SUMI = hx("#1B1D29")
FIN = hx("#F8EFDC")


def keyed(*keys):
    k = np.array(keys, float)
    return PchipInterpolator(k[:, 0], k[:, 1])(TG)


@lru_cache(1)
def motion():
    heading = np.radians(keyed((-1, -12), (0, -24), (1, -36), (1.75, -44), (2.1, -10), (2.36, 34), (2.5, 62),
                               (2.7, 58), (3, 40), (3.6, 26), (4.4, 18), (5.75, 14)))
    bend = keyed((-1, .3), (0, .45), (1, .9), (1.75, .8), (2.1, 3.2), (2.36, 5.4), (2.46, 2.8), (2.56, -3.6),
                 (2.7, -1.2), (2.86, 1.7), (3.1, -.7), (3.5, .45), (4.2, .5), (5.75, .45))
    amp = keyed((-1, 2.7), (1.7, 2.7), (2.1, .9), (2.4, .5), (2.6, 2.2), (2.9, 3.4), (3.5, 2.4), (4.4, 1.9), (5.75, 1.9))
    freq = keyed((-1, .95), (1.7, .95), (2.1, .5), (2.4, .3), (2.5, 2.2), (2.8, 2.9), (3.2, 1.6), (4, .9), (5.75, .75))
    speed = keyed((-1, 150), (1.5, 135), (1.9, 70), (2.25, 20), (2.4, 30), (2.5, 900), (2.62, 1650), (2.8, 1200),
                  (3.2, 520), (3.8, 210), (4.4, 110), (5, 80), (5.75, 80))
    squash = keyed((-1, 1), (1.9, 1), (2.3, 1.07), (2.42, 1.08), (2.56, .9), (2.75, .93), (3.1, 1), (5.75, 1))
    tuck = keyed((-1, 1), (2.1, 1.1), (2.42, 1.2), (2.6, .42), (3.2, .6), (4.2, .86), (5.75, .8))
    phase = np.cumsum(freq) / 120
    limit = L / (1.35 * np.maximum(WIDTH, 18))
    env = (.12 + AX) ** 1.35
    kappa = bend[:, None] * (.2 + 1.3 * AX) + amp[:, None] * env * np.sin(2 * np.pi * (1.05 * AX - phase[:, None]))
    kappa = limit * np.tanh(kappa / limit)
    theta = heading[:, None] - cumtrapz(kappa, AX, axis=1, initial=0)
    rel = -L * cumtrapz(np.stack((np.cos(theta), np.sin(theta)), -1), AX, axis=1, initial=0)
    mid = int(np.searchsorted(AX, .45))
    rel -= rel[:, mid:mid + 1]
    v = np.stack((np.cos(theta[:, mid]), np.sin(theta[:, mid])), -1) * speed[:, None]
    centre = np.cumsum(v, axis=0) / 120
    centre -= centre[int(np.searchsorted(TG, 0.0))]
    lag = centre.copy()
    for i in range(1, len(lag)):
        lag[i] = lag[i - 1] + (centre[i] - lag[i - 1]) * (1 / 120) / .32
    return dict(spine=centre[:, None] + rel, theta=theta, kappa=kappa, centre=centre, lag=lag,
                squash=squash, tuck=tuck)


def in_time(a, t):
    x = np.clip((t - TG[0]) * 120, 0, len(TG) - 1.001)
    i = int(x)
    return a[i] * (1 - (x - i)) + a[i + 1] * (x - i)


def angle(u, times):
    return np.interp(times, TG, motion()["theta"][:, int(np.searchsorted(AX, u))])


class Fish:
    def __init__(self, t):
        m = motion()
        self.t = t
        self.p = in_time(m["spine"], t)
        self.th = in_time(m["theta"], t)
        self.kappa = in_time(m["kappa"], t)
        self.w = WIDTH * float(in_time(m["squash"], t))
        self.tuck = float(in_time(m["tuck"], t))
        self.n = np.stack((-np.sin(self.th), np.cos(self.th)), -1)

    def at(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        w = np.interp(u, AX, self.w) * v
        return np.stack((np.interp(u, AX, self.p[:, 0]) + w * np.interp(u, AX, self.n[:, 0]),
                         np.interp(u, AX, self.p[:, 1]) + w * np.interp(u, AX, self.n[:, 1])), -1)


class Camera:
    def __init__(self, t):
        m = motion()
        fish = Fish(t)
        head = fish.at(.11, 0.0)
        e = np.clip(t / 2.1, 0, 1)
        e = e * e * (3 - 2 * e)
        drift = np.interp(t, (3.6, 5), (-60., -125.))              # the frame falls slowly behind, so the fish is still going somewhere
        self.focus = head * (1 - e) + (in_time(m["lag"], t) + np.array((drift, 0.))) * e
        self.anchor = np.array((1090., 540.)) * (1 - e) + np.array((960., 540.)) * e
        self.z = float(np.exp(np.interp(t, (0, 2.1, 2.55, 4.4, 5), np.log((2.45, 1.0, .9, .97, .97)))))

    def xy(self, p):
        return self.anchor + self.z * (np.asarray(p) - self.focus)


# --------------------------------------------------------------------------------------
# paper and brush
# --------------------------------------------------------------------------------------
def norm(a):
    return (a - a.mean()) / (a.std() + 1e-6)


@lru_cache(1)
def stock():
    r = np.random.default_rng(11)
    big = norm(gaussian_filter(r.standard_normal((H // 4, W // 4)), 22))
    big = np.repeat(np.repeat(big, 4, 0), 4, 1)
    big = gaussian_filter(big, 4)
    mid = norm(gaussian_filter(r.standard_normal((H, W)), 9))
    fibre = norm(gaussian_filter(r.standard_normal((H, W)), (.5, 7)) + gaussian_filter(r.standard_normal((H, W)), (7, .5)))
    grain = r.standard_normal((H, W))
    y, x = np.mgrid[0:H, 0:W]
    vig = 1 - .42 * (((x - W / 2) / (W * .62)) ** 2 + ((y - H / 2) / (H * .62)) ** 2)
    tone = (1 + .13 * big + .04 * mid + .05 * fibre + .012 * grain) * vig
    return (INDIGO[None, None] * tone[..., None]).astype(np.float32), big.astype(np.float32), mid.astype(np.float32), fibre.astype(np.float32)


def wob(n, seed, step=0):
    """Smooth 1-D noise, about +-1, a pure function of (seed, step)."""
    s = np.linspace(0, 1, n)
    out = 0
    for i, f in enumerate((1.3, 3.1, 6.7)):
        ph = math.modf(math.sin(seed * 12.9898 + step * 78.233 + i * 37.719) * 43758.5453)[0] * 6.283
        out = out + np.sin(s * f * 6.283 + ph) / (1 + i * .7)
    return out / 1.9


def jiggle(seed, step, a):
    return a * math.sin(seed * 3.1 + step * 17.7), a * math.cos(seed * 5.3 + step * 13.1)


def brush(c, pts, width, seed, step, col, a=1.0, taper=(.14, .2), boil=.8, press=None):
    """A stroke as a filled ribbon: it thickens and thins, and is redrawn a little differently on every drawing."""
    n = len(pts)
    if n < 3:
        return
    s = np.linspace(0, 1, n)
    w = width * np.minimum(1, s / max(taper[0], 1e-3)) ** .7 * np.minimum(1, (1 - s) / max(taper[1], 1e-3)) ** .7
    w = w * (1 + .3 * wob(n, seed, 0)) * (1 + .08 * wob(n, seed + 3, step))
    if press is not None:
        w = w * press
    p = np.asarray(pts) + boil * np.stack((wob(n, seed + 1, step + 1), wob(n, seed + 2, step + 1)), -1)
    d = np.gradient(p, axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1])[:, None], 1e-6)
    side = np.stack((-d[:, 1], d[:, 0]), -1) * (w[:, None] / 2)
    c.set_source_rgba(*col, a)
    poly(c, np.vstack((p + side, (p - side)[::-1])))
    c.fill()


def over(img, layer):
    """Premultiplied surface over a float image."""
    img *= 1 - alpha_of(layer)[..., None]
    img += rgb_of(layer)


def layer():
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    return s, cairo.Context(s)


# --------------------------------------------------------------------------------------
# water: gold lines that part around the fish and curl in its wake
# --------------------------------------------------------------------------------------
@lru_cache(1)
def vortices():
    out = []
    sign = 1
    for i, ti in enumerate(np.arange(-.9, 2.0, .53)):
        out.append((ti, *tail_tip(ti), sign * 1.05, 62))
        sign = -sign
    for ti, st, r in ((2.5, 3.1, 175), (2.66, -2.5, 140), (2.9, 1.7, 110), (3.3, -1.1, 80), (3.8, .9, 70)):
        out.append((ti, *tail_tip(ti), st, r))
    return out


TAIL_SHAPE = PchipInterpolator((0, .3, .6, .78, .92, 1), (.60, .66, .92, 1.0, .90, .70))
U_OUT = np.concatenate((AX[AX < .93], np.linspace(.93, 1, 30)))


def closing(u):
    """The flanks close in a curve over the root of the tail, as the scaled skin does on a real koi."""
    return np.sqrt(np.clip(1 - (np.clip(np.asarray(u, float) - .93, 0, None) / .07) ** 2, 0, 1))


def tail_rays(t, count=33):
    fish = Fish(t)
    s = np.linspace(0, 1, 26)
    rays = []
    for j, k in enumerate(np.linspace(-1, 1, count)):
        shape = float(TAIL_SHAPE(abs(k))) * (1 + .002 * math.sin(j * 2.3))
        length = L * .5 * shape
        r = s * shape                                           # distance along the ray: all rays bend alike, so they never cross
        base = angle(1.0, t)
        late = angle(1.0, t - .19 * r)
        late = base + 1.35 * np.tanh((late - base) / 1.35)      # a whip bends the fin, it must never coil it
        ang = late + math.pi - k * math.radians(38) * (.5 + .5 * r)
        ang = ang + .22 * np.sin(2 * np.pi * (1.15 * r - 1.1 * t) + k * 1.4) * r
        d = np.stack((np.cos(ang), np.sin(ang)), -1)
        rays.append(fish.at(.945, k * .92) + length * cumtrapz(d, s, axis=0, initial=0))
    return rays


@lru_cache(512)
def tail_tip(t):
    p = tail_rays(float(t), 3)[1][-1]
    return float(p[0]), float(p[1])


def water(cam, fish, t, step):
    s, c = layer()
    z = cam.z
    x0, x1 = cam.focus[0] + (np.array((-60., W + 60.)) - cam.anchor[0]) / z
    y0, y1 = cam.focus[1] + (np.array((-160., H + 160.)) - cam.anchor[1]) / z
    sp = 27.0
    q = fish.at(np.linspace(.0, 1, 120), 0.0)
    qw = np.interp(np.linspace(.0, 1, 120), AX, fish.w)
    v = [x for x in vortices() if x[0] < t]
    n = int(min(620, (x1 - x0) * z / 5))
    x = np.linspace(x0, x1, n)
    tilt = -.14
    k0 = int((y0 - tilt * x1 - 60) / sp) - 2
    k1 = int((y1 - tilt * x0 + 60) / sp) + 2
    tail = np.array(tail_tip(2.44))
    for k in range(k0, k1):
        y = k * sp + tilt * x + 34 * np.sin(x * .0031 + k * .19 + .25 * t) + 16 * np.sin(x * .0079 - k * .11 + .5 * t + 1.7) \
            + 7 * np.sin(x * .021 + k * .37 - .7 * t)
        p = np.stack((x, y), -1)
        for ti, vx, vy, st, r0 in v:
            a = t - ti
            d = p - (vx, vy)
            r2 = (d ** 2).sum(1)
            rr = r0 + 130 * a ** .6
            rot = st * (1 - math.exp(-a * 6)) * math.exp(-a * .3) * np.exp(-r2 / (rr * rr))
            co, si = np.cos(rot), np.sin(rot)
            p = np.stack((vx + d[:, 0] * co - d[:, 1] * si, vy + d[:, 0] * si + d[:, 1] * co), -1)
        press = 1.0
        for ring, amp in ((2.44, 30.0), (2.62, 18.0)):
            if t > ring:
                a = t - ring
                d = p - tail
                r = np.hypot(d[:, 0], d[:, 1]) + 1e-3
                f = np.exp(-((r - 980 * a) / (90 + 90 * a)) ** 2) * math.exp(-a * .8)
                p = p + d / r[:, None] * (amp * np.sin((r - 980 * a) / 42) * f)[:, None]
                press = press + 1.7 * f
        d2 = ((p[:, None] - q[None]) ** 2).sum(2)
        near = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(n), near]) + 1e-3
        h = qw[near] * 1.05 + 18
        push = (p - q[near]) / dist[:, None] * (h * np.exp(-(dist / (h * 1.7)) ** 2))[:, None]
        p = p + gaussian_filter1d(push, 7, axis=0, mode="nearest")
        tone = .5 + .42 * (math.sin(k * 2.399) * .5 + .5) ** 2
        width = 1.9 * z ** .8 * (.72 + .28 * math.sin(k * 1.7))
        gap = np.sin(x * .0057 + k * 2.1) + .6 * np.sin(x * .0137 + k * .7)
        keep = gap > -1.15
        edges = np.flatnonzero(np.diff(np.concatenate(((0,), keep.astype(int), (0,)))))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > 8:
                seg_press = press[a:b] if isinstance(press, np.ndarray) else None
                brush(c, cam.xy(p[a:b]), width, 5000 + k + a, step, GOLD_DIM, tone, taper=(.06, .08), boil=.7, press=seg_press)
    _, big, _, fibre = stock()
    return s, (.82 + .1 * fibre + .08 * big)


# --------------------------------------------------------------------------------------
# fins
# --------------------------------------------------------------------------------------
def smooth_line(pts, k=6):
    n = len(pts)
    x = np.arange(n)
    q = np.linspace(0, n - 1, (n - 1) * k + 1)
    return np.stack((PchipInterpolator(x, pts[:, 0])(q), PchipInterpolator(x, pts[:, 1])(q)), -1)


def lifted(c, pts, width, seed, step, col, a):
    """A line the brush keeps lifting from."""
    n = len(pts)
    on = wob(n, seed, 0) + .6 * np.sin(np.arange(n) * 8.0 / max(n, 1) * 6.283 + seed) > -.05
    edges = np.flatnonzero(np.diff(np.concatenate(((0,), on.astype(int), (0,)))))
    for e0, e1 in zip(edges[::2], edges[1::2]):
        if e1 - e0 > 5:
            brush(c, pts[e0:e1], width, seed + e0, step, col, a, taper=(.25, .35), boil=.5)


def fin(paint, ink, rays, z, seed, step, density=1.0, ray_w=1.1, tint=None, veil=.34):
    tips = smooth_line(np.array([r[-1] for r in rays]))
    root = np.mean([r[0] for r in rays], axis=0)
    reach = max(float(np.hypot(*(r[-1] - root))) for r in rays) + 1
    wash = cairo.RadialGradient(root[0], root[1], 0, root[0], root[1], reach)
    if tint is None:
        wash.add_color_stop_rgba(0, *hx("#F7DCA8"), .94 * density)
        wash.add_color_stop_rgba(.45, *FIN, (.42 + .6 * veil) * density)
        wash.add_color_stop_rgba(1, *FIN, veil * density)
    else:
        wash.add_color_stop_rgba(0, *tint, density)
        wash.add_color_stop_rgba(1, *tint, density * .6)
    paint.set_source(wash)
    poly(paint, np.vstack((rays[0], tips, rays[-1][::-1])))
    paint.fill()
    for j in range(0, len(rays) - 1, 2):
        paint.set_source_rgba(*FIN, (.09 + .05 * math.sin(j * 2.1 + seed)) * density)
        poly(paint, np.vstack((rays[j], rays[j + 1][::-1])))
        paint.fill()
    for j, r in enumerate(rays):
        wide = (1.25 if j in (0, len(rays) - 1) else .8 + .25 * math.sin(j * 1.9 + seed)) * ray_w
        brush(ink, r, wide * z ** .8, seed + j * 7, step, GOLD, .92, taper=(.02, .18), boil=.6)
    lifted(ink, tips, 1.5 * z ** .8, seed + 99, step, GOLD, .85)


PECTORAL = PchipInterpolator((0, .25, .55, .8, .93, 1), (.40, .62, .92, 1.0, .93, .74))


def pectoral(fish, side, t):
    base = fish.at(.215, side * .93)
    s = np.linspace(0, 1, 22)
    rays = []
    open_ = fish.tuck * (1 + .12 * math.sin(2 * np.pi * .7 * t + side))
    for j, k in enumerate(np.linspace(0, 1, 14)):
        a = math.radians(10 + 62 * k * open_)
        length = L * .37 * float(PECTORAL(k))
        ang = angle(.2, t - .22 * s) + math.pi - side * a * (.65 + .35 * s)
        ang = ang + .13 * np.sin(2 * np.pi * (1.1 * s - .9 * t) + k * 2.2 + side) * s
        d = np.stack((np.cos(ang), np.sin(ang)), -1)
        rays.append(base + fish.n[int(np.searchsorted(AX, .215))] * 0 + length * cumtrapz(d, s, axis=0, initial=0))
    return rays


def ventral(fish, side, t):
    base = fish.at(.52, side * .86)
    s = np.linspace(0, 1, 16)
    rays = []
    for k in np.linspace(0, 1, 9):
        a = math.radians(8 + 42 * k * fish.tuck)
        length = L * .2 * (.62 + .38 * math.sin(math.pi * k ** .8))
        ang = angle(.5, t - .2 * s) + math.pi - side * a
        ang = ang + .1 * np.sin(2 * np.pi * (s - t) + k * 2) * s
        d = np.stack((np.cos(ang), np.sin(ang)), -1)
        rays.append(base + length * cumtrapz(d, s, axis=0, initial=0))
    return rays


def dorsal_rays(ink, rays, z, step):
    """From above the back fin is a row of rays standing on the spine: filled in, it reads as a stain."""
    for j, r in enumerate(rays):
        brush(ink, r, 1.0 * z ** .8, 90 + j * 7, step, GOLD, .62, taper=(.02, .3), boil=.3)
    brush(ink, smooth_line(np.array([r[-1] for r in rays])), 1.1 * z ** .8, 190, step, hx("#F7DCA8"), .7, taper=(.2, .2), boil=.3)


def dorsal(fish):
    us = np.linspace(.31, .74, 19)
    kappa = np.interp(us, AX, fish.kappa)
    lean = np.tanh(kappa * .9) * .9 + .1
    rays = []
    for u, side in zip(us, lean):
        h = .042 * L * math.sin(math.pi * (u - .3) / .45) ** .6 * (.45 + .55 * abs(side))
        base = fish.at(u, 0.0)
        tip = fish.at(u + .035, 0.0) + np.array((np.interp(u, AX, fish.n[:, 0]), np.interp(u, AX, fish.n[:, 1]))) * h * np.tanh(side * 2.5)
        rays.append(base + (tip - base) * np.linspace(0, 1, 9)[:, None])
    return rays


# --------------------------------------------------------------------------------------
# the body: every scale is painted, and colour patches follow the scales as they do on a real koi
# --------------------------------------------------------------------------------------
ROWS, ACROSS = 35, 15


@lru_cache(1)
def scales():
    r = np.random.default_rng(23)
    us = np.linspace(.205, .975, ROWS)
    du = us[1] - us[0]
    phis = np.radians(np.linspace(-82, 82, ACROSS))
    dphi = phis[1] - phis[0]
    steps = ((.325, -.2, .092, .82), (.565, .24, .098, .78), (.80, -.14, .08, .86))
    ink_spots = ((.448, .66, .032, .34), (.68, -.6, .03, .36), (.905, .5, .03, .55), (.26, .78, .02, .22), (.56, -.72, .02, .2))
    out = []
    for i, u in enumerate(us):
        for j, ph in enumerate(phis):
            ph = ph + (dphi / 2 if i % 2 else 0)
            if abs(ph) > math.radians(88):
                continue
            v = math.sin(ph)
            e = .10 * math.sin(u * 23 + v * 3.1) + .09 * math.sin(u * 41 - v * 5.3 + 1.2) + .07 * math.sin(u * 9 + v * 7.7 + 2.1)
            d = min(((u - a) / ra) ** 2 + ((v - b) / rb) ** 2 for a, b, ra, rb in steps) + e
            kind = 1 if d < 1 else 0
            if min(((u - a) / ra) ** 2 + ((v - b) / rb) ** 2 for a, b, ra, rb in ink_spots) + e * .6 < 1:
                kind = 2 if r.random() > .22 else kind
            out.append((i, u, ph, kind, float(r.random()), float(r.random())))
    return out, du, dphi


def paint_scales(c, ink, fish, cam, z, step):
    table, du, dphi = scales()
    a = np.radians(np.linspace(-90, 90, 11))
    ca, sa = np.cos(a), np.sin(a)
    for i, u, ph, kind, r1, r2 in reversed(table):
        uu = u + du * 1.22 * ca
        vv = np.sin(np.clip(ph + dphi * .60 * sa, -1.56, 1.56))
        rim = cam.xy(fish.at(uu, vv))
        back_ = cam.xy(fish.at(np.array((u - du * .3,)), np.array((math.sin(ph),))))
        if kind == 1:
            base = (.87 + .05 * r1, .22 + .07 * r1, .10 + .03 * r2)
            lit = (.96, .38 + .08 * r2, .19)
            line = (.50, .09, .03)
        elif kind == 2:
            base = (.09, .10, .15)
            lit = (.125, .13, .165)
            line = (.30, .32, .40)
        else:
            base = (.86 + .05 * r1, .82 + .05 * r1, .73 + .05 * r2)
            lit = (.985, .96, .90)
            line = (.62, .53, .38)
        c.set_source_rgb(*base)
        poly(c, np.vstack((back_, rim)))
        c.fill()
        inner = cam.xy(fish.at(u + du * (.20 + .86 * ca), np.sin(np.clip(ph + dphi * .43 * sa, -1.56, 1.56))))
        c.set_source_rgb(*lit)
        poly(c, np.vstack((back_, inner)))
        c.fill()
        brush(c, rim, 1.25 * z ** .85, 900 + i * 31 + int(r1 * 999), step, line, .8, taper=(.1, .1), boil=.35)
        if r2 > .72:
            shine = inner[3:8]
            brush(ink, shine, 1.3 * z ** .8, 300 + int(r2 * 999), step, GOLD, .75, taper=(.3, .3), boil=.3)


def rim_shade(fish, cam, z):
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(s)
    c.set_operator(cairo.OPERATOR_ADD)
    uu = AX
    for v in (.985, .96, .925, .88, .82, .73, .6):
        shift = np.array((4., 6.)) * z * (1 - v)
        vv = v * np.minimum(1, (uu / max(.02 * (1 - v) * 4, 1e-3)) ** .5 + .0)
        left = cam.xy(fish.at(uu, vv)) - shift
        right = cam.xy(fish.at(uu, -vv)) - shift
        c.set_source_rgba(0, 0, 0, 1 / 7)
        poly(c, np.vstack((left, right[::-1])))
        c.fill()
    s.flush()
    inside = np.ndarray((H, W), np.uint8, s.get_data(), strides=(s.get_stride(), 1)).astype(np.float32) / 255
    return gaussian_filter(inside, 2.2 * z)


def paint_head(c, ink, fish, cam, z, step):
    n = 60
    vs = np.linspace(1, -1, n)
    edge = .172 + .05 * np.abs(vs) ** 1.5
    us = AX[AX <= .222]
    top = fish.at(us, np.ones_like(us))
    low = fish.at(us, -np.ones_like(us))
    outline = np.vstack((low[::-1], top, fish.at(edge, vs)))
    # the shadow the head throws on the first scales
    back = cam.xy(fish.at(edge + .006, vs))
    m = cairo.ImageSurface(cairo.FORMAT_A8, W // 2, H // 2)
    g = cairo.Context(m)
    g.scale(.5, .5)
    g.set_line_width(26 * z ** .9)
    g.set_source_rgba(0, 0, 0, 1)
    poly(g, back, close=False)
    g.stroke()
    m.flush()
    soft = gaussian_filter(np.ndarray((H // 2, W // 2), np.uint8, m.get_data(), strides=(m.get_stride(), 1)).astype(np.float32), 6.5 * z ** .9)
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, W // 2)
    buf = np.zeros((H // 2, stride), np.uint8)
    buf[:, :W // 2] = soft.astype(np.uint8)
    mask = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_A8, W // 2, H // 2, stride)
    c.save()
    poly(c, cam.xy(np.vstack((fish.at(U_OUT, closing(U_OUT)), fish.at(U_OUT[::-1], -closing(U_OUT[::-1]))))))
    c.clip()
    c.scale(2, 2)
    c.set_source_rgba(.20, .13, .16, .52)
    c.mask_surface(mask, 0, 0)
    c.restore()
    c.set_source_rgb(*hx("#F7F0E2"))
    poly(c, cam.xy(outline))
    c.fill()
    # the red crown: a blob in (u, v) with a hand-painted wandering edge
    a = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    wander = 1 + .12 * np.sin(a + 2.2) + .06 * np.sin(a * 2 + .4) + .035 * np.sin(a * 3 + 1.1) + .012 * np.sin(a * 7 + 2.2) + .012 * wob(90, 77, step)
    crown = cam.xy(fish.at(.112 + .056 * wander * np.cos(a), -.05 + .56 * wander * np.sin(a)))
    c.set_source_rgb(*hx("#DF3B1B"))
    poly(c, crown)
    c.fill()
    inner = cam.xy(fish.at(.118 + .036 * wander * np.cos(a), -.12 + .34 * wander * np.sin(a)))
    c.set_source_rgba(*hx("#F05E2E"), .5)
    poly(c, inner)
    c.fill()
    # cheek and brow lines, gill covers, pores, nostrils, lips
    for side in (1, -1):
        s = np.linspace(0, 1, 30)
        vv = side * (.99 - .80 * s)
        gill = fish.at(.172 + .05 * np.abs(vv) ** 1.5, vv)
        brush(ink, cam.xy(gill), 3.2 * z ** .85, 40 + side, step, GOLD, 1, taper=(.04, .7))
        vv = side * (.93 - .55 * s)
        cheek = fish.at(.146 + .05 * np.abs(vv) ** 1.5, vv)
        brush(ink, cam.xy(cheek), 1.5 * z ** .85, 44 + side, step, UMBER, .75, taper=(.2, .6))
        brow = fish.at(.05 + .06 * s, side * (.46 + .18 * s ** 1.4))
        brush(ink, cam.xy(brow), 1.4 * z ** .85, 46 + side, step, UMBER, .6, taper=(.3, .5))
        for j, (u, v) in enumerate(zip(np.linspace(.135, .19, 6), np.linspace(.84, .55, 6))):
            p = cam.xy(fish.at(u, side * v))
            c.set_source_rgba(*UMBER, .65)
            c.arc(p[0], p[1], 1.5 * z ** .8, 0, 6.283)
            c.fill()
        nostril = fish.at(.036 + .006 * np.sin(np.linspace(0, 5, 14)), side * (.42 + .06 * np.cos(np.linspace(0, 5, 14))))
        brush(ink, cam.xy(nostril), 1.5 * z ** .85, 50 + side, step, UMBER, 1, taper=(.1, .3), boil=.3)
    # the crown is painted, so its brush marks show
    c.save()
    poly(c, crown)
    c.clip()
    for j in range(16):
        u0 = .072 + .08 * (j * .618 % 1)
        v0 = -.05 + .75 * ((j * .381 % 1) - .5)
        q = np.linspace(0, 1, 12)
        mark = fish.at(u0 + .045 * q, v0 + .05 * np.sin(q * 3 + j))
        tone = hx("#B92A12") if j % 3 else hx("#F4693A")
        brush(c, cam.xy(mark), (5 + 5 * (j % 4)) * z ** .85, 1200 + j, step, tone, .26, taper=(.3, .4), boil=.5)
    c.restore()
    vs = np.linspace(-.92, .92, 22)
    brush(ink, cam.xy(fish.at(.017 + .012 * (1 - vs ** 2), vs)), 2.3 * z ** .85, 54, step, UMBER, 1, taper=(.25, .25))


def eye(c, fish, cam, z, side, step):
    """Small and set into the head, as on a real koi: the skin of the brow comes down over its inner third."""
    p = cam.xy(fish.at(.09, side * .86))
    r = .0073 * L * z
    jx, jy = jiggle(side + 2, step, .25 * z ** .5)
    x, y = p[0] + jx, p[1] + jy
    out = fish.n[int(np.searchsorted(AX, .09))] * side
    along = np.array((-out[1], out[0]))
    centre = np.array((x, y))
    g = cairo.RadialGradient(x, y, r * .8, x, y, r * 1.5)      # the socket: a soft shade in the skin, never a ring
    g.add_color_stop_rgba(0, *hx("#3B2A18"), .38)
    g.add_color_stop_rgba(1, *hx("#3B2A18"), 0)
    c.set_source(g)
    c.arc(x, y, r * 1.5, 0, 6.283)
    c.fill()
    k = np.linspace(-1.2, 1.2, 25)
    lid = centre + np.outer(k * r, along) - np.outer(r * (.42 - .2 * k * k), out)
    rim = math.atan2(out[1], out[0]) + np.linspace(1.75, -1.75, 24)
    c.save()
    poly(c, np.vstack((lid, centre + r * 1.6 * np.stack((np.cos(rim), np.sin(rim)), -1))))
    c.clip()
    g = cairo.RadialGradient(x, y, r * .4, x, y, r)
    g.add_color_stop_rgb(0, *hx("#E2B04E"))
    g.add_color_stop_rgb(.7, *hx("#B47F2C"))
    g.add_color_stop_rgb(1, *hx("#6E4A16"))
    c.set_source(g)
    c.arc(x, y, r, 0, 6.283)
    c.fill()
    q = centre + out * r * .08
    c.set_source_rgb(*hx("#090A11"))
    c.arc(q[0], q[1], r * .5, 0, 6.283)
    c.fill()
    c.set_source_rgba(1, 1, 1, .85)
    c.arc(q[0] - r * .16, q[1] - r * .2, r * .1, 0, 6.283)
    c.fill()
    c.restore()
    brush(c, lid[2:-2], 1.25 * z ** .85, 60 + side, step, hx("#4A3420"), .9, taper=(.3, .3), boil=.2)


def barbels(ink, fish, cam, z, t, step):
    s = np.linspace(0, 1, 14)
    for side in (1, -1):
        for u, v, length, seed in ((.016, .78, .095, 70), (.032, .97, .05, 72)):
            base = fish.at(u, side * v)
            ang = angle(.0, t - .25 * s) + side * np.radians(48 + 78 * s) + .16 * np.sin(5 * s - 4 * t + side) * s
            d = np.stack((np.cos(ang), np.sin(ang)), -1)
            p = base + L * length * cumtrapz(d, s, axis=0, initial=0)
            brush(ink, cam.xy(p), 3.2 * z ** .85, seed + side, step, GOLD, 1, taper=(.01, .75), boil=.4)


def roughs(c, fish, cam, z, step):
    """The animator's rough: loose chalk lines under the clean line, different on every drawing."""
    chalk = hx("#A9C2EA")
    for side in (1, -1):
        for j, off in enumerate((1.07, 1.16)):
            p = cam.xy(fish.at(AX, side * off * np.ones_like(AX)))
            brush(c, p, 1.5 * z ** .7, 700 + j * 5 + side, step, chalk, .34 - .1 * j, taper=(.12, .12), boil=4.5 + 2.5 * j)
    th = np.linspace(.4, 6.0, 50)
    head = cam.xy(fish.at(.1, 0.0)) + np.stack((np.cos(th), np.sin(th)), -1) * (.135 * L * z)
    brush(c, head, 1.4 * z ** .7, 720, step, chalk, .3, taper=(.2, .2), boil=5)
    action = np.vstack((fish.p[0] + (fish.p[0] - fish.p[8]) * 2.4, fish.p[8:]))
    brush(c, cam.xy(action), 1.4 * z ** .7, 722, step, chalk, .24, taper=(.2, .1), boil=4)


def contour(ink, fish, cam, z, step):
    for side in (1, -1):
        for a, b, seed in ((.0, .6, 80), (.54, .978, 84)):
            keep = (U_OUT >= a) & (U_OUT <= b)
            p = cam.xy(fish.at(U_OUT[keep], side * closing(U_OUT[keep])))
            press = .62 + .62 * (.5 + .5 * np.sin(np.linspace(0, 1, len(p)) * 8.5 + seed * side))
            brush(ink, p, 4.8 * z ** .85, seed + side, step, GOLD, 1, taper=(.1, .16), boil=1.1, press=press)
    nose = np.linspace(0, 1, 36) ** 2 * .1                      # each flank tapers out at the snout, so one more stroke carries the line round it
    p = np.vstack((fish.at(nose[::-1], np.ones_like(nose)), fish.at(nose[1:], -np.ones_like(nose[1:]))))
    brush(ink, cam.xy(p), 4.6 * z ** .85, 88, step, GOLD, 1, taper=(.3, .3), boil=.8)


# --------------------------------------------------------------------------------------
# the wave thrown by the tail, and its drops
# --------------------------------------------------------------------------------------
WHIP = (2.38, 2.60)


@lru_cache(1)
def droplets():
    r = np.random.default_rng(5)
    m = motion()
    end = sweep()[0][-1]
    out = []
    for i in range(280):
        ti = r.uniform(WHIP[0], end)
        tip = np.array(tail_tip(round(ti * 240) / 240))
        ahead = np.array(tail_tip(round(ti * 240) / 240 + 1 / 120)) - tip
        ahead /= np.hypot(*ahead) + 1e-6
        away = tip - in_time(m["centre"], ti)
        away /= np.hypot(*away) + 1e-6
        flung = .75 * away + .65 * ahead                        # water leaves a swung tail along its path, not straight out
        flung /= np.hypot(*flung) + 1e-6
        a = r.normal(0, .3)
        dirn = np.array((flung[0] * math.cos(a) - flung[1] * math.sin(a), flung[0] * math.sin(a) + flung[1] * math.cos(a)))
        size = 1.8 + 8.5 * r.random() ** 3.2
        out.append((ti, tip + away * r.uniform(30, 150) + dirn * r.uniform(0, 40), dirn * r.uniform(350, 1900), r.uniform(2.6, 4.6), r.uniform(.45, 1.5), size, r.random()))
    return out


def splash(cam, t, step):
    s, c = layer()
    z = cam.z
    for ti, p0, v0, drag, life, size, tone in droplets():
        a = t - ti
        if a <= 0:
            continue
        pos = p0 + v0 * (1 - math.exp(-drag * a)) / drag
        if a < life:
            speed = np.hypot(*v0) * math.exp(-drag * a)
            r = size * (1 - (a / life) ** 3) * z
            if r < .5:
                continue
            p = cam.xy(pos)
            d = v0 / (np.hypot(*v0) + 1e-6)
            stretch = 1 + speed / 340
            nrm = np.array((-d[1], d[0]))
            ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)  # angle 0 is the point of the tail
            run = -np.cos(ang)
            wide = np.sin(ang) * np.abs(np.sin(ang / 2)) ** min(3.2, (stretch - 1) * 1.6) * (1 + .6 * min(1, stretch - 1))
            shape = p + np.outer(np.where(run < 0, stretch, 1) * run * r, d) + np.outer(wide * r, nrm)
            col = SHELL if tone > .12 else GOLD
            c.set_source_rgb(*col)
            poly(c, shape)
            c.fill()
        elif size > 4.5 and a < life + 1.4:
            p = cam.xy(pos)
            b = a - life
            r = (8 + 95 * b ** .6) * z
            th = np.linspace(0, 2 * np.pi, 46)
            ring = np.stack((p[0] + r * np.cos(th), p[1] + r * .96 * np.sin(th)), -1)
            brush(c, ring, 1.6 * z ** .8 * (1 - b / 1.4), int(tone * 999), step, GOLD, .85 * (1 - b / 1.4), taper=(.2, .2), boil=.6)
    return s


@lru_cache(1)
def strokes():
    """Three loaded strokes, each a sheaf of bristles; a few loose hairs; solid claws of foam on the outside."""
    r = np.random.default_rng(17)
    hairs = []
    for m, (off, a, b, width) in enumerate(((.66, .18, 1.0, 34.), (-.02, .24, .9, 28.), (-.74, .36, .78, 16.))):
        count = max(6, int(width / 1.2))
        for j in range(count):
            hairs.append((m, off, a, b - r.uniform(0, .05), width, (j + .5) / count * 2 - 1, width / count * 1.9,
                          GOLD if r.random() < .1 else SHELL, r.uniform(0, 6.28), r.uniform(5, 12), m * 100 + j))
    flecks = []
    for j in range(18):
        a = r.uniform(.03, .85)
        flecks.append((r.uniform(-1.15, 1.25), a, a + r.uniform(.04, .16), r.uniform(.9, 2.2), GOLD if r.random() < .1 else SHELL, j))
    claws = [(r.uniform(.24, .94), r.uniform(.6, .84), 12 + 18 * r.random() ** 1.5, j) for j in range(10)]
    return hairs, flecks, claws


@lru_cache(1)
def sweep():
    m = motion()
    times = np.linspace(WHIP[0], WHIP[1], 120)
    tips = np.array([tail_tip(round(x * 240) / 240) for x in times])
    tips = gaussian_filter1d(tips, 4, axis=0, mode="nearest")
    away = tips - np.array([in_time(m["centre"], x) for x in times])
    away /= np.maximum(np.hypot(away[:, 0], away[:, 1])[:, None], 1e-6)
    away = gaussian_filter1d(away, 4, axis=0, mode="nearest")
    # the tip doubles back at the end of the whip: the water keeps going, so the stroke stops before that bend
    d = np.gradient(tips, axis=0)
    bent = np.abs(np.gradient(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))) > np.hypot(d[:, 0], d[:, 1]) / 140
    bent[:60] = False
    cut = max(70, int(np.argmax(bent)) - 6) if bent.any() else 120
    k = 4
    q = np.linspace(0, cut - 1, (cut - 1) * k + 1)
    return np.interp(q, np.arange(120), times), smooth_line(tips[:cut], k), smooth_line(away[:cut], k)


def crest(cam, t, step):
    """The sheet of water the tail throws: brush strokes that run dry along their length and, as the water falls, altogether."""
    if t < WHIP[0]:
        return None, None
    times, tips, away = sweep()
    i_now = int(np.searchsorted(times, t))
    if i_now < 24:
        return None, None
    n = len(times)
    age = np.maximum(t - times, 0)
    run = 26 + 210 * (1 - np.exp(-2.6 * age))
    span = np.linspace(0, 1, n)
    env = np.minimum(1, age / .06 + .2) * np.exp(-age * .55)
    thick = 118 * (.38 + .62 * np.sin(np.pi * (span * .96 + .02)) ** .7) * env
    s, c = layer()
    z = cam.z
    hairs, flecks, claws = strokes()
    dry = seg_(t, 2.85, 3.85)
    retreat = max(0.0, t - 2.95) * .4

    # the body of the water under the strokes
    idx = np.arange(int(retreat * n), i_now)
    wash = None
    if len(idx) > 8:
        m = cairo.ImageSurface(cairo.FORMAT_A8, W // 2, H // 2)
        g = cairo.Context(m)
        g.scale(.5, .5)
        g.set_source_rgba(0, 0, 0, 1)
        hi = tips[idx] + away[idx] * (run[idx] + .95 * thick[idx])[:, None]
        lo = tips[idx] + away[idx] * (run[idx] - 1.0 * thick[idx])[:, None]
        poly(g, cam.xy(np.vstack((hi, lo[::-1]))))
        g.fill()
        m.flush()
        wash = gaussian_filter(np.ndarray((H // 2, W // 2), np.uint8, m.get_data(), strides=(m.get_stride(), 1)).astype(np.float32) / 255, 13 * z)
        wash = wash.repeat(2, 0).repeat(2, 1)[:H, :W] * (1 - dry)

    for m_, off, a, b, width, d, w, col, ph, freq, seed in hairs:
        i0, i1 = int((a + retreat) * n), min(i_now, int(b * n))
        if i1 - i0 < 12:
            continue
        idx = np.arange(i0, i1)
        u = np.clip((idx - a * n) / ((b - a) * n), 0, 1)
        swell = np.minimum(1, u / .1) ** .7 * np.minimum(1, (1 - u) / .5) ** .8
        mid = off + .26 * np.sin(2 * np.pi * (u * (.7 + .25 * m_) + m_ * .37))     # they wander together and apart, so they never read as concentric wires
        swell = swell * (.82 + .18 * np.sin(2 * np.pi * (u * 1.3 + m_ * .6)))
        p = tips[idx] + away[idx] * (run[idx] + mid * thick[idx] + d * .5 * width * swell)[:, None]
        streak = .55 * np.sin(u * freq * 6.283 + ph) + .45 * np.sin(u * freq * 2.7 * 6.283 + ph * 2.3)
        on = streak > -1.25 + 1.5 * u ** 1.6 + .55 * d * d + 2.3 * dry   # solid where the brush is loaded, flying white where it runs dry
        edges = np.flatnonzero(np.diff(np.concatenate(((0,), on.astype(int), (0,)))))
        for e0, e1 in zip(edges[::2], edges[1::2]):
            if e1 - e0 > 6:
                brush(c, cam.xy(p[e0:e1]), w * (1 - .5 * dry) * z ** .85, 2000 + seed, step, col, .96 * (1 - .6 * dry), taper=(.1, .24), boil=.4)
    if dry < .8:
        for off, a, b, w, col, j in flecks:
            i0, i1 = int((a + retreat) * n), min(i_now, int(b * n))
            if i1 - i0 < 12:
                continue
            idx = np.arange(i0, i1)
            p = tips[idx] + away[idx] * (run[idx] + off * thick[idx])[:, None]
            brush(c, cam.xy(p), w * z ** .85, 2200 + j, step, col, .8 * (1 - dry), taper=(.3, .4), boil=.6)
    for u, reach, size, j in claws:
        i = int(u * n)
        if i >= i_now:
            continue
        grow = min(1.0, age[i] / .14)
        if grow < .1 or age[i] > .7:
            continue
        root = tips[i] + away[i] * (run[i] + thick[i] * reach)
        a0 = math.atan2(away[i, 1], away[i, 0])
        ahead = tips[min(i + 8, n - 1)] - tips[max(i - 8, 0)]
        sign = 1 if away[i, 0] * ahead[1] - away[i, 1] * ahead[0] > 0 else -1   # all the claws lean the way the water is going
        q = np.linspace(0, grow, 22)
        th = a0 + sign * (.75 + q * (1.1 + .7 * (j * .618 % 1)))   # peeling forward off the crest, not standing on it like thorns
        life = (1 - age[i] / .7) ** .7
        rad = size * (1 - .62 * q) * life
        p = root + np.cumsum(np.stack((np.cos(th), np.sin(th)), -1) * (rad * .22)[:, None], axis=0)
        brush(c, cam.xy(p), size * .62 * life * z ** .85, 2400 + j, step, SHELL, .96, taper=(.03, .92), boil=.4)
    return s, wash


# --------------------------------------------------------------------------------------
# one drawing
# --------------------------------------------------------------------------------------
def fish_shadow(fish, rays_all, cam):
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(s)
    c.translate(*(np.array((26., 36.)) * cam.z))
    outline = np.vstack((fish.at(U_OUT, closing(U_OUT)), fish.at(U_OUT[::-1], -closing(U_OUT[::-1]))))
    c.set_source_rgba(0, 0, 0, 1)
    poly(c, cam.xy(outline))
    c.fill()
    c.set_source_rgba(0, 0, 0, .45)
    for rays in rays_all:
        poly(c, np.vstack((rays[0], np.array([r[-1] for r in rays[1:]]), rays[-1][::-1])))
        c.fill()
    s.flush()
    a = np.ndarray((H, W), np.uint8, s.get_data(), strides=(s.get_stride(), 1)).astype(np.float32) / 255
    return gaussian_filter(a[::2, ::2], 8 * cam.z).repeat(2, 0).repeat(2, 1)[:H, :W]


@lru_cache(2)
def drawing(step):
    t = step / RATE
    fish, cam = Fish(t), Camera(t)
    z = cam.z
    base, big, mid, fibre = stock()
    img = base.copy()

    lines, tex = water(cam, fish, t, step)
    a = alpha_of(lines) * tex
    img = img * (1 - a[..., None]) + rgb_of(lines) * tex[..., None]

    tail = [cam.xy(r) for r in tail_rays(t)]
    pecs = [[cam.xy(r) for r in pectoral(fish, side, t)] for side in (1, -1)]
    vents = [[cam.xy(r) for r in ventral(fish, side, t)] for side in (1, -1)]
    img *= 1 - .62 * fish_shadow(fish, [tail, *pecs, *vents], cam)[..., None]

    top, sheet = crest(cam, t, step)
    if top is not None:
        fade = seg_(t, 3.05, 3.95)
        m = np.clip((norm_cached(0) * .5 + .5 - (fade * 1.3 - .3)) / .3, 0, 1) if fade > 0 else np.ones((H, W), np.float32)
        m = m * m * (3 - 2 * m)
        if sheet is not None:
            k = (.15 * sheet * m * (.85 + .15 * fibre))[..., None]
            img = img * (1 - k) + np.array(hx("#C4D8EC"), np.float32) * k
        a = alpha_of(top) * m
        img = img * (1 - a[..., None]) + rgb_of(top) * m[..., None]

    paint_s, paint = layer()
    ink_s, ink = layer()
    paint.translate(*jiggle(1, step, 1.3))

    # smear drawings: older tails stay on the page for a moment during the whip
    if WHIP[0] - .03 < t < WHIP[1]:
        for back_ in (3, 2, 1):
            old = tail_rays(t - back_ / 24, 17)
            for j, r in enumerate(old):
                brush(paint, cam.xy(r), 1.9 * z ** .8, 400 + j + back_, step, GOLD, .28 + .12 * (3 - back_), taper=(.02, .4), boil=.6)

    rough_s, rough = layer()
    roughs(rough, fish, cam, z, step)
    over(img, rough_s)

    fin(paint, paint, tail, z, 11, step, 1.0, 1.5, veil=.16)    # its rays start under the body, so they are painted before it
    for side, rays in zip((1, -1), vents):
        fin(paint, ink, rays, z, 21 + side, step, .9)
    for side, rays in zip((1, -1), pecs):
        fin(paint, ink, rays, z, 31 + side, step, 1.0, 1.5)

    outline = np.vstack((fish.at(U_OUT, closing(U_OUT)), fish.at(U_OUT[::-1], -closing(U_OUT[::-1]))))
    paint.set_source_rgb(*hx("#CFC6B4"))
    poly(paint, cam.xy(outline))
    paint.fill()
    body_ink_s, body_ink = layer()
    paint.save()
    body_ink.save()
    for c_ in (paint, body_ink):
        poly(c_, cam.xy(outline))
        c_.clip()
    paint_scales(paint, body_ink, fish, cam, z, step)
    # the scales melt into the gold at the root of the fin
    near, far = cam.xy(fish.at(np.array((.885, 1.)), np.zeros(2)))
    melt = cairo.LinearGradient(*near, *far)
    melt.add_color_stop_rgba(0, *hx("#F7DCA8"), 0)
    melt.add_color_stop_rgba(.55, *hx("#F7DCA8"), .42)
    melt.add_color_stop_rgba(1, *hx("#F7DCA8"), .96)
    rear = U_OUT[U_OUT >= .885]
    rear = cam.xy(np.vstack((fish.at(rear, 1.3 * np.ones_like(rear)), fish.at(rear[::-1], -1.3 * np.ones_like(rear)))))
    for c_, mode in ((paint, cairo.OPERATOR_OVER), (body_ink, cairo.OPERATOR_DEST_OUT)):
        c_.save()
        poly(c_, rear)                                          # a curled body can bring the head across the gradient's far side
        c_.clip()
        c_.set_operator(mode)
        c_.set_source(melt)
        c_.paint()
        c_.restore()
    body_ink.restore()
    paint.restore()
    paint_head(paint, body_ink, fish, cam, z, step)

    # paint, then the roundness of the body as an indigo wash on top of it
    paint_s.flush()
    mask = rim_shade(fish, cam, z)
    body = np.zeros((H, W), np.uint8)
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(s)
    c.set_source_rgba(0, 0, 0, 1)
    poly(c, cam.xy(outline))
    c.fill()
    s.flush()
    body = np.ndarray((H, W), np.uint8, s.get_data(), strides=(s.get_stride(), 1)).astype(np.float32) / 255
    shade = (1 - mask) * body
    a = alpha_of(paint_s)
    rgb = rgb_of(paint_s)
    wash = np.array(hx("#1A2C52"), np.float32)
    k = (shade * .78)[..., None]
    rgb = rgb * (1 - k) + wash * a[..., None] * k
    rgb *= (1 + .018 * fibre + .02 * mid)[..., None]
    img = img * (1 - a[..., None]) + rgb

    over(img, body_ink_s)

    # dorsal fin stands on the back
    dors_s, dors = layer()
    drays = [cam.xy(r) for r in dorsal(fish)]
    dorsal_rays(dors, drays, z, step)
    over(img, dors_s)

    eye_s, eyes = layer()
    contour(eyes, fish, cam, z, step)
    barbels(eyes, fish, cam, z, t, step)
    for side in (1, -1):
        eye(eyes, fish, cam, z, side, step)
    over(img, eye_s)
    tex = (.86 + .10 * fibre + .08 * mid)[..., None]
    a = alpha_of(ink_s)
    img = img * (1 - a[..., None]) + rgb_of(ink_s) * tex

    # the water is thrown behind the fish: no drop may land on the body
    drops = splash(cam, t, step)
    clear = (1 - gaussian_filter(body, 1.2))[..., None]
    img = img * (1 - alpha_of(drops)[..., None] * clear) + rgb_of(drops) * clear
    return surface(np.clip(img, 0, 1))


def seg_(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


@lru_cache(1)
def norm_cached(_):
    r = np.random.default_rng(31)
    return np.clip(norm(gaussian_filter(r.standard_normal((H, W)), 30)) * .5, -1, 1).astype(np.float32)


def handdrawn(t):
    return drawing(min(59, max(0, int(float(t) * RATE + 1e-7))))

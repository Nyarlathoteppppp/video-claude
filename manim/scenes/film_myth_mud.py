"""The wet mud of 鬼夜哭, the character cut in it, the bird's print beside it and the twig that cut it. These are things, not drawings: the mud is one height field in film pixels, every mark in it a real dent with a raised lip, lit from one side with real cast shadows, and only then turned into ink: what faces away and what lies in shadow takes ink, what faces the light is wiped to paper.

The character is measured from the oracle-bone form in a square of 300 (see GLYPH). Cang Jie cuts only the spine and three teeth, four strokes, the teeth a little splayed so they answer the toes of the print. His last stroke is the middle tooth, cut from the left and ending just past the spine. In the ghost's vision and only there, the right side of the form starts to come, on the far side of the spine: never crossing it.

    python film_myth_mud.py look        # the height field shaded, no ink, to judge the marks: keys/mud_look.png
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree

from film_myth_wash import OUT, band
from ink_grade import fft_noise, sm

WORLD_W, WORLD_H = 2700, 1600
CHAR_SCALE = 4.0                 # world pixels to one unit of the 300-square glyph
CHAR_AT = (1450.0, 620.0)        # where glyph point (97, 80), the spine at the last stroke, lies in the world
TRACK_AT = (880.0, 640.0)        # the palm of the bird's print

# Glyph points as measured, y down, in the 300 square. Cang Jie's teeth are splayed a little from the measure.
SPINE = [(97, 22), (95.5, 78), (95, 132)]
TOOTH_TOP = [(43, 12), (72, 19), (97, 23)]
TOOTH_MID = [(42, 80), (72, 82), (100, 82)]      # the last stroke
TOOTH_LOW = [(40, 146), (68, 137), (95, 130)]
FOUR = [("spine", SPINE), ("top", TOOTH_TOP), ("low", TOOTH_LOW), ("mid", TOOTH_MID)]     # in the order they are cut
# The vision, on the far side of the spine, in the director's skeleton (film pixels of CAM_K3, here in glyph units): a small wedge leaning on the spine, its point well short of the long stroke; then the long stroke, in from the right along a hook at the top, down, and stopping in the air. Never more than this: no branch, no foot, no whole character.
WEDGE = [(101.25, 47.25), (131.5, 65.5), (101.25, 87.5)]
SHAFT = [(179.0, 7.0), (152.75, 5.0), (147.0, 57.5), (141.5, 122.5)]
# Where the brush stops, as a fraction of the stroke, how much it left there, how big: an uneven beat, two long and a short on the long stroke.
VISION = (
    dict(pts=WEDGE, soft=5.0, base=11.0, stops=((0.27, 0.55, 1.0), (0.485, 0.6, 0.85))),
    dict(pts=SHAFT, soft=16.0, base=11.0, stops=((0.15, 0.7, 1.1), (0.60, 0.9, 1.35), (0.72, 0.45, 0.8))),
)


def unit(v):
    """A slow random line with mean 0 and spread 1. A short sample of it has a mean of its own, many times its spread: left in, it scales or shifts the whole mark."""
    return (v - v.mean()) / (v.std() + 1e-6)


def g2w(p):
    return (CHAR_AT[0] + (p[0] - 97) * CHAR_SCALE, CHAR_AT[1] + (p[1] - 80) * CHAR_SCALE)


def dense(points, step=1.0, wobble=0.0, seed=0):
    """A smooth line through these points at even steps, with a hand's slight waver."""
    p = np.array(points, np.float64)
    seg = np.hypot(*np.diff(p, axis=0).T)
    along = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(along[-1] / step), 4)
    s = np.linspace(0, along[-1], n)
    if len(p) >= 3:
        # Catmull-Rom through the points
        out = []
        P = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
        for t in s:
            i = min(np.searchsorted(along, t, side="right") - 1, len(seg) - 1)
            u = (t - along[i]) / max(seg[i], 1e-6)
            a, b, c, d = P[i], P[i + 1], P[i + 2], P[i + 3]
            out.append(0.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (3 * b - a - 3 * c + d) * u ** 3))
        q = np.array(out)
    else:
        q = np.stack([np.interp(s, along, p[:, 0]), np.interp(s, along, p[:, 1])], 1)
    if wobble:
        r = np.random.default_rng(seed)
        sig = 34.0 / step
        nx = gaussian_filter(r.normal(0, 1, n), sig)
        ny = gaussian_filter(r.normal(0, 1, n), sig)
        q = q + wobble * np.stack([unit(nx), unit(ny)], 1)
    return q


class Mud:
    def __init__(self, seed=5, w=WORLD_W, h=WORLD_H):
        self.w, self.h, self.seed = w, h, seed
        H = 9.0 * fft_noise(h, w, 4.0, seed + 1) + 1.1 * fft_noise(h, w, 3.2, seed + 2)
        H += 0.05 * band(h, w, 3.0, 14, seed + 3) + 0.012 * band(h, w, 0.6, 2.2, seed + 4)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.random.default_rng(seed + 7)
        for _ in range(int(w * h / 42000)):
            # pits where a bubble came up, a grain of grit standing out
            x, y, rad = r.uniform(0, w), r.uniform(0, h), r.uniform(1.2, 3.6)
            x0, x1, y0, y1 = int(max(x - 3 * rad, 0)), int(min(x + 3 * rad, w)), int(max(y - 3 * rad, 0)), int(min(y + 3 * rad, h))
            if x1 > x0 and y1 > y0:
                sign = -1.0 if r.uniform() < 0.7 else 1.0
                H[y0:y1, x0:x1] += sign * r.uniform(0.25, 0.9) * np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (rad * rad))
        self.H = np.float32(H)
        self.rng = r
        self.xx, self.yy = xx, yy

    def _window(self, q, margin):
        x0, x1 = int(max(q[:, 0].min() - margin, 0)), int(min(q[:, 0].max() + margin, self.w))
        y0, y1 = int(max(q[:, 1].min() - margin, 0)), int(min(q[:, 1].max() + margin, self.h))
        return x0, x1, y0, y1

    def cut(self, q, width, depth, lip=0.6, seed=0, pile=1.0, start=0.3, crumbs=1.0, grainy=1.0, chew_k=0.10):
        """A mark drawn through the mud along the line q (world pixels), width (half width) and depth given along it. The mud it displaces stands up in a broken lip on both sides and piles up where it ends."""
        r = np.random.default_rng(seed)
        n = len(q)
        along = np.linspace(0, 1, n)
        wv = np.interp(along, np.linspace(0, 1, len(width)), width)
        dv = np.interp(along, np.linspace(0, 1, len(depth)), depth)
        chew = gaussian_filter(r.normal(0, 1, n), 5)
        chew = unit(chew)
        wv = wv * (1 + chew_k * chew)
        wander = gaussian_filter(r.normal(0, 1, n), 30)
        dv = dv * (1 + 0.10 * unit(wander))
        m = float(wv.max() * 3.2 + 8)
        x0, x1, y0, y1 = self._window(q, m)
        xx, yy = self.xx[y0:y1, x0:x1], self.yy[y0:y1, x0:x1]
        tree = cKDTree(q)
        dist, idx = tree.query(np.stack([xx.ravel(), yy.ravel()], 1), distance_upper_bound=m, workers=2)
        far = ~np.isfinite(dist)
        idx = np.minimum(idx, n - 1)
        wp, dp = wv[idx], dv[idx]
        t = dist / wp
        t[far] = 9.0
        # which side of the line, for grain running along the cut
        tan = np.gradient(q, axis=0)
        tan = tan / (np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-6)
        rel = np.stack([xx.ravel() - q[idx, 0], yy.ravel() - q[idx, 1]], 1)
        side = tan[idx, 0] * rel[:, 1] - tan[idx, 1] * rel[:, 0]
        u = along[idx] * n
        grain = np.sin(side * 1.1 + 1.6 * np.sin(u * 0.05) + 0.9 * np.sin(u * 0.13 + 1)) * 0.06 + np.sin(side * 2.3 + 2.1 * np.sin(u * 0.09)) * 0.03
        body = np.where(t < 1, np.clip(1 - t * t, 0, 1) ** 0.8, 0.0) * dp
        floor = body * (1 + grainy * grain * (t < 0.9))
        h, w = yy.shape
        noise_a = band(h, w, 1.6, 6, seed + 21)
        noise_b = band(h, w, 6.0, 26, seed + 22)
        lip_h = lip * dp * np.clip(0.62 + 0.28 * noise_b.ravel(), 0.2, 1.2) * np.exp(-(((t - 1.42) / 0.66) ** 2))
        lip_h = lip_h * (1 + 0.04 * noise_a.ravel())
        d = (lip_h - floor).reshape(h, w)
        d[far.reshape(h, w)] = 0
        self.H[y0:y1, x0:x1] += np.float32(d)
        # where it ends the mud is pushed up in front of the point
        end, last = q[-1], tan[-1]
        if pile:
            c = end + last * wv[-1] * 1.1
            self.heap(c, wv[-1] * 2.0, dv[-1] * 0.85 * pile, seed + 31, along=last)
        if start:
            self.heap(q[0] - tan[0] * wv[0] * 0.3, wv[0] * 1.2, dv[0] * 0.28 * start, seed + 32)
        # crumbs thrown off to the sides
        k = int(0.018 * n * crumbs)
        for _ in range(k):
            i = int(r.uniform(0, n))
            sgn = 1 if r.uniform() < 0.5 else -1
            nrm = np.array((-tan[i, 1], tan[i, 0]))
            p = q[i] + nrm * sgn * wv[i] * r.uniform(1.6, 3.0)
            self.crumb(p, r.uniform(0.9, 2.8), r.uniform(0.6, 1.8))

    def heap(self, c, rad, height, seed, along=None):
        rad = float(rad)
        x0, x1, y0, y1 = int(max(c[0] - 3 * rad, 0)), int(min(c[0] + 3 * rad, self.w)), int(max(c[1] - 3 * rad, 0)), int(min(c[1] + 3 * rad, self.h))
        xx, yy = self.xx[y0:y1, x0:x1], self.yy[y0:y1, x0:x1]
        dx, dy = xx - c[0], yy - c[1]
        if along is not None:
            a = dx * along[0] + dy * along[1]
            b = -dx * along[1] + dy * along[0]
            q = (a / (rad * 0.7)) ** 2 + (b / (rad * 1.15)) ** 2
        else:
            q = (dx * dx + dy * dy) / (rad * rad)
        h, w = yy.shape
        n = 0.2 * band(h, w, 1.6, 5, seed) + 0.5 * band(h, w, 4, 12, seed + 1)
        self.H[y0:y1, x0:x1] += np.float32(height * np.exp(-q) * (1 + 0.32 * n))

    def crumb(self, c, rad, height):
        x0, x1, y0, y1 = int(max(c[0] - 3 * rad, 0)), int(min(c[0] + 3 * rad + 1, self.w)), int(max(c[1] - 3 * rad, 0)), int(min(c[1] + 3 * rad + 1, self.h))
        if x1 <= x0 or y1 <= y0:
            return
        xx, yy = self.xx[y0:y1, x0:x1], self.yy[y0:y1, x0:x1]
        self.H[y0:y1, x0:x1] += np.float32(height * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (rad * rad)))

    def press_toe(self, palm, angle, length, seed, width=15.0, depth=9.0):
        """One toe of the bird: a long pressed groove with soft pads in it, thinner toward the end, then the claw's thin scratch."""
        r = np.random.default_rng(seed)
        bend = r.uniform(-0.06, 0.06)
        pts = []
        for u in np.linspace(0, 1, 10):
            a = angle + bend * u
            pts.append((palm[0] + np.cos(a) * length * u, palm[1] + np.sin(a) * length * u))
        q = dense(pts, 1.0, wobble=0.5, seed=seed)
        n = len(q)
        pads = np.linspace(0, 1, 60)
        wid = width * (1 - 0.40 * pads) * (1 + 0.13 * np.cos(pads * 2 * np.pi * 3.0 + 0.5))
        dep = depth * (0.72 + 0.14 * np.cos(pads * 2 * np.pi * 3.0 + 0.5)) * (1.0 - 0.2 * pads)
        self.cut(q, wid, dep, lip=0.34, seed=seed + 1, pile=0.10, start=0, crumbs=0.1, grainy=0.0)
        # the claw
        a = angle + bend
        tip = (q[-1][0], q[-1][1])
        k = width / 23.0
        claw = dense([tip, (tip[0] + np.cos(a + 0.05) * 22 * k, tip[1] + np.sin(a + 0.05) * 22 * k), (tip[0] + np.cos(a + 0.12) * 46 * k, tip[1] + np.sin(a + 0.12) * 46 * k)], 1.0)
        self.cut(claw, [6.0 * k, 3.6 * k, 1.8 * k], [3.4 * k ** 0.5, 2.4 * k ** 0.5, 0.7 * k ** 0.5], lip=0.28, seed=seed + 2, pile=0.0, start=0, crumbs=0.15 * k, grainy=0.0)

    def print_(self, palm, facing, seed, scale=1.0):
        """The three-toed print, toes toward `facing` radians, each toe splayed from the palm the way the teeth of the character are."""
        for k, (off, length, w) in enumerate(((-0.5, 195, 23), (0.0, 245, 26), (0.5, 190, 23))):
            self.press_toe(palm, facing + off, length * scale, seed + 10 * k, width=w * scale, depth=10.0 * scale ** 0.5)
        self.heap(palm, 36 * scale, -7.0 * scale ** 0.5, seed + 77)

    def char(self, seed=100):
        for i in range(len(FOUR)):
            self.char_stroke(i, seed)

    def char_line(self, i, seed=100):
        """The line stroke i of the character follows, in the world."""
        return dense([g2w(p) for p in FOUR[i][1]], 1.0, wobble=1.3, seed=seed + i)

    def char_stroke(self, i, seed=100):
        name = FOUR[i][0]
        q = self.char_line(i, seed)
        # the point goes in narrow, the cut widens as the hand leans on it
        n = 40
        u = np.linspace(0, 1, n)
        press = np.minimum(u / 0.10, 1) ** 0.7
        wid = 14.0 * (0.62 + 0.38 * press) * (1 + 0.06 * np.sin(u * 7 + i))
        dep = 10.5 * (0.5 + 0.5 * press)
        self.cut(q, wid, dep, lip=0.55, seed=seed + 10 * i, pile=1.1 if name == "mid" else 0.8, start=0.5, crumbs=0.5, grainy=0.55)

    def grown(self, seed=170):
        """The vision's strokes, shallower than the cuts, deepest where the brush came down and hardly there where it left."""
        for k in range(len(VISION)):
            self.grown_stroke(k, seed)

    def grown_stroke(self, k, seed=170):
        st = vision_strokes(seed)[k]
        stop = int(0.93 * len(st["q"]))      # the dry hairs at the end leave no groove
        dep = 6.0 * (1 + 0.3 * np.exp(-st["u"] * len(st["u"]) / 30)) * (1 - 0.95 * sm((st["u"] - 0.78) / 0.15))
        self.cut(st["q"][:stop], 0.95 * st["w"][:stop], dep[:stop], lip=0.32, seed=seed + 10 + k, pile=0.0, start=0.0, crumbs=0.2, grainy=0.3)

    def shade(self, light=(-0.36, -0.93), elevation=38.0, shadow_steps=26):
        """The surface in light. Returns (slope light, 0 when flat, + toward the light), cast shadow 0..1, cavity (how far below the neighbourhood), and the height."""
        H = self.H
        gy, gx = np.gradient(gaussian_filter(H, 0.6))
        lx, ly = light
        norm = np.hypot(lx, ly)
        lx, ly = lx / norm, ly / norm
        e = np.radians(elevation)
        lz = np.tan(e)
        s = (-gx * lx - gy * ly) / (np.sqrt(1 + gx * gx + gy * gy))
        # march toward the light: anything higher along the way shades this point
        occ = np.zeros_like(H)
        tanE = 0.82
        for i in range(1, shadow_steps):
            dx, dy = int(round(lx * i)), int(round(ly * i))
            shifted = np.roll(np.roll(H, -dy, axis=0), -dx, axis=1)
            occ = np.maximum(occ, shifted - H - i * tanE)
        shadow = np.clip((occ - 0.35) / 1.3, 0, 1)
        shadow = gaussian_filter(shadow, 0.7)
        cav = H - gaussian_filter(H, 9)
        self.broad = np.float32(gaussian_filter(-gaussian_filter(H, 10) * 0 + (-np.gradient(gaussian_filter(H, 16))[1] * ly - np.gradient(gaussian_filter(H, 16))[0] * lx), 6))
        return np.float32(s), np.float32(shadow), np.float32(cav), H


def crop_to_film(a, x0, y0, zoom=1.0, W=1920, H=1080):
    """A part of a world field as a film-sized one. zoom 1 is one world pixel to one film pixel."""
    w, h = W / zoom, H / zoom
    im = Image.fromarray(np.float32(a), "F")
    return np.asarray(im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h)), np.float32)


def ink_of(s, shadow, cav, amount=1.0):
    """The relief as ink. What turns from the light and what lies in shadow takes ink; what faces it is left paper. Only about a fifth of a wash on the flat."""
    away = np.maximum(-s - 0.05, 0)
    toward = np.maximum(s - 0.08, 0)
    d = 1.7 * away ** 1.05 + 0.95 * shadow + 0.32 * np.maximum(-cav - 0.5, 0) ** 0.9
    lift = sm(toward / 0.42)
    return amount * d, lift


# ---------------------------------------------------------------- the twig

def _polyline(path, step=0.5, soft=7.0):
    """Even steps along straight lengths, the corners between them rounded only a little: a branch changes direction at its nodes."""
    p = np.array(path, np.float64)
    seg = np.hypot(*np.diff(p, axis=0).T)
    along = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, along[-1], step)
    q = np.stack([np.interp(t, along, p[:, 0]), np.interp(t, along, p[:, 1])], 1)
    sig = soft / step
    keep = q.copy()
    q = np.stack([gaussian_filter1d(q[:, 0], sig, mode="nearest"), gaussian_filter1d(q[:, 1], sig, mode="nearest")], 1)
    q[:int(sig)] = keep[:int(sig)]
    return q, along[1:-1] / along[-1]


def vision_strokes(seed=170):
    """Each new stroke as a line in the world at 1 px steps with its half width along it: heavy where the brush came down, wandering in width, a little wider at each stop, by how much differing from stop to stop, thin where it leaves. u is the fraction along it."""
    out = []
    for k, st in enumerate(VISION):
        q, _ = _polyline([g2w(p) for p in st["pts"]], 1.0, st["soft"])
        n = len(q)
        rg = np.random.default_rng(seed + k)
        q = q + 0.9 * np.stack([unit(gaussian_filter1d(rg.normal(0, 1, n), 30)), unit(gaussian_filter1d(rg.normal(0, 1, n), 30))], 1)
        u = np.linspace(0, 1, n)
        w = st["base"] * (1 + 0.16 * unit(gaussian_filter1d(rg.normal(0, 1, n), 46)) + 0.05 * unit(gaussian_filter1d(rg.normal(0, 1, n), 9)))
        w = w * (1 + 0.42 * np.exp(-u * n / 24))
        w = w * (1 - 0.45 * sm((u - 0.84) / 0.16))
        for at, much, size in st["stops"]:
            w = w * (1 + 0.28 * much * np.exp(-((u - at) * n / (13 * size)) ** 2))
        out.append(dict(q=q, u=u, w=w, stops=st["stops"]))
    return out


def twig_shape(path, radius, W=1920, H=1080, seed=40, stubs=(), jag_k=2.8, step=0.5):
    """The outline of a dead branch along path (film pixels), whose vertices are its nodes: thick where it swells at a node, narrower between, snapped at the far end, worn round at the point. stubs are broken side shoots: (fraction along, angle off the line, length, radius against the branch). Returns the distance outside it (negative inside) at film size, and its limbs for the brush, the main one first, each as dict(q, r, nodes): its line, its half widths, the indices of its nodes."""
    q, nodes = _polyline(path, step)
    n = len(q)
    px = np.arange(n) * step
    L = float(px[-1])
    u = px / L
    rg = np.random.default_rng(seed)
    r = np.interp(u, np.linspace(0, 1, len(radius)), radius).astype(np.float64)
    r = r * (1 + 0.05 * unit(gaussian_filter1d(rg.normal(0, 1, n), 110 / step)) + 0.022 * unit(gaussian_filter1d(rg.normal(0, 1, n), 2.2 / step)))
    swell = np.zeros(n)
    for kn in nodes:
        reach = float(np.interp(kn, u, r)) * 1.25
        swell += rg.uniform(0.20, 0.36) * np.exp(-((px - kn * L) / reach) ** 2)
    r = r * (0.95 + swell)
    r_end = float(r[-1])
    r = r * (0.46 + 0.54 * np.sqrt(np.clip(px / (2.4 * r[0]), 0, 1)))
    m = float(r.max() + 12)
    sdf = np.full((H, W), 1e3, np.float32)
    x0, x1 = int(max(q[:, 0].min() - m, 0)), int(min(q[:, 0].max() + m, W))
    y0, y1 = int(max(q[:, 1].min() - m, 0)), int(min(q[:, 1].max() + m, H))
    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist, idx = cKDTree(q).query(np.stack([xx.ravel(), yy.ravel()], 1), distance_upper_bound=m + 2, workers=2)
        idx = np.minimum(idx, n - 1)
        dist = np.minimum(dist, m + 2)
        tan = np.gradient(q, axis=0)
        tan = tan / (np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-6)
        rel = np.stack([xx.ravel() - q[idx, 0], yy.ravel() - q[idx, 1]], 1)
        side = tan[idx, 0] * rel[:, 1] - tan[idx, 1] * rel[:, 0]
        fwd = tan[idx, 0] * rel[:, 0] + tan[idx, 1] * rel[:, 1]
        d = dist - r[idx]
        d = np.where((idx == 0) & (fwd < 0), np.maximum(d, -fwd), d)
        rows = 64
        jag = gaussian_filter1d(rg.normal(0, 1, rows), 3.0)
        jag = (jag - jag.min()) / (jag.max() - jag.min() + 1e-6)
        v = np.clip(side / r[idx], -1, 1)
        limit = L - r_end * jag_k * jag[np.clip(np.int32((v + 1) / 2 * (rows - 1)), 0, rows - 1)]
        d = np.maximum(d, idx * step - limit)
        d = np.where((idx == n - 1) & (fwd > 0), np.maximum(d, fwd), d)
        sdf[y0:y1, x0:x1] = np.float32(d.reshape(yy.shape))
    limbs = [dict(q=q, r=r, nodes=[int(kn * (n - 1)) for kn in nodes])]
    for k, (at, ang, length, rel_r) in enumerate(stubs):
        i = int(at * (n - 1))
        t = q[min(i + 4, n - 1)] - q[max(i - 4, 0)]
        b = np.arctan2(t[1], t[0]) + ang
        tip = q[i] + np.array((np.cos(b), np.sin(b))) * length
        mid = q[i] + np.array((np.cos(b + 0.14), np.sin(b + 0.14))) * length * 0.55
        sd, more = twig_shape([tuple(q[i]), tuple(mid), tuple(tip)], [r[i] * rel_r, r[i] * rel_r * 0.82], W, H, seed + 7 + k, jag_k=1.3, step=step)
        sdf = np.minimum(sdf, sd)
        limbs.extend(more)
    return sdf, limbs


def twig_shadow(path, radius, lift, W=1920, H=1080, light=(0.5, 0.5)):
    """The shadow a twig throws on the mud: lift is how high it is over the mud along its length, in pixels, so it touches where lift is 0 and comes away further on, softer the higher it is. It stops short of the snapped end."""
    q, _ = _polyline(path, 1.0)
    n = len(q)
    u = np.linspace(0, 1, n)
    r = np.interp(u, np.linspace(0, 1, len(radius)), radius)
    l = np.interp(u, np.linspace(0, 1, len(lift)), lift)
    keep = n - int(1.6 * r[-1])
    shift = np.array(light)[None] * (l + r * 0.5)[:, None]
    p = (q + shift)[:keep]
    r, l = r[:keep], l[:keep]
    m = float(r.max() + 6 + 0.5 * l.max())
    x0, x1 = int(max(p[:, 0].min() - m, 0)), int(min(p[:, 0].max() + m, W))
    y0, y1 = int(max(p[:, 1].min() - m, 0)), int(min(p[:, 1].max() + m, H))
    out = np.zeros((H, W), np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist, idx = cKDTree(p).query(np.stack([xx.ravel(), yy.ravel()], 1), workers=2)
    soft = 2.4 + 0.24 * l[idx]
    a = sm((r[idx] + soft - dist) / (2 * soft)) * (1 - 0.35 * l[idx] / (l.max() + 1e-6))
    out[y0:y1, x0:x1] = np.float32(a.reshape(yy.shape))
    return out


def main():
    if sys.argv[1] == "look":
        m = Mud()
        m.print_(TRACK_AT, np.pi, 300)
        m.char()
        s, sh, cav, H = m.shade()
        a = 200 + 170 * s - 90 * sh + 8 * np.minimum(cav, 0)
        cx0, cy0 = 300, 100
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(a, 0, 255))).crop((cx0, cy0, cx0 + 1920, cy0 + 1080)).save(os.path.join(OUT, "mud_look.png"))


if __name__ == "__main__":
    main()

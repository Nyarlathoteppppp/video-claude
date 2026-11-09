"""Everything the myth's Blender scenes need that takes scipy: mud with footprints, and the running ink the river's banks share.
Runs in the render environment and leaves .npy files for ink3d_myth.py to load.

    python ink3d_myth_prep.py tracks|grow|all
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

from ink_film import OUT

TEX = os.path.join(OUT, "3d", "myth", "tex")


def spectral(n, m, beta, seed):
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    s = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
    s[0, 0] = 0
    v = np.fft.irfft2(s, s=(n, m))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def run_ink(mask, cell, seed, count=420, ground_row=8, rock=None):
    """Ink running down a wall from the bottom edges of the strokes: each trickle wanders where the stone leans, thins, and may stop.
    mask is in world orientation: row 0 at the bottom. Returns (wetness, trail centres)."""
    r = np.random.default_rng(seed)
    n, m = mask.shape
    trail = np.zeros((n, m), np.float32)
    edge = (mask > .5) & (np.roll(mask, 3, 0) < .5)                            # stroke pixels with bare stone just below
    ys, xs = np.nonzero(edge)
    order = r.permutation(len(ys))[:count]
    lean = None
    if rock is not None:
        lean = np.gradient(gaussian_filter(rock, 5), axis=1)
    for k in order:
        y, x = float(ys[k]), float(xs[k])
        width = r.uniform(.9, 3.6) * (.3 + r.random() ** 2.4 * 2.0)
        life = r.random() ** 1.2 * (y * .8) + 90
        heavy = r.random() < .3
        if heavy:
            life = y + 50
            width = r.uniform(3, 6.5)
        drift = r.normal(0, .08)
        step = 0
        while y > 1 and step < life:
            step += 1
            yi, xi = int(y), int(np.clip(x, 0, m - 1))
            w = width * (1 - .55 * step / life) * (1 + .25 * math.sin(step * .07 + k))
            if lean is not None:
                drift = drift * .96 + float(np.clip(lean[yi, xi], -.6, .6)) * .05 + r.normal(0, .012)
            x = float(np.clip(x + drift, 2, m - 3))
            y -= 1.0
            a, b = max(int(x - w), 0), min(int(x + w) + 1, m)
            d = np.abs(np.arange(a, b) - x) / max(w, .5)
            trail[yi, a:b] = np.maximum(trail[yi, a:b], 1 - d ** 2)
    return gaussian_filter(trail, .8)


def bird_print(size):
    """A bird's foot pressed in mud: three toes forward and one back."""
    s = size
    yy, xx = np.mgrid[-s:s, -s:s].astype(np.float32) / s
    out = np.zeros_like(xx)
    for ang, length, w in ((math.pi / 2, .96, .082), (math.pi / 2 + .74, .78, .072), (math.pi / 2 - .74, .78, .072), (-math.pi / 2, .42, .058)):
        ca, sa = math.cos(ang), math.sin(ang)
        t = xx * ca + yy * sa
        d = np.abs(-xx * sa + yy * ca)
        wid = w * (1.15 - .85 * np.clip(t / length, 0, 1))
        out = np.maximum(out, sm((wid - d) / (wid * .55)) * sm(t / .02) * sm((length - t) / .07))
    pad = sm((.095 - np.hypot(xx, yy)) / .05)
    return np.maximum(out, pad)


def paw_print(size, cloven=False):
    """A beast's paw: a broad pad and four toes."""
    s = size
    yy, xx = np.mgrid[-s:s, -s:s].astype(np.float32) / s
    out = sm((.34 - np.hypot(xx / 1.15, (yy + .22) / .92)) / .08)
    for ox, oy, rx, ry in ((-.47, .17, .13, .19), (-.17, .47, .13, .21), (.17, .47, .13, .21), (.47, .17, .13, .19)):
        out = np.maximum(out, sm((1 - np.hypot((xx - ox) / rx, (yy - oy) / ry)) / .3))
    return out


def prep_tracks():
    """Mud by the water, a trail of bird and beast prints winding away from the lens. Row 0 is nearest the camera."""
    cell = .012
    n, m = 2600, 2200                # 31 × 26 units
    r = np.random.default_rng(91)
    mud = 0.05 * spectral(n, m, 3.3, 92) + .004 * spectral(n, m, 2.2, 93) + .0006 * spectral(n, m, 1.2, 94)
    # the ripples a river leaves when it draws back, bending where the mud did
    zr, xr = np.mgrid[:n, :m].astype(np.float32) * cell
    mud = mud + .004 * np.sin((zr * 1.0 + xr * .35 + 1.8 * spectral(n, m, 3.5, 95)) * 2.6)
    prints = np.zeros((n, m), np.float32)
    rim = np.zeros((n, m), np.float32)
    def stamp(stencil, x, y, angle, depth=1.0):
        h, w = stencil.shape
        ca, sa = math.cos(angle), math.sin(angle)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        xx, yy = xx - w / 2, yy - h / 2
        s = map_coordinates(stencil, [xx * sa + yy * ca + h / 2, xx * ca - yy * sa + w / 2], order=1)
        x0, y0 = int(x / cell - w / 2), int(y / cell - h / 2)
        ya, yb, xa, xb = max(y0, 0), min(y0 + h, n), max(x0, 0), min(x0 + w, m)
        if ya < yb and xa < xb:
            prints[ya:yb, xa:xb] = np.maximum(prints[ya:yb, xa:xb], s[ya - y0:yb - y0, xa - x0:xb - x0] * depth)
    # the bird comes in big and close, the beast crosses behind it
    t = np.linspace(0, 1, 34)
    px = 12.6 + 4.2 * np.sin(t * 5.2 + .4) - 1.6 * t
    py = 0.7 + 29.0 * t ** 1.25
    side = 1
    for i in range(len(t) - 1):
        ang = math.atan2(py[i + 1] - py[i], px[i + 1] - px[i]) - math.pi / 2
        size = int(.62 * (1 - .15 * t[i]) / cell)
        side = -side
        ox, oy = math.cos(ang) * .24 * side, math.sin(ang) * .24 * side
        stamp(bird_print(size), px[i] + ox, py[i] + oy, ang + r.normal(0, .12), r.uniform(.85, 1))
    t2 = np.linspace(0, 1, 24)
    qx = 19.5 - 11 * t2 + 1.6 * np.sin(t2 * 7)
    qy = 8.5 + 22.0 * t2
    for i in range(len(t2) - 1):
        ang = math.atan2(qy[i + 1] - qy[i], qx[i + 1] - qx[i]) - math.pi / 2
        side = -side
        stamp(paw_print(int(.42 / cell)), qx[i] + math.cos(ang) * .42 * side, qy[i] + math.sin(ang) * .42 * side, ang + r.normal(0, .1), r.uniform(.8, 1))
    prints = gaussian_filter(prints, 1.6)
    # the mud pushed out of a print stands up round it
    rim = np.clip(gaussian_filter(prints, 9) * 1.5 - prints, 0, 1)
    h = mud - .24 * prints + .035 * rim
    # water at the far left, the bank sloping into it
    z, x = np.mgrid[:n, :m].astype(np.float32) * cell
    bank = 7.4 + 3.0 * np.sin(z / 5.1 + .7) + .24 * z + .9 * spectral(n, m, 3.3, 96)
    h = h + .70 * np.clip((x - bank) / 6, -1.2, 1) + .003 * spectral(n, m, 2.0, 97)
    grit = spectral(n, m, .8, 98)
    alb = .68 + .07 * spectral(n, m, 3.0, 99) + .004 * grit + .08 * np.clip(rim, 0, 1)
    alb = alb - .14 * sm((.2 - h) / .35)                    # wet near the water
    np.save(os.path.join(TEX, "tracks_h.npy"), h.astype(np.float32))
    np.save(os.path.join(TEX, "tracks_a.npy"), np.clip(alb, .04, .9).astype(np.float32))
    np.save(os.path.join(TEX, "tracks_p.npy"), prints.astype(np.float32))
    print("tracks", h.min(), h.max())


def prep_grow():
    """Mud with a sign being made in it: chains of prints come in from three sides, their toes lengthening print by print until they cross and run on as the strokes of 鸟."""
    cell = .012
    n, m = 2050, 2800                # 24.6 × 33.6 units. Row 0 nearest the camera
    r = np.random.default_rng(111)
    S = 2
    im = Image.new("L", (m // S, n // S), 0)
    d = ImageDraw.Draw(im)
    def world_px(x, y):
        return x / cell / S, y / cell / S
    def stroke(pts, w0, w1):
        pts = np.asarray(pts, float)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        t = np.r_[0, np.cumsum(seg)]
        u = np.linspace(0, t[-1], max(int(t[-1] / .22), 3))
        xs, ys = np.interp(u, t, pts[:, 0]), np.interp(u, t, pts[:, 1])
        # a claw does not draw a straight line
        jit = np.cumsum(r.normal(0, .018, (len(u), 2)), 0)
        xs, ys = xs + jit[:, 0] - jit[:, 0].mean(), ys + jit[:, 1] - jit[:, 1].mean()
        for i in range(len(u) - 1):
            f = i / max(len(u) - 2, 1)
            w = (w0 + (w1 - w0) * f) * (1 + .18 * math.sin(i * .9))
            d.line([world_px(xs[i], ys[i]), world_px(xs[i + 1], ys[i + 1])], fill=255, width=max(int(w / cell / S), 1))
    def foot(x, y, ang, grow):
        """A bird's foot, its toes drawn out by grow."""
        for da, L in ((0, .96), (.74, .78), (-.74, .78)):
            L2 = L * (1 + grow)
            a0 = ang + da
            bend_ = da * -.10 * grow
            pts = [(x + math.cos(a0 + bend_ * q) * L2 * q, y + math.sin(a0 + bend_ * q) * L2 * q) for q in np.linspace(0, 1, 7)]
            stroke(pts, .12, .035)
        stroke([(x, y), (x - math.cos(ang) * .42, y - math.sin(ang) * .42)], .09, .035)
    # the sign itself: 鸟, large, in the middle of the mud
    cx, cy, size = 16.2, 10.6, 12.5
    def to_world(q):
        return cx + (q[0] - .5) * size, cy + (.5 - q[1]) * size
    from ink_oracle import ORACLE
    for st in ORACLE["鸟"]:
        if len(st) == 1:
            x, y = to_world(st[0])
            px_, py_ = world_px(x, y)
            d.ellipse((px_ - 16, py_ - 16, px_ + 16, py_ + 16), fill=255)
            continue
        fine = []
        for a, b in zip(st[:-1], st[1:]):
            for q in np.linspace(0, 1, 4)[:-1]:
                fine.append(to_world((a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)))
        fine.append(to_world(st[-1]))
        stroke(fine, .15, .085)
    # chains coming in. Each ends with its toes already part of a stroke
    chains = [((6.6, .2), (10.2, 6.6), 7), ((26.6, 1.2), (21.2, 5.6), 6), ((29.6, 20.0), (23.2, 14.6), 5), ((4.6, 22.0), (10.0, 14.6), 5)]
    for (x0, y0), (x1, y1), k in chains:
        ang = math.atan2(y1 - y0, x1 - x0)
        for i in range(k):
            f = i / (k - 1)
            side = (-1) ** i * .3
            x = x0 + (x1 - x0) * f - math.sin(ang) * side
            y = y0 + (y1 - y0) * f + math.cos(ang) * side
            foot(x, y, ang + r.normal(0, .07), 5.6 * f ** 2.6)
    prints = np.asarray(im.resize((m, n), Image.BICUBIC), np.float32) / 255
    prints = gaussian_filter(prints, 1.6)
    prints = np.clip((prints - .25) / .5, 0, 1)
    mud = .05 * spectral(n, m, 3.3, 112) + .004 * spectral(n, m, 2.2, 113) + .0006 * spectral(n, m, 1.2, 114)
    rim = np.clip(gaussian_filter(prints, 9) * 1.5 - prints, 0, 1)
    h = mud - .2 * prints + .03 * rim
    alb = .68 + .07 * spectral(n, m, 3.0, 115) + .004 * spectral(n, m, .8, 116) + .08 * np.clip(rim, 0, 1)
    np.save(os.path.join(TEX, "grow_h.npy"), h.astype(np.float32))
    np.save(os.path.join(TEX, "grow_a.npy"), np.clip(alb, .04, .9).astype(np.float32))
    np.save(os.path.join(TEX, "grow_p.npy"), prints.astype(np.float32))
    print("grow", h.min(), h.max())


if __name__ == "__main__":
    os.makedirs(TEX, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("tracks", "all"):
        prep_tracks()
    if what in ("grow", "all"):
        prep_grow()

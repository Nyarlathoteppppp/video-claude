"""The ground, the water and the paint of 八 · 河床 (ink3d_bed.py). Runs in the render environment, not Blender, because it needs fonts and scipy.

    python ink3d_bed_tex.py ground [--n=3400] [--tex=8192]     # rock, sand and far mountains: bed_{h,a}.npy, far_{h,a}.npy
    python ink3d_bed_tex.py sunrise|recede                     # that water level's paint: SET_w.npy, SET_near.npy (needs ground)
    python ink3d_bed_tex.py flood                              # the flood's surface, with what still stands out of it: flood_{h,a,g}.npy (needs ground)
    python ink3d_bed_tex.py low                                # the water's paint for ink3d_bed_low.py: tex_low/low_w.npy

The world is built from one camera. The river is drawn first in the picture itself (river_canvas): the character 我, worn by the water, with a river coming down
to it from the far mountains and going out at the bottom of the frame. Every spot on the ground then looks up where it falls in that picture, so the channel is 我 from up there and
only from up there; from anywhere else it is just a river among cliffs. Banks are shaped from the same picture: a far bank (above the water in the frame) may be a cliff,
because it faces us, but a near bank has to stay low enough not to hide the water behind it (see limit in ground).
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
from ink3d import OUT3D, W, H, erode, ink_albedo, lerp, massif, ridged, sm, spectral
from ink3d_map_tex import Painter, up
from ink_film import CORPUS, glyph_mask

TEXDIR = os.path.join(OUT3D, "bed", "tex")
HANZI = [c for c in CORPUS if "一" <= c <= "鿿"]

# the one camera everything is built from: at (0, 0, CAM_Z), looking along +y, tipped down by PITCH, lens 24
CAM_Z, PITCH, FOCAL = 60.0, math.radians(14.0), 1280.0
COS, SIN = math.cos(PITCH), math.sin(PITCH)
# the near ground: a square SIZE wide, centred at (0, CY). Beyond it, far_ground takes over
SIZE, CY = 840.0, 360.0
# the far ground
FAR_SIZE, FAR_CY = 3400.0, 2200.0
LEVEL = dict(sunrise=.6, recede=4.5, flood=29.0)
# sharp words for the part of the river just below the camera: centre x, y and size
NEAR = (30.0, 180.0, 240.0)

PAD_X, PAD_Y = 500, 200


def world_of(u, v):
    """The point on the ground seen at pixel (u, v)."""
    dz = -FOCAL * SIN + (H / 2 - v) * COS
    t = -CAM_Z / dz
    return t * (u - W / 2), t * (FOCAL * COS + (H / 2 - v) * SIN)


def screen_of(X, Y, z=0.0):
    zc = Y * COS + (CAM_Z - z) * SIN
    return W / 2 + FOCAL * X / zc, H / 2 - FOCAL * (Y * SIN - (CAM_Z - z) * COS) / zc, zc


def river_canvas():
    """The river as the camera is to see it, in pixels. The glyph narrows toward the top as anything does going away, its edges wander as a river's do,
    and real water joins it: the river from the far mountains into the top of the long stroke, two ways out at the bottom of the frame, a tributary at either side."""
    CW, CH = W + 2 * PAD_X, H + 2 * PAD_Y
    m = glyph_mask("我", "kaih", 900, 10)
    vv, uu = np.mgrid[:CH, :CW].astype(np.float32)
    uu, vv = uu - PAD_X, vv - PAD_Y
    r = np.random.default_rng(401)

    def wander(sigma, amp):
        n = gaussian_filter(r.normal(size=(CH, CW)).astype(np.float32), sigma)
        return n / n.std() * amp

    wu, wv = wander(75, 11) + wander(20, 3), wander(75, 8) + wander(20, 2.5)
    V0, V1, UC, S = 380.0, 1040.0, 870.0, 1.09
    t = (vv + wv - V0) / (V1 - V0)
    py = 92 + 793 * np.where(t > 0, np.maximum(t, 1e-4) ** .92, t)
    px = 452 + (uu + wu - UC) / (S * (.70 + .42 * t))
    M = map_coordinates(m, [py, px], order=1, cval=0).astype(np.float32)

    paths = [
        # from the far mountains, widening as it comes
        [(853, 400, 36), (872, 366, 30), (925, 342, 26), (980, 318, 20), (1000, 298, 15), (965, 281, 11), (932, 268, 8), (945, 256, 6), (968, 246, 4), (962, 238, 2.5)],
        # out at the bottom, the long stroke's way and the short one's
        [(1258, 1020, 70), (1270, 1060, 80), (1300, 1150, 100), (1340, 1260, 120)],
        [(677, 990, 46), (660, 1060, 52), (620, 1150, 60), (560, 1260, 70)],
        # a tributary into the far left of the long crossing stroke
        [(470, 800, 22), (380, 770, 18), (300, 790, 14), (190, 760, 10), (60, 730, 8), (-120, 690, 6), (-400, 650, 5)],
        # another feeding the dot, and the dot's way down into the main stroke
        [(1060, 450, 16), (1150, 410, 13), (1250, 420, 10), (1380, 380, 8), (1520, 395, 6), (1700, 350, 5), (2000, 330, 4)],
        [(935, 445, 12), (985, 448, 12)],
    ]
    im = Image.new("L", (CW, CH), 0)
    d = ImageDraw.Draw(im)
    for path in paths:
        for (x0, y0, w0), (x1, y1, w1) in zip(path[:-1], path[1:]):
            for i in range(14):
                a = i / 14
                x, y, w_ = x0 + (x1 - x0) * a + PAD_X, y0 + (y1 - y0) * a + PAD_Y, (w0 + (w1 - w0) * a) / 2
                d.ellipse((x - w_, y - w_, x + w_, y + w_), fill=255)
    P = gaussian_filter(np.asarray(im, np.float32) / 255, 2.5)
    P = map_coordinates(P, [vv + wv * .5 + PAD_Y, uu + wu * .5 + PAD_X], order=1, mode="nearest").astype(np.float32)
    M = np.maximum(M, P) > .5

    rows = np.arange(CH, dtype=np.float32)[:, None] * np.ones((1, CW), np.float32)
    # how far, straight up the picture, to the water's lower edge above; and straight down to its upper edge below
    last = np.maximum.accumulate(np.where(M, rows, -1e4), axis=0)
    gap = rows - last
    nxt = np.minimum.accumulate(np.where(M, rows, 1e5)[::-1], axis=0)[::-1]
    down = nxt - rows
    # and, inside the water, how far straight up and straight down to its edges
    last_dry = np.maximum.accumulate(np.where(~M, rows, -1e4), axis=0)
    nxt_dry = np.minimum.accumulate(np.where(~M, rows, 1e5)[::-1], axis=0)[::-1]
    return dict(M=M.astype(np.float32), ins=distance_transform_edt(M).astype(np.float32), out=distance_transform_edt(~M).astype(np.float32), gap=gap, down=down,
                above=rows - last_dry, below=nxt_dry - rows)


def look(arr, u, v):
    return map_coordinates(arr, [v + PAD_Y, u + PAD_X], order=1, mode="nearest").astype(np.float32)


def grid(N, size, cy):
    y, x = np.mgrid[:N, :N].astype(np.float32)
    return (x / (N - 1) - .5) * size, (y / (N - 1) - .5) * size + cy


def strata(h, tilt, seed):
    """Layers of uneven thickness, tilted a little: each is a level tread and a riser, so the walls come down in steps."""
    r = np.random.default_rng(seed)
    edges = np.concatenate([[-30], np.cumsum(r.choice([1.4, 2.2, 3.2, 4.8, 7.0], 60, p=[.2, .3, .25, .17, .08])) - 30])
    i = np.interp(h + tilt, edges, np.arange(len(edges)))
    f = i - np.floor(i)
    return (np.interp(np.floor(i) + sm((f - .28) / .72), np.arange(len(edges)), edges) - tilt).astype(np.float32)


def peaks_at(spots):
    """Mountains placed where they are to appear in the picture: (u, v of the foot, height, radius)."""
    return [(*world_of(u, v), h, rad) for u, v, h, rad in spots]


NEAR_PEAKS = [(60, 1000, 26, 18), (230, 1060, 20, 16), (1760, 1010, 26, 20), (1560, 1090, 16, 14),
              (110, 700, 46, 36), (280, 560, 40, 40), (1800, 660, 44, 40), (1600, 560, 32, 46),
              (250, 430, 60, 70), (560, 380, 44, 70), (20, 470, 70, 70), (1300, 365, 50, 80), (1600, 420, 66, 90), (1840, 440, 60, 80), (720, 360, 34, 50)]


def ground(N, T):
    c = river_canvas()
    X, Y = grid(N, SIZE, CY)
    cell = SIZE / N
    u, v, zc = screen_of(X, Y)
    behind = zc < 5
    u, v = np.where(behind, -1e4, u), np.where(behind, 2e4, v)
    ins, out = look(c["ins"], u, v), look(c["out"], u, v)
    Mw, gap, down, above, below = look(c["M"], u, v), look(c["gap"], u, v), look(c["down"], u, v), look(c["above"], u, v), look(c["below"], u, v)
    ph = FOCAL * COS / np.maximum(zc, 5)        # pixels a unit of height lifts a spot by, in the picture
    w0 = LEVEL["sunrise"]
    n1, n2, n3 = spectral(N, 2.8, 411), spectral(N, 2.0, 412), spectral(N, 1.5, 413)

    # the river's own floor: deep in the middle, a low shelf of sand on the near side, a steeper one under the far bank, bars where it runs wide
    near = (w0 + .6) - below * (.7 / ph)
    far = 2.8 * np.exp(-above / (8 + 3 * np.clip(n2, -1.5, 1.5)))
    side = 1.6 * np.exp(-ins / 8)
    bars = 1.9 * sm((n1 * .8 + .5 * n2 - .85) / .5) * sm((ins - 22) / 20)
    floor_ = np.maximum.reduce([-.8 + bars + .12 * n3, near, far, side])

    # the rock: placed by hand as ink3d_look does, on a broad shelf, with the bank rising from the water's edge
    rid = ridged(N, 414, base=1.0, octaves=3, warp=.05)
    rock = np.maximum(18 + 5 * n1 * .6 + 6 * (rid - .35), massif(X, Y, peaks_at(NEAR_PEAKS), 415))
    run = 34 + 12 * np.clip(spectral(N, 2.6, 416), -1.5, 1.5)
    h = rock * sm(out / run) ** .75
    # layers cut the walls low down; higher up it is only mountain
    layered = strata(h, 1.8 * spectral(N, 3.0, 417) + .012 * X + .007 * Y, 418)
    h = layered + (h - layered) * sm((h - 34) / 14)
    h = erode(h, cell, 419, wavelength=30, octaves=7, gain=.5, strength=.2)
    h = np.maximum(h, 0) * (1 - Mw)

    # a near bank must not stand in front of the water above it in the picture. The limit grows with the distance from that water, so the rock rises away from it as the back of a ridge,
    # and whatever is between two reaches of the river ends up a ridge showing us its cliff, as tall as it can be without hiding the reach behind it
    limit = gaussian_filter(np.minimum(.72 * gap / ph, 90).astype(np.float32), 4.0 / cell) + .05       # the picture's columns must not show in the rock
    h = limit * (1 - np.exp(-h / limit))
    edge = np.where(gap < down, w0 + .6, np.where(down < 60, 2.8, 1.6)).astype(np.float32)
    edge = gaussian_filter(edge, .7 / cell)
    full = floor_ * Mw + (edge + h) * (1 - Mw)
    soft = gaussian_filter(full, .35 / cell)
    full = soft + (full - soft) * sm((full - 3) / 3)                            # sand is smooth, rock is not
    full = full * sm((CY + SIZE / 2 - 50 - Y + 45 * spectral(N, 2.8, 420)) / 80)    # behind this the far ground takes over

    # the paint: ink3d_look's rock, and the shore
    a = ink_albedo(full, 431, cell)
    a = np.clip(a * .92 + .04, 0, 1)
    sand = 1 - sm((h - .5) / 1.6)
    z = full - w0
    tone = .58 + .24 * sm((z - .1) / .7)                                        # wet and dark at the water, paler as it dries
    tone = tone + .05 * n3
    # the marks each falling level left: thin lines along the sand
    lines = sum(np.exp(-((z + .03 * n2 - lv) / .035) ** 2) * (.5 + .5 * sm((n2 + .4) / .6)) for lv in (.12, .24, .4, .62, .88, 1.2, 1.6))
    tone = tone - .20 * np.clip(lines, 0, 1)
    # stones: thicker toward the bank
    r = np.random.default_rng(432)
    stones = gaussian_filter((r.random((N, N)) < .012 * sm(z / 1.1) * (.4 + .6 * sm((n1 + .3) / .8))).astype(np.float32), .9)
    tone = tone - np.clip(stones * 6.5, 0, 1) * .28
    a = a * (1 - sand) + np.clip(tone, .05, 1) * sand
    np.save(os.path.join(TEXDIR, "bed_h.npy"), full.astype(np.float32))
    A = up(np.clip(a, .03, 1), T)
    np.save(os.path.join(TEXDIR, "bed_a.npy"), (A * 255).astype(np.uint8))
    np.save(os.path.join(TEXDIR, "bed_floor.npy"), np.asarray(Image.fromarray(full.astype(np.float32)).resize((2048, 2048), Image.BILINEAR), np.float32))
    np.save(os.path.join(TEXDIR, "bed_m.npy"), np.asarray(Image.fromarray(Mw).resize((2048, 2048), Image.BILINEAR), np.float32))
    print("bed", full.shape, A.shape, "h", float(full.min()), float(full.max()))
    far_ground(c)


# the land beyond the shelf, as the river cuts on through it: foothills to either side. What stands behind them are ridge_layers, in ink3d_bed.py
FAR_PEAKS = [(-290, 640, 44, 70), (-420, 720, 60, 90), (-240, 800, 40, 70), (300, 650, 50, 80), (420, 760, 66, 100), (250, 840, 44, 70),
             (-420, 1050, 76, 130), (-650, 1000, 90, 160), (430, 1100, 84, 140), (680, 1000, 96, 170)]


def far_ground(c, N=1300, T=4096):
    X, Y = grid(N, FAR_SIZE, FAR_CY)
    cell = FAR_SIZE / N
    u, v, zc = screen_of(X, Y)
    out = look(c["out"], u, v)
    h = massif(X, Y, FAR_PEAKS, 441)
    h = np.maximum(h, 18 + 6 * np.clip(spectral(N, 2.6, 442), -1.5, 1.5))
    h = h * sm(out / (10 + 4 * np.clip(spectral(N, 2.6, 443), -1.5, 1.5))) ** .7
    h = erode(h, cell, 444, wavelength=70, octaves=6, gain=.45, strength=.14)
    h = np.maximum(h, 0) * sm((Y - 520) / 140) * sm((FAR_SIZE / 2 - np.abs(X)) / 160)
    a = ink_albedo(h, 445, cell)
    np.save(os.path.join(TEXDIR, "far_h.npy"), h.astype(np.float32))
    np.save(os.path.join(TEXDIR, "far_a.npy"), (up(np.clip(a, .03, 1), T) * 255).astype(np.uint8))
    print("far", h.shape, float(h.max()))


def lic(noise, ux, uy, steps, step):
    n = noise.shape[0]
    y, x = np.mgrid[:n, :n].astype(np.float32)
    out = noise.copy()
    px, py, qx, qy = x.copy(), y.copy(), x.copy(), y.copy()
    for _ in range(steps):
        px, py = px + lerp(ux, px, py) * step, py + lerp(uy, px, py) * step
        qx, qy = qx - lerp(ux, qx, qy) * step, qy - lerp(uy, qx, qy) * step
        out += lerp(noise, px, py) + lerp(noise, qx, qy)
    return out / out.std()


def current(Q):
    """Which way the water runs: down the picture toward us, and along the strokes where it has found them."""
    Mw = np.load(os.path.join(TEXDIR, "bed_m.npy"))
    Mw = np.asarray(Image.fromarray(Mw).resize((Q, Q), Image.BILINEAR), np.float32)
    cell = SIZE / Q
    ins = distance_transform_edt(Mw > .5).astype(np.float32) * cell
    gy, gx = np.gradient(gaussian_filter(ins, 2.0 / cell), cell)
    g = np.hypot(gx, gy) + 1e-5
    tx, ty = -gy / g, gx / g
    flip = ty > 0
    tx, ty = np.where(flip, -tx, tx), np.where(flip, -ty, ty)
    curl = spectral(Q, 3.2, 451) * .4
    bx, by = np.sin(curl), -np.cos(curl)
    w = sm(gaussian_filter(Mw, 3))
    ux, uy = bx * (1 - w) + tx * w, by * (1 - w) + ty * w
    m = np.hypot(ux, uy) + 1e-5
    return gaussian_filter(ux / m, 2), gaussian_filter(uy / m, 2), Mw


def current_of(R):
    """The flood's current: straight at us, bending around what stands in it."""
    n = R.shape[0]
    gy, gx = np.gradient(R, SIZE / n)
    curl = spectral(n, 3.2, 469) * .4
    ux, uy = np.sin(curl) - gx * .9, -np.cos(curl) - gy * .3
    m = np.hypot(ux, uy) + 1e-5
    return gaussian_filter(ux / m, 2), gaussian_filter(uy / m, 2)


def walk_words(ux, uy, Q, Mw, seeds, r, ymax=430.0):
    """Short lines of very small words carried along by the current."""
    out = []
    c = 0
    for _ in range(seeds):
        x, y = r.uniform(-150, 200), r.uniform(50, ymax)
        size = r.uniform(1.3, 2.3) * (1 + y / 260)
        for _s in range(r.integers(5, 14)):
            ix, iy = int((x + SIZE / 2) / SIZE * Q), int((y - CY + SIZE / 2) / SIZE * Q)
            if not (0 <= ix < Q and 0 <= iy < Q) or Mw[iy, ix] < .5:
                break
            vx, vy = float(ux[iy, ix]), float(uy[iy, ix])
            out.append((HANZI[c % len(HANZI)], x, y, size, r.uniform(.45, .95), math.degrees(math.atan2(-vy, -vx)) - 90))
            c += 1
            x, y = x + vx * size * 1.2, y + vy * size * 1.2
    return out


def tide_words(r):
    """The words the falling water left: stranded along its lines on the sand, thin as a line of debris from up here, words when you look."""
    z = np.load(os.path.join(TEXDIR, "bed_floor.npy")) - LEVEL["sunrise"]
    Mw = np.load(os.path.join(TEXDIR, "bed_m.npy"))
    n = z.shape[0]
    gy, gx = np.gradient(gaussian_filter(z, 2))
    cx, cy, size = NEAR
    occ = set()
    out = []
    c = 40
    for lv in (.24, .4, .62, .88, 1.2):
        for _ in range(26000):
            x, y = r.uniform(cx - size / 2, cx + size / 2), r.uniform(cy - size / 2, cy + size / 2)
            ix, iy = int((x + SIZE / 2) / SIZE * n), int((y - CY + SIZE / 2) / SIZE * n)
            if abs(z[iy, ix] - lv) > .05 or Mw[iy, ix] < .3:
                continue
            key = (int(x / 1.7), int(y / 1.7))
            if key in occ:
                continue
            occ.add(key)
            tx, ty = -gy[iy, ix], gx[iy, ix]
            if ty < 0:
                tx, ty = -tx, -ty
            out.append((HANZI[c % len(HANZI)], x, y, r.uniform(1.25, 1.7), r.uniform(.5, .9), math.degrees(math.atan2(ty, tx)) - 90))
            c += 1
    return out


def water(which):
    """The water's face at this level: bright paper with the current's lines pulled through it, darker where it runs fast at this level, a dark thread and a little foam at the shore."""
    Q = 2048
    level = LEVEL[which]
    ux, uy, Mw = current(Q)
    z = np.load(os.path.join(TEXDIR, "bed_floor.npy"))
    depth = level - z
    fine = lic(spectral(Q, .4, 452), ux, uy, 24, 1.0)
    broad = lic(spectral(Q, 1.4, 453), ux, uy, 34, 1.8)
    lines = sm((fine * .6 + broad * .7 - 1.0) / .7)
    dark = sm((broad * .9 + spectral(Q, 2.6, 454) * .5 - .2) / 1.3)
    shore = np.exp(-np.maximum(depth, 0) / .10)
    foam = np.exp(-(np.maximum(depth, 0) - .22) ** 2 / .02) * sm((fine + .9) / .7)
    if which == "sunrise":
        paint = .995 - .09 * lines - .03 * dark
        paint = paint - .22 * shore + .10 * foam
    else:
        paint = .90 - .34 * lines - .16 * dark
        paint = paint - .30 * shore + .50 * foam
    r = np.random.default_rng(455)
    marks = walk_words(ux, uy, Q, Mw, 2600, r, ymax=430.0)
    p = Painter(4096, SIZE, 0.0, CY)
    cx, cy, size = NEAR
    q = Painter(6144, size, cx, cy)
    for ch, x, y, s, k, ang in marks:
        p.char(ch, x, y, s, "kai", k, ang)
        if abs(x - cx) < size / 2 + 4 and abs(y - cy) < size / 2 + 4:
            q.char(ch, x, y, s, "kai", k, ang)
    if which == "sunrise":
        for ch, x, y, s, k, ang in tide_words(r):
            q.char(ch, x, y, s, "kai", k, ang)
    A = up(np.clip(paint, .05, 1), 4096, Image.BILINEAR) * (1 - (.20 if which == "sunrise" else .34) * np.clip(p.ink, 0, 1))
    np.save(os.path.join(TEXDIR, f"{which}_w.npy"), (np.clip(A, .05, 1) * 255).astype(np.uint8))
    np.save(os.path.join(TEXDIR, f"{which}_near.npy"), (np.clip(q.ink, 0, 1) * 255).astype(np.uint8))
    print(which, A.shape, "marks", len(marks))


def flood(N=3400, T=8192):
    """The flood: the canyon full to over its rim, only the mountains' heads out, and nothing to see but what the water is doing. Painted the way 马远 painted water: row after row of swells, each a pulled line with a white crest
    and a wash darkening behind it, crowding together toward the far gap and parting around whatever stands in the way, white where it tears on rock. Words ride in the lines."""
    Q = 4096
    level = LEVEL["flood"]
    rock = np.load(os.path.join(TEXDIR, "bed_h.npy"))
    a_rock = np.load(os.path.join(TEXDIR, "bed_a.npy")).astype(np.float32) / 255
    Xq, Yq = grid(Q, SIZE, CY)
    Rq = np.asarray(Image.fromarray(rock).resize((Q, Q), Image.BILINEAR), np.float32)
    above = np.maximum(gaussian_filter(Rq, 10 / (SIZE / Q)) - (level - 6), 0)
    # the rows' long bends, and the scallops along each of them, which vary across the stream and hardly at all along it
    tile = spectral(2048, 2.6, 462)
    warp = spectral(Q, 3.6, 461) * 18 + lerp(tile, Xq * 1.3 + 700, Yq * .1 + 300) * 1.5
    # every swell a row, wider apart near us; the rows bend upstream in front of rock
    phi = 2.83 * np.sqrt(np.maximum(Yq + warp - 2.6 * above, 1))
    f = phi - np.floor(phi)
    press = sm((spectral(Q, 2.0, 465) + .7) / 1.2)
    line = np.exp(-((f - .06) / (.035 + .04 * press)) ** 2)                 # the pulled line, just behind the crest
    crest = np.exp(-(np.minimum(f, 1 - f) / .05) ** 2)
    wash = sm((f - .1) / .75)                                                # darkening to the next crest
    streak = sm((lic(spectral(Q // 2, .4, 466), *current_of(gaussian_filter(Rq, 8)[::2, ::2]), 24, 1.0) - 1.0) / .8)
    streak = up(streak, Q, Image.BILINEAR)
    d = np.maximum(level - Rq, 0)
    tear = np.exp(-d / 1.6)
    far = sm((Yq - 170) / 190)                                               # the rows give way to plain tone before they turn to moire
    paint = .97 - (.20 * wash + .05 * streak) * (1 - .55 * far) - .60 * line * (1 - far) ** 1.4
    paint = paint + (1 - paint) * np.clip(crest * .6 * (1 - far) + tear * .95, 0, 1)
    swell = (f - .5) * -1.1 * (1 - far) * sm((Yq - 34) / 6)
    S = level + swell + .6 * spectral(Q, 2.9, 467)
    Hq = np.maximum(up(rock, Q, Image.BILINEAR), S)
    water_ = sm((S - Rq + .1) / .4)
    r = np.random.default_rng(468)
    p = Painter(T, SIZE, 0.0, CY)
    c = 0
    Nq = Q - 1
    for _ in range(2400):
        x, y = r.uniform(-120, 160), r.uniform(36, 240)
        size = r.uniform(1.0, 1.7) * (.55 + y / 110)
        ang = r.normal(0, 7)
        rad = math.radians(ang)
        for _s in range(r.integers(5, 15)):
            ix, iy = int((x + SIZE / 2) / SIZE * Nq), int((y - CY + SIZE / 2) / SIZE * Nq)
            if not (0 <= ix <= Nq and 0 <= iy <= Nq):
                break
            p.char(HANZI[c % len(HANZI)], x, y, size, "kai", r.uniform(.5, .9), ang)
            c += 1
            x, y = x + math.cos(rad) * size * 1.25, y + math.sin(rad) * size * 1.25
    Pa = up(np.clip(paint, .03, 1), T, Image.BILINEAR) * (1 - .5 * np.clip(p.ink, 0, 1))
    Wa = up(water_, T, Image.BILINEAR)
    A = a_rock * (1 - Wa) + Pa * Wa
    np.save(os.path.join(TEXDIR, "flood_h.npy"), np.asarray(Image.fromarray(Hq).resize((N, N), Image.BILINEAR), np.float32))
    np.save(os.path.join(TEXDIR, "flood_a.npy"), (np.clip(A, .03, 1) * 255).astype(np.uint8))
    np.save(os.path.join(TEXDIR, "flood_g.npy"), (np.asarray(Image.fromarray(water_).resize((2048, 2048), Image.BILINEAR), np.float32) * .86 * 255).astype(np.uint8))
    # how it will lie in the frame, to check it without rendering
    print("rock percentiles under Y=600", np.percentile(rock[grid(N, SIZE, CY)[1] < 600], [50, 75, 90, 95, 99]))
    Ap = gaussian_filter(A[::2, ::2], 1.0)
    pv = np.ones((540, 960), np.float32)
    rows = np.arange(120, 540)
    for v in rows:
        dep = math.atan2(v * 2 - H / 2, FOCAL) + PITCH
        if dep <= .001:
            continue
        Yp = (CAM_Z - level) / math.tan(dep)
        xs = (np.arange(960) * 2 - W / 2) * math.hypot(Yp, CAM_Z - level) / FOCAL
        iy = int((Yp - CY + SIZE / 2) / SIZE * (T // 2 - 1))
        ix = ((xs + SIZE / 2) / SIZE * (T // 2 - 1)).astype(int)
        if 0 <= iy < T // 2:
            ok = (ix >= 0) & (ix < T // 2)
            pv[v][ok] = Ap[iy, ix[ok]]
    Image.fromarray((pv * 255).astype(np.uint8)).save(os.path.join(TEXDIR, "flood_preview.png"))
    print("flood", Hq.shape, float(water_.mean()))


LOW_LEVEL = 1.02        # as LEVEL in ink3d_bed_low.py
LOW_FLOW = (40.0, -20.0, 180.0)


def low(Q=3072):
    """The low shot's water, on the first canyon's ground: what the current draws on it. It runs down the canyon and bends along whatever stands out of it, so the long lines show the flow, thicker in the wake of rocks,
    and a broken rim of froth follows the shore. Pale where this says so: everything else about the water is left to the real reflections."""
    h = np.load(os.path.join(OUT3D, "bed", "tex_low", "bed_h.npy"))
    cx, cy, size = LOW_FLOW
    y, x = np.mgrid[:Q, :Q].astype(np.float32)
    X, Y = (x + .5) / Q * size + cx - size / 2, (y + .5) / Q * size + cy - size / 2
    N, G = h.shape[0], 440.0
    H = map_coordinates(h, [(Y + G / 2) / G * (N - 1), (X + G / 2) / G * (N - 1)], order=1).astype(np.float32)
    cell = size / Q
    wet = H < LOW_LEVEL
    d = distance_transform_edt(wet).astype(np.float32) * cell
    gy, gx = np.gradient(gaussian_filter(d, 1.2 / cell), cell)
    g = np.hypot(gx, gy) + 1e-5
    tx, ty = -gy / g, gx / g
    bx, by = .28, -.96
    flip = tx * bx + ty * by < 0
    tx, ty = np.where(flip, -tx, tx), np.where(flip, -ty, ty)
    w = np.exp(-d / 5.0)
    ux, uy = bx * (1 - w) + tx * w, by * (1 - w) + ty * w
    m = np.hypot(ux, uy) + 1e-5
    ux, uy = gaussian_filter(ux / m, 3), gaussian_filter(uy / m, 3)
    fine = lic(spectral(Q, .35, 481), ux, uy, 40, 2.0)
    broad = lic(spectral(Q, 1.4, 482), ux, uy, 40, 3.0)
    lines = sm((fine * .62 + broad * .55 - 1.25) / .75) * (.45 + .55 * sm((broad + 1) / 2))
    wake = np.exp(-d / 3.0)
    n = spectral(Q, 2.2, 483)
    depth = np.maximum(LOW_LEVEL - H, 0)
    rim = np.exp(-(depth / .05) ** 2) * sm((n + fine * .35 + .3) / .7) * wet
    paint = np.clip(lines * (.30 + .4 * wake) + rim * .75, 0, 1) * wet
    np.save(os.path.join(TEXDIR, "..", "tex_low", "low_w.npy"), (paint * 255).astype(np.uint8))
    print("low", paint.shape, float(paint.mean()), "wet", float(wet.mean()))


if __name__ == "__main__":
    q = argparse.ArgumentParser()
    q.add_argument("which")
    q.add_argument("--n", type=int, default=3400)
    q.add_argument("--tex", type=int, default=8192)
    args = q.parse_args()
    os.makedirs(TEXDIR, exist_ok=True)
    if args.which == "ground":
        ground(args.n, args.tex)
    elif args.which == "low":
        low()
    elif args.which == "flood":
        flood(args.n, args.tex)
    else:
        water(args.which)

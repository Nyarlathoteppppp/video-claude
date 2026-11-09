"""The first shot of 鬼夜哭, 5.0 to 14.0: one painting, one camera, slowly closer. The bank of k1_wide with a mountain beyond it in layers of wet ink going down into mist, a bird on a dead branch, a horse on the same ground as the man, who kneels small below them with his twig. One after another they become characters where they stand: the ink further from the strokes goes first, into the paper; what is left draws in; the strokes go dark with it. The horse turns on end as it goes, because the beasts of the oracle bones are written standing up.

Any moment of it is a function of the film second, so stills and motion are the same code.

    python film_myth_world.py glyphs           # the characters alone: keys/_x_glyphs.png
    python film_myth_world.py still            # x1_world, x2_shan, x3_niao, x4_ma_turn, x45, x5_ma, x6_page in myth/
    python film_myth_world.py at 9.3 name      # one moment

    python film_myth_world.py seq [--only a,b] # 5.0 up to 而天雨粟 at 30 fps into seq_raw4/, as film_myth_seq.py names its frames

From 14.0 the camera goes down onto the mud, past him, until the print is the print of the close shot; from 15.5 he writes there (film_myth_write.py). The characters are the outlines of the Shang oracle forms, read at render time from a file kept on the render machine and out of this repository (see GLYPHS): SVG paths on a square of 300, y down, even-odd for the holes.
"""
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

import ink_grade

_fft0, _fields = ink_grade.fft_noise, {}


def _fft(h, w, beta, seed):
    key = (h, w, float(beta), seed)
    if key not in _fields:
        a = _fft0(h, w, beta, seed)
        a.flags.writeable = False
        _fields[key] = a
    return _fields[key]


ink_grade.fft_noise = _fft

from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, gaussian_filter, gaussian_filter1d, label, map_coordinates, maximum_filter
from scipy.spatial import cKDTree

from film_myth import paint_clouds, tidy
from film_myth_mud import Mud, TRACK_AT, dense, ink_of, twig_shadow, unit
from film_myth_wash import INK_ABSORB, OUT, Sheet, Wash, blob, develop, grow, lumps, noise, sm
from film_myth_wide import bristle, brush, cached, cangjie, ground_relief, painted_twig, ridge
from film_myth_write import Writer

W, H = 1920, 1080
MYTH = os.path.dirname(OUT)
SS = 3
GLYPHS = os.path.expanduser(os.environ.get("ORACLE_GLYPHS", "~/render_work/glyphs/oracle_glyphs.json"))
PAPER = 95      # the sheet of k1_wide, so that the cut to the bank is onto the same paper


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def grid_():
    return np.mgrid[0:H, 0:W].astype(np.float32)


# ---------------------------------------------------------------- the characters

# where each stands at the end, how big
AT = dict(shan=("山", (430, 305), 1.45), niao=("鳥", (1340, 320), 1.35), ma=("馬", (830, 605), 1.5),
          ri=("日", (850, 128), 0.46), yue=("月", (1060, 196), 0.46), shui=("水", (470, 700), 0.46), mu=("木", (1190, 690), 0.46), yu=("雨", (1530, 250), 0.46))
SMALL = ("ri", "yue", "shui", "mu", "yu")
_TOK = re.compile(r"[MmLlHhVvCcSsQqAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _arc(x0, y0, rx, ry, phi, fa, fs, x, y, n=24):
    phi = np.radians(phi)
    c, s = np.cos(phi), np.sin(phi)
    dx, dy = (x0 - x) / 2, (y0 - y) / 2
    x1, y1 = c * dx + s * dy, -s * dx + c * dy
    rx, ry = abs(rx), abs(ry)
    L = x1 * x1 / (rx * rx) + y1 * y1 / (ry * ry)
    if L > 1:
        rx, ry = rx * np.sqrt(L), ry * np.sqrt(L)
    q = max(0.0, (rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1) / (rx * rx * y1 * y1 + ry * ry * x1 * x1))
    k = np.sqrt(q) * (-1 if fa == fs else 1)
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    cx, cy = c * cx1 - s * cy1 + (x0 + x) / 2, s * cx1 + c * cy1 + (y0 + y) / 2
    a0 = np.arctan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    a1 = np.arctan2((-y1 - cy1) / ry, (-x1 - cx1) / rx)
    da = a1 - a0
    if fs == 0 and da > 0:
        da -= 2 * np.pi
    if fs == 1 and da < 0:
        da += 2 * np.pi
    return [(cx + rx * np.cos(a0 + da * t) * c - ry * np.sin(a0 + da * t) * s, cy + rx * np.cos(a0 + da * t) * s + ry * np.sin(a0 + da * t) * c) for t in np.linspace(0, 1, n)[1:]]


def svg_subpaths(p):
    """Each closed part of an SVG path as a list of points."""
    t = _TOK.findall(p)
    i, out, cur, x, y, cmd, px, py = 0, [], [], 0.0, 0.0, None, None, None

    def num():
        nonlocal i
        v = float(t[i])
        i += 1
        return v

    while i < len(t):
        if re.match("[A-Za-z]", t[i]):
            cmd = t[i]
            i += 1
        if cmd in "Zz":
            if cur:
                out.append(cur)
                x, y = cur[0]
                cur = []
            px = None
            continue
        rel, C = cmd.islower(), cmd.upper()
        if C == "M":
            if cur:
                out.append(cur)
            a, b = num(), num()
            x, y = (x + a, y + b) if rel else (a, b)
            cur = [(x, y)]
            cmd, px = ("l" if rel else "L"), None
        elif C == "L":
            a, b = num(), num()
            x, y = (x + a, y + b) if rel else (a, b)
            cur.append((x, y))
            px = None
        elif C == "H":
            a = num()
            x = x + a if rel else a
            cur.append((x, y))
            px = None
        elif C == "V":
            b = num()
            y = y + b if rel else b
            cur.append((x, y))
            px = None
        elif C in "CS":
            if C == "C":
                a1, b1 = num(), num()
                if rel:
                    a1, b1 = x + a1, y + b1
            else:
                a1, b1 = (2 * x - px, 2 * y - py) if px is not None else (x, y)
            a2, b2, a, b = num(), num(), num(), num()
            if rel:
                a2, b2, a, b = x + a2, y + b2, x + a, y + b
            for u in np.linspace(0, 1, 17)[1:]:
                v = 1 - u
                cur.append((v ** 3 * x + 3 * v * v * u * a1 + 3 * v * u * u * a2 + u ** 3 * a, v ** 3 * y + 3 * v * v * u * b1 + 3 * v * u * u * b2 + u ** 3 * b))
            px, py, x, y = a2, b2, a, b
        elif C == "Q":
            a1, b1, a, b = num(), num(), num(), num()
            if rel:
                a1, b1, a, b = x + a1, y + b1, x + a, y + b
            for u in np.linspace(0, 1, 13)[1:]:
                v = 1 - u
                cur.append((v * v * x + 2 * v * u * a1 + u * u * a, v * v * y + 2 * v * u * b1 + u * u * b))
            x, y, px = a, b, None
        elif C == "A":
            rx, ry, ph, fa, fs = num(), num(), num(), num(), num()
            a, b = num(), num()
            if rel:
                a, b = x + a, y + b
            cur += _arc(x, y, rx, ry, ph, int(fa), int(fs), a, b)
            x, y, px = a, b, None
        else:
            raise ValueError(cmd)
    if cur:
        out.append(cur)
    return out


def glyph_mask(name):
    """The character at film size, exact: 0 to 1, how much of each pixel the outline covers, even-odd."""
    with open(GLYPHS, encoding="utf8") as fh:
        data = json.load(fh)
    ch, c, S = AT[name]
    v = data[ch]
    p = v if isinstance(v, str) else (v.get("path") or v.get("d"))
    bits = np.zeros((H * SS, W * SS), bool)
    for part in svg_subpaths(p):
        q = [((c[0] + (a - 150) * S) * SS, (c[1] + (b - 150) * S) * SS) for a, b in part]
        xs, ys = [a for a, _ in q], [b for _, b in q]
        x0, x1, y0, y1 = int(max(min(xs) - 2, 0)), int(min(max(xs) + 3, W * SS)), int(max(min(ys) - 2, 0)), int(min(max(ys) + 3, H * SS))
        im = Image.new("1", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(im).polygon([(a - x0, b - y0) for a, b in q], fill=1)
        bits[y0:y1, x0:x1] ^= np.asarray(im, bool)
    return np.asarray(Image.fromarray(np.uint8(bits) * 255).resize((W, H), Image.BOX), np.float32) / 255


def nf(c, S):
    return lambda p: (c[0] + (p[0] - 150) * S, c[1] + (p[1] - 150) * S)


# ---------------------------------------------------------------- brushes

def _field(pts, width, seed, tremble):
    q = dense(pts, 1.0, wobble=tremble, seed=seed)
    n = len(q)
    wv = gaussian_filter1d(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(width)), width).astype(np.float64), max(n / 24, 1.0), mode="nearest")
    m = float(wv.max() + 7)
    x0, x1 = int(max(q[:, 0].min() - m, 0)), int(min(q[:, 0].max() + m, W))
    y0, y1 = int(max(q[:, 1].min() - m, 0)), int(min(q[:, 1].max() + m, H))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist, idx = cKDTree(q).query(np.stack([xx.ravel(), yy.ravel()], 1), distance_upper_bound=m + 2, workers=2)
    idx = np.minimum(idx, n - 1)
    dist = np.minimum(dist, m + 2)
    tan = np.gradient(q, axis=0)
    tan = tan / (np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-6)
    side = tan[idx, 0] * (yy.ravel() - q[idx, 1]) - tan[idx, 1] * (xx.ravel() - q[idx, 0])
    return (x0, x1, y0, y1), yy.shape, dist, idx, np.clip(side / wv[idx], -1, 1), idx / max(n - 1, 1), wv, n


def wet_stroke(sheet, pts, width, seed, side=0.0, ends=(1.0, 1.0), tremble=0.9):
    """Where one loaded stroke of a big soft brush puts its ink, film size, about 0 to 1: the side the belly of the brush rides on takes more (side, -1 to 1), the hairs leave faint streaks along it, and it can run from full to thin (ends). The wash does the rest."""
    (x0, x1, y0, y1), shape, dist, idx, v, uf, wv, n = _field(pts, width, seed, tremble)
    r = np.random.default_rng(seed)
    rows, cols = 48, max(n, 8)
    hair = unit(gaussian_filter(r.normal(0, 1, (rows, cols)), (1.0, cols / 6)))
    T = map_coordinates(hair, [(v + 1) / 2 * (rows - 1), idx.astype(np.float64)], order=1, mode="nearest")
    tooth = sheet.tooth[y0:y1, x0:x1].ravel()
    body = np.clip((wv[idx] - dist - 0.7 * tooth) / 2.2, 0, 1)
    tone = (1 + side * v) * (0.86 + 0.14 * T) * np.interp(uf, np.linspace(0, 1, len(ends)), ends)
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = np.float32((body * np.maximum(tone, 0)).reshape(shape))
    return out


# ---------------------------------------------------------------- the horse's shape

HORSE_G, HORSE_X0, HORSE_U = 781.0, 785.0, 270.0      # film y of the ground, film x of the chest's zero, film pixels to a horse's height at the withers
# In heights at the withers, x forward being left, y up, the near side all the way round. Body as long as it is high, head four tenths of that, the neck arched and running into the withers, knee and hock, small hooves.
HORSE_NEAR = [
    (-0.135, 1.345), (-0.06, 1.315), (0.03, 1.24), (0.12, 1.14), (0.20, 1.05), (0.27, 1.00),      # poll, crest, withers
    (0.40, 0.955), (0.55, 0.945), (0.68, 0.965), (0.80, 0.985), (0.92, 0.95), (1.00, 0.875),         # back, loin, croup, dock
    (1.035, 0.79), (1.03, 0.70), (1.00, 0.60), (0.975, 0.50), (1.005, 0.405), (0.975, 0.35),         # buttock, gaskin, the point of the hock
    (0.965, 0.22), (0.972, 0.125), (0.962, 0.085), (0.950, 0.045), (0.943, 0.0),                     # cannon, fetlock, pastern, heel
    (0.886, 0.0), (0.897, 0.05), (0.908, 0.085), (0.915, 0.125), (0.915, 0.22), (0.915, 0.36), (0.905, 0.42),
    (0.855, 0.52), (0.78, 0.60), (0.72, 0.625),                                                      # gaskin, stifle, flank
    (0.62, 0.555), (0.50, 0.50), (0.38, 0.475), (0.30, 0.47),                                         # belly
    (0.27, 0.465), (0.25, 0.40), (0.232, 0.36), (0.242, 0.315), (0.242, 0.28), (0.225, 0.22), (0.222, 0.14), (0.232, 0.105), (0.212, 0.055), (0.205, 0.0),      # elbow, forearm, knee, cannon, fetlock, heel
    (0.148, 0.0), (0.156, 0.05), (0.164, 0.085), (0.168, 0.12), (0.168, 0.22), (0.156, 0.285), (0.158, 0.325), (0.170, 0.365), (0.135, 0.47),
    (0.10, 0.54), (0.055, 0.62), (0.04, 0.70),                                                       # breast, point of shoulder
    (0.02, 0.76), (-0.035, 0.87), (-0.075, 0.97), (-0.10, 1.06), (-0.115, 1.11),                      # under the neck, the throat
    (-0.15, 1.09), (-0.205, 1.06), (-0.275, 1.02), (-0.31, 0.99), (-0.34, 0.985), (-0.37, 0.995),    # jaw, chin, lip
    (-0.388, 1.025), (-0.38, 1.065), (-0.335, 1.13), (-0.275, 1.215), (-0.215, 1.29), (-0.165, 1.335),      # muzzle, nose, forehead
    (-0.128, 1.455), (-0.098, 1.335),                                                                # the ear
]
HORSE_FAR_FORE = [(0.20, 0.60), (0.185, 0.45), (0.15, 0.32), (0.152, 0.285), (0.14, 0.22), (0.14, 0.14), (0.15, 0.105), (0.128, 0.055), (0.121, 0.0), (0.068, 0.0), (0.074, 0.05), (0.082, 0.085), (0.085, 0.12), (0.08, 0.22), (0.07, 0.29), (0.07, 0.335), (0.07, 0.45), (0.08, 0.58)]
HORSE_FAR_HIND = [(0.82, 0.66), (0.845, 0.52), (0.872, 0.41), (0.84, 0.35), (0.825, 0.22), (0.832, 0.125), (0.822, 0.085), (0.812, 0.04), (0.805, 0.0), (0.748, 0.0), (0.756, 0.05), (0.77, 0.085), (0.777, 0.125), (0.775, 0.22), (0.775, 0.36), (0.765, 0.43), (0.72, 0.52), (0.68, 0.62)]


def hf(p):
    return (HORSE_X0 + p[0] * HORSE_U, HORSE_G - p[1] * HORSE_U)


def shape_of(points, closed=True, smooth=1.0):
    """Filled, with a distance out from it: negative inside. Through its points on a smooth line."""
    p = [tuple(q) for q in points]
    if closed:
        p = p + [p[0]]
    q = dense(p, smooth)
    im = Image.new("L", (W * 2, H * 2), 0)
    ImageDraw.Draw(im).polygon([(a * 2, b * 2) for a, b in q], fill=255)
    m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255
    inside = m > 0.5
    return np.float32(distance_transform_edt(~inside) - distance_transform_edt(inside))


BRANCH = dict(path=[(1076, 420), (1150, 406), (1256, 442), (1366, 434), (1478, 448), (1548, 376), (1580, 270), (1620, 150), (1636, -40)], radius=[3.0, 4.5, 6.0, 8.0, 10.5, 13.0, 15.5, 18.0, 20.0], stubs=((0.09, -0.75, 50, 0.7), (0.26, 0.85, 44, 0.6), (0.42, 0.9, 64, 0.55), (0.60, -0.9, 96, 0.5)), seed=66)


# ---------------------------------------------------------------- the painting

def largest_part(fine, coarse):
    """A wash's near-load fires at noise peaks far from what is being painted: only what is joined to the main body is it."""
    lab, _ = label(gaussian_filter(fine + coarse, 2.0) > 0.04)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return sm((gaussian_filter(np.float32(lab == big), 5.0) - 0.05) / 0.3)


def main_ridge():
    """Through the three peaks of the character, so that each peak has its stroke to go to."""
    to = nf(AT["shan"][1], AT["shan"][2])
    pts = [(-20, 530), (80, 478), (170, 410), (238, 300), to((64, 75)), (362, 268), to((163, 41)), (518, 276), to((246, 79)), (622, 340), (672, 424), (752, 496), (880, 540)]
    return ridge([(float(a), float(b)) for a, b in pts], 71, jag=4)


def paint_mountain(sheet, grid):
    """As k2 is painted: more than one layer of wet ink with mist between, the nearest darkest. Each goes down into wet paper and is lost. The tops are not ruled off: here the edge holds against dry sky, there the paper was damp and it goes soft, and no line is drawn round any of it."""
    yy, xx = grid_()
    to = nf(AT["shan"][1], AT["shan"][2])
    R = main_ridge()
    # the far range behind it, pale, showing past it on both sides
    R0 = ridge([(-20, 420), (70, 345), (150, 300), (240, 345), (330, 300), (470, 250), (590, 268), (660, 300), (740, 372), (840, 440), (960, 480)], 72, jag=6)
    soft0 = sm((noise(sheet, 3.5, 90) + 0.1) / 0.5)
    foot0 = 462 + 24 * noise(sheet, 3.6, 77)
    env0 = 0.95 * sm((yy - R0 + 18) / 10) * (1 - sm((yy - foot0) / 80)) * (1 - sm((xx - 610) / 110))
    wet0 = 1.1 * sm((yy - R0 + 18 + 26 * soft0) / 12) * (1 - sm((yy - foot0 - 110) / 80)) * (1 - sm((xx - 700) / 110))
    f0, c0 = cached("xc_m0b", lambda: paint_clouds(sheet, env0, reserve=sm((R0 - yy) / 2.4) * (1 - soft0), seed=81, grid=grid, heavy=0.22, blooms=7, wet_env=wet0).density())
    k = largest_part(f0, c0)
    in_front = 1 - 0.9 * sm((yy - R + 8) / 12)      # not through the near one
    f0, c0 = f0 * k * in_front, c0 * k * in_front
    # the mountain itself: the peaks only, going into mist well above the foot. More of the way round them the paper was damp, so the edge goes soft more often than it holds
    soft = sm((noise(sheet, 3.4, 91) + 0.55) / 0.55)
    foot = 352 + 26 * noise(sheet, 3.6, 76)
    fade_x = 1 - sm((xx - 560) / 140)
    env = 1.02 * sm((yy - R + 22) / 10) * (1 - sm((yy - foot) / 118)) * fade_x
    wet = 1.15 * sm((yy - R + 22 + 30 * soft) / 12) * (1 - sm((yy - foot - 300) / 120)) * (1 - sm((xx - 680) / 150))
    # what is dark on it: the faces away from the light, off each peak
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for gx, gy, run, lean in ((64, 75, 170, 58), (163, 41, 210, 66), (246, 79, 150, 52)):
        px, py = to((gx, gy))
        d.line([(px + 12, py + 20), (px + lean * 0.55, py + run * 0.55), (px + lean, py + run)], fill=255, width=26)
    rib = gaussian_filter(np.asarray(im, np.float32) / 255, 15)
    rib = rib / rib.max()
    fm, cm = cached("xc_m1h", lambda: settled(paint_clouds(sheet, env, reserve=sm((R - yy) / 4.0) * (1 - soft), seed=70, grid=grid, heavy=0.50, near=0.55 * rib * (1 - sm((R - yy) / 2.4)), blooms=18, wet_env=wet), 0.55))
    k = largest_part(fm, cm)
    fm, cm = fm * k, cm * k
    # a lower ridge in front of them, across the middle of the mass, with its own top and its own mist under it
    R3 = ridge([(120, 476), (232, 410), (345, 436), (472, 392), (590, 428), (692, 496), (790, 548)], 75, jag=3)
    soft3 = sm((noise(sheet, 3.4, 94) + 0.35) / 0.55)
    foot3 = R3 + 34 + 18 * noise(sheet, 3.5, 95)
    env3 = 0.98 * sm((yy - R3 + 20) / 10) * (1 - sm((yy - foot3) / 62)) * sm((xx - 90) / 90) * (1 - sm((xx - 690) / 110))
    wet3 = 1.1 * sm((yy - R3 + 20 + 26 * soft3) / 12) * (1 - sm((yy - R3 - 270) / 110)) * sm((xx - 60) / 90) * (1 - sm((xx - 770) / 110))
    f3, c3 = cached("xc_m1g", lambda: settled(paint_clouds(sheet, env3, reserve=sm((R3 - yy) / 4.0) * (1 - soft3), seed=78, grid=grid, heavy=0.56, blooms=9, wet_env=wet3), 0.55))
    k = largest_part(f3, c3)
    f3, c3 = f3 * k, c3 * k
    # the water standing between them: the peaks' ink thins just above that ridge and gives way under it
    gap = 1 - 0.62 * np.exp(-((yy - R3 + 30) / 30) ** 2) * sm((xx - 90) / 60) * (1 - sm((xx - 700) / 90))
    under3 = 1 - 0.75 * sm((yy - R3 + 4) / 14) * sm((xx - 90) / 60) * (1 - sm((xx - 700) / 90))
    fm, cm = fm * gap * under3 + f3, cm * gap * under3 + c3
    # a low spur nearer still, across the left, its top holding and its foot lost in the river's mist
    R2 = ridge([(-20, 532), (110, 506), (240, 530), (360, 548), (500, 584), (640, 610)], 73, jag=6)
    env2 = 1.1 * sm((yy - R2 + 22) / 8) * (1 - sm((yy - R2 - 12) / 50)) * (1 - sm((xx - 400) / 150))
    wet2 = 1.1 * sm((yy - R2 + 22) / 8) * (1 - sm((yy - R2 - 80) / 60)) * (1 - sm((xx - 520) / 130))
    f2, c2 = cached("xc_m2", lambda: paint_clouds(sheet, env2, reserve=sm((R2 - yy) / 2.2), seed=74, grid=grid, heavy=0.50, blooms=8, wet_env=wet2).density())
    k = largest_part(f2, c2)
    f2, c2 = f2 * k, c2 * k
    # the near one's ink gives way to the spur's
    behind = 1 - 0.8 * sm((yy - R2 + 4) / 10) * (1 - sm((xx - 480) / 100))
    fine, coarse = f0 + fm * behind + f2, c0 + cm * behind + c2
    # small chips of bare paper shut in between the layers read as flecks, not as mist: closed over from round them
    low_ = gaussian_filter(fine + coarse, 1.5) < 0.10
    lab, _ = label(low_)
    chips = np.float32(low_ & (np.bincount(lab.ravel())[lab] < 2200))
    chips = np.minimum(gaussian_filter(chips, 3.0) * 2.5, 1.0)
    fine, coarse = fine * (1 - chips) + chips * gaussian_filter(fine, 12.0), coarse * (1 - chips) + chips * gaussian_filter(coarse, 12.0)
    # and the two that are still joined to the open sky, where the front ridge meets the peaks and by the horse's nose
    for cx, cy, r_ in ((340, 405, 40), (650, 450, 72)):
        m_ = sm((r_ - np.hypot(xx - cx, yy - cy)) / 16.0)
        fine = fine + m_ * np.maximum(gaussian_filter(fine, 16.0) - fine, 0)
        coarse = coarse + m_ * np.maximum(gaussian_filter(coarse, 16.0) - coarse, 0)
    # what the rock does: a very few dry pulls down the shadow faces, along the slope; nothing round the edge
    r = np.random.default_rng(79)
    dry = np.zeros((H, W), np.float32)
    for k_, (gx, gy, n, run, lean) in enumerate(((64, 75, 2, 150, 52), (163, 41, 3, 190, 60), (246, 79, 2, 130, 46))):
        px, py = to((gx, gy))
        for i in range(n):
            a = (i + 0.5) / n
            x0_, y0_ = px + 8 + 26 * a, py + 16 + 46 * a
            ln, le = run * r.uniform(0.55, 1.0), lean * r.uniform(0.7, 1.1)
            dry = dry + bristle(sheet, [(x0_, y0_), (x0_ + le * 0.5, y0_ + ln * 0.52), (x0_ + le, y0_ + ln)], (r.uniform(2.5, 4), r.uniform(6, 10), r.uniform(1.5, 3)), 400 + 10 * k_ + i, tone=r.uniform(0.7, 1.05), dry=r.uniform(0.82, 0.94), tremble=1.0)
    lines = gaussian_filter(dry, 0.7) * sm((yy - R - 3) / 3)
    return fine, coarse, lines


def paint_far(sheet, grid):
    """The far bank running off to the right, as k1_wide has it: thin, its foot going down wet paper into the river's mist, broken where the mist stands up."""
    yy, xx = grid_()
    Rf = ridge([(640, 478), (820, 462), (980, 474), (1180, 456), (1400, 482), (1640, 500), (1920, 512)], 11, jag=7)
    gap = 1 - 0.85 * blob(xx, yy, 1260, 480, 90, 70)
    far = 1.25 * sm((yy - Rf + 22) / 8) * (1 - sm((yy - Rf - 14) / 46)) * sm((xx - 620) / 110) * (1 - sm((xx - 1500) / 300)) * gap
    far_wet = 1.1 * sm((yy - Rf + 22) / 8) * (1 - sm((yy - Rf - 100) / 80)) * sm((xx - 640) / 120) * (1 - sm((xx - 1620) / 240))
    return tidy(*cached("xc_far2", lambda: paint_clouds(sheet, far, reserve=sm((Rf - yy) / 2.2), seed=41, grid=grid, heavy=0.34, blooms=7, wet_env=far_wet).density()))


MAN_AT, MAN_S = (1450.0, 905.0), 0.46
PALM, PRINT_S = (1046.0, 920.0), 0.47      # the print the camera goes down to: the same size against the man as in k1_wide, and placed so that the close shot's print and the point of his twig land where they are in k1_close


def paint_ground(sheet, grid, res):
    """The bank of k1_wide: its edge a line at the water, its ground wet ink fading toward the foot of the frame, a dark place under the horse and a dark place under the man, the one sheet of mud under both. res is what was left dry for the man."""
    yy, xx = grid_()
    shore = ridge([(0, 642), (420, 630), (900, 642), (1360, 624), (1920, 642)], 3, jag=6)
    water = sm((shore - yy) / 2.2)
    land = sm((yy - shore + 24) / 8)
    mx, my = MAN_AT
    pool = 1.20 * blob(xx, yy, 940, 800, 320, 40) + 0.75 * blob(xx, yy, 680, 836, 230, 30) + 0.92 * blob(xx, yy, mx - 30, my + 8, 230, 36) + 0.66 * blob(xx, yy, mx - 230, my + 18, 150, 22)
    g_env = land * (0.52 * (1 - sm((yy - shore - 40) / 170)) * (0.45 + 0.55 * sm((noise(sheet, 3.6, 92) + 0.3) / 0.8)) + 0.60 * pool + 0.44 * blob(xx, yy, 250, 760, 330, 92, -0.05))
    g_wet = land * (1 - sm((yy - 962) / 60)) * 1.1
    fg, cg = cached("xc_ground3", lambda: paint_clouds(sheet, g_env, reserve=np.maximum(water, res), seed=42, grid=grid, heavy=0.60, blooms=6, wet_env=g_wet).density())
    fg, cg = tidy(fg, cg)
    low = sm((yy - 815) / 30)      # its near edge dried on damp paper: no hard white lobes down here
    fg, cg = fg * (1 - low) + gaussian_filter(fg, 7.0) * low, cg * (1 - low) + gaussian_filter(cg, 7.0) * low
    clear = 1 - sm((yy - 945) / 50)      # the foot of the frame is the subtitle's
    fg, cg = fg * clear, cg * clear
    # the mud of the close shot, seen from a man's height, with the prints going off from under his twig
    def relief_():
        prints = [(PALM[0], PALM[1], np.pi), (905, 884, np.pi - 0.08)]
        rel, lift, inside = ground_relief(prints, scale=PRINT_S)
        present = sm((yy - shore - 6) / 26) * inside * (1 - 0.6 * sm((yy - 985) / 60))
        return rel * present, lift * present
    rel, lift = cached("xc_relief3", relief_)
    k = 0.8 * (1 - 0.2 * lift)
    fg, cg = fg * k, cg * k
    # the weights on it, soft, the same side as the mud's shadows
    fg = fg + 0.30 * blob(xx, yy, mx + 12, my + 6, 92, 8) + 0.30 * blob(xx, yy, hf((0.20, 0))[0], HORSE_G + 5, 70, 7) + 0.30 * blob(xx, yy, hf((0.92, 0))[0], HORSE_G + 5, 60, 7)
    r = np.random.default_rng(16)
    strokes = []
    for _ in range(9):      # the water: bare paper, and a few long dry lines
        y, x, n = r.uniform(530, 616), r.uniform(560, 1450), r.uniform(120, 420)
        strokes.append(dict(pts=[(x, y), (x + n / 2, y + r.uniform(-1.5, 1.5)), (x + n, y + r.uniform(-2, 2))], w=r.uniform(1.2, 2.0), dark=r.uniform(0.22, 0.42), breaks=-0.4, tremble=0.4))
    return fg, cg, rel, 1.5 * brush(sheet, strokes, seed=60)


def man():
    """Cang Jie as k1_wide has him, on the paper of k1_wide, arm and twig and all, at this size. A fine line stays a fine line."""
    path = os.path.join(OUT, "_wash_xc_man.npz")
    if os.path.exists(path) and os.environ.get("MYTH_FRESH") != "1":
        z = np.load(path)
        return np.float32(z["ink"]), np.float32(z["twig"]), np.float32(z["fill"])
    sh = Sheet(seed=95)
    x0, y0 = 1340.0, 930.0
    i, t, fl = cangjie(sh, x0, y0, split=True, hands=True)
    box = (int(x0 - 480), int(y0 - 420), int(x0 + 200), int(y0 + 60))
    tw, th = int((box[2] - box[0]) * MAN_S), int((box[3] - box[1]) * MAN_S)
    small = lambda a: np.asarray(Image.fromarray(np.float32(a), "F").crop(box).resize((tw, th), Image.LANCZOS), np.float32)
    px, py = int(MAN_AT[0] - (x0 - box[0]) * MAN_S), int(MAN_AT[1] - (y0 - box[1]) * MAN_S)
    out = []
    for a in (i, t, fl):
        b = np.zeros((H, W), np.float32)
        b[py:py + th, px:px + tw] = np.maximum(small(a), 0)
        out.append(b)
    out[2] = np.clip(out[2], 0, 1)
    np.savez_compressed(path, ink=np.float16(out[0]), twig=np.float16(out[1]), fill=np.float16(out[2]))
    return out


def settled(wash, k=0.35):
    """density(), with the sheet's pull on the edge turned down. At the size of a mountain that pull is the ragged edge of a wash; at the size of a bird's back it is the same scallop all the way round, which is what a stroke does not have."""
    h, w = wash.sheet.h, wash.sheet.w
    out = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = wash.sheet.warp
    for i in (0, 1):
        a = wash.fix[i] + wash.d[i] + np.maximum(wash.c[i], 0)
        if wash.shape != (h, w):
            a = np.maximum(grow(a, h, w), 0)
        out.append(map_coordinates(a, [yy + k * dy, xx + k * dx], order=1, mode="nearest"))
    return out[0], out[1]


def held(sheet, a, sdf, seed):
    """What a wash leaves against a reserve is a fringe all the way round, which reads as fur. Here its edge is settled: along some lengths it holds on a clean line, along others it goes soft, as the paper happened to be damp or dry."""
    a = a + (gaussian_filter(a, 1.6) - a) * sm(1 - np.abs(sdf) / 6.0)
    soft = sm((noise(sheet, 3.2, seed) + 0.2) / 0.6)
    return a * sm(-(sdf + 1.5 - 2.5 * soft) / (1.2 + 5.0 * soft))


def paint_bird(sheet, grid=0.67):
    """As Bada Shanren would: the head one wet blot, the beak two dry strokes, the eye a small ring with its dot high up, the body three strokes with the paper left between them, the tail a few dry pulls, the legs a line each. Nothing is filled in."""
    S = AT["niao"][2]
    to = nf(AT["niao"][1], S)
    P = lambda *pts: [to(p) for p in pts]
    Wd = lambda *w: tuple(v * S for v in w)
    yy, xx = grid_()
    ex, ey = to((124, 40))
    eye_d = np.hypot((xx - ex) / 1.18, (yy - ey) / 0.90)      # not a circle: longer than it is high
    eye_r = 3.0 * S
    def make():
        wash = Wash(sheet, s=grid, reserve=sm((eye_r + 0.6 - eye_d) / 1.2))
        # the breast, thin, some water in it, so it dries with a soft inside
        wash.load(wet_stroke(sheet, P((120, 68), (116, 112), (138, 170), (186, 224)), Wd(5, 20, 30, 5), 201, side=-0.7, ends=(0.6, 1, 0.9, 0.3)), 0.14, fine=0.62, coarse=0.08)
        wash.run(22, evap=2.5e-4)
        # the back: one long stroke off the nape, the belly of the brush along the outside, running out toward the tail
        wash.load(wet_stroke(sheet, P((170, 56), (202, 104), (222, 164), (234, 224)), Wd(10, 24, 30, 5), 202, side=0.85, ends=(1, 0.9, 0.7, 0.3)), 0.10, fine=1.55, coarse=0.24)
        wash.run(18, evap=2.5e-4)
        # the wing: dark, laid while the back is damp, so it runs into it above and holds below
        wash.load(wet_stroke(sheet, P((164, 100), (182, 150), (208, 200), (228, 244)), Wd(6, 24, 15, 2), 203, side=0.6, ends=(0.8, 1, 1, 0.4)), 0.09, fine=2.2, coarse=0.34)
        wash.run(18, evap=2.5e-4)
        # the head: one blot, turned once
        wash.load(wet_stroke(sheet, P((108, 48), (128, 26), (158, 22), (180, 44)), Wd(11, 19, 19, 12), 204, side=0.45, ends=(0.8, 1, 1)), 0.10, fine=2.2, coarse=0.32)
        wash.load(wet_stroke(sheet, P((150, 50), (172, 58), (182, 76)), Wd(13, 13, 7), 205, ends=(1, 0.7)), 0.10, fine=1.5, coarse=0.24)
        wash.run(16, evap=2.5e-4)
        wash.run(150, evap=7e-4)
        return settled(wash)
    fine, coarse = cached("xc_bird5", make)
    fine, coarse = gaussian_filter(fine, 0.5), gaussian_filter(coarse, 0.5)
    stroke = lambda pts, w, sd, tone, dr=0.55, tr=0.5: bristle(sheet, pts, w, sd, tone=tone, dry=dr, tremble=tr)
    dry = np.zeros((H, W), np.float32)
    for k, (pts, w, tone, dr) in enumerate((
        # the beak: upper and lower, a hair of paper between them where they part
        (P((112, 40), (92, 44), (66, 54)), Wd(4.2, 3.2, 0.7), 2.3, 0.5),
        (P((110, 57), (92, 58), (76, 59)), Wd(2.8, 2.2, 0.6), 1.9, 0.62),
        # the tail
        (P((212, 226), (222, 268), (232, 312)), Wd(6, 7, 1.6), 2.3, 0.62),
        (P((198, 230), (200, 272), (194, 306)), Wd(5, 6, 1.4), 2.1, 0.68),
        (P((226, 222), (246, 262), (264, 296)), Wd(4, 5, 1.2), 1.9, 0.74),
        # the long feathers at the wing's end, broken
        (P((190, 180), (206, 212), (224, 236)), Wd(1.6, 3.4, 1.0), 1.5, 0.8),
        # where the brush was lifted off the back
        (P((222, 150), (234, 188), (236, 218)), Wd(2, 4, 1.2), 1.3, 0.86),
    )):
        dry = dry + stroke(pts, w, 220 + k, tone, dr)
    dry = gaussian_filter(dry, 0.6)
    # the eye: a thin ring, and the dot up in it, looking up the way his birds do
    ring = 1.9 * np.exp(-((eye_d - eye_r) / (0.42 * S)) ** 2) * (0.5 + 0.5 * sm((noise(sheet, 3.0, 247) + 0.5) / 0.8))
    px, py = ex + 0.5 * S, ey - 1.9 * S
    pupil = 2.6 * sm((1.35 * S - np.hypot(xx - px, yy - py)) / 0.8)      # hung from the top of the ring: he is not staring
    lines = dry + ring + pupil
    # the legs: one thin line each, and a toe or two round the branch
    for k, pts in enumerate((P((150, 200), (146, 222), (136, 238)), P((172, 212), (168, 226), (156, 241)), P((136, 238), (126, 242)), P((156, 241), (146, 246)))):
        lines = lines + bristle(sheet, pts, (1.5, 1.2, 0.9), 240 + k, tone=2.0, dry=0.2, tremble=0.3)
    return fine, coarse, lines      # nothing cut out behind it: ink over dry ink only goes darker


def paint_horse(sheet, grid=0.67):
    """As Xu Beihong's: no fill. A thin wash where the barrel and the neck turn from the light, the paper left bare along the back, on the shoulder and over the croup; then the structure in wet strokes dropped on damp paper, thick to thin, darkest under the neck, along the belly and behind the thigh; the legs one stroke each with the knee and hock left pale; a small hoof. The mane and tail in the dry brush."""
    body = shape_of([hf(p) for p in HORSE_NEAR])
    far_sdf = np.minimum(shape_of([hf(p) for p in HORSE_FAR_FORE]), shape_of([hf(p) for p in HORSE_FAR_HIND]))
    keep = sm(-(body - 1) / 2.5)
    far_shape = sm(-(far_sdf - 1) / 2.5)
    far_keep = far_shape * (1 - keep)
    yy, xx = grid_()
    Bl = lambda a, b, rx, ry, ang=0.0: blob(xx, yy, *hf((a, b)), rx, ry, ang)
    P = lambda *pts: [hf(p) for p in pts]
    def make():
        wash = Wash(sheet, s=grid, reserve=1 - sm(-(body - 3) / 3))
        light = np.clip(0.85 * Bl(0.20, 0.86, 40, 34, 0.8) + 0.85 * Bl(0.50, 0.94, 150, 11) + 0.85 * Bl(0.86, 0.92, 50, 20, 0.2) + 0.7 * Bl(-0.22, 1.20, 20, 15, 0.9), 0, 1)
        # damp: the barrel and neck, not the legs
        damp = keep * (1 - sm(((HORSE_G - yy) / HORSE_U - 0.62) / -0.08))
        wash.load(damp * lumps(sheet, damp, 133, level=0.30, soft=0.25, amount=0.34), 0.17)
        wash.run(10, evap=1e-4)
        thin = keep * (1 - light) * sm((yy - hf((0, 1.5))[1]) / 1)
        wash.load(lumps(sheet, thin, 131, level=0.30, soft=0.30, amount=0.16), 0.05, fine=0.46, coarse=0.06)
        wash.run(22, evap=1.5e-4)
        # (path, half widths in px, side, water, fine, coarse): a stroke at a time, thick to thin
        for k, (pts, wd, sd, wat, fi, co) in enumerate((
            (P((0.03, 1.22), (0.12, 1.08), (0.22, 0.97)), (9, 18, 22), -0.5, 0.11, 1.0, 0.16),                       # under the mane
            (P((-0.10, 1.09), (-0.05, 0.93), (0.03, 0.76), (0.09, 0.60)), (6, 15, 20, 13), 0.5, 0.12, 1.75, 0.28),    # under the neck, down to the breast
            (P((0.10, 0.70), (0.17, 0.58), (0.22, 0.47)), (16, 22, 15), 0.2, 0.12, 1.45, 0.24),                       # the arm
            (P((0.30, 0.97), (0.34, 0.80), (0.29, 0.62), (0.27, 0.50)), (4, 12, 13, 8), 0.3, 0.10, 0.95, 0.15),       # the back of the shoulder
            (P((0.30, 0.74), (0.46, 0.70), (0.62, 0.74), (0.72, 0.80)), (22, 30, 30, 18), 0.8, 0.13, 0.72, 0.10),     # the barrel: one broad thin stroke, full on its belly side, running out toward the back
            (P((0.28, 0.52), (0.48, 0.52), (0.72, 0.64)), (14, 18, 8), 0.75, 0.11, 1.95, 0.32),                       # the belly
            (P((0.40, 0.56), (0.385, 0.65), (0.40, 0.74)), (7, 8, 3), 0.0, 0.09, 0.70, 0.10),                         # where the ribs turn: three short strokes up out of the belly, wet into it
            (P((0.49, 0.54), (0.485, 0.63), (0.505, 0.73)), (7, 8, 3), 0.0, 0.09, 0.62, 0.09),
            (P((0.58, 0.57), (0.585, 0.65), (0.61, 0.74)), (6, 7, 3), 0.0, 0.09, 0.55, 0.08),
            (P((0.63, 0.90), (0.67, 0.76), (0.71, 0.65)), (5, 15, 10), -0.3, 0.10, 0.80, 0.13),                       # the flank
            (P((0.92, 0.94), (1.01, 0.84), (1.02, 0.70), (0.98, 0.56)), (8, 20, 22, 13), -0.7, 0.12, 1.80, 0.30),     # the buttock
            (P((0.78, 0.76), (0.85, 0.62), (0.90, 0.47)), (14, 25, 15), -0.2, 0.12, 1.30, 0.22),                      # the thigh
            (P((-0.13, 1.30), (-0.25, 1.18), (-0.36, 1.05)), (8, 10, 8), 0.0, 0.08, 0.85, 0.13),                      # the face
            (P((-0.11, 1.13), (-0.20, 1.08), (-0.31, 1.00)), (12, 9, 5), 0.5, 0.08, 1.55, 0.25),                      # the jaw
            (P((-0.33, 1.07), (-0.37, 1.01)), (8, 7), 0.0, 0.07, 1.90, 0.32),                                         # the muzzle
            (P((0.21, 0.46), (0.20, 0.31), (0.195, 0.13), (0.18, 0.05)), (20, 13, 10, 10), 0.3, 0.09, 1.85, 0.32),     # the near fore, one stroke
            (P((0.89, 0.50), (0.95, 0.37), (0.94, 0.13), (0.92, 0.05)), (18, 12, 10, 10), 0.3, 0.09, 1.85, 0.32),      # the near hind
        )):
            wash.load(wet_stroke(sheet, pts, wd, 260 + k, side=sd, tremble=0.7), wat, fine=fi, coarse=co)
            wash.run(9, evap=1.5e-4)
        wash.run(30, evap=2e-4)
        for k, (a, b, rx, ry) in enumerate(((0.195, 0.30, 7, 9), (0.95, 0.37, 7, 9), (0.33, 0.82, 16, 22), (0.80, 0.82, 22, 12))):      # knee, hock: clean water, the bone shows pale
            wash.dab(*hf((a, b)), rx, ry, 0.0, water=0.24, seed=290 + k, rough=0.5)
            wash.run(9, evap=1.5e-4)
        wash.run(210, evap=6.5e-4)
        return wash.density()      # the sheet's pull stays: it is what keeps the inside from going smooth like an airbrush, and held() below settles the edge
    fine, coarse = cached("xc_horse5", make)
    def far_():
        wash = Wash(sheet, s=grid, reserve=1 - far_shape)
        for k, (pts, wd) in enumerate((
            (P((0.13, 0.58), (0.11, 0.30), (0.11, 0.13), (0.095, 0.05)), (22, 13, 10, 10)),
            (P((0.78, 0.60), (0.81, 0.40), (0.80, 0.13), (0.78, 0.05)), (20, 13, 10, 10)),
        )):
            wash.load(wet_stroke(sheet, pts, wd, 280 + k, side=0.45, ends=(1.0, 0.8, 0.65, 0.8), tremble=0.7), 0.13, fine=0.85, coarse=0.14)
            wash.run(12, evap=1.5e-4)
        wash.run(30, evap=2e-4)
        for k, (a, b) in enumerate(((0.11, 0.31), (0.81, 0.37))):      # the far knee and hock: clean water here too
            wash.dab(*hf((a, b)), 6, 9, 0.0, water=0.2, seed=296 + k, rough=0.5)
            wash.run(9, evap=1.5e-4)
        wash.run(120, evap=6.5e-4)
        return wash.density()
    f2, c2 = cached("xc_horse_far3", far_)
    clip = held(sheet, np.ones_like(fine), body, 294)
    clip_far = held(sheet, np.ones_like(fine), far_sdf, 295) * (1 - clip)
    fine, coarse = held(sheet, fine, body, 294), held(sheet, coarse, body, 294)
    fine, coarse = fine + held(sheet, f2, far_sdf, 295) * (1 - clip), coarse + held(sheet, c2, far_sdf, 295) * (1 - clip)
    # the dry brush: mane and tail are the blackest things on it
    stroke = lambda pts, w, sd, tone, dr=0.5, tr=0.9: bristle(sheet, pts, w, sd, tone=tone, dry=dr, tremble=tr)
    mane = np.zeros((H, W), np.float32)
    for k, (pts, w, tone, dr) in enumerate((
        (P((-0.125, 1.335), (-0.03, 1.285), (0.08, 1.165), (0.21, 1.03)), (5, 11, 12, 5), 2.6, 0.42),
        (P((-0.08, 1.31), (0.0, 1.21), (0.04, 1.10), (0.06, 0.99)), (4, 9, 9, 2), 2.5, 0.62),
        (P((-0.03, 1.27), (0.05, 1.16), (0.10, 1.05), (0.12, 0.93)), (4, 9, 8, 2), 2.5, 0.66),
        (P((0.04, 1.20), (0.12, 1.09), (0.17, 0.99), (0.18, 0.89)), (4, 8, 7, 2), 2.4, 0.70),
        (P((0.11, 1.11), (0.20, 1.02), (0.25, 0.93)), (3, 7, 2), 2.2, 0.72),
        (P((-0.14, 1.34), (-0.195, 1.26), (-0.205, 1.20)), (4, 6, 1.5), 2.4, 0.5),      # the forelock
    )):
        mane = mane + stroke(pts, w, 180 + k, tone, dr)
    tail = np.zeros((H, W), np.float32)
    for k, (pts, w, tone, dr) in enumerate((
        (P((1.00, 0.87), (1.09, 0.80), (1.13, 0.62), (1.14, 0.40), (1.12, 0.18)), (5, 10, 12, 9, 2), 2.6, 0.55),
        (P((1.00, 0.86), (1.06, 0.76), (1.08, 0.56), (1.07, 0.32), (1.05, 0.14)), (4, 8, 9, 6, 1.5), 2.4, 0.66),
        (P((1.01, 0.85), (1.12, 0.74), (1.18, 0.52), (1.19, 0.32), (1.21, 0.16)), (4, 7, 7, 5, 1.5), 2.3, 0.72),
        (P((1.00, 0.85), (1.05, 0.68), (1.04, 0.48), (1.02, 0.30)), (3, 6, 5, 1.5), 2.0, 0.76),
    )):
        tail = tail + stroke(pts, w, 190 + k, tone, dr, 1.2)
    # what is said in line, broken and thick to thin: the top of the back only here and there, the cheek, the shoulder, the stifle, the tendons
    drawn = np.zeros((H, W), np.float32)
    for k, (pts, w, tone, dr) in enumerate((
        (P((0.30, 0.985), (0.42, 0.952), (0.55, 0.945), (0.70, 0.968)), (1.2, 2.6, 2.2, 1.0), 1.25, 0.82),   # the back
        (P((0.80, 0.985), (0.92, 0.95), (1.00, 0.875)), (1.4, 3.0, 1.6), 1.5, 0.7),                           # the croup
        (P((-0.165, 1.335), (-0.275, 1.215), (-0.335, 1.13), (-0.385, 1.05)), (1.2, 2.0, 2.0, 1.4), 1.25, 0.76),      # the nose
        (P((-0.105, 1.115), (-0.15, 1.15), (-0.20, 1.19), (-0.17, 1.26)), (1, 2.2, 1.8, 1), 1.4, 0.55),       # the cheek
        (P((0.25, 0.98), (0.20, 0.82), (0.12, 0.66)), (1.2, 3.0, 1.4), 1.25, 0.7),                            # the shoulder blade
        (P((0.72, 0.62), (0.69, 0.76), (0.72, 0.90)), (1, 2.6, 1), 1.1, 0.8),                                 # the flank
        (P((0.84, 0.60), (0.88, 0.47), (0.90, 0.40)), (1, 2.2, 1), 1.3, 0.7),                                 # the gaskin
        (P((0.225, 0.285), (0.222, 0.20), (0.225, 0.12)), (1, 1.8, 1), 1.7, 0.45),                            # the tendons
        (P((0.965, 0.35), (0.97, 0.22), (0.97, 0.13)), (1, 1.8, 1), 1.7, 0.45),
    )):
        drawn = drawn + stroke(pts, w, 200 + k, tone, dr, 0.6)
    # the hoof: small. One press of the loaded brush, black where it meets the ground, gone by the pastern, no outline
    hoof = np.zeros((H, W), np.float32)
    hh = 0.046
    for k, (l, r, tone) in enumerate(((0.148, 0.205, 1.0), (0.886, 0.943, 1.0), (0.068, 0.121, 0.62), (0.748, 0.805, 0.62))):
        sdf = shape_of([hf(p) for p in ((l, 0.0), (r, 0.0), (r - 0.006, hh), (l + 0.012, hh))], smooth=0.5)
        up = np.clip((HORSE_G - yy) / (hh * HORSE_U), 0, 1)
        hoof = np.maximum(hoof, tone * (2.3 - 1.3 * up) * sm(-(sdf - 0.3 + 0.5 * sheet.tooth) / 1.3))
    ear = bristle(sheet, P((-0.15, 1.335), (-0.128, 1.44)), (5, 1.5), 208, tone=1.9, dry=0.3, tremble=0.2) + bristle(sheet, P((-0.108, 1.335), (-0.125, 1.44)), (4, 1.5), 209, tone=1.6, dry=0.4, tremble=0.2)
    lines = gaussian_filter(mane + tail + ear, 0.7) + gaussian_filter(drawn, 0.5) + gaussian_filter(hoof, 0.5)
    # the eye and the nostril
    ex, ey = hf((-0.205, 1.225))
    nx, ny = hf((-0.345, 1.06))
    lines = lines + 2.4 * sm((3.4 - np.hypot(xx - ex, (yy - ey) * 1.25)) / 1.2) + 1.6 * sm((2.4 - np.hypot(xx - nx, (yy - ny) * 1.4)) / 1.2)
    return fine, coarse, lines, clip + clip_far


def paint_world(grid=0.67):
    """All that is painted, cached: the mountain, the far bank, the ground, the man, the branch, the bird, the horse."""
    sheet = Sheet(seed=PAPER)
    m_ink, m_twig, m_fill = man()
    ik, ke = cached("xc_branch", lambda: painted_twig(sheet, BRANCH))
    return dict(sheet=sheet, mtn=paint_mountain(sheet, grid), far=paint_far(sheet, grid), ground=paint_ground(sheet, grid, m_fill), man=(m_ink, m_twig, m_fill), branch=(ik, ke), bird=paint_bird(sheet, grid), horse=paint_horse(sheet, grid))


# ---------------------------------------------------------------- the ink going

def along_strokes(inside, start):
    """How far along the strokes of a character each pixel is from where it starts, in pixels; beyond the outline, that of the nearest pixel of it."""
    ys, xs = np.nonzero(inside)
    y0, y1, x0, x1 = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
    ins = inside[y0:y1, x0:x1]
    cur = ins & start[y0:y1, x0:x1]
    assert cur.any()
    far = np.full(ins.shape, -1.0, np.float32)
    far[cur] = 0
    step = 0
    while True:
        step += 1
        nxt = binary_dilation(cur, np.ones((3, 3), bool), mask=ins)
        new = nxt & ~cur
        if not new.any():
            break
        far[new] = step
        cur = nxt
    apart = far < 0      # a stroke not joined to the rest: as far as it is, straight
    if apart.any():
        far[apart] = distance_transform_edt(~start)[y0:y1, x0:x1][apart]
    out = np.zeros(inside.shape, np.float32)
    out[y0:y1, x0:x1] = far
    idx = distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return out[idx[0], idx[1]]


class Target:
    """What the ink of one thing draws in to: the distance from its character, and which way that is."""

    def __init__(self, mask):
        inside = mask > 0.5
        self.d = np.float32(distance_transform_edt(~inside))
        holes = binary_fill_holes(inside) & ~inside
        lab, _ = label(holes)
        small = holes & (np.bincount(lab.ravel())[lab] < 1000)      # an eye is not an edge the ink grows from
        self.depth = np.float32(distance_transform_edt(inside | small))
        self.deep = np.maximum(maximum_filter(self.depth, 45), 1.0)      # how thick the character is hereabouts
        smooth = gaussian_filter(self.d, 7.0)
        gy, gx = np.gradient(smooth)
        norm = np.hypot(gx, gy) + 1e-3
        self.gx, self.gy = gx / norm, gy / norm


_soft, _blur = {}, {}


def _once(table, a, sigma):
    key = id(a)
    if key not in table:
        table[key] = gaussian_filter(a, sigma)
    return table[key]


def withdraw(sheet, arrays, tg, tau, reach, seed, pull=26.0, rest=0.03, stain=0.012, tide=0.55, soften=0.12, end=4.0, spread=0.0):
    """How far gone this thing is at tau, 0 to 1. The paper drinks from the far side of it toward the character, on an edge that wanders along the sheet, and ink piles a little at that edge as it goes back: the tide line. What is left behind the edge sinks too, and the rest of it moves in. spread: how far it goes soft as it sinks, wet ink going into wet paper. stain: what the paper keeps, soft, with no edge in it."""
    if tau <= 0:
        return list(arrays)
    d = tg.d
    front = end + (reach - end) * (1 - ease(tau / 0.86))
    wander = (0.30 * reach * (1 - ease(tau / 0.9))) * noise(sheet, 3.0, seed) + (3 + 14 * (1 - tau)) * noise(sheet, 2.6, seed + 1) + 3.0 * (1 - tau) * sheet.warp[0]
    x = d + wander - front
    soft = 6 + soften * front
    keep = 1 - sm(x / soft)
    crest = np.exp(-(x / 8.0) ** 2) * (1 - ease((tau - 0.45) / 0.45))
    near = np.exp(-d / 30.0)
    thin = 1 - (1 - rest) * ease((tau - 0.04 - 0.16 * near) / 0.82)      # nearest the strokes it stays longest
    move = pull * ease(tau / 0.8) * sm(d / 22.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = [yy + tg.gy * move, xx + tg.gx * move]
    how_soft = spread * ease((tau - 0.25) / 0.5)
    out = []
    for a in arrays:
        src = a if how_soft <= 0 else a + how_soft * (_once(_blur, a, 5.0) - a)
        b = map_coordinates(src, coords, order=1, mode="nearest") * keep * (1 + tide * crest) * thin
        if stain > 0:
            b = b + _once(_soft, a, 12.0) * stain * ease(tau) * sm(1 - d / 150.0)
        out.append(np.float32(b))
    return out


def gather(C, tg, tau, seed, sheet):
    """The strokes going dark as the ink comes to them: a thin line down the middle first, widening; not everywhere at once. By the end all of it, the seep past the outline too."""
    if tau <= 0:
        return np.zeros_like(C)
    a = ease((tau - 0.14 + 0.09 * np.clip(noise(sheet, 3.0, seed), -1.4, 1.4)) / 0.68)
    grow_ = np.maximum(sm((tg.depth - (1 - a) * tg.deep) / 2.5), ease((a - 0.86) / 0.14))
    return C * grow_ * (0.42 + 0.58 * a) * (a > 0)


# When each thing goes
T_SHAN = (6.5, 8.5)
T_BRANCH = (8.5, 9.1)
T_NIAO = (8.6, 10.5)
T_MA = (10.5, 13.0)
T_SMALL = dict(ri=(13.0, 13.55), shui=(13.12, 13.67), yue=(13.24, 13.79), mu=(13.36, 13.91), yu=(13.45, 14.0))
ZOOM_FIX, ZOOM_RATE = (900.0, 520.0), 0.012     # the slow push: this fraction per second, about one point that stays where it is


def tau_of(t, span):
    return float(np.clip((t - span[0]) / (span[1] - span[0]), 0, 1))


class World:
    def __init__(self):
        self.P = paint_world()
        self.sheet = self.P["sheet"]
        self.chars()
        self.tg = {name: Target(self.masks[name]) for name in ("shan", "niao")}
        self.tg["ma"] = Target(self.masks["ma"])
        self.tg["small"] = {name: Target(self.masks[name]) for name in SMALL}
        inside = self.masks["ma"] > 0.5
        self.ma_holes = np.float32(gaussian_filter(np.float32(binary_fill_holes(inside) & ~inside), 1.0))
        self.ma_deep = np.float32(np.maximum(maximum_filter(self.tg["ma"].depth, 21), 1.0))
        # the character starts at the one place on its strokes nearest the middle of the horse's body, and runs out along them from there
        on_horse = gaussian_filter(self.P["horse"][0] + self.P["horse"][1] + self.P["horse"][2], 2.0) > 0.25
        ys_, xs_ = np.nonzero(on_horse)
        yo, xo = np.nonzero(on_horse & inside)
        at_ = np.argmin(np.hypot(xo - xs_.mean(), yo - ys_.mean()))
        yq, xq = np.mgrid[0:H, 0:W]
        self.ma_along = along_strokes(inside, np.hypot(xq - xo[at_], yq - yo[at_]) < 18)
        # what stays when the ground has gone: the dark place he is kneeling in, with the prints by his twig
        yy, xx = grid_()
        mx, my = MAN_AT
        near_him = blob(xx, yy, mx - 70, my + 12, 330, 44)
        self.stays = sm((near_him + 0.16 * noise(self.sheet, 3.4, 93) * sm(near_him / 0.15) - 0.22) / 0.34)      # the noise only where he is: its peaks far off from him were leaving marks on the page
        self.stays_marks = np.maximum(self.stays, sm((blob(xx, yy, 960, 902, 190, 60) - 0.2) / 0.3))      # the prints stay too
        # the pool and the line of the shore go with the mountain, not with the horse: nothing with an edge is left behind by then
        fg0, cg0, rl0, wl0 = self.P["ground"]
        wl0 = wl0 * (1 - sm((xx - 1640) / 30))      # the water lines stop short of the note column
        shore = ridge([(0, 642), (420, 630), (900, 642), (1360, 624), (1920, 642)], 3, jag=6)
        left = sm((760 - xx + 70 * noise(self.sheet, 3.4, 96)) / 240)
        top = 1 - sm((yy - shore - 30) / 110)
        pond = np.maximum(left, top) * (1 - self.stays)
        line_left = sm((800 - xx) / 30)      # all of the water line on that side of the horse, not part of it
        self.pond = (fg0 * pond, cg0 * pond, rl0 * pond, wl0 * line_left)
        self.field = (fg0 * (1 - pond), cg0 * (1 - pond), rl0 * (1 - pond), wl0 * (1 - line_left))

    def chars(self):
        """One loaded stroke for every character, on this sheet: the outline exact, the paper's tooth a pixel in or out along it; along some lengths, where the paper was damp, a thin stain out into the fibres; the ink not one flat black inside, deep at the edge where it settled, a little thinner here and there in the middle. The big characters near black, the small ones paler."""
        sheet = self.sheet
        self.masks = {name: glyph_mask(name) for name in AT}
        self.ink = {}
        tooth = np.clip(sheet.tooth, -1.8, 1.8)
        grit = 0.32 * tooth + 0.22 * noise(sheet, 2.2, 98)
        damp = sm((noise(sheet, 3.4, 100) - 0.35) / 0.5)      # most of the way round the paper was dry
        for k, name in enumerate(AT):
            inside = self.masks[name] > 0.5
            in_d = distance_transform_edt(inside)
            sdf = gaussian_filter(np.float32(distance_transform_edt(~inside) - in_d), 0.6)
            little = name in SMALL
            amp = 0.6 if little else 1.0
            core = sm((-sdf + amp * grit + 0.5) / 0.9)
            thin = sm((noise(sheet, 3.2, 101 + k) + 0.4) / 1.2) * sm(in_d / (4.0 * amp))
            body = (1.55 - 0.3 * thin if little else 2.75 - 0.5 * thin) + 0.05 * tooth
            seep = damp * np.exp(-np.maximum(sdf, 0) / (amp * (0.9 + 2.0 * damp))) * (1 - core) * (0.7 + 0.3 * sm((tooth + 0.8) / 1.6))
            self.ink[name] = np.float32(body * core + (0.3 if little else 0.42) * seep)

    def at(self, t, him=True):
        P, sheet = self.P, self.sheet
        # the mountain
        ts = tau_of(t, T_SHAN)
        fm, cm, lm = withdraw(sheet, P["mtn"], self.tg["shan"], ts, 620.0, 301, pull=16.0, rest=0.02, stain=0.008, tide=0.25, soften=0.28, spread=0.7)
        ff, cf = withdraw(sheet, P["far"], self.tg["shan"], tau_of(t, (T_SHAN[0], T_SHAN[1] + 0.3)), 1300.0, 302, pull=0.0, rest=0.0, stain=0.0, tide=0.0, soften=0.4)
        # the branch goes first, then the bird
        tb, tn = tau_of(t, T_BRANCH), tau_of(t, T_NIAO)
        ik, ke = P["branch"]
        ib, kb = withdraw(sheet, (ik, ke), self.tg["niao"], tb, 720.0, 303, pull=0.0, rest=0.0, stain=0.0)
        bf, bc, bl = P["bird"]
        bf, bc, bl = withdraw(sheet, (bf, bc, bl), self.tg["niao"], tn, 170.0, 304, pull=12.0, rest=0.015, stain=0.006, tide=0.25, soften=0.28, spread=0.6)
        # the horse
        tm = tau_of(t, T_MA)
        hf_, hc, hl, hcov = P["horse"]
        hf_, hc, hl = withdraw(sheet, (hf_, hc, hl), self.tg["ma"], tm, 200.0, 305, pull=12.0, rest=0.012, stain=0.0, tide=0.14, soften=0.5, spread=0.8)
        hl = hl * (1 - ease((tm - 0.25) / 0.35))      # the dry brush has no water to go anywhere with: it just goes
        late = 1 - ease((tm - 0.5) / 0.3) * sm(self.tg["ma"].d / 70.0)
        clear = 1 - self.ma_holes * ease((tm - 0.72) / 0.2)
        hf_, hc, hl = hf_ * late * clear, hc * late * clear, hl * clear
        over_h = hcov * (1 - ease((tm - 0.05) / 0.55))
        # the ground: all of it but the place he is kneeling
        fg0, cg0, rl0, wl0 = P["ground"]
        tg_ = tau_of(t, (T_MA[0] + 0.1, T_MA[1]))
        fp, cp, rp, wp = withdraw(sheet, self.pond, self.tg["shan"], ts, 1500.0, 307, pull=0.0, rest=0.0, stain=0.0, tide=0.0, soften=0.45)
        fg, cg, rl, wl = withdraw(sheet, self.field, self.tg["ma"], tg_, 980.0, 306, pull=0.0, rest=0.0, stain=0.0, tide=0.0, soften=0.4)
        fg, cg, rl, wl = fg + fp, cg + cp, rl + rp, wl + wp
        fg, cg, rl = np.maximum(fg, fg0 * self.stays), np.maximum(cg, cg0 * self.stays), np.maximum(rl, rl0 * self.stays_marks)
        # what is in front of what, while it is still a picture
        over_br = ke * (1 - ease(tb / 0.5))
        man_ink, man_twig, man_fill = P["man"]
        if not him:      # the page without him: the ground closed over from round where it was painted round him
            have = gaussian_filter(1 - man_fill, 12.0) + 1e-3
            fg = fg * (1 - man_fill) + man_fill * gaussian_filter(fg * (1 - man_fill), 12.0) / have
            cg = cg * (1 - man_fill) + man_fill * gaussian_filter(cg * (1 - man_fill), 12.0) / have
            man_ink, man_twig, man_fill = 0.0, 0.0, 0.0
        behind = (1 - over_h) * (1 - man_fill)
        fine = (fm + ff) * behind * (1 - over_br) + fg * behind + hf_ + bf
        coarse = (cm + cf) * behind * (1 - over_br) + cg * behind + hc + bc
        # the characters
        c_shan = gather(self.ink["shan"], self.tg["shan"], ts, 311, sheet)
        c_niao = gather(self.ink["niao"], self.tg["niao"], tn, 312, sheet)
        # the character comes up where it lies on the horse first, and runs out along its own strokes from there: nothing floats on bare paper
        a_ma = float(ease((tm - 0.16) / 0.60))
        width = np.maximum(sm((self.tg["ma"].depth - (1 - (0.45 + 0.55 * a_ma)) * self.ma_deep + 1.0) / 2.0), ease((a_ma - 0.86) / 0.14))      # a dark line down each stroke, widening; never so thin it breaks into pieces
        c_ma = self.ink["ma"] * width * (0.5 + 0.5 * a_ma) * sm((self.ma_along.max() * 1.1 * ease((tm - 0.14) / 0.6) - self.ma_along) / 30.0)
        small = np.zeros((H, W), np.float32)
        for name in SMALL:
            u = tau_of(t, T_SMALL[name])
            if u > 0:
                small = small + gather(self.ink[name], self.tg["small"][name], 0.14 + 0.86 * u, 320 + len(name), sheet)
        line = lm * behind + bl + hl + wl * behind + rl * behind + np.minimum(ib * (1 - man_fill), 2.7) + np.minimum(man_ink + man_twig, 2.7)
        chars = np.minimum(c_shan + c_niao + c_ma + small, 2.75)
        return develop(sheet, fine, coarse, line=np.minimum(line, 2.75) + chars)

    def shot(self, t, name=None):
        f = self.at(t)
        z = np.exp(ZOOM_RATE * (t - 5.0))
        cx, cy = ZOOM_FIX
        box = (cx - cx / z, cy - cy / z, cx + (W - cx) / z, cy + (H - cy) / z)
        im = Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).resize((W, H), Image.BICUBIC, box=box)
        if name:
            path = os.path.join(MYTH, name + ".png")
            im.save(path)
            print(path, flush=True)
        return im


# ---------------------------------------------------------------- 14.0 to 15.5: down onto the mud

PUSH0, PUSH1 = 14.0, 464 / 30      # the last frame of the push; the ratio stays under 2.2 a second
Z_END = 2.98                        # the close shot's print against the print on the page
C_PALM = (560.0, 490.0)             # where that print's palm is in the close shot
STRETCH = 1.52                      # the page's ground is seen at a slant and the close shot from above: this much more height to the same width, by the end
T_WASH = (14.05, 15.20)             # the wash on the mud, from round what is already on the ground, thin at first
T_MAN_OUT = (14.22, 14.82)          # he goes out past the edge
PAD_X, PAD_Y = 1500, 1300           # the close shot's wash goes on past its frame, so that while it is still small it has no edge to show


def cam_of(t):
    """How far in, how much the ground has opened, where on the frame the palm of the print is. Even in ratio, taken up and given up over a moment."""
    z0 = np.exp(ZOOM_RATE * (PUSH0 - 5.0))
    L = np.log(Z_END / z0)
    TA, TB, T = 0.15, 0.22, PUSH1 - PUSH0
    r = (L - ZOOM_RATE * TA / 2) / (T - TA / 2 - TB / 2)
    u = float(np.clip(t - PUSH0, 0, T))
    if u < TA:
        area = ZOOM_RATE * u + (r - ZOOM_RATE) * u * u / (2 * TA)
    elif u < T - TB:
        area = (ZOOM_RATE + r) / 2 * TA + r * (u - TA)
    else:
        area = L - r * (T - u) ** 2 / (2 * TB)
    p = area / L
    x0, y0 = ZOOM_FIX[0] + (PALM[0] - ZOOM_FIX[0]) * z0, ZOOM_FIX[1] + (PALM[1] - ZOOM_FIX[1]) * z0
    return z0 * np.exp(area), 1 + (STRETCH - 1) * float(ease((p - 0.25) / 0.6)), x0 + (C_PALM[0] - x0) * p ** 1.5, y0 + (C_PALM[1] - y0) * p ** 1.5


def _aff(src, dst, sx, sy):
    return (1 / sx, 0, src[0] - dst[0] / sx, 0, 1 / sy, src[1] - dst[1] / sy)


def _warp_f(a, tx):
    return np.asarray(Image.fromarray(np.float32(a), "F").transform((W, H), Image.AFFINE, tx, Image.BICUBIC), np.float32)


class Push:
    def __init__(self, world, writer):
        sheet = world.sheet
        self.paper = np.float32(sheet.rgb)
        self.base = Image.fromarray(np.uint8(np.clip(world.at(PUSH0, him=False) + 0.5, 0, 255)))
        self.paper_im = Image.fromarray(np.uint8(np.clip(self.paper + 0.5, 0, 255)))
        ink, twig_on_page, fill = world.P["man"]
        self.man_d, self.man_f = np.minimum(ink + twig_on_page, 2.7), fill
        # the close shot is the writer's, the mud still bare
        self.wr = writer
        sc = writer.sheet
        yy, xx = grid_()
        self.xx, self.yy = xx, yy
        rel0, lift0 = writer.blank
        # past its frame the wash alone, reflected: the print is only in the frame itself
        rf = lambda a: np.pad(a, ((PAD_Y, PAD_Y), (PAD_X, PAD_X)) + ((0, 0),) * (a.ndim - 2), mode="reflect")
        D_far = rf(0.55 * writer.D_wash * (1 - 0.6 * lift0) + gaussian_filter(np.minimum(rel0 * writer.present, 0.2), 14.0))
        self.pad = np.uint8(np.clip(develop(SimpleNamespace(valley=rf(sc.valley), rgb=rf(sc.rgb)), D_far, np.zeros_like(D_far), grain=0) + 0.5, 0, 255))
        self.edge = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy])
        # how far the wash has reached, a quarter the size: it is soft
        Hq, Wq = (H + 2 * PAD_Y) // 4, (W + 2 * PAD_X) // 4
        r = np.random.default_rng(340)
        n = gaussian_filter(r.standard_normal((Hq, Wq)), 34) * 34 * 3.5 + 0.55 * gaussian_filter(r.standard_normal((Hq, Wq)), 13) * 13 * 3.5
        self.reach_n = np.float32(190 * n)
        # what is drawn on the page stays on it while the wash comes up under it, until it has gone past the edge
        lum = np.log(np.maximum(np.float32(self.base).mean(2), 1.0))
        L = lum - gaussian_filter(maximum_filter(lum, 61), 18)
        by, bx = grid_()
        box = lambda cx, cy: sm((bx - cx + 170) / 30) * (1 - sm((bx - cx - 50) / 30)) * sm((by - cy + 70) / 24) * (1 - sm((by - cy - 70) / 24))
        first = box(PALM[0], PALM[1])
        self.marks = Image.fromarray(np.float32(np.minimum(L + 0.07, 0) * (1 - first)), "F")
        # the wash starts from the prints and from the twig in his hand. The page and the close shot stay in one fixed relation, so these can be carried over once
        on_ground = np.maximum((L < -0.25) * np.maximum(first, box(905.0, 884.0)), np.clip(twig_on_page, 0, 1))
        to_page = (4 / Z_END, 0, PALM[0] - (PAD_X + C_PALM[0]) / Z_END, 0, 4 / (Z_END * STRETCH), PALM[1] - (PAD_Y + C_PALM[1]) / (Z_END * STRETCH))
        seeds = np.asarray(Image.fromarray(np.float32(on_ground), "F").transform((Wq, Hq), Image.AFFINE, to_page, Image.BILINEAR), np.float32) > 0.25
        assert seeds.any()
        self.reach_d = np.float32(distance_transform_edt(~seeds, sampling=(4 * 1.25, 4)))
        qy, qx = np.mgrid[0:Hq, 0:Wq]
        self.bord = np.float32(ease(np.minimum.reduce([qx, Wq - 1 - qx, qy, Hq - 1 - qy]) / 40.0))

    def frame(self, t):
        if t > PUSH1 + 1e-6:
            return self.wr.image(t)
        coming = t >= T_WASH[0]
        if coming:
            k_t = cam_of(t)[0] / Z_END
            C_t = self.wr.film(t)
            back = self.pad[PAD_Y:PAD_Y + H, PAD_X:PAD_X + W]
            fe = ((1 - ease(self.edge / 70.0)) * (1 - float(ease((k_t - 0.86) / 0.12))))[..., None]
            pad = Image.fromarray(self.pad)
            pad.paste(Image.fromarray(np.uint8(np.clip(C_t * (1 - fe) + back * fe + 0.5, 0, 255))), (PAD_X, PAD_Y))
            g = float(ease((t - T_WASH[0]) / (T_WASH[1] - T_WASH[0])))
            M_c = Image.fromarray(np.float32(sm((-100 + 2500 * g + self.reach_n - self.reach_d) / 620.0) * g * self.bord), "F")
        total = 0
        SHUTTER = np.linspace(-0.22, 0.22, 7)      # at this pace an edge moves too far in a frame to be drawn sharp
        for dt in SHUTTER:
            t_ = t + dt / 30
            z, a, X, Y = cam_of(t_)
            tx = _aff(PALM, (X, Y), z, z * a)
            f = np.float32(self.base.transform((W, H), Image.AFFINE, tx, Image.BICUBIC))
            if coming:
                k, b = z / Z_END, a / STRETCH
                txc = _aff((C_PALM[0] + PAD_X, C_PALM[1] + PAD_Y), (X, Y), k, k * b)
                f_c = np.float32(pad.transform((W, H), Image.AFFINE, txc, Image.BICUBIC))
                txq = _aff(((C_PALM[0] + PAD_X) / 4, (C_PALM[1] + PAD_Y) / 4), (X, Y), k * 4, k * b * 4)
                M = np.clip(np.asarray(M_c.transform((W, H), Image.AFFINE, txq, Image.BILINEAR), np.float32), 0, 1)[..., None]
                keep = 1 - float(ease((k - 0.84) / 0.14))
                mk = np.asarray(self.marks.transform((W, H), Image.AFFINE, tx, Image.BICUBIC), np.float32)[..., None] * keep
                f = np.exp(np.log(np.maximum(f, 1.0)) * (1 - M) + (np.log(np.maximum(f_c, 1.0)) + mk) * M)
            # he goes out past the edge: he is nearer than the ground we are going down to
            e = float(ease((t_ - T_MAN_OUT[0]) / (T_MAN_OUT[1] - T_MAN_OUT[0])))
            if e < 1:
                zm = z * (1 + 0.25 * e)
                to = (X + z * (MAN_AT[0] - PALM[0]) + 1250 * e, Y + z * (MAN_AT[1] - PALM[1]) + 260 * e)
                txm = _aff(MAN_AT, to, zm, zm)
                fill = np.clip(_warp_f(self.man_f, txm), 0, 1)[..., None]
                d = np.maximum(_warp_f(self.man_d, txm), 0)
                pap = np.float32(self.paper_im.transform((W, H), Image.AFFINE, tx, Image.BICUBIC))
                f = (f * (1 - fill) + pap * fill) * np.exp(-d[..., None] * INK_ABSORB)
            total = total + f
        return Image.fromarray(np.uint8(np.clip(total / len(SHUTTER) + 0.5, 0, 255)))


def sequence(only=None):
    """Into seq_raw4/: the frames of the slowed opening carry the new film's numbers, and seq_raw/ keeps the older shots this one still uses."""
    import ink_subs
    RAW = os.path.join(MYTH, "seq_raw4")
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    frames = [n for n in range(150, int(round(ink_subs.at("而天雨粟") * 30))) if only is None or n in only]
    page = any(n / 30 <= PUSH1 + 1e-6 for n in frames)
    w = World() if page else None
    wr = Writer() if any(n / 30 >= PUSH0 - 1e-6 for n in frames) else None
    push = Push(w, wr) if page and wr is not None else None
    print(f"setup {time.time() - t0:.0f}s", flush=True)
    for k, n in enumerate(frames):
        t, t1 = n / 30, time.time()
        im = w.shot(t) if t < PUSH0 - 1e-6 else (wr.image(t) if t > PUSH1 + 1e-6 else push.frame(t))
        im.save(os.path.join(RAW, f"f{n:05d}.png"), compress_level=1)
        if k % 15 == 0:
            print(f"  frame {n} (t {t:.2f}) {time.time() - t1:.2f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- the painting again. Its own clock is the old one, 30.0 to 34.5, frames 900 to 1034 of seq_raw/; in the film it starts at 鬼何以哭 (film_myth_seq.source)

T_BACK = (30.0, 31.0)
SHIFT_UP = 60                       # the whole painting a little higher: the subtitle's row is under the prints, not on them
CODA_Z = 1.06                       # how much closer from 31.0 to 34.5


def core_of(D, q=0.8):
    """Where a thing is darkest: the ink comes up from here first."""
    b = gaussian_filter(D, 3.0)
    return np.float32(b > np.quantile(b[b > 0.03], q))


class Again:
    """The ridge of ghosts sinks into the sheet and the painting of the first shot comes back up out of it: withdraw, run backward, each thing from where it is darkest. Then a very slow push, and the mist and the water drift a little, one way only."""

    def __init__(self, world):
        self.w = world
        P, sheet = world.P, world.sheet
        self.sheet = sheet
        self.paper = np.float32(sheet.rgb)
        self.paper4 = np.float32(Sheet(seed=99).rgb)
        last = np.float32(np.asarray(Image.open(os.path.join(MYTH, "seq_raw", "f00899.png")).convert("RGB")))
        L4 = np.log(np.maximum(last, 1.0)) - np.log(np.maximum(self.paper4, 1.0))
        self.L4 = [np.float32(L4[..., i]) for i in range(3)]
        self.tg4 = Target(core_of(-L4.mean(2), 0.75))
        self.mtn, self.far_, self.bird, self.branch = P["mtn"], P["far"], P["bird"], P["branch"]
        self.horse = P["horse"][:3]
        self.hcov = P["horse"][3]
        self.man = P["man"]
        fp, cp, _, wp = world.pond
        ff, cf, _, wf = world.field
        self.ground = (ff, cf, fp, cp, P["ground"][2], wf + wp)
        self.tg = dict(
            mtn=Target(core_of(self.mtn[0] + self.mtn[1])), far=Target(core_of(self.far_[0] + self.far_[1])), bird=Target(core_of(self.bird[0] + self.bird[1])),
            branch=Target(core_of(self.branch[0], 0.6)), horse=Target(core_of(self.horse[0] + self.horse[1])), man=Target(core_of(self.man[0] + self.man[1], 0.6)),
            ground=Target(core_of(ff + cf + fp + cp, 0.85)))
        yy, xx = grid_()
        self.yy, self.xx = yy, xx
        self.foot = 1 - sm((yy - 1015) / 50)
        # how far each layer has drifted by the end: the mist round the mountain's feet, the far hills less, the pool one way and the water lines the other
        vary = 0.75 + 0.25 * np.clip(noise(sheet, 3.4, 350), -1.5, 1.5)
        self.dx_mtn = 24 * sm((yy - 400) / 110) * (1 - sm((yy - 720) / 60)) * vary
        self.dx_far = 13 * (0.8 + 0.2 * np.clip(noise(sheet, 3.4, 351), -1.5, 1.5))
        self.dx_pool = -11 * vary
        self.dx_line = 15 * (0.8 + 0.2 * np.clip(noise(sheet, 3.0, 352), -1.5, 1.5))

    def drift(self, a, dx, s):
        return np.float32(map_coordinates(a, [self.yy, self.xx - dx * s], order=1, mode="nearest"))

    def back(self, arrays, tg, t, span, reach, seed, **kw):
        tau = 1 - tau_of(t, span)
        if tau >= 1:
            return [np.zeros_like(a) for a in arrays], 1.0
        out = withdraw(self.sheet, arrays, tg, tau, reach, seed, stain=0.0, **kw)
        there = float(ease((1 - tau) / 0.12))
        return [a * there for a in out], tau

    def frame(self, t):
        fg_, cg_, pf, pc, rl, wl = self.back(self.ground, self.tg["ground"], t, (30.25, 30.9), 1400.0, 361, pull=0.0, rest=0.0, tide=0.0, soften=0.4)[0]
        (fm, cm, lm), _ = self.back(self.mtn, self.tg["mtn"], t, (30.3, 30.95), 620.0, 362, pull=16.0, rest=0.02, tide=0.25, soften=0.28, spread=0.7)
        (ff, cf), _ = self.back(self.far_, self.tg["far"], t, (30.3, 31.0), 1300.0, 363, pull=0.0, rest=0.0, tide=0.0, soften=0.4)
        (ib, kb), tb = self.back(self.branch, self.tg["branch"], t, (30.4, 30.95), 720.0, 364, pull=0.0, rest=0.0, tide=0.3)
        (bf, bc, bl), _ = self.back(self.bird, self.tg["bird"], t, (30.4, 31.0), 170.0, 365, pull=12.0, rest=0.015, tide=0.25, soften=0.28, spread=0.6)
        (hf_, hc, hl), th = self.back(self.horse, self.tg["horse"], t, (30.35, 31.0), 300.0, 366, pull=12.0, rest=0.012, tide=0.14, soften=0.5, spread=0.8)
        (mi, mt, _), tm_ = self.back(self.man, self.tg["man"], t, (30.5, 31.0), 260.0, 367, pull=8.0, rest=0.01, tide=0.2, soften=0.3, spread=0.4)
        s = float(np.clip((t - 30.5) / 4.0, 0, 1))
        if s > 0:
            fm, cm, lm = self.drift(fm, self.dx_mtn, s), self.drift(cm, self.dx_mtn, s), self.drift(lm, self.dx_mtn, s)
            ff, cf = self.drift(ff, self.dx_far, s), self.drift(cf, self.dx_far, s)
            pf, pc = self.drift(pf, self.dx_pool, s), self.drift(pc, self.dx_pool, s)
            wl = self.drift(wl, self.dx_line, s)
        over_h = self.hcov * (1 - ease((th - 0.05) / 0.55))
        over_br = self.branch[1] * (1 - ease(tb / 0.5))
        man_fill = self.man[2] * (1 - ease(tm_ / 0.6))
        behind = (1 - over_h) * (1 - man_fill)
        fine = (fm + ff) * behind * (1 - over_br) + (fg_ + pf) * behind + hf_ + bf
        coarse = (cm + cf) * behind * (1 - over_br) + (cg_ + pc) * behind + hc + bc
        line = lm * behind + bl + hl + wl * behind + rl * behind + np.minimum(ib * (1 - man_fill), 2.7) + np.minimum(mi + mt, 2.7)

        def up(a):
            b = np.zeros_like(a)
            b[:-SHIFT_UP] = a[SHIFT_UP:]
            return b * self.foot

        f1 = develop(self.sheet, up(fine), up(coarse), line=up(np.minimum(line, 2.75)))
        u = tau_of(t, T_BACK)
        if u < 1:
            tau4 = tau_of(t, (30.0, 30.65))
            L = withdraw(self.sheet, self.L4, self.tg4, tau4, self.tg4.d.max() * 0.8, 368, pull=12.0, rest=0.0, stain=0.0, tide=0.18, soften=0.3, spread=0.6) if tau4 < 1 else None
            gone = 1 - float(ease((tau4 - 0.84) / 0.16))
            I4 = self.paper4 * (np.exp(np.stack(L, 2) * gone) if L is not None else 1.0)
            mix = ease(u)
            f1 = I4 * f1 / (self.paper ** (1 - mix) * self.paper4 ** mix)
        z = CODA_Z ** max((t - T_BACK[1]) / (34.5 - T_BACK[1]), 0.0)
        cx, cy = ZOOM_FIX
        box = (cx - cx / z, cy - cy / z, cx + (W - cx) / z, cy + (H - cy) / z)
        return Image.fromarray(np.uint8(np.clip(f1 + 0.5, 0, 255))).resize((W, H), Image.BICUBIC, box=box)


def coda(only=None):
    RAW = os.path.join(MYTH, "seq_raw")
    t0 = time.time()
    a = Again(World())
    print(f"setup {time.time() - t0:.0f}s", flush=True)
    for n in range(900, 1035):
        if only is not None and n not in only:
            continue
        t1 = time.time()
        a.frame(n / 30).save(os.path.join(RAW, f"f{n:05d}.png"), compress_level=1)
        if n % 15 == 0:
            print(f"  frame {n} (t {n / 30:.2f}) {time.time() - t1:.2f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


STILLS = dict(x1_world=5.0, x2_shan=8.5, x3_niao=10.5, x5_ma=13.0, x6_page=14.0)


def main():
    what = sys.argv[1]
    only = set(int(v) for v in sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    if what == "seq":
        sequence(only)
        return
    if what == "coda":
        coda(only)
        return
    if what == "glyphs":
        b = np.full((H, W), 236.0)
        for name in AT:
            b = b * (1 - 0.92 * glyph_mask(name))
        Image.fromarray(np.uint8(b)).convert("RGB").save(os.path.join(OUT, "_x_glyphs.png"))
        return
    t0 = time.time()
    w = World()
    print(f"painted in {time.time() - t0:.0f}s", flush=True)
    if what == "at":
        w.shot(float(sys.argv[2]), sys.argv[3])
    elif what == "still":
        names = sys.argv[2:] or list(STILLS)
        for name in names:
            t1 = time.time()
            w.shot(STILLS[name], name)
            print(f"  {name} {time.time() - t1:.1f}s", flush=True)


if __name__ == "__main__":
    main()

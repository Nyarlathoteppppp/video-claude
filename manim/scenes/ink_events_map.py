"""五 · 字的位置, happening. 男到女, 王到后: the two marches are set down track by track, the same way, and then a wet track runs along each at the same moment and pace, so that it can be seen they are parallel.
一个字的位置, 便是一串数字: the numbers beside each place come up one at a time, wet.
"""
import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter, label

from ink_events_lib import _X, _Y, H, W, box, crop8, fill, plate
from ink_grade import fft_noise, sm

LUM = np.array([0.30, 0.59, 0.11], np.float32)
_r, _c = {}, {}


def _dark(P, frac, sigma=24.0):
    L = (P * LUM).sum(-1)
    return L < gaussian_filter(L, sigma) * frac


def _items_of(ink, min_px=30):
    """The pieces of ink, as (ya, yb, xa, xb, mask), with the hole around each."""
    lab, n = label(ink, np.ones((3, 3), bool))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < min_px:
            continue
        out.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1, i))
    return out, lab


def _route_prep():
    P = plate("map/route")
    dark = _dark(P, 0.5)
    rows = []
    for ya, yb, xa, xb in ((625, 712, 738, 1148), (445, 505, 788, 1120)):
        m = np.zeros((H, W), bool)
        m[ya:yb, xa:xb] = True
        ink = binary_opening(dark & m, np.ones((2, 2), bool))
        items, lab = _items_of(ink)
        items = sorted(items, key=lambda q: q[2])
        rows.append((items, lab))
    hole = np.zeros((H, W), bool)
    for items, lab in rows:
        for it in items:
            hole |= lab == it[4]
    hole = binary_dilation(hole, iterations=5)
    C = fill(P, hole, sigma=9.0, offset=(52, 16), feather=1.6).astype(np.float32)
    # each march: its chevrons, each with what it adds to the clean ground and its order from back to point
    tracks = []
    for items, lab in rows:
        line = []
        for ya, yb, xa, xb, i in items:
            pad = 7
            ya, yb, xa, xb = max(ya - pad, 0), min(yb + pad, H), max(xa - pad, 0), min(xb + pad, W)
            order = (_X[ya:yb, xa:xb] - xa) / max(xb - xa, 1)
            line.append((ya, yb, xa, xb, order.astype(np.float32), P[ya:yb, xa:xb] - C[ya:yb, xa:xb]))
        tracks.append(line)
    # the two ways, as a wet track on the ground from character to character, through the chevrons' middles
    ends = ((698, 1198, 15.0), (742, 1172, 9.5))
    ways = []
    for line, (x0, x1, wide) in zip(tracks, ends):
        cx = np.array([(q[2] + q[3]) / 2 for q in line])
        cy = np.array([(q[0] + q[1]) / 2 for q in line])
        ways.append(_way(np.polyfit(cx, cy, 1), x0, x1, wide, 40 + len(ways)))
    _r.update(C=C, tracks=tracks, ways=ways)


def _way(fit, x0, x1, wide, seed):
    """A wet track along a straight line. Returns its box, how much it wets each pixel, and how far along it each pixel is. Wetting multiplies: what is under it still shows."""
    yrow = np.polyval(fit, _X)
    d = _Y - yrow
    along = np.clip((_X - x0) / (x1 - x0), 0, 1)
    rough = gaussian_filter(fft_noise(H, W, 1.5, seed), (2.0, 16.0))
    rough = rough / rough.std()
    fine = gaussian_filter(fft_noise(H, W, 1.2, seed + 1), 1.1)
    fine = fine / fine.std()
    w = wide * (0.9 + 0.22 * rough) * (0.45 + 0.55 * sm(along / 0.08)) * (0.5 + 0.5 * sm((1 - along) / 0.1))
    core = np.exp(-((d + 1.6 * rough) / np.maximum(w, 1.0)) ** 2)
    wet = core * (0.82 + 0.18 * fine) * sm(along / 0.02) * sm((1 - along) / 0.02) * (_X > x0 - 2) * (_X < x1 + 2)
    y0, y1 = np.polyval(fit, [x0, x1])
    ya, yb = int(min(y0, y1) - wide * 3), int(max(y0, y1) + wide * 3)
    xa, xb = int(x0) - 3, int(x1) + 4
    return ya, yb, xa, xb, wet[ya:yb, xa:xb].astype(np.float32), along[ya:yb, xa:xb].astype(np.float32)


ROW_T = ((0.32, 1.22), (1.32, 2.22))      # when each march is set down: 男到女, 王到后
WAY_T = (2.45, 3.3)                       # 走的是同一个方向: both ways at once, at one pace
WET = np.array([0.46, 0.52, 0.5], np.float32)      # what the wet track leaves of the light under it


def route(frame, t, dur):
    if not _r:
        _route_prep()
    r = _r
    out = r["C"].copy()
    for (t0, t1), line in zip(ROW_T, r["tracks"]):
        step = (t1 - t0) / len(line)
        for i, (ya, yb, xa, xb, order, delta) in enumerate(line):
            p = (t - t0 - i * step) / (step * 1.3)
            if p <= 0:
                continue
            out[ya:yb, xa:xb] += delta * sm((p * 1.5 - order) / 0.5)[..., None]
    if t > WAY_T[0]:
        p = (t - WAY_T[0]) / (WAY_T[1] - WAY_T[0])
        for ya, yb, xa, xb, wet, along in r["ways"]:
            front = sm((p * 1.15 - along) / 0.15)
            tip = np.exp(-(((p * 1.15 - 0.075) - along) / 0.09) ** 2) * sm((1.25 - p) / 0.3)        # the wet edge, wettest
            a = (wet * np.clip(front * 0.8 + tip * 0.3, 0, 1.1))[..., None]
            out[ya:yb, xa:xb] *= 1 - a * (1 - WET)
    return crop8(out, box((1.0, 1.05), (0.5, 0.5), t, dur))


# Each number, as (row's top, row's bottom, left, right), the near place first. Read off the still: they tilt, so a row's box is the height of the whole row.
GROUPS = (
    (((867, 909), (270, 384), (395, 512), (523, 622)), ((908, 953), (266, 352), (365, 480), (493, 596))),                 # 男
    (((692, 727), (1200, 1302), (1313, 1419), (1428, 1532)), ((728, 762), (1208, 1310), (1322, 1430), (1440, 1532))),     # 女
    (((548, 577), (414, 492), (504, 590), (601, 680)), ((576, 607), (405, 481), (492, 572), (582, 654))),                 # 王
    (((452, 492), (1172, 1252), (1254, 1320), (1322, 1398)), ((484, 524), (1156, 1248), (1250, 1322), (1324, 1404))),     # 后
)


def _coords_prep():
    P = plate("map/coords")
    # the numbers' own ink and the lit lip beside it, and no more: everything between the digits stays the still's own ground
    G = (P * LUM).sum(-1)
    near = gaussian_filter(G, 12.0)
    where = np.zeros((H, W), bool)
    for rows in GROUPS:
        for row in rows:
            (ya, yb), toks = row[0], row[1:]
            for xa, xb in toks:
                where[ya - 2:yb + 2, xa - 2:xb + 2] = True
    ink = where & (G < near * 0.8)
    hole = binary_dilation(ink, np.ones((3, 3), bool), iterations=4)
    C = fill(P, hole, sigma=5.0, offset=(11, 13), feather=1.2, fine=2.2)
    C = C.astype(np.float32)
    n = fft_noise(H, W, 1.8, 44)
    items = []
    t = 0.36
    for g, rows in enumerate(GROUPS):
        for row in rows:
            (ya, yb), toks = row[0], row[1:]
            ya, yb = ya - 5, yb + 5
            for xa, xb in toks:
                xa, xb = xa - 5, xb + 5
                hy, hx = yb - ya, xb - xa
                # soft at its edges: a number sets down its own ink, not the ground of the still around it
                edge = np.minimum(np.minimum(_Y[:hy, 0][:, None], _Y[:hy, 0][::-1][:, None]), np.minimum(_X[0, :hx][None], _X[0, :hx][::-1][None]))
                delta = (P[ya:yb, xa:xb] - C[ya:yb, xa:xb]) * sm(edge / 5.0)[..., None]
                wet = np.stack([gaussian_filter(delta[..., c], 1.6) for c in range(3)], -1)
                order = ((_X[ya:yb, xa:xb] - xa) / max(hx, 1)).astype(np.float32)
                items.append((t, ya, yb, xa, xb, order, delta, wet))
                t += 0.125 if g == 0 else 0.092
        t += 0.03
    _c.update(C=C, items=items, n=n)


def coords(frame, t, dur):
    if not _c:
        _coords_prep()
    c = _c
    out = c["C"].copy()
    for t0, ya, yb, xa, xb, order, delta, wet in c["items"]:
        a = t - t0
        if a <= 0:
            continue
        p = a / 0.2
        al = sm((p * 1.4 - order - 0.14 * c["n"][ya:yb, xa:xb]) / 0.4)[..., None]
        soak = math.exp(-a / 0.55)          # as it comes it is a little wider and darker than it will dry to
        out[ya:yb, xa:xb] += (delta * (1 - soak) + wet * 1.35 * soak) * al
    return crop8(out, box((1.0, 1.05), (0.5, 0.5), t, dur))

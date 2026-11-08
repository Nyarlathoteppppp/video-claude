"""六 · 回头看, happening.
笔断意连: the threads between the characters go across one at a time. 下一个字, 我来猜: in the next cell the candidates come and go, unsure, and settle on 家. 猜错了, 就改几个数: 家 thins away, 故 comes up in its place, the water runs down from it to the numbers, and a few of them change.
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter
from scipy.spatial import cKDTree

from ink_events_lib import _X, _Y, H, W, box, crop8, fill, membrane, plate, region_from, stroke_order, to8
from ink_grade import fft_noise, sm
from ink_move import sample

LUM = np.array([0.30, 0.59, 0.11], np.float32)
_l, _g, _f = {}, {}, {}


def _along(points, step=1.0):
    """A polyline as points about a pixel apart, with how far along each one is."""
    p = np.array(points, np.float64)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.arange(0, s[-1], step)
    # a little smoothing, so the corridor follows the curve and not its corners
    x, y = np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])
    k = np.hanning(25)
    k = k / k.sum()
    pad = 12
    x = np.convolve(np.pad(x, pad, mode="edge"), k, "same")[pad:-pad]
    y = np.convolve(np.pad(y, pad, mode="edge"), k, "same")[pad:-pad]
    return np.stack([x, y], -1), q


# ---- 笔断意连 ---------------------------------------------------------------------------------------------------------------------------------------------
# Each thread as it lies on the still, from where it starts: (points, starts, ends, half width). 低 and 头 into 思 first, then from 思 across to 举, 望, 明, 月.
THREADS = (
    (((682, 230), (692, 270), (705, 315), (712, 365), (707, 430), (692, 480), (675, 525), (668, 540)), 0.24, 0.92, 9.0),
    (((690, 408), (700, 430), (703, 455), (692, 490), (672, 535)), 0.72, 1.14, 7.0),
    (((682, 552), (700, 548), (735, 535), (760, 515), (785, 470), (795, 410), (802, 355), (812, 290), (827, 255), (852, 247)), 0.98, 1.68, 6.5),
    (((682, 570), (735, 575), (765, 575), (800, 567), (848, 575)), 1.46, 1.92, 8.0),
    (((684, 592), (715, 598), (747, 602), (790, 583), (852, 587)), 1.56, 2.02, 5.5),
    (((680, 596), (715, 602), (737, 622), (757, 650), (785, 677), (807, 710), (825, 732), (854, 742)), 1.84, 2.42, 8.5),
    (((690, 627), (725, 640), (757, 652), (772, 705), (787, 775), (795, 845), (810, 895), (830, 922), (849, 920)), 2.16, 2.96, 8.5),
    (((695, 152), (770, 170), (850, 196)), 2.72, 3.14, 4.5),
    (((710, 357), (790, 380), (846, 396)), 2.82, 3.22, 4.5),
)
LIAN_BOX = (118, 950, 650, 874)       # y0, y1, x0, x1 on the still
FAN = ((682, 536), (760, 528), (848, 556), (848, 604), (832, 700), (800, 730), (760, 716), (690, 640))


def _lian_prep():
    P = plate("attn/lian")
    y0, y1, x0, x1 = LIAN_BOX
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pts, idx, arc, wide = [], [], [], []
    for i, (points, _, _, hw) in enumerate(THREADS):
        q, s = _along(points)
        pts.append(q)
        idx.append(np.full(len(q), i))
        arc.append(s)
        wide.append(hw)
    pts, idx, arc, wide = np.concatenate(pts), np.concatenate(idx), np.concatenate(arc), np.array(wide)
    dist, near = cKDTree(pts).query(np.stack([xx.ravel(), yy.ravel()], -1), distance_upper_bound=16.0)
    near = np.where(np.isfinite(dist), near, 0)
    dist, near = dist.reshape(yy.shape), near.reshape(yy.shape)
    which = idx[near]
    # the thread and the pale lip each side of it, but never the characters it comes out of: what is black and thick is theirs. Where the threads fan out of 思 their lips run together, so the whole fan goes.
    L = (P * LUM).sum(-1)
    chars = binary_dilation(binary_opening(L < 74.0, iterations=4), iterations=2)
    inside = dist < wide[which] + 4.0
    hole = np.zeros((H, W), bool)
    hole[y0:y1, x0:x1] = inside
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(FAN, fill=255)
    hole |= np.asarray(im) > 0
    hole &= ~chars
    # the paper's tone is taken with the characters out of it too, or the hole beside a stroke comes up grey
    paper = fill(P, hole | chars, sigma=7.0, offset=(0, 280), feather=1.3, fine=6.0)     # its grain from clear of the threads, past the other column
    m = np.clip(gaussian_filter(hole.astype(np.float32), 1.3) * 2.0, 0, 1)[..., None]
    C = P * (1 - m) + paper * m
    _l.update(delta=(C - P)[y0:y1, x0:x1].astype(np.float32), which=which, lives=inside, arc=arc[near].astype(np.float32),
              lens=np.array([arc[idx == i].max() for i in range(len(THREADS))], np.float32),
              t0=np.array([t[1] for t in THREADS], np.float32), t1=np.array([t[2] for t in THREADS], np.float32))


def lian(frame, t, dur):
    if not _l:
        _lian_prep()
    d = _l
    SOFT = 30.0
    front = np.clip((t - d["t0"]) / (d["t1"] - d["t0"]), 0, 1.2) * (d["lens"] + SOFT)
    hide = np.where(d["lives"], 1 - sm((front[d["which"]] - d["arc"]) / SOFT), 1.0)
    layer = d["delta"] * hide[..., None]
    y0, y1, x0, x1 = LIAN_BOX
    # the frame's pixels that can look at it
    ya, yb, xa, xb = max(y0 - 28, 0), min(y1 + 28, H), max(x0 - 28, 0), min(x1 + 28, W)
    ys, xs = region_from("attn/lian", t, dur, (ya, yb, xa, xb))
    got = sample(layer, ys - y0, xs - x0)
    out = frame.copy()
    out[ya:yb, xa:xb] = np.clip(frame[ya:yb, xa:xb] + got + 0.5, 0, 255).astype(np.uint8)
    return out


# ---- 下一个字, 我来猜 ---------------------------------------------------------------------------------------------------------------------------------------
GY0, GY1, GX0, GX1 = 470, 905, 670, 1115
# (character, comes, goes, how dark): quick at first, then fewer and slower, and then it stays
TRIES = (("念", 0.18, 0.42, 0.5), ("亲", 0.4, 0.66, 0.6), ("故", 0.64, 1.0, 0.8), ("亲", 0.98, 1.14, 0.46), ("故", 1.12, 1.4, 0.82), ("家", 1.38, 1.58, 0.72), ("故", 1.56, 1.72, 0.56))
SETTLE = 1.7


def _vapour(ch, seed):
    """A character as it hangs in the air on this still: on the plane 家 is on, its strokes thick and smoking."""
    from ink_film import font
    s = 512
    im = Image.new("L", (s, s), 0)
    ImageDraw.Draw(im).text((s / 2, s / 2), ch, 255, font=font("kaih", 340), anchor="mm")
    h, w = GY1 - GY0, GX1 - GX0
    ang = math.radians(-6.5)
    sx, sy = 0.86, 1.1          # to the size 家 has here
    cx, cy = 932.0 - GX0, 690.0 - GY0
    ca, sa = math.cos(ang), math.sin(ang)
    a, b = ca / sx, sa / sx
    d, e = -sa / sy, ca / sy
    g = np.asarray(im.transform((w, h), Image.AFFINE, (a, b, s / 2 - a * cx - b * cy, d, e, s / 2 - d * cx - e * cy), Image.BICUBIC), np.float32) / 255
    return (gaussian_filter(g, 10.0) + 0.45 * gaussian_filter(g, 26.0)).astype(np.float32)


def _guess_prep():
    P = plate("attn/guess")
    hole = np.zeros((H, W), bool)
    hole[488:886, 690:1094] = True      # 家, and the square of odd grain the still has round it
    tone = membrane(P, hole, feel=6.0)
    detail = P - np.stack([gaussian_filter(P[..., c], 30.0) for c in range(3)], -1)
    x, y = _X.astype(np.int32), _Y.astype(np.int32)
    low, high = detail[np.clip(y + 188, 0, H - 1), x], detail[np.clip(y + 400, 0, H - 1), x]       # the same paper, below
    k = sm((_Y - 636.0) / 38.0)[..., None]
    # the grain comes in from the pale paper to the left over some way, not at the hole's edge
    grain = (0.25 + 0.75 * sm((_X - 690.0) / 110.0))[..., None]
    m = np.clip(gaussian_filter(hole.astype(np.float32), 7.0) * 1.8, 0, 1)[..., None]
    C = (P * (1 - m) + (tone + (high * (1 - k) + low * k) * grain) * m).astype(np.float32)
    LP, LC = (P * LUM).sum(-1), gaussian_filter((C * LUM).sum(-1), 9.0)
    dark = gaussian_filter(1 - LP / LC, 2.0)
    jia = (sm((dark - 0.14) / 0.62))[GY0:GY1, GX0:GX1] * m[GY0:GY1, GX0:GX1, 0]
    ink = P[(dark > 0.72) & hole].mean(0)
    faces = {"家": jia.astype(np.float32)}
    for i, ch in enumerate("念亲故"):
        faces[ch] = _vapour(ch, 60 + i)
    n = fft_noise(GY1 - GY0, GX1 - GX0, 2.3, 60)
    _g.update(C=C, faces=faces, ink=ink.astype(np.float32), n=(n / n.std()).astype(np.float32))


def _shift(a, dx, dy):
    return np.roll(np.roll(a, dy, 0), dx, 1)


def _thin(a, e):
    """A mask as it is at density e: its thin parts go first, and its middle stays dark."""
    return np.clip(a * (2 - e) - (1 - e), 0, 1) * min(e * 3, 1.0)


def guess(frame, t, dur):
    if not _g:
        _guess_prep()
    g = _g
    keep = np.ones((GY1 - GY0, GX1 - GX0), np.float32)
    for i, (ch, a, b, dens) in enumerate(TRIES):
        if a < t < b + 0.1:
            e = sm((t - a) / 0.07) * (1 - sm((t - b + 0.02) / 0.1)) * dens
            if ch == "家":
                keep = keep * (1 - _thin(g["faces"][ch], e))
            else:
                # it condenses from the middle of its strokes, and it does not sit still until it is the one
                f = _shift(g["faces"][ch], int(round(3 * math.sin(i * 2.4 + t * 9))), int(round(2 * math.cos(i * 1.7 + t * 7))))
                keep = keep * (1 - np.clip(sm((f - (0.86 - 0.5 * e)) / 0.42) * (0.8 + 0.26 * g["n"]) * min(e * 2.2, 1.0), 0, 1))
    if t > SETTLE - 0.03:
        keep = keep * (1 - _thin(g["faces"]["家"], (0.6 + 0.4 * sm((t - SETTLE) / 0.5)) * sm((t - SETTLE + 0.03) / 0.08)))
    out = g["C"].copy()
    q = keep[..., None]
    out[GY0:GY1, GX0:GX1] = out[GY0:GY1, GX0:GX1] * q + g["ink"] * (1 - q)
    return crop8(out, box((1.0, 1.05), (0.5, 0.5), t, dur))


# ---- 猜错了, 就改几个数 -----------------------------------------------------------------------------------------------------------------------------------
# The water from 故 down to the numbers and on into their reflection: its path on the still, its start, and which clean paper to its right its grain comes from
RUNS = (
    (((655, 440), (650, 500), (655, 540), (645, 600), (638, 650), (632, 690), (650, 740), (665, 800), (675, 900), (690, 925)), 1.22, 270),
    (((682, 440), (695, 500), (700, 550), (695, 600), (700, 680), (702, 700), (705, 760), (710, 830), (725, 925)), 1.08, 220),
    (((755, 450), (760, 520), (775, 590), (775, 650), (782, 690), (800, 760), (815, 850), (825, 925)), 1.32, 150),
    (((835, 450), (820, 520), (825, 600), (845, 680), (852, 700), (862, 760), (890, 840), (915, 905)), 1.16, 85),
)
SHORE = 925        # where the water's edge is on this still
# The numbers that change: where the changed one is, and a dark digit on the still that shows what was there before.
CHANGES = (
    ((714, 756, 738, 800), (748, 760, 774, 802), 1.74),     # the 3 in 0.367
    ((820, 698, 848, 736), (854, 698, 882, 736), 1.92),     # the 4 of 40.56
    ((595, 828, 622, 880), (630, 826, 657, 878), 2.1),      # the 0 of 0.09
    ((352, 726, 390, 774), (390, 722, 428, 770), 2.26),     # the 6 of -1.67
    ((172, 782, 202, 830), (136, 782, 166, 830), 2.42),     # the 9 of 1.79
    ((310, 846, 338, 902), (396, 838, 424, 894), 2.56),     # the 7 of 7.14
)
GU = (205, 455, 626, 874)       # y0, y1, x0, x1: 故 and its wet ground
JIA = (446, 650, 945, 1140)     # 家, thinned to nothing much on the still


def _corridor(points, hw, mirror=False):
    q, _ = _along(points)
    if mirror:
        q = q[q[:, 1] > SHORE - 110] * 1.0
        q[:, 1] = 2 * SHORE - q[:, 1]
    m = np.zeros((H, W), np.uint8)
    im = Image.fromarray(m)
    ImageDraw.Draw(im).line([tuple(p) for p in q], fill=255, width=int(hw * 2))
    return np.asarray(im) > 0


def _fix_prep():
    P = plate("attn/fix")
    L = (P * LUM).sum(-1)
    digits = binary_dilation(L < gaussian_filter(L, 14.0) * 0.62, iterations=2)
    C = P
    wet = []
    for points, t0, dx in RUNS:
        above = _corridor(points, 24.0) & (_Y < 690)
        below = _corridor(points, 16.0) & (_Y >= 690) & (_Y < SHORE) & ~digits
        under = _corridor(points, 16.0, mirror=True) & (_Y >= SHORE) & ~digits
        C = fill(C, above, sigma=9.0, offset=(0, dx), feather=2.0, fine=7.0)
        wet.append(above | below | under)
    # among the numbers the grain comes from the bare paper above them, now that it is clean: anything nearer would bring digits with it
    for points, t0, dx in RUNS:
        below = _corridor(points, 16.0) & (_Y >= 690) & (_Y < SHORE) & ~digits
        C = fill(C, below, sigma=7.0, offset=(-262, 0), feather=1.6, fine=4.0)
        under = _corridor(points, 16.0, mirror=True) & (_Y >= SHORE) & ~digits
        C = fill(C, under, sigma=7.0, offset=(-500, 0), feather=1.6, fine=4.0)
    wet = [gaussian_filter(w.astype(np.float32), 2.0) for w in wet]
    y0, y1, x0, x1 = GU
    hole = np.zeros((H, W), bool)
    hole[y0:y1, x0:x1] = True
    C = fill(C, hole, sigma=16.0, offset=(238, 0), feather=6.0, fine=16.0)
    # the changed numbers, as they were
    spots = np.zeros((H, W), bool)
    for (xa, ya, xb, yb), _, _ in CHANGES:
        spots[ya:yb, xa:xb] = True
    C = fill(C, spots, sigma=8.0, offset=(-190, 0), feather=2.0, fine=3.0).astype(np.float32)
    near = gaussian_filter(L, 14.0)
    for (xa, ya, xb, yb), (sa, sb, sc, sd), _ in CHANGES:
        a = np.clip((1 - L[sb:sd, sa:sc] / near[sb:sd, sa:sc]) / 0.55, 0, 1)
        a = np.asarray(Image.fromarray(a).resize((xb - xa, yb - ya), Image.BILINEAR))[..., None]
        C[ya:yb, xa:xb] = C[ya:yb, xa:xb] * (1 - a) + np.array((30, 23, 17), np.float32) * a
    # 故's order
    ink = (L < gaussian_filter(L, 30.0) * 0.5)[y0:y1, x0:x1]
    order = stroke_order(ink, gap=16.0)
    # the wet ground round it goes off softly, not to the straight edge the still has at its right
    halo = np.clip(gaussian_filter(binary_dilation(ink, iterations=12).astype(np.float32), 14.0) * 1.6, 0, 1)
    # 家 at the strength it had
    ja, jb, jc, jd = JIA
    r = gaussian_filter(L, 4.0) / np.maximum(gaussian_filter(L, 42.0), 1.0)
    edge = np.minimum(np.minimum(_Y[ja:jb, 0][:, None] - ja, jb - _Y[ja:jb, 0][:, None]), np.minimum(_X[0, jc:jd][None] - jc, jd - _X[0, jc:jd][None]))
    r = 1 - np.clip(1 - r[ja:jb, jc:jd], 0, 0.8) * sm(edge / 30.0)
    wobble = fft_noise(H, W, 1.9, 77)
    wobble = (wobble / wobble.std())[430:1080, 590:950]
    run = np.stack(wet, 0)[:, 430:1080, 590:950].astype(np.float32)
    _f.update(P=P, C=C, order=order, halo=halo.astype(np.float32), top=float(order[ink].max()), r=r.astype(np.float32), run=run, wobble=wobble.astype(np.float32), rows=_Y[430:1080, 590:950].copy(), n=fft_noise(H, W, 1.3, 78))


def fix(frame, t, dur):
    if not _f:
        _fix_prep()
    f = _f
    out = f["C"].copy()
    # the right character, stroke by stroke, its ground wetting with it
    y0, y1, x0, x1 = GU
    p = (t - 0.78) / 0.86
    if p > 0:
        a = (sm((p * (f["top"] + 60) - f["order"]) / 60.0) * f["halo"])[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + f["P"][y0:y1, x0:x1] * a
    # the water
    if t > RUNS[1][1]:
        acc = np.zeros(f["rows"].shape, np.float32)
        for i, (_, t0, _) in enumerate(RUNS):
            front = 436 + (t - t0) * 410.0 * (1 + 0.5 * sm((t - t0 - 0.4) / 0.6))
            acc = np.maximum(acc, f["run"][i] * sm((front + 16 * f["wobble"] - f["rows"]) / 34.0))
        a = acc[..., None]
        out[430:1080, 590:950] = out[430:1080, 590:950] * (1 - a) + f["P"][430:1080, 590:950] * a
    # a few of the numbers change: what was there thins, and what it is now comes up pale
    for (xa, ya, xb, yb), _, t0 in CHANGES:
        if t > t0:
            a = sm((t - t0 + 0.09 * f["n"][ya:yb, xa:xb]) / 0.26)[..., None]
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - a) + f["P"][ya:yb, xa:xb] * a
    # the wrong one, dark, thinning to what the still has left of it
    k = 4.2 * (1 - sm((t - 0.28) / 0.72))
    if k > 0.01:
        ja, jb, jc, jd = JIA
        out[ja:jb, jc:jd] = out[ja:jb, jc:jd] * np.power(f["r"], k)[..., None]
    return crop8(out, box((1.0, 1.05), (0.5, 0.5), t, dur))

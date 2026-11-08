"""The wide shots of 鬼夜哭: the river bank (k1_wide), the rain of grain over the bare hills (k2) and the ridge of small ghosts under the moon (k4_wide). Same ink, same mud and same ghost as the close shots in film_myth.py, which is what runs them:

    python film_myth.py k1_wide
    python film_myth.py k2
    python film_myth.py k4_wide

Painted washes are kept in keys/_wash_*.npz, so what is drawn over them can be changed in seconds. MYTH_FRESH=1 paints again.
"""
import os
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

from film_myth import _clear, lift_figure, open_mouths, paint_clouds, qing_wash, report, tidy
from film_myth_ghost import Ghost, _egg, _seg, _smin
from film_myth_mud import Mud, dense, ink_of, twig_shape, unit
from film_myth_wash import INK_ABSORB, OUT, Sheet, Wash, band, blob, develop, lumps, noise, save, sm

W, H = 1920, 1080
GRAIN_ABSORB = np.array((0.10, 0.34, 1.0), np.float32)     # the thinnest straw-coloured glaze, on the grain only: no red, and it can be set to nothing


def grid_():
    return np.mgrid[0:H, 0:W].astype(np.float32)


def cached(name, make):
    path = os.path.join(OUT, f"_wash_{name}.npz")
    if os.path.exists(path) and os.environ.get("MYTH_FRESH") != "1":
        z = np.load(path)
        return np.float32(z["fine"]), np.float32(z["coarse"])
    fine, coarse = make()
    np.savez_compressed(path, fine=np.float16(fine), coarse=np.float16(coarse))
    return fine, coarse


def ridge(points, seed, jag=14.0):
    """A skyline: y for each x through these points, its jags sharp, because these hills have nothing growing on them."""
    r = np.random.default_rng(seed)
    p = np.array(points, np.float64)
    y = gaussian_filter1d(np.interp(np.arange(W), p[:, 0], p[:, 1]), 14)
    return np.float32(y + jag * unit(gaussian_filter1d(r.normal(0, 1, W), 30)) + 0.42 * jag * unit(gaussian_filter1d(r.normal(0, 1, W), 8)) + 0.12 * jag * unit(gaussian_filter1d(r.normal(0, 1, W), 2.2)))


def brush(sheet, strokes, seed=0, ss=3):
    """Lines of a fine, half-dry brush: going in quickly, thinning and thickening, trembling, skipping, leaving thin. Each stroke is a dict of pts, w (px), and optionally dark, tremble, breaks. Returns 0..1 at film size, which a caller scales to a density."""
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for i, st in enumerate(strokes):
        sd = seed + 13 * i
        q = dense(st["pts"], 0.7, wobble=st.get("tremble", 0.7), seed=sd)
        n = len(q)
        r = np.random.default_rng(sd + 1)
        u = np.linspace(0, 1, n)
        press = np.clip(0.8 + 0.34 * unit(gaussian_filter1d(r.normal(0, 1, n), max(n / 12, 2))), 0.3, 1.45)
        press = press * np.minimum(u / 0.05 + 0.25, 1) ** 0.6 * np.minimum((1 - u) / 0.2 + 0.08, 1) ** 0.8
        skip = unit(gaussian_filter1d(r.normal(0, 1, n), max(n / 24, 2))) < st.get("breaks", -1.2)
        val = int(255 * st.get("dark", 1.0))
        for k in range(n):
            if skip[k]:
                continue
            rad = 0.5 * st["w"] * press[k] * ss
            x, y = q[k] * ss
            d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=val)
    a = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255
    a = gaussian_filter(a, 0.45)
    yy, xx = grid_()
    dx, dy = sheet.warp
    a = map_coordinates(a, [yy + 0.55 * dy, xx + 0.55 * dx], order=1)
    dry = 1 - 0.42 * sm((band(H, W, 0.6, 2.4, seed + 7) - 0.5) / 1.1)
    return np.float32(a * dry)


# ---------------------------------------------------------------- k1_wide

def bristle(sheet, pts, width, seed, tone=1.6, dry=0.6, tremble=1.0):
    """One stroke of a broad hair brush on dry paper. The hairs run along the stroke; ink runs out along it, sooner at its edges and on hairs that had less, and the paper's tooth decides where each hair skips: solid where the brush was full, split into flying white where it was not. width is half widths along the stroke."""
    q = dense(pts, 1.0, wobble=tremble, seed=seed)
    n = len(q)
    u = np.linspace(0, 1, n)
    wv = np.interp(u, np.linspace(0, 1, len(width)), width).astype(np.float64)
    m = float(wv.max() + 4)
    out = np.zeros((H, W), np.float32)
    x0, x1 = int(max(q[:, 0].min() - m, 0)), int(min(q[:, 0].max() + m, W))
    y0, y1 = int(max(q[:, 1].min() - m, 0)), int(min(q[:, 1].max() + m, H))
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist, idx = cKDTree(q).query(np.stack([xx.ravel(), yy.ravel()], 1), distance_upper_bound=m + 2, workers=2)
    idx = np.minimum(idx, n - 1)
    dist = np.minimum(dist, m + 2)
    tan = np.gradient(q, axis=0)
    tan = tan / (np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-6)
    side = tan[idx, 0] * (yy.ravel() - q[idx, 1]) - tan[idx, 1] * (xx.ravel() - q[idx, 0])
    v = np.clip(side / wv[idx], -1, 1)
    r = np.random.default_rng(seed)
    rows, cols = 64, max(n, 8)
    hair = unit(gaussian_filter(r.normal(0, 1, (rows, cols)), (1.0, cols / 5)))
    fine = unit(gaussian_filter(r.normal(0, 1, (rows, cols)), (0.6, cols / 18)))
    short = unit(gaussian_filter(r.normal(0, 1, (rows, cols)), (0.7, max(cols / 60, 2))))
    patch = unit(gaussian_filter(r.normal(0, 1, (rows, cols)), (9, cols / 9)))
    vv, uu = (v + 1) / 2 * (rows - 1), idx.astype(np.float64)
    pick = lambda a: map_coordinates(a, [vv, uu], order=1, mode="nearest")
    T = 0.66 * pick(hair) + 0.40 * pick(fine) + 0.26 * pick(short) + 0.46 * pick(patch)
    tooth = sheet.tooth[y0:y1, x0:x1].ravel()
    fine_stroke = float(np.clip(wv.max() / 10, 0.15, 1))      # a thin line is not broken up by the paper the way a broad one is
    T = T + 0.42 * fine_stroke * tooth
    uf = idx / max(n - 1, 1)
    thr = -2.6 + 3.7 * dry * uf ** 1.25 + 1.0 * np.abs(v) ** 4
    cover = sm((T - thr) / 0.42)
    body = np.clip((wv[idx] - dist - 1.2 * fine_stroke * tooth) / (1.8 if fine_stroke == 1 else 1.1), 0, 1)
    dens = tone * (0.74 + 0.10 * T) * (1 - 0.30 * uf * dry)
    out[y0:y1, x0:x1] = np.float32((cover * body * dens).reshape(yy.shape))
    return out


def painted_twig(sheet, stick, grid=0.67):
    """The branch as it is painted, the way the man on the bank is: one wet stroke of thin grey along it, darks dropped in at the nodes and down the side away from the light, clean water here and there; when that is dry a half-dry brush pulled along it one length at a time, so it breaks where the branch is jointed. Nothing is lit and nothing is highlighted: the pale of it is only thin ink. Returns the ink and how far it covers what is under it."""
    seed = stick.get("seed", 40)
    sdf, limbs = twig_shape(stick["path"], stick["radius"], seed=seed, stubs=stick.get("stubs", ()))
    keep = sm(-(sdf - 1.5) / 3.5)
    r = np.random.default_rng(seed + 300)
    frame = []
    for limb in limbs:
        q = limb["q"]
        tan = np.gradient(q, axis=0)
        tan = tan / (np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-6)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        nrm = nrm * np.sign(nrm[:, 0] * 0.36 + nrm[:, 1] * 0.93 + 1e-6)[:, None]      # toward the side away from the light
        frame.append((q, limb["r"], tan, nrm, limb["nodes"]))
    wash = Wash(sheet, s=grid, reserve=1 - keep)
    wash.load(keep, 0.24)
    wash.run(8, evap=1e-4, note="branch")
    wash.load(keep * lumps(sheet, keep, seed + 1, level=0.30, soft=0.25, amount=0.42), 0.08, fine=0.46, coarse=0.09)
    wash.run(26, evap=1.2e-4, note="branch")
    q, rad, tan, nrm, nodes = frame[0]
    spots = list(nodes) + [int(v * (len(q) - 1)) for v in r.uniform(0.04, 0.96, 7)]
    for k, i in enumerate(spots):
        x, y = q[i] + nrm[i] * 0.34 * rad[i]
        big = i in nodes
        wash.dab(x, y, rad[i] * (1.5 if big else r.uniform(1.4, 2.8)), rad[i] * (0.8 if big else 0.5), float(np.arctan2(tan[i, 1], tan[i, 0])), water=0.15, fine=1.15 if big else 0.75, coarse=0.2 if big else 0.12, seed=seed + 10 + k, rough=0.5)
        wash.run(7, evap=1.2e-4)
    for k in range(3):
        i = int(r.uniform(0.15, 0.9) * (len(q) - 1))
        x, y = q[i] - nrm[i] * 0.25 * rad[i]
        wash.dab(x, y, rad[i] * 1.2, rad[i] * 0.45, float(np.arctan2(tan[i, 1], tan[i, 0])), water=0.24, seed=seed + 40 + k, rough=0.5)
        wash.run(8, evap=1.2e-4)
    wash.run(240, evap=7e-4, note="branch dries")
    fine, coarse = wash.density()
    ink = (fine + coarse) * keep
    # the half-dry brush, a length at a time: heavier on some lengths than others, no line drawn round it
    line = lambda fr, a, b, off, k=6: [tuple(fr[0][i] + fr[3][i] * off * fr[1][i]) for i in np.linspace(a, b, k).astype(int)]
    n = len(q)
    joints = [0] + list(nodes) + [n - 1]
    dry = np.zeros((H, W), np.float32)
    for j, (a, b) in enumerate(zip(joints[:-1], joints[1:])):
        span = b - a
        load = float(r.uniform(0.72, 1.3))
        # the pull down the shaded side starts past the joint and is lifted before the next, somewhere different each time
        a2, b2 = a + int(r.uniform(0.02, 0.14) * span), b - int(r.uniform(0.0, 0.16) * span)
        mid = (a2 + b2) // 2
        dry = dry + bristle(sheet, line(frame[0], a2, b2, float(r.uniform(0.28, 0.46))), (0.30 * rad[a2], 0.54 * rad[mid], 0.34 * rad[b2]), seed + 60 + j, tone=1.55 * load, dry=float(r.uniform(0.5, 0.74)), tremble=1.1)
        # one short, broken touch on the lit side, not on every length
        if r.uniform() < 0.7:
            c0 = a + int(r.uniform(0.1, 0.5) * span)
            c1 = min(c0 + int(r.uniform(0.2, 0.42) * span), b)
            dry = dry + bristle(sheet, line(frame[0], c0, c1, float(r.uniform(-0.8, -0.55)), k=4), (max(0.08 * rad[c0], 0.9), max(0.14 * rad[c0], 1.2), 0.8), seed + 70 + j, tone=1.15, dry=0.8, tremble=0.9)
        # a crack in the bark, wandering
        c0 = a + int(r.uniform(0.05, 0.45) * span)
        c1 = min(c0 + int(r.uniform(0.25, 0.5) * span), b)
        dry = dry + bristle(sheet, line(frame[0], c0, c1, float(r.uniform(-0.3, 0.2)), k=4), (max(0.05 * rad[c0], 0.8), max(0.09 * rad[c0], 1.0), 0.7), seed + 80 + j, tone=1.0, dry=0.8, tremble=1.0)
    # each node one dark touch into it from the shaded side, leaning, not a ring
    for j, i in enumerate(nodes):
        lean_ = float(r.uniform(-0.5, 0.5))
        reach = float(r.uniform(-0.45, 0.25))
        pts = [tuple(q[i] + nrm[i] * v * rad[i] + tan[i] * (lean_ * (v - 0.8)) * rad[i]) for v in (0.8, (0.8 + reach) / 2, reach)]
        dry = dry + bristle(sheet, pts, (max(0.17 * rad[i], 1.5), max(0.13 * rad[i], 1.3), max(0.05 * rad[i], 0.8)), seed + 100 + j, tone=1.3, dry=0.5, tremble=0.6)
    for j, fr in enumerate(frame[1:]):
        m = len(fr[0]) - 1
        dry = dry + bristle(sheet, line(fr, 0, m, 0.22, k=4), (0.7 * fr[1][0], 0.62 * fr[1][m // 2], 0.42 * fr[1][m]), seed + 120 + j, tone=1.65, dry=0.5, tremble=0.6)
    dry = dry * sm(-(sdf - 2.5) / 3.0)      # the snapped end stays snapped, and no touch sticks out past the edge like a thorn
    ink = gaussian_filter(ink + dry, 0.9)
    # it covers the ground only well inside its edge: covering it right up to the edge leaves a bright line round the branch
    return ink, sm(-(sdf + 1.5) / 3.0)


def cangjie_shape(x0, y0, sheet):
    """The man from behind and a little to his left, down on his heels, head bent, one long sleeve reaching to the ground. Distances, film pixels; the outline wanders as a wet edge does."""
    yy, xx = grid_()
    xx = xx + 3.0 * noise(sheet, 3.2, 120)
    yy = yy + 3.0 * noise(sheet, 3.2, 121)
    P = lambda a, b: (x0 + a, y0 + b)
    robe = _seg(xx, yy, P(-20, -222), P(22, -74), 70, 112)
    robe = _smin(robe, _seg(xx, yy, P(-92, -232), P(62, -226), 26, 34), 30)
    robe = _smin(robe, _seg(xx, yy, P(-104, -16), P(122, -18), 22, 30), 34)
    sleeve = _seg(xx, yy, P(-92, -232), P(-182, -146), 30, 33)
    sleeve = _smin(sleeve, _seg(xx, yy, P(-182, -146), P(-262, -70), 33, 25), 12)
    robe = _smin(robe, sleeve, 14)
    neck = _seg(xx, yy, P(-36, -248), P(-52, -292), 25, 21)
    c = P(-60, -322)
    a = np.radians(58)
    head = _egg(xx, yy, c, (np.cos(a), np.sin(a)), 37, 45)
    knot = np.hypot(xx - (x0 - 40), yy - (y0 - 374)) - 15
    wrist = _seg(xx, yy, P(-258, -70), P(-296, -38), 14, 13)
    hand = _seg(xx, yy, P(-296, -36), P(-324, -22), 14, 12)
    return dict(robe=robe, neck=neck, head=head, knot=knot, wrist=wrist, hand=hand, all=np.minimum.reduce([robe, neck, head, knot, wrist, hand]))


def cangjie(sheet, x0, y0, split=False, hands=True):
    """Painted the way a splashed-ink figure is: first the robe in one wet mass with the darks dropped into it and clean water after them, so its insides run and its edges hold; when that is dry, a few broad strokes of a half-dry brush down the folds, the hair nearly solid, and almost nothing else. No face: the head is turned away. Returns the ink and what is to hide of the ground behind him."""
    t0 = time.time()
    S = cangjie_shape(x0, y0, sheet)
    yy, xx = grid_()
    keep = sm(-(S["robe"] - 3) / 4)
    wash = Wash(sheet, s=0.67, reserve=1 - keep)
    # most of the back in a thin grey, the shoulder toward the light left nearly paper
    light = blob(xx, yy, x0 - 60, y0 - 232, 112, 74, 0.3)
    wash.load(keep, 0.26)
    wash.run(10, evap=1e-4)
    wash.load(lumps(sheet, keep * (1.15 - 1.1 * light), 131, level=0.42, soft=0.22, amount=0.34), 0.08, fine=0.44, coarse=0.08)
    wash.run(36, evap=1.2e-4, note="robe")
    D = lambda a, b, rx, ry, ang, water, fine, coarse, sd: wash.dab(x0 + a, y0 + b, rx, ry, ang, water=water, fine=fine, coarse=coarse, seed=sd, rough=0.5)
    for args in ((84, -128, 46, 118, 0.12, 0.24, 1.7, 0.30, 11), (44, -44, 96, 38, -0.05, 0.22, 1.5, 0.28, 12), (-96, -110, 40, 26, -0.72, 0.20, 1.5, 0.26, 13), (-226, -88, 30, 20, -0.74, 0.18, 1.2, 0.20, 14), (98, -18, 54, 18, 0.0, 0.20, 1.9, 0.32, 15), (-46, -34, 52, 16, 0.08, 0.20, 1.2, 0.20, 16), (-8, -168, 44, 20, 0.5, 0.12, 0.6, 0.10, 17), (-170, -150, 62, 24, -0.74, 0.16, 0.85, 0.14, 18)):
        D(*args)
        wash.run(16, evap=1.2e-4)
    wash.run(46, evap=2e-4, note="darks")
    for args in ((-16, -196, 26, 18, 0.2, 0.30, 0, 0, 21), (40, -118, 18, 26, 0.0, 0.30, 0, 0, 22), (-150, -152, 16, 12, -0.7, 0.26, 0, 0, 23), (-6, -74, 22, 14, 0.1, 0.28, 0, 0, 24)):
        D(*args)
        wash.run(18, evap=1.2e-4)
    wash.run(320, evap=6e-4, note="robe dries")
    fine, coarse = wash.density()
    wet = (fine + coarse) * sm(-(S["robe"] - 2) / 3)
    # the dry brush, over it
    P = lambda *pts: [(x0 + a, y0 + b) for a, b in pts]
    dry = np.zeros((H, W), np.float32)
    strokes = (
        (P((58, -254), (112, -196), (134, -120), (136, -56), (116, -8)), (9, 20, 24, 22, 11), 41, 1.9, 0.62),      # the far side of the back, one long pull
        (P((-20, -232), (6, -176), (20, -108), (8, -30)), (5, 13, 16, 8), 42, 1.25, 0.84),                          # the fold down the middle
        (P((-58, -150), (-68, -96), (-92, -30)), (5, 12, 7), 43, 1.35, 0.72),                                      # under the arm, the knee's shadow
        (P((-92, -254), (-150, -204), (-208, -150), (-276, -86)), (7, 14, 15, 7), 44, 1.75, 0.66),                  # along the top of the sleeve
        (P((-104, -180), (-158, -116), (-218, -88), (-268, -52)), (4, 11, 11, 5), 45, 1.45, 0.62),                  # and under it
        (P((-124, -10), (-44, 3), (40, -6), (150, -14)), (5, 11, 9, 6), 46, 1.65, 0.64),                            # the hem on the ground
        (P((-92, -250), (-48, -264), (4, -252)), (5, 11, 6), 47, 1.45, 0.55),                                      # the collar
    )
    for pts, w, sd, tone, d in strokes:
        dry = dry + bristle(sheet, pts, w, sd, tone=tone, dry=d)
    dry = gaussian_filter(dry, 0.7)      # a real bristle has no stair-steps along its edge when you lean in
    # the hair: strokes round the head, close enough to close up into one black mass that breaks only at its edge, a paler dry pull across it where the light is; the knot small on top; a sliver of paper left on the side the face is turned to
    cx, cy = x0 - 55, y0 - 322
    hair = np.zeros((H, W), np.float32)
    for k, (a0, a1, rad, w, sd) in enumerate(((-150, 40, 31, 13, 51), (-170, 70, 22, 15, 52), (-180, 90, 12, 15, 53), (-170, 150, 3, 12, 58), (-165, -40, 35, 7, 54), (0, 92, 35, 8, 55))):
        ang = np.radians(np.linspace(a0, a1, 7))
        hair = hair + bristle(sheet, [(cx + rad * np.cos(t), cy + rad * 1.16 * np.sin(t)) for t in ang], (w * 0.7, w, w * 0.8), sd, tone=2.5, dry=0.22, tremble=0.5)
    hair = np.minimum(hair, 2.5)
    ang = np.radians(np.linspace(-165, -60, 6))
    sheen = bristle(sheet, [(cx + 24 * np.cos(t), cy + 24 * 1.16 * np.sin(t)) for t in ang], (3, 9, 3), 59, tone=1.0, dry=0.86, tremble=0.5)
    hair = np.maximum(hair - 1.1 * sheen, 0.0)
    hair = hair + bristle(sheet, [(x0 - 50, y0 - 370), (x0 - 40, y0 - 379), (x0 - 29, y0 - 371)], (6, 9, 6), 56, tone=2.5, dry=0.15, tremble=0.3)
    hair = gaussian_filter(hair, 0.5)
    # the neck: a thin grey, darkening up into the hair
    neck_tone = sm(-S["neck"] / 2.5) * (0.10 + 0.55 * sm((y0 - 262 - yy) / 40))
    # skin is the paper, drawn with a thin wet line only where it needs one
    thin = (
        (P((-88, -294), (-84, -270), (-74, -252)), (1.6, 2.6, 2.0), 61, 1.5, 0.35),      # the two sides of the neck
        (P((-22, -292), (-14, -268), (-14, -246)), (1.8, 3.0, 2.2), 62, 1.7, 0.35),
        (P((-262, -80), (-290, -58), (-316, -42), (-330, -30)), (1.6, 2.2, 2.0, 1.4), 63, 1.5, 0.35),      # one thin line along the top of the wrist, no outline all round
    )
    fine_lines = np.zeros((H, W), np.float32)
    for pts, w, sd, tone, d in (thin if hands else thin[:2]):
        fine_lines = fine_lines + bristle(sheet, pts, w, sd, tone=tone, dry=d, tremble=0.6)
    # the wrist and the closed hand are one short loaded stroke of grey, not a drawn shape: the fingers two darker touches where they close on the branch
    fist = bristle(sheet, P((-264, -64), (-296, -42), (-334, -20)), (8, 12, 13), 64, tone=0.8, dry=0.16, tremble=0.5)
    fist = fist + bristle(sheet, P((-318, -40), (-326, -14)), (3.2, 4.2), 66, tone=1.5, dry=0.2, tremble=0.4)
    fist = fist + bristle(sheet, P((-304, -36), (-312, -10)), (2.8, 3.8), 67, tone=1.25, dry=0.3, tremble=0.4)
    hand_tone = gaussian_filter(fist, 0.6)
    twig_line = bristle(sheet, [(x0 - 332, y0 - 20), (x0 - 378, y0 - 2), (x0 - 392, y0 + 2), (x0 - 452, y0 + 26)], (3.4, 3.0, 3.2, 1.8), 58, tone=2.1, dry=0.2, tremble=0.5)
    twig_line = gaussian_filter(twig_line, 0.6)
    fill = sm(-((S["all"] if hands else np.minimum.reduce([S["robe"], S["neck"], S["head"], S["knot"]])) - 1) / 2.5)
    ink = wet + dry + hair + fine_lines + neck_tone + (hand_tone if hands else 0)
    print(f"  the man in {time.time() - t0:.0f}s", flush=True)
    if split:
        return ink, twig_line, fill
    return ink + twig_line, fill


def k1_fig(grid):
    """The man alone on a clean sheet, to judge him."""
    sheet = Sheet(seed=95)
    ink, fill = cangjie(sheet, 1340.0, 930.0)
    f = develop(sheet, np.zeros_like(ink), np.zeros_like(ink), line=np.minimum(ink, 2.7))
    save(f, "k1_fig")


def film_to_ground(xx, yy, yh, h, f, cx, yref):
    s = np.maximum((yy - yh) / h, 0.12)
    return (xx - 960) / s + cx, yref - f / s, s


def ground_relief(prints, yh=520.0, h=560.0, f=610.0, scale=1.0, marks=None):
    """The bank as a surface seen from a man's height: the same mud as the close shot, the same kind of prints, laid back in perspective."""
    t0 = time.time()
    mw, mh = 3600, 2200
    cx, yref = mw / 2, mh + 560.0
    m = Mud(seed=8, w=mw, h=mh)
    for i, (x, y, facing) in enumerate(prints):
        if x is None:
            continue
        s = (y - yh) / h
        m.print_(((x - 960) / s + cx, yref - f / s), facing, 300 + 40 * i, scale=scale)
    if marks is not None:
        marks(m, lambda x, y: ((x - 960) / ((y - yh) / h) + cx, yref - f / ((y - yh) / h)))
    s, sh, cav, _ = m.shade()
    yy, xx = grid_()
    X, Y, sc = film_to_ground(xx, yy, yh, h, f, cx, yref)
    look = lambda a: map_coordinates(gaussian_filter(a, 0.9), [Y, X], order=1, mode="nearest")
    relief, lift = ink_of(look(s), look(sh), look(cav))
    inside = sm((1700 - np.abs(X - cx)) / 200) * sm((Y - 30) / 120)
    print(f"  ground in {time.time() - t0:.0f}s", flush=True)
    return relief, lift, inside


def k1_wide(grid):
    sheet, D, coarse, water_lines, ink, twig, fill = k1_parts(grid)
    f = develop(sheet, D, coarse, line=1.5 * water_lines + np.minimum(ink + twig, 2.7))
    report(f, "k1_wide")
    save(f, "k1_wide")


def k1_parts(grid, hands=True, marks=None):
    sheet = Sheet(seed=95)
    yy, xx = grid_()
    x0, y0 = 1340.0, 930.0
    S = cangjie_shape(x0, y0, sheet)
    res = sm(-(S["all"] - 1) / 2.0)
    # the far bank: its top hard against a dry sky, its foot fading down wet paper into the river mist
    Rf = ridge([(0, 410), (300, 372), (640, 420), (980, 392), (1300, 440), (1920, 470)], 11, jag=8)
    sky = sm((Rf - yy) / 2.2)
    far = 1.2 * sm((yy - Rf + 24) / 8) * (1 - sm((yy - Rf - 16) / 56)) * (1 - sm((xx - 1020) / 520))
    far_wet = 1.1 * sm((yy - Rf + 24) / 8) * (1 - sm((yy - Rf - 120) / 90)) * (1 - sm((xx - 1200) / 480))
    ff, cf = cached("k1wf", lambda: paint_clouds(sheet, _clear(xx, yy, far), reserve=sky, seed=41, grid=grid, heavy=0.56, blooms=8, wet_env=_clear(xx, yy, far_wet)).density())
    # the near bank: its edge a hard line at the water, its ground fading toward the foot of the frame, a pool of dark round the man
    shore = ridge([(0, 664), (420, 640), (900, 650), (1360, 628), (1920, 646)], 3, jag=6)
    water = sm((shore - yy) / 2.2)
    land = sm((yy - shore + 24) / 8)
    pool = blob(xx, yy, 1370, 900, 400, 58) + 0.7 * blob(xx, yy, 1120, 930, 220, 40)
    g_env = land * (0.56 * (1 - sm((yy - shore - 40) / 170)) + 0.62 * pool + 0.46 * blob(xx, yy, 180, 800, 330, 110, -0.05))
    g_wet = land * (1 - sm((yy - 960) / 70)) * 1.1
    fg, cg = cached("k1wg", lambda: paint_clouds(sheet, g_env, reserve=np.maximum(water, res), seed=42, grid=grid, heavy=0.62, blooms=14, wet_env=g_wet).density())
    # the lobe out at the right edge goes: the column of big characters ends there
    edge = 1 - sm((xx - 1700) / 70) * (1 - sm((yy - 840) / 50))
    fine, coarse = ff + fg * edge, cf + cg * edge
    fine, coarse = tidy(fine, coarse)
    prints = [(872, 906, np.pi + 0.05), (560, 812, np.pi - 0.08), (352, 742, np.pi + 0.06), (214, 700, np.pi - 0.04)]
    # marks: what he has cut there by now, in the place of the print nearest him
    relief, lift, inside = ground_relief(prints if marks is None else [(None, None, None)] + prints[1:], marks=marks)
    present = sm((yy - shore - 6) / 26) * inside * (1 - 0.6 * sm((yy - 985) / 60))
    k = 0.8 * (1 - 0.5 * lift * present)
    fine, coarse = fine * k, coarse * k
    # his weight on the ground, soft, the same side as the mud's shadows
    fine = fine + 0.40 * blob(xx, yy, x0 + 24, y0 + 12, 190, 16)
    r = np.random.default_rng(16)
    lines = []
    # the water: bare paper, and a few long dry lines, left of him
    for _ in range(7):
        y, x, n = r.uniform(500, 618), r.uniform(70, 880), r.uniform(120, 460)
        lines.append(dict(pts=[(x, y), (x + n / 2, y + r.uniform(-1.5, 1.5)), (x + n, y + r.uniform(-2, 2))], w=r.uniform(1.2, 2.0), dark=r.uniform(0.22, 0.42), breaks=-0.4, tremble=0.4))
    water_lines = brush(sheet, lines, seed=60)
    ink, twig, fill = cangjie(sheet, x0, y0, split=True, hands=hands)
    if not hands:
        # the plate was painted round his wrist and hand: closed over from what is round it, or the paper left there would read as one
        hole = sm(-(np.minimum(S["wrist"], S["hand"]) - 6) / 4)
        have = gaussian_filter(1 - hole, 12.0) + 1e-3
        fine = fine * (1 - hole) + hole * gaussian_filter(fine * (1 - hole), 12.0) / have
        coarse = coarse * (1 - hole) + hole * gaussian_filter(coarse * (1 - hole), 12.0) / have
    fine, coarse = fine * (1 - fill), coarse * (1 - fill)
    D = fine + relief * present * (1 - fill)
    return sheet, D, coarse, water_lines, ink, twig, fill


# ---------------------------------------------------------------- k2

def grain(f, sheet, cx, cy, L, ang, seed, glaze=0.34):
    """One grain, drawn in place, a solid thing: plump, blunt at the germ, a little drawn out at the other end, one deep crease down its length, a hard dark side and a small hard light, lit from the same side as everything else."""
    R = int(L * 0.74) + 4
    x0, x1, y0, y1 = int(max(cx - R, 0)), int(min(cx + R + 1, W)), int(max(cy - R, 0)), int(min(cy + R + 1, H))
    if x1 <= x0 or y1 <= y0:
        return
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ca, sa = np.cos(ang), np.sin(ang)
    half = L / 2
    u = ((xx - cx) * ca + (yy - cy) * sa) / half
    v = (-(xx - cx) * sa + (yy - cy) * ca) / half
    bulge = 0.03 * np.sin(u * 3.1 + r.uniform(0, 6))
    inside = np.abs(u) < 1
    hw = 0.44 * np.clip(1 - np.abs(u) ** 2.3, 0, 1) ** 0.60 * (1 - 0.15 * u) * inside
    a = np.clip((hw - np.abs(v - bulge)) * half + 0.5, 0, 1) * inside
    vv = np.clip((v - bulge) / np.maximum(hw, 1e-3), -1, 1)
    z = np.sqrt(np.maximum(1 - vv * vv, 0))
    # the normal also turns along the length toward each end: a rounded body, not a tube
    nu = np.clip(u, -1, 1) ** 3 * 0.9
    nx, ny = -sa * vv + ca * nu, ca * vv + sa * nu
    nz = np.maximum(z * np.sqrt(np.maximum(1 - nu * nu, 0.05)), 0.02)
    lit = np.clip((nx * -0.30 + ny * -0.72 + nz * 0.62) / np.sqrt(nx * nx + ny * ny + nz * nz), 0, 1)
    tone = 0.26 + 1.75 * (1 - lit) ** 1.45
    off = r.uniform(0.04, 0.22) * (1 if r.uniform() < 0.5 else -1)
    fold = np.exp(-((vv - off) / 0.11) ** 2) * (1 - np.abs(u) ** 5)
    tone = tone + 1.35 * fold
    tone = tone + 0.8 * np.exp(-((u + 0.86) / 0.11) ** 2 - (vv / 0.5) ** 2)
    tone = tone + 0.05 * np.sin(vv * 11 + 2.0 * u + r.uniform(0, 6)) * (L > 30)
    # the outline, darkest on the side away from the light
    tone = tone + 0.9 * sm((np.abs(vv) - 0.80) / 0.20) * sm((L - 9) / 10)
    spec = np.exp(-((nx - -0.20) / 0.20) ** 2 - ((ny - -0.42) / 0.20) ** 2) * (L > 22)
    tone = tone * (1 - 0.72 * spec)
    if L >= 40:
        # a small brush of hairs at the far end only, on the near ones
        for i in range(7):
            b = r.uniform(-0.34, 0.34)
            n = r.uniform(0.05, 0.11) * L
            ax, ay = cx + ca * half * 0.94 + sa * b * L * 0.04, cy + sa * half * 0.94 - ca * b * L * 0.04
            bx, by = ax + np.cos(ang + b) * n, ay + np.sin(ang + b) * n
            hair = np.clip((0.75 - _seg(xx, yy, (ax, ay), (bx, by), 0.5, 0.25)) / 0.9, 0, 1)
            tone = np.where(hair > a, 1.1, tone)
            a = np.maximum(a, hair)
    if L > 60:
        # the nearest ones are a touch out of the plane
        a = gaussian_filter(a, L / 180)
    paper = sheet.rgb[y0:y1, x0:x1]
    obj = paper * np.exp(-np.clip(tone, 0, 2.6)[..., None] * INK_ABSORB - glaze * GRAIN_ABSORB)
    f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - a[..., None]) + obj * a[..., None]


def hill_strokes(R, spans, seed, w=4.0, dark=0.85):
    """Broken lines along the top of a hill, and a few dry strokes down its face."""
    r = np.random.default_rng(seed)
    out = []
    for a, b in spans:
        out.append(dict(pts=[(float(x), float(R[x]) + 1.0) for x in range(a, b, 26)], w=w, dark=dark, breaks=-0.75, tremble=0.6))
    for _ in range(sum(b - a for a, b in spans) // 240):
        a, b = spans[int(r.integers(len(spans)))]
        x = int(r.uniform(a, b))
        n = r.uniform(26, 96)
        lean = r.choice((-1, 1)) * r.uniform(8, 46)
        y = float(R[x]) + r.uniform(4, 16)
        out.append(dict(pts=[(x, y), (x + lean * 0.5, y + n * 0.55), (x + lean, y + n)], w=r.uniform(1.4, 2.6), dark=dark * r.uniform(0.32, 0.55), breaks=-0.5, tremble=0.6))
    return out


def grain_plan(heads):
    """Every grain of the frame as (x, y, length, angle, seed), far ones first. None of the big ones over a head."""
    r = np.random.default_rng(31)
    placed = []
    def free(x, y, L):
        if x > 1640 and y < 760:
            return False
        if L > 26:
            for c in heads:
                if np.hypot(x - c[0], y - c[1]) < 120 + L:
                    return False
        for px, py, pl in placed:
            if np.hypot(x - px, y - py) < 0.62 * (L + pl) + 8:
                return False
        return True
    plan = [(6, 72, 112, -40, 920), (18, 36, 54, -20, 920), (42, 19, 30, -10, 900), (84, 10, 16, -10, 880)]
    for count, lo, hi, ya, yb in plan:
        done = 0
        for _ in range(count * 80):
            if done >= count:
                break
            L = r.uniform(lo, hi)
            x, y = r.uniform(-10, W + 10), r.uniform(ya, yb)
            if not free(x, y, L):
                continue
            placed.append((x, y, L))
            done += 1
    out = []
    for x, y, L in sorted(placed, key=lambda p: p[2]):
        ang = np.pi / 2 + r.normal(0, 0.62)
        out.append((x, y, L, ang, int(r.integers(1 << 30))))
    return out


def k2_parts(grid):
    sheet = Sheet(seed=97)
    yy, xx = grid_()
    Ra = ridge([(0, 468), (300, 366), (560, 470), (820, 400), (1120, 300), (1420, 450), (1640, 530), (1920, 590)], 21, jag=13)
    Rb = ridge([(0, 610), (260, 560), (640, 646), (980, 568), (1330, 628), (1660, 668), (1920, 700)], 22, jag=11)
    env_a = 1.0 * sm((yy - Ra + 26) / 8) * (1 - sm((yy - Ra - 60) / 190))
    sky_a = sm((Ra - yy) / 2.4)
    fa, ca = cached("k2a", lambda: paint_clouds(sheet, _clear(xx, yy, env_a), reserve=sky_a, seed=50, grid=grid, heavy=0.50, blooms=12).density())
    # Where the grain keeps clear of a head was worked out for these three: kept, so that not one grain moves.
    heads = []
    people = []
    for x, s, flip, seed, far_s in ((470, 0.34, False, 3, 0.30), (900, 0.42, True, 5, 0.36), (1330, 0.30, False, 7, 0.25)):
        y = float(Rb[x]) + 84 + 606 * s
        c = Ghost(0.08, scale=s, at=(1100, 1010), to=(x + 220 * s * (-1 if flip else 1), y), seed=seed, flip=flip).parts["skull"]
        heads.append(c)
        # the same ghost as k4_wide, small, with its mouth still shut, its head where the old one's was
        people.append(Ghost(1.0, scale=far_s, at=(1098, 330), to=tuple(c), seed=seed, flip=flip, opens=False, form="phantom"))
    res = np.zeros_like(xx)
    near = np.maximum.reduce([0.95 * np.exp(-(np.maximum(g.sdf, 0) / 62) ** 2) * g.fade() for g in people])
    env_b = 1.22 * sm((yy - Rb + 26) / 8) * (1 - sm((yy - Rb - 130) / 150))
    sky_b = sm((Rb - yy) / 2.4)
    fb, cb = cached("k2b_shroud", lambda: paint_clouds(sheet, np.maximum(env_b, near * sm((yy - Rb + 26) / 8)), reserve=np.maximum(sky_b, res), seed=51, grid=grid, heavy=0.92, near=near * (1 - sky_b), blooms=20).density())
    spans_a = [(40, 540), (700, 1230), (1290, 1560)]
    spans_b = [(0, 330), (560, 790), (930, 1180), (1400, 1720)]
    lines = 1.25 * brush(sheet, hill_strokes(Ra, spans_a, 80, w=3.2, dark=0.62), seed=81) + 1.9 * brush(sheet, hill_strokes(Rb, spans_b, 82, w=4.2, dark=0.9), seed=83)
    return dict(sheet=sheet, Ra=Ra, Rb=Rb, sky_a=sky_a, sky_b=sky_b, fa=fa, ca=ca, fb=fb, cb=cb, people=people, heads=heads, lines=lines)


def k2_mist(sheet, m):
    return m * (0.88 + 0.12 * sm((noise(sheet, 3.4, 77) + 1.0) / 2.0))


K2_MIST = (0.66, 0.74, 0.54)      # pale: the first time you see them they should be hardly there


def k2(grid):
    P = k2_parts(grid)
    sheet = P["sheet"]
    fine, coarse = tidy(P["fa"] + P["fb"], P["ca"] + P["cb"])
    # the sky: the faintest bank of cloud far up on the left, or it is only a blank
    for g, mist in zip(P["people"], K2_MIST):
        fine, coarse = lift_figure(sheet, g, fine, coarse, edge=0.0, rim_amount=0.14, mist=k2_mist(sheet, mist))
    f = develop(sheet, fine, coarse, line=P["lines"])
    # the grain
    drawn = 0
    for x, y, L, ang, seed in grain_plan(P["heads"]):
        grain(f, sheet, x, y, L, ang, seed)
        drawn += 1
    print(f"  {drawn} grains", flush=True)
    report(f, "k2")
    save(f, "k2")


# ---------------------------------------------------------------- k4_wide

def trace(sheet, R, x, half, tall, strength, lean, seed):
    """A tall thin place where the ink is thinner, with no edge you could follow and no parts: what to multiply the ink by."""
    yy, xx = grid_()
    cy = float(R[x]) + 6 - tall / 2
    q = np.hypot((xx - x - lean * (yy - cy)) / half, (yy - cy) / (tall / 2))
    cut = strength * sm((1 + 0.28 * noise(sheet, 2.8, seed) - q) / 0.8) * (0.72 + 0.28 * sm((noise(sheet, 3.2, seed + 1) + 0.6) / 1.2))
    return 1 - cut * (1 - sm((yy - R[None, :]) / 4.0))


def k4_wide(grid):
    P = k4_parts(grid)
    sheet = P["sheet"]
    fine, coarse = P["fc"], P["cc"]
    for k in P["lifted"] + P["traces"]:
        fine, coarse = fine * k, coarse * k
    fine, coarse = open_mouths(sheet, fine, coarse, np.maximum.reduce([g.mouth() for g in P["ghosts"]]), grid, warp=0.25, soften=0.8)
    # the ridge goes on over them: whatever is lower than it is behind it
    fine, coarse = fine + P["over_f"], coarse + P["over_c"]
    fine, coarse = fine * P["floor"], coarse * P["floor"]
    fine, coarse = tidy(fine, coarse)
    f = develop(sheet, fine, coarse, line=P["lines"])
    report(f, "k4_wide")
    save(f, "k4_wide")


def k4_parts(grid):
    sheet = Sheet(seed=99)
    yy, xx = grid_()
    R = ridge([(0, 650), (260, 566), (520, 612), (860, 476), (1120, 520), (1400, 596), (1640, 632), (1920, 700)], 41, jag=9)
    # The two that howl: x, size, lean, how far down the ridge cuts them (0.31: at about the knee). The others are not figures at all, see trace().
    ghosts = []
    big = ((905, 0.36, 0.0, 0.37, 3, (64, 70)), (438, 0.27, 0.06, 0.36, 21, (56, 74)))      # the outline's wander differs, so the same egg would come out rounder on the small one: a sheet ghost
    for x, sc, lean, sunk, seed, head in big:
        ghosts.append(Ghost(1.0, scale=sc, at=(1100, 1010), to=(x + 200 * sc, float(R[x]) + sunk * 1000 * sc), seed=seed, lean=lean, form="phantom", behind=R, head=head))
    # x, half width, height, strength, lean: unevenly, a pair, one alone, three together, and none on the right. You take them for ghosts only because of the big ones.
    traces = ((196, 15, 64, 0.50, 0.05), (240, 22, 104, 0.72, -0.06), (676, 14, 58, 0.36, 0.08), (1168, 26, 128, 0.78, -0.05), (1216, 14, 62, 0.42, 0.06), (1266, 18, 88, 0.58, 0.03))
    near = np.maximum.reduce([0.9 * np.exp(-(np.maximum(g.sdf, 0) / (40 + 90 * g.scale)) ** 2) * g.fade() for g in ghosts])
    behind = sm((yy - (R - 190)) / 90) * (1 - sm((yy - R - 4) / 10))
    clouds = np.clip(0.95 * behind, 0, 1.2) + 0.5 * blob(xx, yy, 1120, 290, 400, 56, 0.05)
    clouds = _clear(xx, yy, np.clip(clouds, 0, 1.25)) * (1 - 0.92 * sm((yy - R - 2) / 6))
    near = _clear(xx, yy, near) * (1 - sm((yy - R - 2) / 6))
    # This plate is the mountain as it was approved, and it is kept: it was painted round the figures that used to stand here, so where those left bare paper the ink is first closed over from its surroundings.
    fc, cc = cached("k4wc7", lambda: paint_clouds(sheet, clouds, seed=60, grid=grid, heavy=0.92, near=near, blooms=20).density())
    for x, sc, lean, sunk, seed, arm in ((905, 0.36, 0.0, 0.31, 3, 0.90), (430, 0.27, 0.06, 0.33, 21, 1.00)):
        was = Ghost(1.0, scale=sc, at=(1100, 1010), to=(x + 200 * sc, float(R[x]) + sunk * 1000 * sc), seed=seed, arm=arm, lean=lean, behind=R)
        hole = sm((8.0 - was.sdf) / 4.0) * sm((was.fade_all() - 0.05) / 0.15)
        have = gaussian_filter(1 - hole, 14.0) + 1e-3
        fc = fc * (1 - hole) + hole * gaussian_filter(fc * (1 - hole), 14.0) / have
        cc = cc * (1 - hole) + hole * gaussian_filter(cc * (1 - hole), 14.0) / have
    env_r = 1.18 * sm((yy - R + 24) / 8) * (1 - sm((yy - R - 70) / 170))
    sky = sm((R - yy) / 2.2)
    fr, cr = cached("k4wr", lambda: paint_clouds(sheet, env_r, reserve=sky, seed=61, grid=grid, heavy=0.82, blooms=14).density())
    # the moon is not painted: a very thin wet grey round a place it did not reach, its edge lost for most of the way round, and a wisp over it
    moon_c, moon_r = (640.0, 176.0), 60.0
    dist = np.hypot(xx - moon_c[0], yy - moon_c[1])
    halo = 1.15 * np.exp(-(np.maximum(dist - moon_r * 0.8, 0) / 210) ** 2) * (1 - 0.35 * sm((dist - 190) / 120))
    halo_wet = 1.2 * np.exp(-(np.maximum(dist - moon_r * 0.8, 0) / 300) ** 2)
    fm, cm = cached("k4wm", lambda: paint_clouds(sheet, halo, seed=62, grid=grid, heavy=0.17, blooms=8, wet_env=halo_wet).density())
    wisp = 1.0 * blob(xx, yy, 700, 206, 300, 15, -0.04) + 0.8 * blob(xx, yy, 500, 138, 190, 11, 0.03)
    fw, cw = cached("k4ww", lambda: paint_clouds(sheet, wisp, seed=63, grid=grid, heavy=0.17, blooms=3, wet_env=1.1 * wisp).density())
    # the far band on the left: laid on paper wetted well past it, so it has no end you can point at
    band = 1.05 * blob(xx, yy, 60, 330, 300, 44, -0.05)
    band_wet = 1.25 * blob(xx, yy, 80, 332, 560, 96, -0.05)
    fb, cb = cached("k4wb", lambda: paint_clouds(sheet, band, seed=64, grid=grid, heavy=0.34, blooms=4, wet_env=band_wet).density())
    fw, cw = 0.30 * fw + 0.62 * fb, 0.30 * cw + 0.62 * cb
    fm, cm = 0.34 * fm, 0.34 * cm
    soft = 16 + 46 * sm((noise(sheet, 3.4, 45) + 0.4) / 1.0)      # how lost the edge is, going round: most of the way it is hardly an edge
    disc = sm((moon_r + 3.0 * noise(sheet, 3.0, 44) - dist) / soft + 0.5)
    fm, cm = fm * (1 - 0.9 * disc), cm * (1 - 0.9 * disc)
    # the ridge's wash stops short of the valley floor: the valley is empty
    valley = 1 - sm((yy - R - 180) / 50)
    fr, cr = fr * valley, cr * valley
    one = np.ones_like(fc)
    lifted = [lift_figure(sheet, g, one, one, edge=0.0, rim_amount=0.2, mist=0.84 + 0.14 * sm((noise(sheet, 3.4, 77) + 1.0) / 2.0))[0] for g in ghosts]
    trace_k = [trace(sheet, R, *tr, seed=140 + 2 * i) for i, tr in enumerate(traces)]
    # nothing below the ridge's own wash: the valley is empty. And one dark drop by the big one's head goes, it reads as a second moon
    floor = (1 - sm((yy - R - 170) / 50)) * (1 - 0.82 * blob(xx, yy, 1026, 290, 44, 40))
    lines = 1.4 * brush(sheet, hill_strokes(R, [(0, 470), (600, 1180), (1290, 1900)], 90, w=3.2, dark=0.8), seed=91)
    return dict(sheet=sheet, R=R, ghosts=ghosts, traces_at=traces, fc=fc, cc=cc, lifted=lifted, traces=trace_k, over_f=fr + fm + fw, over_c=cr + cm + cw, floor=floor, lines=lines)

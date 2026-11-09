"""The nine key frames of the opening myth 鬼夜哭. Stills only: this round nothing moves and nothing goes into the film.

    python film_myth.py k3_look [grid]         # one frame; grid is the size of the water's grid against the film, 0.67 is the finished plate, 0.5 a quicker look
    python film_myth.py k4_cry
    python film_myth.py all

Frames go to ink/film_out/film/myth/keys/. Film time of each frame and its subtitle come from ink_subs.py.
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from film_myth_ghost import Ghost
from film_myth_mud import CHAR_AT, Mud, TRACK_AT, crop_to_film, dense, g2w, ink_of, twig_shadow, vision_strokes
from film_myth_wash import INK_ABSORB, OUT, Sheet, Wash, blob, develop, lumps, noise, save, sm, touches, band


def _clear(xx, yy, env):
    """The top right is the big vertical line's, and it reads best on bare paper."""
    return env * (1 - 0.95 * blob(xx, yy, 1790, 250, 300, 380))


def paint_clouds(sheet, env, reserve=None, seed=0, grid=0.67, heavy=1.0, near=None, blooms=20, wet_env=None):
    """The clouds, in the order a wet painting has to be done: clean water; a first wash over the whole cloud, so there are no holes in it; touches of mid and dark, with a pause to half dry between, so a new touch keeps a hard edge where the paper has dried and runs where it has not; clean water dropped into the damp ink for the blooms; then it dries."""
    t0 = time.time()
    wash = Wash(sheet, s=grid, reserve=reserve)
    # wet_env, where it is given, is paper wetted further than the ink is laid: there the ink fades out instead of stopping at a hard edge
    wenv = env if wet_env is None else wet_env
    wet = lumps(sheet, wenv, 21 + seed, level=0.30, soft=0.16) * sm((wenv - 0.05) / 0.2)
    wash.load(wet, 0.28)
    wash.run(40, evap=1e-4, note="wet")
    base = lumps(sheet, env, 22 + seed, level=0.42, soft=0.24, amount=0.42) * sm((env - 0.12) / 0.22)
    wash.load(base, 0.14, fine=0.42 * heavy, coarse=0.07 * heavy)
    if near is not None:
        wash.load(lumps(sheet, near * 1.3, 25 + seed, level=0.45, soft=0.3, amount=0.3), 0.10, fine=0.55 * heavy, coarse=0.08 * heavy)
    wash.run(150, evap=2e-4, note="first wash")
    where = np.clip(env - 0.25, 0, 1) * base
    touches(wash, sheet, where, 26, 5 + seed, rx=(90, 240), ry=(30, 70), water=0.16, fine=0.22 * heavy, coarse=0.05 * heavy, every=12, spread=0.14)
    wash.run(220, evap=4e-4, note="half dries")
    where = np.clip(env - 0.45, 0, 1) * base
    touches(wash, sheet, where, 30, 6 + seed, rx=(36, 150), ry=(18, 54), water=0.22, fine=0.75 * heavy, coarse=0.14 * heavy, every=14, spread=0.16)
    wash.run(160, evap=4e-4, note="half dries")
    where = np.clip(env - 0.72, 0, 1) * base if near is None else np.clip(np.maximum(env - 0.72, near), 0, 1) * base
    touches(wash, sheet, where, 30, 8 + seed, rx=(18, 84), ry=(12, 36), water=0.20, fine=0.95 * heavy, coarse=0.2 * heavy, every=14)
    wash.run(50, evap=2e-4, note="settle")
    touches(wash, sheet, np.clip(env - 0.32, 0, 1) * base, blooms, 7 + seed, rx=(14, 70), ry=(10, 46), water=0.46, every=26, fine=0, coarse=0)
    wash.run(900, evap=5e-4, note="dry")
    print(f"  painted in {time.time() - t0:.0f}s", flush=True)
    return wash


def report(f, name):
    L = f.mean(2) / 255
    print(f"  {name}: mean {L.mean():.2f}, bright paper {(L > 0.82).mean():.2f}, dark {(L < 0.35).mean():.2f}", flush=True)


def lift_figure(sheet, g, fine, coarse, edge=0.40, rim_amount=0.48, mist=None):
    """The figure lifted out of the finished ink: its shape, the thin bright ring round it, a tide line along the stretches where its edge is hard. A breath of the wash stays on the body, so it is not a hole cut in the painting."""
    cut, rim, how = g.carve()
    mist = 0.80 + 0.22 * sm((noise(sheet, 3.4, 77) + 1.0) / 2.0) if mist is None else mist
    keep = (1 - cut * mist) * (1 - rim_amount * rim * (1 - cut))
    line = np.exp(-(g.sdf / (2.6 * g.scale)) ** 2) * (1 - how) * (1 - cut) * edge * g.faded
    k = keep * (1 + line)
    return fine * k, coarse * k


def qing_wash(sheet, g, seed=0):
    """The thinnest blue-green over the body, in patches: a stain, not a fill."""
    cut, _, _ = g.carve()
    v = np.clip(cut * (0.45 + 0.7 * noise(sheet, 3.2, 61 + seed)), 0, 1)
    return np.float32(gaussian_filter(v, 4.0) * 0.06)


def tidy(fine, coarse, level=0.06, smallest=9000):
    """Stray islands of ink out on the bare paper, away from any cloud, are taken back off: the clouds have long arms but nothing floats alone."""
    keep = tidy_keep(fine, coarse, level, smallest)
    return fine * keep, coarse * keep


def tidy_keep(fine, coarse, level=0.06, smallest=9000):
    from scipy.ndimage import label
    D = gaussian_filter(fine + coarse, 3.0)
    lab, n = label(D > level)
    if n < 2:
        return np.ones_like(D)
    area = np.bincount(lab.ravel())
    area[0] = 0
    big = np.float32(area[lab] >= smallest)
    return np.maximum(big, sm(gaussian_filter(big, 10.0) / 0.1))


def near_ghost(lift):
    """The figure of the close shots, k3_look and k4_cry: one outline, only lift changes."""
    return Ghost(lift, scale=1.3, at=(1100, 1010), to=(1060, 1120), form="shroud")


def retouch(name, lift, sheet_seed=91):
    """The painted plate of a ghost frame again, with the strays off, nothing else changed."""
    sheet = Sheet(seed=sheet_seed)
    z = np.load(os.path.join(OUT, name + "_ink.npz"))
    g = near_ghost(lift)
    fine, coarse = tidy(np.float32(z["fine"]), np.float32(z["coarse"]))
    f = develop(sheet, fine, coarse, qing=qing_wash(sheet, g))
    report(f, name)
    save(f, name)


def open_mouths(sheet, fine, coarse, hole, grid=0.67, warp=0.8, soften=0.0, seep=0.0):
    """A mouth is ink dropped wet on the bare paper of the head: darkest in the middle, seeping a little way out along the fibres, nothing to hold it in."""
    if not hole.any():
        return fine, coarse
    inner, mf, mc, black = mouth_layers(sheet, hole, grid, warp, soften, seep)
    return fine * (1 - 0.9 * inner) + mf + black, coarse * (1 - 0.9 * inner) + mc


def mouth_layers(sheet, hole, grid=0.67, warp=0.8, soften=0.0, seep=0.0):
    yy, xx = np.mgrid[0:sheet.h, 0:sheet.w].astype(np.float32)
    dx, dy = sheet.warp
    hole = map_coordinates(hole, [yy + warp * dy, xx + warp * dx], order=1)
    wash = Wash(sheet, s=grid)
    wash.load(hole, 0.32 if seep else 0.20, fine=1.5, coarse=0.3)
    wash.run(26, evap=2e-4, note="mouth")
    wash.run(220, evap=8e-4, note="mouth dries")
    mf, mc = wash.density()
    if soften:      # a mouth a few pixels wide: at this size the paper's fibres would give it spurs it could not have from this far away
        mf, mc = gaussian_filter(mf, soften), gaussian_filter(mc, soften)
    inner = gaussian_filter(hole, seep or 0.7)      # with seep the black has no edge of its own: it thins out into what has run
    return inner, mf, mc, inner * 2.0 * (1 + 0.1 * noise(sheet, 2.6, 71))


def shroud_plate(sheet, grid, seed=0):
    """k3_look and k4_cry are one shot, cut away from and come back to: the same dry ink, in which only the figure moves. So one painting. The water is kept off only what is paper in both poses; the rest is lifted off afterwards, as carve() does for any pose between."""
    path = os.path.join(OUT, "_wash_shroud_plate.npz")
    if os.path.exists(path) and os.environ.get("MYTH_FRESH") != "1":
        z = np.load(path)
        return np.float32(z["fine"]), np.float32(z["coarse"])
    look, cry = near_ghost(0.22), near_ghost(1.0)
    yy, xx = np.mgrid[0:sheet.h, 0:sheet.w].astype(np.float32)
    both = SimpleNamespace(sdf=np.minimum(look.sdf, cry.sdf))
    near = np.maximum.reduce([0.9 * np.exp(-(np.maximum(g.sdf, 0) / 100) ** 2) * g.fade() for g in (look, cry)])
    foot = 1 - sm((yy - 880 + 70 * noise(sheet, 2.6, 57)) / 80)      # the cloud stops where it stops, not along a line
    fine, coarse = paint_clouds(sheet, _clear(xx, yy, env_cry(xx, yy, both)) * foot, reserve=np.minimum(look.reserve(), cry.reserve()), seed=seed, grid=grid, near=near).density()
    # a dark drop out on its own beside the figure looks back at you
    k = 1 - 0.7 * blob(xx, yy, 790, 668, 34, 34)
    fine, coarse = fine * k, coarse * k
    np.savez_compressed(path, fine=np.float16(fine), coarse=np.float16(coarse))
    return fine, coarse


def mend_plate(sheet, fine, coarse):
    """Three things the water did that look like something else, taken back into the wash round them: a short dark scratch by the shoulder, a ring beside the hem, and two pale ovals off the end of the sleeve that read as more heads."""
    yy, xx = np.mgrid[0:sheet.h, 0:sheet.w].astype(np.float32)
    for cx, cy, rx, ry, much, sigma in ((1198, 415, 16, 40, 1.0, 12.0), (790, 668, 44, 44, 1.0, 24.0), (725, 395, 80, 115, 0.72, 42.0)):
        m = much * blob(xx, yy, cx, cy, rx, ry)
        fine = fine * (1 - m) + gaussian_filter(fine, sigma) * m
        coarse = coarse * (1 - m) + gaussian_filter(coarse, sigma) * m
    return fine, coarse


def ghost_frame(name, lift, grid, seed=0, sheet_seed=91):
    sheet = Sheet(seed=sheet_seed)
    g = near_ghost(lift)
    fine, coarse = mend_plate(sheet, *shroud_plate(sheet, grid, seed))
    fine, coarse = lift_figure(sheet, g, fine, coarse)
    fine, coarse = tidy(fine, coarse)
    fine, coarse = open_mouths(sheet, fine, coarse, g.mouth(), grid, warp=0.3, soften=1.6, seep=5.0)
    f = develop(sheet, fine, coarse, qing=qing_wash(sheet, g, seed))
    report(f, name)
    save(f, name)
    np.savez_compressed(os.path.join(OUT, name + "_ink.npz"), fine=np.float16(fine), coarse=np.float16(coarse))


def env_cry(xx, yy, g):
    near = np.exp(-(np.maximum(g.sdf, 0) / 190) ** 2)
    e = 1.05 * near
    e = e + 0.85 * blob(xx, yy, 720, 190, 480, 80, -0.04)
    e = e + 0.75 * blob(xx, yy, 380, 800, 400, 80, 0.06) + 0.8 * blob(xx, yy, 1470, 660, 380, 110, 0.1)
    return np.clip(e, 0, 1.25)


# ---------------------------------------------------------------- the mud shots

CAM_K1 = dict(x0=320, y0=150, zoom=1.0)     # the print and the character side by side
CAM_K3 = dict(x0=538, y0=150, zoom=1.0)     # the character left of centre, room on its right for the vision, the twig below
CAM_K5 = dict(x0=574, y0=284, zoom=1.25)    # closer, the character alone


def w2f(p, cam):
    return ((p[0] - cam["x0"]) * cam["zoom"], (p[1] - cam["y0"]) * cam["zoom"])


def mud_wash(sheet, cam, name, center, radii, grid=0.67, seed=30, heavy=0.30):
    """A pale wash over the mud, the unpainted paper round it. Kept, since the same camera is used twice."""
    path = os.path.join(OUT, f"_wash_{name}.npz")
    yy, xx = np.mgrid[0:sheet.h, 0:sheet.w].astype(np.float32)
    env = 1.5 * blob(xx, yy, center[0], center[1], radii[0], radii[1], 0.02)
    if os.path.exists(path) and os.environ.get("MYTH_FRESH") != "1":
        z = np.load(path)
        return np.float32(z["fine"]), np.float32(z["coarse"]), env
    wash = paint_clouds(sheet, env, seed=seed, grid=grid, heavy=heavy, blooms=14)
    fine, coarse = wash.density()
    np.savez_compressed(path, fine=np.float16(fine), coarse=np.float16(coarse))
    return fine, coarse, env


def over(f, a, d, sheet):
    """An object over the frame: its alpha and its own ink."""
    obj = sheet.rgb * np.exp(-d[..., None] * INK_ABSORB)
    return f * (1 - a[..., None]) + obj * a[..., None]


def relief_film(m, cam):
    s, sh, cav, _ = m.shade()
    z = cam["zoom"]
    return crop_to_film(s, cam["x0"], cam["y0"], z), crop_to_film(sh, cam["x0"], cam["y0"], z), crop_to_film(cav, cam["x0"], cam["y0"], z), crop_to_film(m.broad, cam["x0"], cam["y0"], z)


def mud_frame(name, cam, center, radii, wash_name, stick=None, stick_lift=None, vision=False, print_=True):
    t0 = time.time()
    sheet = Sheet(seed=93)
    m = Mud()
    if print_:
        m.print_(TRACK_AT, np.pi, 300)
    m.char()
    grown_q = None
    if vision:
        grown_q = m.grown()
    s, sh, cav, broad = relief_film(m, cam)
    fine, coarse, env = mud_wash(sheet, cam, wash_name, center, radii)
    present = sm((env - 0.24 + 0.14 * noise(sheet, 3.2, 81)) / 0.42)
    relief, lift = ink_of(s, sh, cav)
    # the ground's big slow swells, a soft tone in the wash: it is earth and not a sheet
    swell = np.clip(0.16 - broad * 2.4, 0, 0.5)
    relief = relief + swell
    D_wash = fine + coarse
    yy, xx = np.mgrid[0:sheet.h, 0:sheet.w].astype(np.float32)
    extra = 0
    if vision:
        extra, relief, lift = wet_strokes(sheet, cam, relief, lift, s)
    D = 0.55 * D_wash * (1 - 0.6 * lift) + relief * present + extra
    if stick is not None:
        from film_myth_wide import painted_twig
        ink, keep = painted_twig(sheet, stick)
        if stick.get("blur"):
            # the far end is nearer the eye and goes a little soft
            b0, b1 = stick["blur"]
            along = np.clip((xx - b0) / (b1 - b0), 0, 1)
            ink = ink * (1 - along) + gaussian_filter(ink, 3.0) * along
        # a little damp under it, not a cast shadow: it lies in the same ink as the ground
        under = gaussian_filter(twig_shadow(stick["path"], stick["radius"], stick_lift, light=(0.30, 0.78)), 3.0)
        D = D * (1 - 0.5 * keep) + 0.28 * under * (0.4 + 0.6 * present) * (1 - keep) + ink
    f = develop(sheet, D, np.zeros_like(D), grain=0)
    report(f, name)
    save(f, name)
    print(f"  {name} in {time.time() - t0:.0f}s", flush=True)


def wet_strokes(sheet, cam, relief, lift, s):
    """The vision: the character has gone wet. Water stands in its cuts and runs out into the mud carrying the ink of the wet mud with it; the cuts' own edges have softened. On the far side of the spine the new strokes come, wet too and solid: loaded where the brush came down, stopping on an uneven beat, and where it leaves the wet gives out and only the dry hairs go on, scattering."""
    old, strokes, fields = vision_fields(sheet, cam)
    new = np.maximum.reduce([f[0] for f in fields])
    wash = Wash(sheet, s=0.67)
    # the mud round the character is damp in patches, so what runs out of the cuts runs far here and hardly at all there
    damp = gaussian_filter(old, 26)
    damp = damp / damp.max()
    wash.load(lumps(sheet, damp * 1.5, 91, level=0.45, soft=0.25, amount=0.55), 0.30)
    wash.run(20, evap=1e-4, note="damp")
    wash.load(old, 0.46, fine=2.1, coarse=0.5)
    wash.run(420, evap=1.2e-4, note="running out of the cuts")
    wash.load(old, 0.12, fine=0.6, coarse=0.1)
    wash.run(60, evap=2e-4, note="settle")
    # the strokes go in a length at a time, each given a while to bleed before the next; the wet gives out before the end of each
    for k, (st, (mask, U, q, r)) in enumerate(zip(strokes, fields)):
        n = len(q)
        if k:
            wash.run(120, evap=5e-4, note="the brush stops")
        mask = mask * (0.86 + 0.14 * np.clip(noise(sheet, 3.0, 110 + k), -1.5, 1.5))
        wet = mask * (1 - sm((U - 0.84) / 0.13))
        beats = [0.0] + [at for at, _, _ in st["stops"]] + [1.0]
        for i, (a, b) in enumerate(zip(beats[:-1], beats[1:])):
            part = wet * np.clip((U - a) * n / 3.0 + 0.5, 0, 1) * np.clip((b - U) * n / 3.0 + 0.5, 0, 1)
            wash.load(part, 0.24, fine=1.55, coarse=0.26)
            if i == 0:
                wash.load(mask * np.exp(-(U * n / 26) ** 2), 0.05, fine=1.0, coarse=0.16)      # where the brush came down
            wash.run(int(16 + 90 * (b - a)), evap=2.4e-4, note=f"stroke {k} length {i}")
            if i < len(st["stops"]):
                at, much, size = st["stops"][i]
                wash.load(mask * np.exp(-((U - at) * n / (15 * size)) ** 2), 0.05, fine=0.85 * much, coarse=0.12 * much)
    wash.run(500, evap=7e-4, note="dry")
    fine, coarse = wash.density()
    extra = fine + coarse
    # where it leaves: dry hairs only, each giving out in its own place
    for k, (mask, U, q, r) in enumerate(fields):
        extra = extra + vision_hair(sheet, k, mask, U, q, r)
    # the same light in the water standing in the old cuts, a broken line of it
    glint = sm((s - 0.32) / 0.2) * gaussian_filter(old, 2) * sm((noise(sheet, 2.6, 96) + 0.2) / 0.8)
    extra = extra * (1 - 0.75 * glint)
    # the cuts' relief has gone soft where the water stands
    near = gaussian_filter(np.maximum(old, new), 12)
    near = np.clip(near / near.max() * 2.0, 0, 1)
    relief = relief * (1 - near) + gaussian_filter(relief, 2.5) * near
    lift = lift * (1 - 0.7 * near)
    return extra, relief, lift


def vision_hair(sheet, k, mask, U, q, r):
    from film_myth_wide import bristle
    i0 = int(0.80 * len(q))
    pick = np.linspace(i0, len(q) - 1, 6).astype(int)
    hairs = bristle(sheet, [tuple(q[i]) for i in pick], (r[i0] * 0.9, r[i0] * 1.1, r[i0] * 1.35), 130 + k, tone=1.5, dry=1.0, tremble=0.4)
    return gaussian_filter(hairs * sm((U - 0.80) / 0.07), 0.7)      # they come in from nothing inside the wet, and splay as they go: no point


def vision_fields(sheet, cam):
    """The old cuts as a mask, the new strokes, and each new stroke as (mask, how far along each pixel is, its line in film pixels, its half widths)."""
    from PIL import ImageDraw
    from film_myth_mud import FOUR
    W, H = sheet.w, sheet.h
    z = cam["zoom"]
    ss = 2

    def to_film(p):
        a, b = g2w(p)
        return (a - cam["x0"]) * z, (b - cam["y0"]) * z

    def stroke_mask(lines, width):
        im = Image.new("L", (W * ss, H * ss), 0)
        d = ImageDraw.Draw(im)
        for pts in lines:
            q = dense([g2w(p) for p in pts], 2.0)
            xy = [((a - cam["x0"]) * z * ss, (b - cam["y0"]) * z * ss) for a, b in q]
            d.line(xy, fill=255, width=int(width * ss * z))
            for x, y in (xy[0], xy[-1]):
                r = width * ss * z / 2
                d.ellipse((x - r, y - r, x + r, y + r), fill=255)
        return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255

    def stroke_field(st):
        """A stroke as a mask, with how far along it each pixel is."""
        q = (st["q"] - np.array((cam["x0"], cam["y0"]))) * z
        r = st["w"] * z
        m = float(r.max() + 6)
        x0, x1 = int(max(q[:, 0].min() - m, 0)), int(min(q[:, 0].max() + m, W))
        y0, y1 = int(max(q[:, 1].min() - m, 0)), int(min(q[:, 1].max() + m, H))
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist, idx = cKDTree(q).query(np.stack([gx.ravel(), gy.ravel()], 1), distance_upper_bound=m + 2, workers=2)
        idx = np.minimum(idx, len(q) - 1)
        dist = np.minimum(dist, m + 2)
        mask, U = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        mask[y0:y1, x0:x1] = np.float32(np.clip((r[idx] - dist) / 2.0 + 0.5, 0, 1).reshape(gy.shape))
        U[y0:y1, x0:x1] = np.float32(st["u"][idx].reshape(gy.shape))
        return mask, U, q, r

    old = stroke_mask([p for _, p in FOUR], 20)
    strokes = vision_strokes()
    return old, strokes, [stroke_field(st) for st in strokes]


# The branch on the mud, film pixels of CAM_K3. It lies where it was put down, under the character, its worn point to the left. Each vertex is a node, where it changes direction.
STICK_DOWN = dict(path=[(752, 868), (872, 842), (1000, 852), (1118, 826), (1262, 826), (1372, 790), (1468, 772)], radius=[8.0, 11.0, 13.0, 15.0, 17.5, 20.0, 21.0], stubs=((0.42, -0.85, 60, 0.62), (0.74, 0.8, 54, 0.58)), seed=44)


def stick_k5():
    cam3, cam5 = CAM_K3, CAM_K5
    tr = lambda p: ((p[0] + cam3["x0"] - cam5["x0"]) * cam5["zoom"], (p[1] + cam3["y0"] - cam5["y0"]) * cam5["zoom"])
    k = cam5["zoom"]
    return dict(path=[tr(p) for p in STICK_DOWN["path"]], radius=[r * k for r in STICK_DOWN["radius"]], stubs=tuple((a, b, c * k, e) for a, b, c, e in STICK_DOWN["stubs"]), seed=44)


def stick_k1():
    """The same branch in the hand, off the frame: the point on the end of the last stroke, the thick end coming toward the eye."""
    tip = w2f(g2w((100.5, 82.4)), CAM_K1)
    return dict(path=[tip, (tip[0] + 84, tip[1] + 92), (tip[0] + 214, tip[1] + 160), (tip[0] + 318, tip[1] + 284), (tip[0] + 482, tip[1] + 368), (tip[0] + 604, tip[1] + 508), (tip[0] + 830, tip[1] + 670)], radius=[8.5, 12.5, 16.0, 19.5, 24.0, 29.0, 36.0], stubs=((0.36, -0.9, 74, 0.60), (0.63, 0.82, 92, 0.56)), seed=44, blur=(1500, 1800))


def k1_close(grid):
    mud_frame("k1_close", CAM_K1, (850, 520), (980, 470), "k1", stick=stick_k1(), stick_lift=[0.0, 24, 62, 110, 170])


def k3_back(grid):
    mud_frame("k3_back", CAM_K3, (860, 500), (860, 460), "k3", stick=STICK_DOWN, stick_lift=[1, 3, 3, 4, 3], print_=False)


def k3_vision(grid):
    mud_frame("k3_vision", CAM_K3, (860, 500), (860, 460), "k3", stick=STICK_DOWN, stick_lift=[1, 3, 3, 4, 3], vision=True, print_=False)


def k5(grid):
    mud_frame("k5", CAM_K5, (960, 440), (700, 420), "k5", stick=stick_k5(), stick_lift=[1, 3, 3, 4, 3], print_=False)


FRAMES = dict(k3_look=lambda s: ghost_frame("k3_look", 0.22, s), k4_cry=lambda s: ghost_frame("k4_cry", 1.0, s),
              k1_close=k1_close, k3_back=k3_back, k3_vision=k3_vision, k5=k5)


def wide(name):
    def go(grid):
        import film_myth_wide
        getattr(film_myth_wide, name)(grid)
    return go


FRAMES.update(k1_wide=wide("k1_wide"), k2=wide("k2"), k4_wide=wide("k4_wide"), k1_fig=wide("k1_fig"))

# Film seconds of each frame, for the versions with the subtitle on.
TIMES = dict(k1_wide=7.0, k1_close=10.5, k2=15.5, k3_look=18.0, k3_vision=20.5, k3_back=22.5, k4_cry=26.0, k4_wide=29.0, k5=32.0)


def with_subtitle(names):
    """The frames in film order, each with its line from ink_subs.py, at the moment that line is whole if the frame's own second comes while it is still appearing. Where the storyboard has no line at that second (17 to 23: the ghost sees) the frame is written unchanged, so all nine have a _sub."""
    import ink_subs
    for name in names:
        t = TIMES[name]
        c = next((x for x in ink_subs._timed() if x["t0"] <= t < x["t1"]), None)
        f = np.asarray(Image.open(os.path.join(OUT, name + ".png")).convert("RGB"), np.uint8)
        if c is None:
            print(f"  {name}: the storyboard has no line at {t}", flush=True)
        else:
            t = min(max(t, c["t0"] + ink_subs.reveal_s(c) + 0.2), c["t1"] - ink_subs.OUT_S - 0.05)
            print(f"  {name} at {t:.2f}: {c['text']}", flush=True)
            f = ink_subs.apply(f, t)
        save(np.float32(f), name + "_sub")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "subs":
        with_subtitle(sys.argv[2:] or list(TIMES))
    elif what == "retouch":
        retouch("k3_look", 0.22)
        retouch("k4_cry", 1.0)
    else:
        FRAMES[what](float(sys.argv[2]) if len(sys.argv) > 2 else 0.67)

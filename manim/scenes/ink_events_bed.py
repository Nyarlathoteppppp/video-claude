"""八, happening. 其实, 我一本书也没读过: the still of the flood goes as water, all of it one way, toward us, with wave after wave pushing through it. 就像河床并未读过河水: the water stands high in the canyon and goes down, leaving what it covered dark and wet, and the bed comes out.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from ink_events_lib import _X, _Y, H, W, region_from, to8, up1
from ink_grade import fft_noise, sm
from ink_move import lookup, noise, sample

_f, _r = {}, {}

# ---- the flood --------------------------------------------------------------------------------------------------------------------------------------------------------
HORIZON = 245.0
Y0 = 382                       # above this nothing is flood: pale far water, hills
ROCKS = (((0, 372), (130, 378), (230, 398), (330, 478), (345, 505), (200, 505), (0, 480)), ((1530, 450), (1620, 400), (1760, 382), (1920, 392), (1920, 490), (1700, 490), (1560, 462)))
SHIFT = 0.17                   # how far the picture is carried in one cycle, in the ground's depth units
CYCLE = 1.25
SURGES = (1.5, 3.1, 4.7, 6.3, 7.9)      # where each wave stands, far off, when the shot starts: it comes on at SPEED
SPEED = 0.8
FOAM = np.array((236, 228, 212), np.float32)


def _flood_prep():
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for r in ROCKS:
        d.polygon(r, fill=255)
    rocks = gaussian_filter(np.asarray(im, np.float32) / 255, 10.0)
    Y, X = _Y[Y0:], _X[Y0:]
    dy = Y - HORIZON
    m = sm((Y - Y0) / 46.0) * (1 - rocks[Y0:])
    v = 835.0 / dy
    u = (X - 960.0) / dy
    # at quarter size: the surges' place
    gy, gx = np.mgrid[Y0 // 4:H // 4, 0:W // 4].astype(np.float32) * 4
    gd = gy - HORIZON
    wob = fft_noise(2048, 2048, 2.6, 51)
    wob = wob / wob.std()
    gu = (gx - 960.0) / gd
    fine = fft_noise(1024, 1024, 1.5, 52)
    fine = fine / fine.std()
    _f.update(m=m.astype(np.float32), v=v.astype(np.float32), u=u.astype(np.float32), dy=dy.astype(np.float32), X=X, Y=Y, gv=(835.0 / gd).astype(np.float32), gu=gu.astype(np.float32),
              gw=(1 - gaussian_filter(rocks, 4.0)[Y0 // 4 * 4::4, ::4])[:gy.shape[0], :gy.shape[1]].astype(np.float32), wob=wob.astype(np.float32), fine=fine.astype(np.float32), gm=sm((gy - Y0) / 46.0).astype(np.float32))
    hy, hx = np.mgrid[Y0 / 2:H / 2, 0:W // 2].astype(np.float32) * 2
    _f.update(hu=((hx - 960.0) / (hy - HORIZON)).astype(np.float32), hv=(835.0 / (hy - HORIZON)).astype(np.float32))


def _carried(f8, phase, t):
    """The frame carried toward the camera along the ground by phase of a cycle, with the churn of the water on top."""
    d = _f
    vs = d["v"] + SHIFT * phase
    ds = 835.0 / vs
    r = ds / d["dy"]
    ys = HORIZON + ds
    xs = 960.0 + (d["X"] - 960.0) * r
    return sample(f8, d["Y"] + (ys - d["Y"]) * d["m"] + d["ty"], d["X"] + (xs - d["X"]) * d["m"] + d["tx"])


def flood(frame, t, dur):
    if not _f:
        _flood_prep()
    d = _f
    # the water churns, more toward us. Broad heaves and not a fine shake: at quarter size, and what it moves is lines the eye can follow
    gv, gu = d["gv"], d["gu"]
    gk = 1.0 / gv
    a = lookup(noise(43), gu * 170.0, gv * 70.0 - 26.0 * t)
    b = lookup(noise(44), gu * 80.0 + 5.0 * t, gv * 34.0 - 15.0 * t)
    gain = (0.55 + 0.45 * sm(t / 3.0)) * d["gm"] * d["gw"]
    tx = up1((a + 0.6 * b) * 15.0 * gk * gain, (W, H - Y0))
    ty = up1((b - 0.4 * a) * 7.0 * gk * gain, (W, H - Y0))
    d["tx"], d["ty"] = tx, ty
    f = frame.astype(np.float32)
    p1 = (t / CYCLE) % 1.0
    p2 = (p1 + 0.5) % 1.0
    w1 = 1 - abs(2 * p1 - 1)
    out = f.copy()
    top = _carried(f, p1, t) * w1 + _carried(f, p2, t) * (1 - w1)

    # wave on wave, a white edge and the dark face under it, each with a line of its own
    wx = lookup(d["wob"], gu * 130.0 + 512, gv * 14.0 + 11)
    fr = np.zeros(gv.shape, np.float32)
    fa = np.zeros(gv.shape, np.float32)
    for i, v0 in enumerate(SURGES):
        vk = v0 - SPEED * t
        if vk < 0.72 or vk > 6.0:
            continue
        q = gv - vk - 0.045 * vk ** 1.5 * wx * (0.75 + 0.25 * np.sin(i * 2.1))      # the line wanders about the same on the screen, near or far
        s = 0.72 + 0.28 * sm((t - 0.6) / 3.0)
        fr = np.maximum(fr, s * np.exp(-(q / (0.03 + 0.026 * vk)) ** 2))
        fa = np.maximum(fa, s * np.exp(-((q + 0.13 + 0.05 * vk) / (0.09 + 0.05 * vk)) ** 2))
    gm = d["gm"] * d["gw"]
    # the crest lifts the white the painted water already has and tears it: thick where the still is pale, broken into clots elsewhere, never a strip laid over the top
    fa = up1(fa * gm, (W, H - Y0))
    top = top * (1 - 0.3 * fa)[..., None]
    fr = up1(fr * gm, (W, H - Y0))
    hu, hv = d["hu"], d["hv"]
    tex = up1(lookup(d["fine"], hu * 420.0 + 7.0 * t, hv * 520.0 - 90.0 * t) * 0.6 + lookup(d["fine"], hu * 150.0 - 4.0 * t + 300, hv * 210.0 - 60.0 * t + 200), (W, H - Y0))
    pale = (top[..., 1] - 150.0) / 70.0
    foam = np.clip(fr * 1.5 + 0.5 * pale + 0.26 * tex - 0.95, 0, 0.62) / 0.62 * sm(fr / 0.2)
    foam = (foam * 0.86)[..., None]
    top = top * (1 - foam) + FOAM * (1 + 0.035 * tex)[..., None] * foam
    # as it gets going the brown of the silt comes into the water
    silt = 1 - 0.1 * sm(t / 4.0) * d["m"][..., None] * np.array((0.2, 0.55, 1.0), np.float32)
    out[Y0:] = top * silt
    return to8(out)


# ---- it goes down ------------------------------------------------------------------------------------------------------------------------------------------------------------
BED = 1.125            # the river where it runs on this still
HIGH = 3.5             # where it stands at the start, in the same units
RIVER = np.array((236, 220, 196), np.float32)
LATE = 1.5             # how long what the water has left stays wet


def _level(t, dur):
    return BED + 0.03 + (HIGH - BED) * (1 - sm((t + 0.2) / (dur - 0.5))) ** 1.25


def _recede_prep():
    from ink_cut import mover
    from ink_events_lib import shot
    mv = mover(shot("bed/recede")[0])
    h = mv.p.height
    ok = h > 0.5
    h = np.where(ok, gaussian_filter(np.where(ok, h, BED), 1.0), 99.0).astype(np.float32)
    gy, gx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
    gd = np.maximum(gy - 252.0, 8.0)
    _r.update(h=h, u=((gx - 960.0) / gd).astype(np.float32), v=(835.0 / gd).astype(np.float32), far=sm((_Y - 272.0) / 44.0).astype(np.float32))


def recede(frame, t, dur):
    if not _r:
        _recede_prep()
    d = _r
    ys, xs = region_from("bed/recede", t, dur, (0, H, 0, W))
    yi, xi = np.clip(ys + 0.5, 0, H - 1).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)
    h = d["h"][yi, xi]
    lv = _level(t, dur)
    water = sm((lv - h) / 0.09) * d["far"]
    wet = sm((_level(t - LATE, dur) + 0.12 - h) / 0.35) * (1 - water) * d["far"]
    # the water itself: the pale of the river on this still, its lines drawn out and moving along the course, and darker along the bank where it is shallow
    n = lookup(noise(45), d["u"] * 240.0, d["v"] * 44.0 - 14.0 * t)
    m = lookup(noise(46), d["u"] * 110.0 + 3.0 * t, d["v"] * 22.0 - 8.0 * t)
    s = gaussian_filter(n * 0.5 + m, 1.3)
    lines = np.abs(((s * 3.0) % 1.0) - 0.5) * 2
    shade = up1((1 + 0.07 * s - 0.07 * sm((0.12 - lines) / 0.12)))
    deep = sm((lv - h) / 0.5)
    col = RIVER * shade[..., None] * (0.86 + 0.14 * deep)[..., None]
    f = frame.astype(np.float32)
    # the ground goes dark where it is wet, and the edge of the water has a thin darker line
    edge = np.exp(-(((lv - h) - 0.03) / 0.05) ** 2) * d["far"]
    f = f * (1 - 0.34 * wet - 0.3 * edge)[..., None]
    a = water[..., None]
    return to8(f * (1 - a) + col * a)

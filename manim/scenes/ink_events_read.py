"""四 · 阅读, happening. The sheets lie on moving water: the water moves round them and they themselves stay flat and only drift. The page of 论坛上的争吵 can now be read. 深夜里没人回复的帖子: the rings go out from a phone that does not move, and its light breathes and goes down a little.
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, distance_transform_edt, gaussian_filter

from ink_events_lib import _X, _Y, H, W, Rings, box, crop8, fill, plate, up1
from ink_grade import fft_noise, sm
from ink_move import lookup, noise, sample

LUM = np.array([0.30, 0.59, 0.11], np.float32)


def poly(points):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(points, fill=255)
    return np.asarray(im) > 0


# ---- the water, looked down on from the camera of these shots: its horizon is far above the frame ----------------------------------------------------------
_GY, _GX = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
_S = (_GY + 1160.0) / 2240.0
_U, _V = (_GX - 960.0) / _S * 0.42, 620.0 / _S


def water_field(t):
    """How far the surface shifts each pixel, and the light it gives."""
    a = lookup(noise(21), _U + 8.0 * t, _V * 1.6 + 15.0 * t)
    b = lookup(noise(22), _U * 0.6 - 7.0 * t, _V * 1.1 + 11.0 * t)
    c = lookup(noise(23, 1.9), _U * 1.5 + 11.0 * t, _V * 2.8 + 20.0 * t)
    dx = (a + 0.6 * b + 0.3 * c) * 3.6 * _S
    dy = (b - 0.4 * a + 0.2 * c) * 1.7 * _S
    return up1(dx), up1(dy), up1((a - b + 0.6 * c) * 0.014)


# ---- the sheets ---------------------------------------------------------------------------------------------------------------------------------------------
class Sheets:
    """Water with sheets of paper on it. Each sheet, with the shadow it throws on the water, is lifted off as one flat piece and laid back a little further on."""

    def __init__(self, still, polys, drifts, write=None):
        P = plate(still)
        L = (P * LUM).sum(-1)
        soft = gaussian_filter(L, 2.0)
        self.sheets = []
        every = np.zeros((H, W), bool)
        lifted = []
        for pts, move in zip(polys, drifts):
            rough = poly(pts)
            near = binary_dilation(rough, iterations=16)
            paper = binary_fill_holes(binary_closing(near & (soft > 128.0), iterations=4)) & near
            shadow = binary_dilation(paper, iterations=9) & near & (soft < 104.0)
            whole = gaussian_filter((paper | shadow).astype(np.float32), 1.2)
            lifted.append((whole, move))
            every |= paper | shadow
        if write:
            P = write(P)
        hole = binary_dilation(every, iterations=9)
        # under a sheet only the rim it uncovers is ever seen: the water's tone there and a little grain, and nothing taken from anywhere, or the writing on a sheet comes along
        grain = gaussian_filter(np.random.default_rng(4).normal(0, 7.0, (H, W)).astype(np.float32), 0.7)
        self.bg = fill(P, hole, sigma=26.0, feather=5.0, fine=0.2).astype(np.float32)
        self.bg += (grain * gaussian_filter(hole.astype(np.float32), 5.0))[..., None]
        # the water is still right up against the paper and takes up its motion over a little way
        self.k = sm((distance_transform_edt(~every).astype(np.float32) - 3.0) / 48.0)
        for whole, move in lifted:
            ys, xs = np.nonzero(whole > 0.01)
            pad = 18
            ya, yb, xa, xb = max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)
            cy, cx = ys.mean(), xs.mean()
            self.sheets.append(dict(ya=ya, yb=yb, xa=xa, xb=xb, cx=cx, cy=cy, move=move, a=whole[ya:yb, xa:xb].astype(np.float32), f=P[ya:yb, xa:xb].astype(np.float32),
                                    yy=_Y[ya:yb, xa:xb], xx=_X[ya:yb, xa:xb]))

    def frame(self, t, dur, z=(1.0, 1.06)):
        dx, dy, lit = water_field(t)
        k = self.k
        out = sample(self.bg, _Y + dy * k, _X + dx * k) * (1 + lit * k)[..., None]
        u = t / max(dur, 1e-3)
        for s in self.sheets:
            mx, my, deg, phase = s["move"]
            # one slow way across the shot, and a very little rise and fall of the swell under it
            ox = mx * u + 0.9 * math.sin(0.8 * t + phase)
            oy = my * u + 1.1 * math.sin(0.95 * t + phase * 1.7)
            r = math.radians(deg * u + 0.08 * math.sin(0.7 * t + phase))
            c, n = math.cos(r), math.sin(r)
            rx, ry = s["xx"] - s["cx"] - ox, s["yy"] - s["cy"] - oy
            sx = s["cx"] + c * rx + n * ry - s["xa"]
            sy = s["cy"] - n * rx + c * ry - s["ya"]
            a = sample(s["a"], sy, sx)[..., None]
            part = out[s["ya"]:s["yb"], s["xa"]:s["xb"]]
            out[s["ya"]:s["yb"], s["xa"]:s["xb"]] = part * (1 - a) + sample(s["f"], sy, sx) * a
        return crop8(out, box(z, (0.5, 0.5), t, dur))


# ---- the page that can be read ----------------------------------------------------------------------------------------------------------------------------------
_F = 1920.0     # lens 36 on a 36 sensor, at 1920
_CAM = np.array((0.75, -6.7, 8.4))
_fw = np.array((0.75, 2.8, 0.0)) - _CAM
_fw = _fw / np.linalg.norm(_fw)
_rt = np.cross(_fw, (0, 0, 1.0))
_rt = _rt / np.linalg.norm(_rt)
_up = np.cross(_rt, _fw)
# as ink3d_read.PAGES2 has it
FORUM = dict(centre=(2.75, 3.9), turn=16.0, width=3.3, corners=(-0.16, 0.06, 0.14, 0.02), tw=1500, th=2100)
# the argument, in the shape of the page: who, what, how far in, how big
POSTS = (("楼主", "你们根本不懂！", 0, 162), ("沙发", "楼上懂个屁", 36, 172), ("板凳", "不懂别乱说", 92, 172), ("4楼", "说谁呢你", 30, 166), ("楼主", "删了吧，没意思", 110, 146), ("6楼", "别删！我还没骂完", 300, 118))


def _project(U, V):
    """Where a point of the sheet, U across and V from the near edge, is in the frame, the sheet's sag at the corners taken into account."""
    f = FORUM
    hgt = f["width"] * f["th"] / f["tw"]
    lx, ly = (U - 0.5) * f["width"], (V - 0.5) * hgt
    nl, nr, fr, fl = f["corners"]
    z = 0.035 + nl * (sm((0.42 - U) / 0.42) * sm((0.42 - V) / 0.42)) ** 1.4 + nr * (sm((U - 0.58) / 0.42) * sm((0.42 - V) / 0.42)) ** 1.4
    z = z + fr * (sm((U - 0.58) / 0.42) * sm((V - 0.58) / 0.42)) ** 1.4 + fl * (sm((0.42 - U) / 0.42) * sm((V - 0.58) / 0.42)) ** 1.4
    c, s = math.cos(math.radians(f["turn"])), math.sin(math.radians(f["turn"]))
    X = f["centre"][0] + c * lx - s * ly
    Y = f["centre"][1] + s * lx + c * ly
    dx, dy, dz = X - _CAM[0], Y - _CAM[1], z - _CAM[2]
    d = dx * _fw[0] + dy * _fw[1] + dz * _fw[2]
    return 960 + _F * (dx * _rt[0] + dy * _rt[1] + dz * _rt[2]) / d, 540 - _F * (dx * _up[0] + dy * _up[1] + dz * _up[2]) / d


def _forum_ink():
    """The thread as it was printed, then soaked: each stroke swollen, soft at its edge, the ink thin in places."""
    from ink_film import font
    tw, th = FORUM["tw"], FORUM["th"]
    im = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(im)
    y = 62
    for who, say, ind, size in POSTS:
        x = 96 + ind
        h = size + 122
        d.rectangle([x, y, tw - 84, y + h], outline=120, width=5)
        d.text((x + 26, y + 12), who, 200, font=font("song", 66))
        d.text((x + 26, y + 88), say, 255, font=font("kaib", size))
        y += h + 26
    a = np.asarray(im, np.float32) / 255
    # the fibres take it unevenly, so no edge is a clean one
    n1, n2 = fft_noise(th, tw, 1.6, 31), fft_noise(th, tw, 1.6, 32)
    n1, n2 = n1 / n1.std() * 2.6, n2 / n2.std() * 2.6
    from ink_move import sample as smp
    a = smp(a, _Yt(th, tw) + n1, _Xt(th, tw) + n2)
    ink = np.clip(a * 0.92 + gaussian_filter(a, 3.0) * 0.5 + gaussian_filter(a, 11.0) * 0.26, 0, 1)
    thin = fft_noise(th, tw, 2.6, 33)
    thin = thin / thin.std()
    return np.clip(ink * (0.84 + 0.13 * thin), 0, 1).astype(np.float32)


def _Yt(h, w):
    return np.mgrid[0:h, 0:w][0].astype(np.float32)


def _Xt(h, w):
    return np.mgrid[0:h, 0:w][1].astype(np.float32)


def _write_forum(P):
    """Wash the page, then lay the thread on it along the sheet's own surface."""
    tw, th = FORUM["tw"], FORUM["th"]
    vv, uu = np.mgrid[0:1:90j, 0:1:64j]
    fx, fy = _project(uu, vv)
    to_page = LinearNDInterpolator(np.stack([fx.ravel(), fy.ravel()], -1), np.stack([uu.ravel(), vv.ravel()], -1))
    xa, xb, ya, yb = int(fx.min()) - 2, int(fx.max()) + 3, int(fy.min()) - 2, int(fy.max()) + 3
    q = to_page(np.stack([_X[ya:yb, xa:xb].ravel(), _Y[ya:yb, xa:xb].ravel()], -1)).reshape(yb - ya, xb - xa, 2).astype(np.float32)
    inside = np.isfinite(q[..., 0])
    U, V = np.nan_to_num(q[..., 0]), np.nan_to_num(q[..., 1])
    # the old writing, blurred past reading, goes. Inside the sheet's own edge all that stays is the paper's tone, from its lightest part, and its grain
    L = (P * LUM).sum(-1)
    edge = np.zeros((H, W), np.float32)
    edge[ya:yb, xa:xb] = np.minimum(np.minimum(U, 1 - U) * tw, np.minimum(V, 1 - V) * th) * inside
    sheet = poly(DRIFT2[1]) & binary_fill_holes(binary_closing(gaussian_filter(L, 2.0) > 128.0, iterations=4))
    sheet = distance_transform_edt(sheet).astype(np.float32)
    keep = np.minimum(sm((edge - 18.0) / 26.0), sm((sheet - 2.0) / 7.0))[..., None]
    light = ((L > gaussian_filter(L, 34.0) * 0.98) & (sheet > 8.0)).astype(np.float32)
    w = gaussian_filter(light, 24.0) + 1e-3
    tone = np.stack([gaussian_filter(P[..., c] * light, 24.0) / w for c in range(3)], -1)
    grit = P - np.stack([gaussian_filter(P[..., c], 1.2) for c in range(3)], -1)
    washed = P * (1 - keep) + (tone + grit) * keep
    # the sheet's far part is a little out of focus, like everything on it
    ink = _forum_ink()
    a = sample(ink, (1 - V) * (th - 1), U * (tw - 1))
    a = gaussian_filter(a, 0.9) * inside
    # torn away near its near left corner: nothing is written on what is not there
    Lw = gaussian_filter((washed * LUM).sum(-1), 2.0)
    a = a * sm((Lw[ya:yb, xa:xb] - 112.0) / 30.0)
    out = washed.copy()
    colour = np.array((34, 42, 40), np.float32)
    part = out[ya:yb, xa:xb]
    k = (a * 0.94)[..., None]
    out[ya:yb, xa:xb] = part * (1 - k) + colour * k
    return out.astype(np.float32)


# ---- the two shots of sheets ---------------------------------------------------------------------------------------------------------------------------------
DRIFT = (
    ((97, 425), (582, 338), (652, 852), (160, 942), (150, 880), (130, 820), (60, 808), (38, 795), (75, 600)),
    ((865, 145), (1135, 185), (1125, 225), (1183, 255), (1115, 455), (960, 460), (700, 400)),
    ((1475, 380), (1705, 680), (1695, 730), (1740, 835), (1385, 945), (1100, 462)),
)
DRIFT2 = (
    ((430, 272), (865, 328), (710, 760), (695, 835), (575, 815), (98, 735)),
    ((935, 290), (1330, 210), (1622, 598), (1245, 690), (1225, 640), (1090, 595), (960, 345)),
)
_sheets = {}


def drift(frame, t, dur):
    if "a" not in _sheets:
        _sheets["a"] = Sheets("read/drift", DRIFT, ((9.0, -5.0, 0.35, 0.3), (-7.0, 5.0, -0.3, 2.1), (6.0, 6.0, 0.4, 4.0)))
    return _sheets["a"].frame(t, dur)


def drift2(frame, t, dur):
    if "b" not in _sheets:
        _sheets["b"] = Sheets("read/drift2", DRIFT2, ((-8.0, 5.0, -0.3, 1.2), (7.0, 5.0, 0.3, 3.0)), write=_write_forum)
    return _sheets["b"].frame(t, dur)


# ---- the phone ----------------------------------------------------------------------------------------------------------------------------------------------------
PHONE = ((622, 478), (640, 458), (850, 438), (880, 480), (968, 760), (965, 782), (660, 782), (636, 640))
SCREEN = ((634, 482), (648, 466), (844, 448), (864, 482), (952, 750), (944, 768), (672, 772), (646, 640))
RING_AT = (770.0, 762.0)
_ph = {}


def _phone_prep():
    P = plate("read/phone")
    phone = poly(PHONE)
    screen = gaussian_filter(poly(SCREEN).astype(np.float32), 2.0)
    still = sm((distance_transform_edt(~phone).astype(np.float32) - 2.0) / 34.0)
    # above the water's edge there is nothing to move: the far water is hardly stirred and the mist and hills not at all
    far = sm((_Y - 280.0) / 240.0)
    spill = np.exp(-(((_X - 790.0) / 240.0) ** 2 + ((_Y - 830.0) / 90.0) ** 2)) * (1 - gaussian_filter(phone.astype(np.float32), 6.0))
    rng = np.random.default_rng(8)
    rings = []
    t0 = -6.0
    while t0 < 4.4:
        rings.append((t0, RING_AT[0], RING_AT[1], 0.34, 1.9 + rng.uniform(-0.25, 0.25)))
        t0 += rng.uniform(1.25, 1.75)
    _ph.update(P=P, phone=phone, screen=screen.astype(np.float32), k=(still * far).astype(np.float32), spill=spill.astype(np.float32), rings=rings, grid=Rings(scale=4))


def phone(frame, t, dur):
    if not _ph:
        _phone_prep()
    d = _ph
    u = t / max(dur, 1e-3)
    # less and less of a stir: nobody is answering
    h = d["grid"].field(d["rings"], t, speed=115.0, wave=16.0, life=5.5, amp=1.0 - 0.45 * sm(u))
    rx, ry, rl = d["grid"].warp(h, gain=9.0, lit=0.075)
    wx, wy, wl = water_field(t * 0.45)      # and hardly any swell under it, on a night like this
    k = d["k"]
    dx, dy = (rx + wx * 0.35) * k, (ry + wy * 0.35) * k
    # nothing above y0 moves
    y0 = 270
    out = d["P"].copy()
    out[y0:] = sample(d["P"], _Y[y0:] + dy[y0:], _X[y0:] + dx[y0:]) * (1 + (rl[y0:] + wl[y0:] * 0.5) * k[y0:])[..., None]
    # the screen's light breathes, slowly and unevenly, and is a little lower at the end than it was
    g = 1 + 0.055 * math.sin(2 * math.pi * t / 2.3 + 0.4) + 0.025 * math.sin(2 * math.pi * t / 1.27) - 0.13 * sm((t - 1.6) / 3.4)
    ya, yb, xa, xb = 420, 960, 480, 1120
    out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 + (g - 1) * (d["screen"][ya:yb, xa:xb] + 0.75 * d["spill"][ya:yb, xa:xb]))[..., None]
    return crop8(out, box((1.0, 1.05), (0.5, 0.5), t, dur))

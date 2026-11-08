"""七 · 水滴石穿, happening. The drop swells and lets go; drops land on the slab, each one a ring and a small step from nonsense toward 低头思故乡, slow at first, then so fast they run together; and in the
last shot the stone is written on, a character at a time, by water alone. Everything is cut out of the finished stills, plus the wrong characters, which are set with the font the slab was cut with.
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter

from ink_events_lib import membrane, _X, _Y, H, W, box, crop8, fill, plate, stroke_order, to8
from ink_grade import fft_noise, sm
from ink_move import sample

# ---- the drop -------------------------------------------------------------------------------------------------------------------------------------------
DC = np.array([940.0, 632.0])       # the drop on its still, and how big
DR = 262.0
TIP = [(650, -2), (690, 100), (715, 200), (735, 300), (760, 368), (790, 425), (820, 452), (860, 473), (900, 484), (940, 490), (985, 486), (1025, 470), (1058, 448), (1078, 422), (1100, 380), (1125, 330), (1150, 250), (1195, 130), (1262, -2)]
LET_GO = 1.42                       # seconds into the shot
FALL_G = 4600.0                     # pixels a second, a second
_d = {}


LUM = np.array([0.30, 0.59, 0.11], np.float32)


def _drop_prep():
    P = plate("drip/drop")
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(TIP, fill=255)
    tip = gaussian_filter(np.asarray(im, np.float32) / 255, 1.2)
    r = np.hypot(_X - DC[0], _Y - DC[1])
    hole = (r < DR + np.where(_Y < 500, 40.0, 15.0)) & (tip < 0.5)      # up beside the stone the ball leaves a darker edge further out
    # behind it the slab, out of focus. Its colours carry on from each side of the ball, the left's blue and the right's pale; its blotches and grain are taken from the bare stone beyond the writing on each side, with what is dark enough to be a stroke held back
    tone = membrane(P, hole | (tip > 0.1), feel=7.0)
    detail = P - np.stack([gaussian_filter(P[..., c], 44.0) for c in range(3)], -1)
    # the only stone with no writing on it is the far right; the left half takes it from higher up, so nothing shows twice side by side
    x, y = _X.astype(np.int32), _Y.astype(np.int32)
    right = detail[np.clip(y, 0, H - 1), np.clip(x + 625, 0, W - 1)]
    left = detail[(y + 330 - 90) % 960 + 90, np.clip(x + 930, 0, W - 1)]
    k = sm((_X - 860.0) / 160.0)[..., None]
    col = tone + (left * (1 - k) + right * k)
    m = np.clip(gaussian_filter(hole.astype(np.float32), 5.0) * 1.7, 0, 1)[..., None]
    bg = P * (1 - m) + col * m
    # the ball's top, under the stone, is never seen in the still: when it falls it is the lower part seen the other way up, which is what a ball of water does
    sprite = P.copy()
    y0, y1 = int(DC[1] - DR - 8), int(DC[1] + DR + 8)
    x0, x1 = int(DC[0] - DR - 8), int(DC[0] + DR + 8)
    s = sprite[y0:y1, x0:x1].copy()
    flip = s[::-1].copy()
    cap = gaussian_filter(np.clip(tip[y0:y1, x0:x1] * 1.0, 0, 1), 3.0)
    cap = np.clip(cap * 1.6, 0, 1)
    # a little more than the stone's edge, for the dark rim the stone throws
    s = s * (1 - cap[..., None]) + flip * cap[..., None]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rr = np.hypot(xx - DC[0], yy - DC[1])
    alpha = sm((DR + 1 - rr) / 3.0)
    _d.update(P=P, tip=tip, bg=bg, s=s, alpha=alpha, sx0=x0, sy0=y0)


def _bead(out, cx, cy, rx, ry, o=1.0):
    """A small bead of water on the stone's tip: it shows what is behind it upside down, darkens at its rim, gathers light low down and has one small highlight."""
    bg = _d["bg"]
    xa, xb, ya, yb = int(cx - rx - 3), int(cx + rx + 4), int(cy - ry - 3), int(cy + ry + 4)
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    u, v = (xx - cx) / rx, (yy - cy) / ry
    r = np.hypot(u, v)
    a = sm((1 - r) * min(rx, ry) / 1.6)[..., None] * o
    sx, sy = np.clip(cx - u * rx * 1.9, 0, W - 1).astype(np.int32), np.clip(cy - v * ry * 1.9, 0, H - 1).astype(np.int32)
    col = bg[sy, sx] * (1.1 + 0.3 * v)[..., None]
    col = col * (1 - 0.68 * sm((r - 0.66) / 0.34) * (0.7 + 0.3 * (1 - v) / 2))[..., None]
    col = col + 170.0 * np.exp(-(((u + 0.3) / 0.15) ** 2 + ((v + 0.42) / 0.11) ** 2))[..., None]
    out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - a) + col * a


def _ball(out, cx, cy, sx, sy, bulge=0.0, blur_y=0.0, o=1.0):
    """The ball onto out, its centre at cx, cy, stretched sx, sy. bulge: what is in its middle is drawn bigger, as a ball that swells does. blur_y: its smear as it falls, in pixels."""
    d = _d
    src = d["s"]
    h, w = src.shape[:2]
    rx, ry = DR * sx + 4, DR * sy + 4 + blur_y
    xa, xb, ya, yb = max(int(cx - rx), 0), min(int(cx + rx) + 2, W), max(int(cy - ry), 0), min(int(cy + ry) + 2, H)
    if xa >= xb or ya >= yb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    acc_c, acc_a = 0, 0
    steps = 6 if blur_y > 1 else 1
    for k in range(steps):
        off = (k / max(steps - 1, 1) - 0.5) * blur_y
        ux, uy = (xx - cx) / sx, (yy - cy - off) / sy
        if bulge:
            rr = np.clip(np.hypot(ux, uy) / DR, 1e-3, 1.2)
            q = rr ** bulge
            ux, uy = ux * q, uy * q
        px, py = ux + DC[0] - d["sx0"], uy + DC[1] - d["sy0"]
        inside = (px >= 0) & (py >= 0) & (px < w - 1) & (py < h - 1)
        c = sample(src, py, px)
        a = sample(d["alpha"], py, px) * inside
        acc_c = acc_c + c * a[..., None]
        acc_a = acc_a + a
    a = np.clip(acc_a / steps, 0, 1)
    c = acc_c / np.maximum(acc_a, 1e-3)[..., None]
    a = (a * o)[..., None]
    out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - a) + c * a


def drop(frame, t, dur):
    """It swells, it sags, it cannot hold, and it goes out of the bottom of the frame faster and faster. A new one starts on the stone behind it."""
    if not _d:
        _drop_prep()
    d = _d
    out = d["bg"].copy()
    swell = 0.84 + 0.30 * sm((t + 0.8) / (LET_GO + 0.8))
    shake = 0.006 * math.sin(2 * math.pi * 5.3 * t) * sm((t - 0.3) / 0.8)
    if t < LET_GO:
        sag = sm((t - 0.2) / (LET_GO - 0.2))
        sx, sy = swell * (1 - 0.03 * sag) * (1 + shake), swell * (1 + 0.13 * sag) * (1 - shake)
        cy = DC[1] - DR + DR * sy
        _ball(out, DC[0], cy, sx, sy, bulge=-0.12 * (swell - 1.0) / 0.14)
    else:
        a = t - LET_GO
        s0 = 1.14
        y0 = DC[1] - DR + DR * s0 * 1.13
        cy = y0 + 0.5 * FALL_G * a * a
        settle = math.exp(-a * 9)
        ring = math.sin(a * 34) * settle
        sx, sy = s0 * (0.97 + 0.03 * (1 - settle)) * (1 - 0.03 * ring), s0 * (1.13 * settle + 1.02 * (1 - settle) + 0.03 * ring)
        _ball(out, DC[0], cy, sx, sy, bulge=-0.12, blur_y=min(FALL_G * a / 30.0 * 0.5, 70.0))
        # what stays on the stone, and starts again
        grow = sm((a - 0.05) / 0.9)
        if grow > 0:
            k = 0.05 + 0.13 * grow
            wob = 1 + 0.12 * math.exp(-a * 5) * math.sin(a * 30)
            _bead(out, DC[0], 480 + DR * k * 0.72, DR * k / wob, DR * k * wob, o=sm(a / 0.08))
    m = d["tip"][..., None]
    out = out * (1 - m) + d["P"] * m
    z0, z1 = 1.0, 1.08
    return crop8(out, box((z0, z1), (0.5, 0.5), t, dur))


# ---- the slab -------------------------------------------------------------------------------------------------------------------------------------------
TILT = math.radians(24)
SC, SS = math.cos(TILT), math.sin(TILT)
CAM = np.array((1.0, -17.0, 11.0))
_f = (np.array((0.1, 3.0, 4.37)) - CAM)
_f = _f / np.linalg.norm(_f)
_r = np.cross(_f, (0, 0, 1))
_r = _r / np.linalg.norm(_r)
_u = np.cross(_r, _f)
FOCAL = 28.0 / 36 * W
Y_OFF = 31.0                # the slab's low forms lift the column this much on the frame: measured off the still
COL_S, STEP, SIZE = 19.2, 4.1, 3.8
TARGET = "低头思故乡"
# what each cell has said before it comes right: each is a little nearer than the last
WRONG = ("的他仁", "了天买", "是里恩", "中古敌", "有么幺")
FINISH = (4.85, 3.2, 2.25, 3.9, 4.4)      # when each comes right, in seconds from the start of the junk shot
START = (3.1, 1.6, 0.3, 1.0, 1.9)
LAST = 5.05                               # the last drop
T_SHOT = 2.6                              # the junk shot's length; the right shot goes on from there
PULL_END = 5.7
PALE = 0.52                               # how dark what is wrong is cut
BOX0, BOX1 = 1.45, 1.278
FIX_Y = 1301.0                            # the point the pull-back holds still, below the frame


def proj(x, s, n=0.0, lift=0.0):
    """Slab coordinates to the still, by the camera the slab was rendered with."""
    x, s = np.asarray(x, np.float64), np.asarray(s, np.float64)
    P = np.stack([x + 0 * s, s * SC - n * SS, s * SS + n * SC + lift], -1) - CAM
    d = P @ _f
    return 960 + FOCAL * (P @ _r) / d, 540 - FOCAL * (P @ _u) / d + Y_OFF, d


def homography(a, b):
    m, v = [], []
    for (x, y), (u, w) in zip(a, b):
        m += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -w * x, -w * y]]
        v += [u, w]
    return np.append(np.linalg.solve(np.array(m, np.float64), np.array(v, np.float64)), 1).reshape(3, 3)


def apply_h(h, x, y):
    d = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / d, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / d


def glyph(ch, kind, size=256):
    from ink_film import font
    s = int(size * 1.5)
    im = Image.new("L", (s, s), 0)
    ImageDraw.Draw(im).text((s / 2, s / 2), ch, 255, font=font(kind, size), anchor="mm")
    return np.asarray(im, np.float32) / 255


def cell_of(i):
    return 0.0, COL_S - STEP * (i + 0.5)


_s = {}
EDGE_L = [(-60, 400), (130, 385), (215, 372), (270, 350), (340, 320), (400, 275), (450, 235), (490, 190), (560, 150), (640, 100), (760, 20), (1090, -200)]      # (y, x) of the sheet's torn edge
EDGE_R = [(-60, 1500), (130, 1530), (200, 1560), (290, 1590), (350, 1615), (400, 1645), (480, 1665), (570, 1705), (640, 1755), (700, 1785), (800, 1875), (1090, 2200)]


def _heal(f):
    """The top corners of the wider frame would show the sheet's torn edge. Past it, the same stone goes on: its tone from just inside, its grain from further in."""
    rows = np.arange(H, dtype=np.float32)
    n = fft_noise(H, 1, 2.6, 71)[:, 0] * 8
    left = np.interp(rows, [a for a, _ in EDGE_L], [b for _, b in EDGE_L]) + 64 + n
    right = np.interp(rows, [a for a, _ in EDGE_R], [b for _, b in EDGE_R]) - 64 + n
    wob = gaussian_filter(fft_noise(H, W, 2.2, 72), 2.0) * 14
    f = fill(f, _X < left[:, None] + wob, sigma=46.0, offset=(37, 250), feather=11.0, fine=46.0)
    return fill(f, _X > right[:, None] + wob, sigma=46.0, offset=(53, -250), feather=11.0, fine=46.0)


def _align(i, dark, q):
    """Where the font's character really lies on the still: the slab's forms have moved it a little from where the camera says."""
    g = glyph(TARGET[i], "kaih")
    s = g.shape[0]
    xa, xb, ya, yb = int(q[:, 0].min()), int(q[:, 0].max()), max(int(q[:, 1].min()), 0), min(int(q[:, 1].max()), H)
    yy, xx = np.mgrid[ya:yb:2, xa:xb:2].astype(np.float32)
    d = dark[ya:yb:2, xa:xb:2]
    to = homography(q, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64))
    frac = 256 / SIZE * (SIZE * 1.5)
    best, arg = -1e9, (0, 0, 1.0)
    for k in (0.94, 0.97, 1.0, 1.03, 1.06, 1.09, 1.12):
        for dy in range(-44, 45, 4):
            for dx in range(-44, 45, 4):
                ux, uy = apply_h(to, xx - dx, yy - dy)
                px, py = (ux - 0.5) * frac / k + s / 2, (uy - 0.5) * frac / k + s / 2
                m = g[np.clip(py, 0, s - 1).astype(np.int32), np.clip(px, 0, s - 1).astype(np.int32)]
                score = (m * d).mean() - 0.55 * (m * (1 - d)).mean()
                if score > best:
                    best, arg = score, (dx, dy, k)
    return arg


def _slab_prep():
    J, R = plate("drip/junk"), plate("drip/right")
    lum = np.array([0.30, 0.59, 0.11], np.float32)
    # the five right ones, found where the font says they are
    glyphs, boxes = [], []
    for i, ch in enumerate(TARGET):
        cx, cs = cell_of(i)
        half = SIZE * 0.68
        xs, ys, _ = proj([cx - half, cx + half, cx + half, cx - half], [cs + half, cs + half, cs - half, cs - half])
        q = np.stack([xs, ys], -1)
        boxes.append(q)
    # the beads the stills caught in the air: not on a stone they keep falling onto
    def beads(f, spots):
        hole = np.zeros((H, W), bool)
        for x, y, rx, ry in spots:
            hole |= ((_X - x) / rx) ** 2 + ((_Y - y) / ry) ** 2 < 1
        return fill(f, hole, sigma=14.0, offset=(-48, 90), feather=2.0), hole
    Rc = fill(R, ((_X - 855) / 34) ** 2 + ((_Y - 790) / 46) ** 2 < 1, sigma=14.0, offset=(44, -150), feather=2.0)
    Rc, rb = beads(Rc, [(985, 630, 22, 30), (905, 695, 26, 30)])
    Jc, _ = beads(J, [(1165, 342, 62, 62)])
    # the writing's bounding bands
    LR = (Rc * lum).sum(-1)
    near = gaussian_filter(LR, 40)
    dark = LR < near * 0.62
    col = np.zeros((H, W), bool)
    col[195:1035, 815:1100] = True
    ink = dark & col
    # small specks are the stone's: keep what is thick enough to be a stroke
    from scipy.ndimage import binary_opening
    ink = binary_opening(ink, np.ones((5, 5), bool))
    hole_r = binary_dilation(ink, iterations=9)
    Rp = fill(Rc, hole_r, sigma=20.0, offset=(0, 340), feather=3.0)
    # where each character really is
    fits = []
    for i in range(5):
        cx, cs = cell_of(i)
        half = SIZE * 0.75
        xs, ys, _ = proj([cx - half, cx + half, cx + half, cx - half], [cs + half, cs + half, cs - half, cs - half])
        fits.append((np.stack([xs, ys], -1), _align(i, (dark & col).astype(np.float32), np.stack([xs, ys], -1))))
    # the beads on 故 sat on its strokes: put the strokes back, from the font and the stroke's own black
    q, (fx, fy, fk) = fits[3]
    to = homography(q, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64))
    g = glyph("故", "kaih")
    frac = 256 / SIZE * (SIZE * 1.5)
    ux, uy = apply_h(to, _X.astype(np.float64) - fx, _Y.astype(np.float64) - fy)
    stroke = sample(g, np.clip((uy - 0.5) * frac / fk + g.shape[0] / 2, 0, g.shape[0] - 1.001).astype(np.float32), np.clip((ux - 0.5) * frac / fk + g.shape[1] / 2, 0, g.shape[1] - 1.001).astype(np.float32))
    stroke = gaussian_filter(np.where((_X > 780) & (_X < 1120) & (_Y > 560) & (_Y < 780), stroke, 0), 1.0)
    black = np.median(Rc[(LR < near * 0.4) & col], axis=0)
    back = (gaussian_filter(rb.astype(np.float32), 2.0) * stroke)[..., None]
    Rc = Rc * (1 - back) + (black + 5.0 * fft_noise(H, W, 1.2, 66)[..., None]) * back
    LJ = (Jc * lum).sum(-1)
    dark_j = (LJ < gaussian_filter(LJ, 40) * 0.8) & col
    dark_j = binary_opening(dark_j, np.ones((3, 3), bool))
    hole_j = binary_dilation(dark_j, iterations=9) | hole_r
    lowj = np.stack([gaussian_filter(Jc[..., c] * (~hole_j), 26.0) / (gaussian_filter((~hole_j).astype(np.float32), 26.0) + 1e-3) for c in range(3)], -1)
    # far inside the column a wider look: there is nothing near
    lowj2 = np.stack([gaussian_filter(Jc[..., c] * (~hole_j), 90.0) / (gaussian_filter((~hole_j).astype(np.float32), 90.0) + 1e-3) for c in range(3)], -1)
    nearw = np.clip(gaussian_filter((~hole_j).astype(np.float32), 26.0) / 0.25, 0, 1)[..., None]
    low = lowj * nearw + lowj2 * (1 - nearw)
    detail = Rp - np.stack([gaussian_filter(Rp[..., c], 26.0) for c in range(3)], -1)
    mj = np.clip(gaussian_filter(hole_j.astype(np.float32), 3.0) * 2, 0, 1)[..., None]
    Jp = Jc * (1 - mj) + (low + detail) * mj
    # the stone's edge out of the frame
    Rc, Rp, Jp = _heal(Rc), _heal(Rp), _heal(Jp)
    ratio = (Rc + 4) / (Rp + 4)
    stone = np.clip((Rp * lum).sum(-1) / (gaussian_filter((Rp * lum).sum(-1), 9.0) + 1), 0.6, 1.4)
    n = (fft_noise(H, W, 2.2, 61) + 1.2 * gaussian_filter(fft_noise(H, W, 3.0, 62), 3.0))
    n = (n - n.min()) / (n.max() - n.min())
    lo, hi = np.percentile(n, [1, 99])
    n = np.clip((n - lo) / (hi - lo), 0, 1).astype(np.float32)

    # each cell's own frame on the still, and where one ends and the next begins
    centres = [float(proj(cell_of(i)[0], cell_of(i)[1])[1]) + fits[i][1][1] for i in range(5)]
    edges = [0] + [int((centres[i] + centres[i + 1]) / 2) for i in range(4)] + [H]
    cells = []
    for i, ch in enumerate(TARGET):
        cx, cs = cell_of(i)
        half = SIZE * 0.75
        xs, ys, _ = proj([cx - half, cx + half, cx + half, cx - half], [cs + half, cs + half, cs - half, cs - half])
        q = np.stack([xs, ys], -1)
        xa, xb = int(q[:, 0].min()) - 4, int(q[:, 0].max()) + 5
        ya, yb = int(q[:, 1].min()) - 4, int(q[:, 1].max()) + 5
        ya, yb = max(ya, edges[i]), min(yb, edges[i + 1])
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        kind = "kaih"
        tile_h = homography(q, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64))
        ax_, ay_, ak_ = fits[i][1]
        ux, uy = apply_h(tile_h, xx - ax_, yy - ay_)
        v = np.clip((yy - q[:, 1].min()) / (q[:, 1].max() - q[:, 1].min()), 0, 1) + 0.09 * n[ya:yb, xa:xb]
        masks = []
        for w in WRONG[i]:
            g = glyph(w, kind)
            s = g.shape[0]
            size_frac = 256 / SIZE * (SIZE * 1.5) / ak_      # glyph pixels across the frame of the cell
            m = sample(g, np.clip((uy - 0.5) * size_frac + s / 2, 0, s - 1.001).astype(np.float32), np.clip((ux - 0.5) * size_frac + s / 2, 0, s - 1.001).astype(np.float32))
            m = gaussian_filter(m, 1.2)
            wear = fft_noise(H, W, 1.3, 63 + i)[ya:yb, xa:xb]
            worn = 0.55 + 0.45 * sm((fft_noise(H, W, 2.2, 53 + i)[ya:yb, xa:xb] + 1.0) / 1.6)      # shallow, and shallower in places
            m = np.clip((m - 0.4 + 0.1 * wear) / 0.34, 0, 1) * worn
            # the stone is cut, not printed: a light lip under each stroke
            lip = np.clip(np.roll(m, 3, 0) - m, 0, 1)
            masks.append((m.astype(np.float32), lip.astype(np.float32)))
        cells.append(dict(xa=xa, xb=xb, ya=ya, yb=yb, v=v.astype(np.float32), masks=masks))
    # the drops: three you can count, then quicker and quicker, until they run together
    r = np.random.default_rng(14)
    drops = []
    first = [2, 4, 3, 1, 2, 4]
    times = [0.3, 0.85, 1.3]
    gap = 0.34
    while times[-1] < 4.55:
        times.append(times[-1] + gap)
        gap = max(gap * 0.72, 0.042)
    for g in (0.09, 0.14, 0.2, 0.3):
        times.append(times[-1] + g)
    for k, t in enumerate(times):
        c = first[k] if k < len(first) else int(r.integers(0, 5))
        cx, cs = cell_of(c)
        drops.append((t, cx + r.uniform(-1.9, 1.9) + (1.25 if k < len(first) else 0), cs + r.uniform(-1.6, 1.6), c))
    # how far each cell has come at each drop: only drops on it move it, in steps, between its start and its finish
    steps = []
    for i in range(5):
        hits = [d[0] for d in drops if START[i] <= d[0] <= FINISH[i]]
        hits = sorted(set(hits[: max(3, len(hits))]))
        # everything lands somewhere: a cell takes a share of every drop in its span, so it steps with what is falling
        steps.append(np.array(hits, np.float32))
    # the small grid the rings are worked out on: slab coordinates of every point of the still
    sg = 4
    gy, gx = np.mgrid[0:H // sg, 0:W // sg].astype(np.float32) * sg
    ax, ay = np.array([-30.0, 30.0, 30.0, -30.0]), np.array([30.0, 30.0, -8.0, -8.0])
    xs, ys, _ = proj(ax, ay)
    hom = homography(np.stack([xs, ys], -1), np.stack([ax, ay], -1))
    gu, gv = apply_h(hom, gx.astype(np.float64), gy.astype(np.float64))
    _s.update(J=Jp, R=Rp, ratio=ratio.astype(np.float32), n=n, stone=stone.astype(np.float32), cells=cells, drops=drops, steps=steps, gu=gu.astype(np.float32), gv=gv.astype(np.float32), sg=sg)
    _flecks(drops)
    return _s


def _flecks(drops):
    """What each drop throws up: flecks of bare paper."""
    r = np.random.default_rng(23)
    out = []
    for t0, x, s, _ in drops:
        n = 26
        ang = r.uniform(0, 2 * math.pi, n)
        up = r.uniform(5, 15, n)
        far = r.uniform(1.5, 7.5, n)
        out.append((t0, x, s, ang, up, far, r.uniform(0.8, 1.1, n), r.uniform(1.0, 2.4, n)))
    _s["flecks"] = out


def _steady(x, ease=0.2):
    x, v = float(np.clip(x, 0, 1)), 1 / (1 - ease)
    return v * x * x / (2 * ease) if x < ease else 1 - v * (1 - x) ** 2 / (2 * ease) if x > 1 - ease else v * (x - ease / 2)


def _view(T):
    """The frame on the slab: starts on the tight frame, and draws back so that the top of the column comes in as it comes right."""
    z = BOX0 * (BOX1 / BOX0) ** _steady(T / PULL_END)
    w, h = W / z, H / z
    y0 = min(FIX_Y - (FIX_Y - 335.2) * BOX0 / z, H - h)
    return 960 - w / 2, y0, 960 + w / 2, y0 + h


def _rings(T):
    """The height of the film of water: as on the slab in Blender, rings going out from each drop."""
    s = _s
    h = np.zeros(s["gu"].shape, np.float32)
    n = 0
    for t0, x, y, _ in s["drops"]:
        a = T - t0
        if 0 < a < 1.25:
            r = np.hypot(s["gu"] - x, s["gv"] - y)
            front = 8.5 * a
            h += np.sin(2 * math.pi * (r - front) / 1.15) * np.exp(-((r - front) / (0.9 + 1.1 * a)) ** 2) * math.exp(-a * 1.9) / (1 + 0.1 * r) * sm(r / 0.5)
            n += 1
    return h, n


def _air(out, T, box_):
    """Drops on their way down, and the flecks of the ones that have landed."""
    s = _s
    g = 34.0
    fast = sm((T - 2.0) / 0.9) * (1 - sm((T - 4.6) / 0.3))
    for t0, x, sl, _ in s["drops"]:
        a = T - t0
        if -0.6 < a < 0:
            lift = 0.5 * g * a * a
            px, py, d = proj(x, sl, lift=lift)
            px, py, d = float(px), float(py), float(d)
            rad = FOCAL * 0.3 / d
            st = 1.15 + 1.1 * min(-a * 2, 1)
            hh, ww = rad * st, rad
            xa, xb, ya, yb = int(px - ww - 2), int(px + ww + 3), int(py - hh - 2), int(py + hh + 3)
            if xa < 0 or ya < 0 or xb >= W or yb >= H:
                continue
            yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
            uy, ux = (yy - py) / hh, (xx - px) / ww
            rr = np.hypot(ux, uy)
            # a ball of water: what is behind it the other way up, a dark rim, one small light
            c = sample(out, np.clip(py - uy * hh * 1.7, 0, H - 1.001), np.clip(px - ux * ww * 1.7, 0, W - 1.001))
            c = c * (1 - 0.62 * np.clip(rr, 0, 1)[..., None] ** 5)
            spot = np.exp(-(((ux + 0.28) / 0.2) ** 2 + ((uy + 0.42) / 0.16) ** 2))[..., None]
            c = c * (1 - spot * 0.85) + np.array([238, 234, 222], np.float32) * spot * 0.85
            al = (sm((1 - rr) / 0.14) * (0.96 - 0.6 * fast))[..., None]
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - al) + c * al
    # the fall itself, once they are too quick to see one by one: a pale thread from above down to where the last one is landing
    if fast > 0.02:
        for t0, x, sl, _ in s["drops"]:
            a = T - t0
            if -0.14 < a <= 0.02:
                x0, y0, _d0 = proj(x, sl, lift=0.5 * g * (a - 0.033) ** 2)
                x1, y1, _d1 = proj(x, sl, lift=0.5 * g * a * a if a < 0 else 0.0)
                n = int(max(abs(float(y1) - float(y0)), 2))
                ys = np.linspace(float(y0), float(y1), n).astype(np.int32)
                xs = np.linspace(float(x0), float(x1), n).astype(np.int32)
                ok = (ys >= 0) & (ys < H) & (xs >= 1) & (xs < W - 1)
                ys, xs = ys[ok], xs[ok]
                o = 0.42 * fast
                c = np.array([232, 228, 214], np.float32)
                for dx, w_ in ((0, 1.0), (1, 0.5), (-1, 0.5)):
                    out[ys, xs + dx] = out[ys, xs + dx] * (1 - o * w_) + c * o * w_
    for t0, x, sl, ang, up, far, sp, size in s["flecks"]:
        a = T - t0
        if not 0 < a < 0.62:
            continue
        dt = a * sp
        dist = far * dt
        z = up * dt - 0.5 * g * dt * dt
        live = z > -0.05
        if not live.any():
            continue
        fx = x + dist * np.cos(ang)
        fs = sl + dist * np.sin(ang) / SC
        px, py, _ = proj(fx[live], fs[live], lift=z[live])
        px, py = px.astype(np.int32), py.astype(np.int32)
        ok = (px >= 1) & (py >= 1) & (px < W - 2) & (py < H - 2)
        px, py = px[ok], py[ok]
        o = (0.85 * (1 - a / 0.62))
        c = np.array([236, 230, 216], np.float32)
        for dy, dx, w_ in ((0, 0, 1.0), (0, 1, 0.6), (1, 0, 0.6), (1, 1, 0.4)):
            out[py + dy, px + dx] = out[py + dy, px + dx] * (1 - o * w_) + c * o * w_


def slab(T):
    """The slab T seconds into 如此千万亿遍, as a frame. T runs straight through both shots, so the join between them does not show."""
    if not _s:
        _slab_prep()
    s = _s
    dl = s["drops"]
    fall = np.array([d[0] for d in dl])
    hits = int(np.searchsorted(fall, T))
    last = fall[hits - 1] if hits else -9.0
    # the wear of the whole slab goes by the drops, not the clock: a step for each, the step's own ramp is short
    E = min(hits / len(dl), 1.0)
    if hits and hits < len(dl):
        E = (hits - 1 + sm((T - last) / 0.1)) / len(dl)
    E = E ** 0.8 if E < 1 else 1.0
    n = s["n"]
    e = sm((E * 1.34 - n) / 0.34)[..., None]
    P = s["J"] * (1 - e) + s["R"] * e
    for i, c in enumerate(s["cells"]):
        xa, xb, ya, yb = c["xa"], c["xb"], c["ya"], c["yb"]
        st = s["steps"][i]
        k = int(np.searchsorted(st, T))
        f = 0.0
        if len(st):
            prev = (k - 1 + sm((T - st[k - 1]) / 0.1)) if k else 0.0
            f = float(np.clip(prev / max(len(st) - 1, 1), 0, 1))
        if T >= FINISH[i]:
            f = 1.0
        # the wrong ones: each drop on the cell moves it on, and it is pale from the start, paler as the right one opens
        masks = c["masks"]
        which = min(int(f * (len(masks) + 0.4)), len(masks) - 1)
        pale = PALE * (1 - sm((f - 0.5) / 0.5))
        region = P[ya:yb, xa:xb]
        a = sm((f * 1.36 - c["v"]) / 0.36)
        if pale > 0.005:
            m, lip = masks[which]
            kk = pale * m * (1 - a) * s["stone"][ya:yb, xa:xb]
            region = region * (1 - kk[..., None] * np.array([0.92, 0.9, 0.86], np.float32)) * (1 + 0.14 * (pale / PALE) * (1 - a[..., None]) * lip[..., None])
        q = s["ratio"][ya:yb, xa:xb]
        region = region * (1 + (q - 1) * a[..., None])
        P[ya:yb, xa:xb] = region
    alive = 0
    h, alive = _rings(T)
    if alive:
        h = h / max(1.0, (alive / 4.0) ** 0.34)      # many rings cross, they do not pile up
        gy, gx = np.gradient(h)
        dx = np.asarray(Image.fromarray(gx * 4.8 * 4).resize((W, H), Image.BILINEAR))
        dy = np.asarray(Image.fromarray(gy * 3.4 * 4).resize((W, H), Image.BILINEAR))
        lit = np.asarray(Image.fromarray(h).resize((W, H), Image.BILINEAR))
        P = sample(P, _Y + dy, _X + dx)
        P = P * (1 + np.clip(lit * 0.12, -0.14, 0.34)[..., None])
        spark = np.asarray(Image.fromarray(np.clip(gy * -0.8 * 4 - 0.12, 0, 1)).resize((W, H), Image.BILINEAR))[..., None]
        P = P * (1 - spark * 0.45) + np.array([232, 228, 214], np.float32) * spark * 0.45
    _air(P, T, None)
    return crop8(P, _view(T))


def junk(frame, t, dur):
    return slab(max(t, -0.3))


def right(frame, t, dur):
    return slab(T_SHOT + t)


# ---- the poem -------------------------------------------------------------------------------------------------------------------------------------------
POEM = ("鬼哭当年缘底事", "早知今夜字成精")
CHANT_T0, CHANT_STEP = 0.28, 0.2
COL_BOX = ((830, 990, 92, 1004), (452, 738, 92, 990))
_c = {}


def _chant_prep():
    P = plate("drip/chant")
    lum = np.array([0.30, 0.59, 0.11], np.float32)
    L = (P * lum).sum(-1)
    around = gaussian_filter(L, 30)
    C = P.copy()
    holes = []
    chars = []
    for b, col in zip(COL_BOX, POEM):
        x0, x1, y0, y1 = b
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        dark = (L < around * 0.72) & m
        from scipy.ndimage import binary_opening, label
        if x0 < 500:      # the wide box is for the chips by the last two; further up it would take in the stone
            dark[:600, :540] = False
        # the characters are the row profile's lumps: seven in each column
        prof = gaussian_filter(dark.sum(1).astype(np.float32), 3.0)
        rows = np.arange(H)
        cuts = np.linspace(y0, y1, 8)
        # let the cuts settle in the valleys of the profile
        for k in range(1, 7):
            lo, hi = int(cuts[k] - 28), int(cuts[k] + 28)
            cuts[k] = lo + prof[lo:hi].argmin()
        for k in range(7):
            ya, yb = int(cuts[k]), int(cuts[k + 1])
            chars.append((x0, x1, ya, yb, dark[ya:yb, x0:x1]))
        # flecks of cut stone beside the last strokes: bare white, in the hole too
        bright = (L > around * 1.28) & m
        holes.append(binary_dilation(dark, iterations=15) | binary_dilation(bright & binary_dilation(dark, iterations=30), iterations=3))
    hole = holes[0] | holes[1]
    C = fill(P, holes[0], sigma=18.0, offset=(0, 420), feather=2.5)
    C = fill(C, holes[1], sigma=18.0, offset=(0, 1060), feather=2.5)      # from the bare stone at the far right: nearer in, it lifts 低头思故乡 and the drop's rim into the hole
    # the order things appear in
    order = np.full((H, W), 1e6, np.float32)
    wet = []
    n = len(chars)
    for i, (x0, x1, ya, yb, ink) in enumerate(chars):
        pad = 14
        xa, xb, y_a, y_b = max(x0 - pad, 0), min(x1 + pad, W), max(ya - pad, 0), min(yb + pad, H)
        full = np.zeros((y_b - y_a, xb - xa), bool)
        full[ya - y_a:yb - y_a, x0 - xa:x1 - xa] = ink
        ord_ = stroke_order(full, gap=8.0)
        top = ord_[full].max() if full.any() else 1.0
        ord_ = ord_ / max(top, 1.0)
        wet.append(dict(xa=xa, xb=xb, ya=y_a, yb=y_b, order=np.minimum(ord_, 2.2).astype(np.float32), ink=gaussian_filter(full.astype(np.float32), 2.0)))
    # a wet patch lands and spreads and dries: each character's own wandering edge
    _c.update(P=P, C=C, chars=wet, wand=gaussian_filter(fft_noise(H, W, 2.6, 81), 3.0))
    return _c


def chant(frame, t, dur):
    """低头思故乡 is there. Then the water writes the rest, stroke by stroke, a wet patch opening where each drop lands."""
    if not _c:
        _chant_prep()
    c = _c
    out = c["C"].copy()
    for i, ch in enumerate(c["chars"]):
        t0 = CHANT_T0 + i * CHANT_STEP
        a = t - t0
        xa, xb, ya, yb = ch["xa"], ch["xb"], ch["ya"], ch["yb"]
        if a <= 0:
            continue
        p = a / 0.36
        # the strokes
        if p < 1.6:
            alpha = sm((p * 1.5 - ch["order"]) / 0.35)
            q = alpha[..., None]
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - q) + c["P"][ya:yb, xa:xb] * q
        else:
            out[ya:yb, xa:xb] = c["P"][ya:yb, xa:xb]
    # where the drops have landed: wet stone, darker, its rim darkest, drying
    for i, ch in enumerate(c["chars"]):
        t0 = CHANT_T0 + i * CHANT_STEP
        a = t - t0 + 0.04
        if not 0 < a < 1.5:
            continue
        cx, cy = (ch["xa"] + ch["xb"]) / 2, (ch["ya"] + ch["yb"]) / 2
        rad = 42 + 130 * (1 - math.exp(-a * 5.5))
        xa, xb, ya, yb = max(int(cx - rad - 44), 0), min(int(cx + rad + 44), W), max(int(cy - rad - 44), 0), min(int(cy + rad + 44), H)
        yy, xx = _Y[ya:yb, xa:xb], _X[ya:yb, xa:xb]
        r = np.hypot(xx - cx, (yy - cy) * 1.08) + 26 * c["wand"][ya:yb, xa:xb]
        dry = sm((1.5 - a) / 1.15) * sm(a / 0.05)
        body = sm((rad - r) / 22.0)
        rim = np.exp(-((r - rad) / 7.0) ** 2)
        k = (1 - (0.17 * body + 0.1 * rim) * dry)[..., None]
        out[ya:yb, xa:xb] *= k
    # the drop in the instant before it lands: a soft dark ball growing smaller
    for i, ch in enumerate(c["chars"]):
        t0 = CHANT_T0 + i * CHANT_STEP
        a = t - t0
        if -0.1 < a <= 0.0:
            cx, cy = (ch["xa"] + ch["xb"]) / 2, (ch["ya"] + ch["yb"]) / 2
            rad = 12 + 300 * (-a) ** 1.1
            cy2 = cy + 900 * (-a)
            xa, xb, ya, yb = max(int(cx - rad - 8), 0), min(int(cx + rad + 8), W), max(int(cy2 - rad - 8), 0), min(int(cy2 + rad + 8), H)
            if xa < xb and ya < yb:
                yy, xx = _Y[ya:yb, xa:xb], _X[ya:yb, xa:xb]
                r = np.hypot(xx - cx, yy - cy2)
                o = (sm((rad - r) / (3 + rad * 0.2)) * 0.5)[..., None]
                out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - o) + (out[ya:yb, xa:xb] * 0.45 + 22) * o
    return crop8(out, box((1.0, 1.05), (0.5, 0.5), t, dur))

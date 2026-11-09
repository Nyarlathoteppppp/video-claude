"""十, happening. 记忆如河水, 性情才是河道: into the dry bed the water comes from far off and runs down the channel stroke by stroke until the bed is a character; it goes on rising, over the ledges, up to the flood, and it falls back to the river. What stays is the shape.
Then, far up where the river starts, a new window lights: gamboge, slowly, with its reflection breaking on the water.
The stills have the three states of the same bed from the same camera. Where each pixel takes the water is worked out from their own water masks and depth.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from ink_color import absorb
from ink_events_lib import _X, _Y, H, W, box, crop8, plate, region_from, shot, up1
from ink_grade import fft_noise, sm
from ink_move import PAPER, sample

LUM = np.array((0.30, 0.59, 0.11), np.float32)
_w, _n = {}, {}

# Where the river comes from on the still: down the thin far reach from under the window to where the water mask begins, as (x, y)
FAR = ((948, 262), (938, 280), (925, 300), (893, 318), (862, 336))
RY0, RY1, RX0, RX1 = 180, 1080, 380, 1420      # the part of the still that ever changes
FX0, FX1 = 120, 1800                           # and of a frame
SOFT1, SOFT2 = 0.022, 0.14
T_DRY, T_FLOOD, T_SHAPE = 0.0, 2.2, 4.0        # each shot's start since the water began


def _levels(T):
    """How far the water has run down the bed (0 to past 1), and how far it stands up the ledges (below 0 to past 1, then back). One rise and one fall over the three shots."""
    x = np.clip((T - 0.08) / 1.8, 0.0, 1.0)
    run = (1 - (1 - x) ** 1.5) * 1.08
    rise = sm((T - 1.55) / 1.45)
    fall = sm((T - 3.2) / 1.5)
    return run, -SOFT2 + (1.14 + SOFT2 + SOFT2) * rise * (1 - fall)


def _arrival(m1, m2):
    """How long the water takes to get to each pixel of the bed, 0 to 1: along the channel from the far end, slowly through what it has to flood to get across a gap in the strokes."""
    gh, gw = H // 4, W // 4
    q1, q2 = m1[::4, ::4][:gh, :gw] > 0.5, m2[::4, ::4][:gh, :gw] > 0.5
    im = Image.new("L", (gw, gh), 0)
    ImageDraw.Draw(im).line([(x / 4, y / 4) for x, y in FAR], fill=255, width=2)
    far = np.asarray(im) > 0
    q = q1 | q2 | far
    cost = np.where(q1 | far, 1.0, 3.2)
    idx = -np.ones((gh, gw), np.int64)
    idx[q] = np.arange(q.sum())
    rows, cols, vals = [], [], []
    for dy, dx, ln in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
        ya, yb = (0, gh - dy), (dy, gh)
        if dx >= 0:
            sa, sb = (slice(*ya), slice(0, gw - dx)), (slice(*yb), slice(dx, gw))
        else:
            sa, sb = (slice(*ya), slice(-dx, gw)), (slice(*yb), slice(0, gw + dx))
        both = q[sa] & q[sb]
        rows.append(idx[sa][both])
        cols.append(idx[sb][both])
        vals.append(ln * (cost[sa][both] + cost[sb][both]) / 2)
    n = int(q.sum())
    G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    src = idx[int(FAR[0][1] / 4), int(FAR[0][0] / 4)]
    d = np.full((gh, gw), np.inf)
    d[q] = dijkstra(G, directed=False, indices=int(src))
    known = np.isfinite(d) & (q1 | far)
    top = np.percentile(d[known], 99.5)
    a = np.minimum(d / top, 1.0) * 0.96
    # everything off the water takes it from the nearest water, a little later
    dist, (iy, ix) = distance_transform_edt(~known, return_indices=True)
    a = a[iy, ix] + 0.0016 * dist
    return up1(a.astype(np.float32))


def _water_prep():
    from ink_cut import mover
    D, F, S = plate("channel/shape_dry"), plate("channel/shape_flood"), plate("channel/shape")
    mf, ms = mover(shot("channel/shape_flood")[0]), mover(shot("channel/shape")[0])
    m1, m2 = np.array(ms.wm, np.float32), np.array(mf.wm, np.float32)
    a1 = _arrival(m1, m2)
    n = gaussian_filter(fft_noise(H, W, 1.8, 61), 1.0)
    a1 = a1 + 0.008 * n / n.std()
    # up the ledges: how high the bed stands above the river, on the settled still, until the flood's own line
    h = np.array(ms.p.height, np.float32)
    inside = m2 > 0.5
    low = np.percentile(h[m1 > 0.9], 50)
    high = np.percentile(h[(m2 > 0.5) & (m1 < 0.1) & (h > 0.5)], 97)
    a2 = np.clip((h - low) / max(high - low, 1e-3), 0, 1)
    a2 = np.where(inside, a2, 1.0)
    a2 = np.where(m1 > 0.5, 0.0, a2)
    a2 = gaussian_filter(a2.astype(np.float32), 1.6)
    n2 = gaussian_filter(fft_noise(H, W, 1.8, 62), 2.0)
    a2 = a2 + 0.018 * n2 / n2.std() * (m1 < 0.5)
    print("channel: river at", low, "flood reaches", high, flush=True)
    _w.update(D=D, S=S, F=F, a1=a1.astype(np.float32), a2=a2.astype(np.float32))


def dry(frame, t, dur):
    if not _w:
        _water_prep()
    d = _w
    run, up = _levels(T_DRY + t)
    R = (slice(RY0, RY1), slice(RX0, RX1))
    a = sm((run - d["a1"][R]) / SOFT1)[..., None]
    r = d["D"][R] * (1 - a) + d["S"][R] * a
    if up > -SOFT2:
        b = sm((up - d["a2"][R]) / SOFT2)[..., None]
        r = r * (1 - b) + d["F"][R] * b
    out = d["D"].copy()
    out[R] = r
    return crop8(out, box((1.23, 1.26), (0.5, 0.6), t, dur))


def _with_other(frame, still, other, t, dur, up, k):
    """Over a frame of still with water in it, the other still where the level says, sampled where the frame's pixels come from."""
    d = _w
    ys, xs = region_from(still, t, dur, (0, H, FX0, FX1))
    yi, xi = np.clip(ys + 0.5, 0, H - 1).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)
    b = sm((up - d["a2"][yi, xi]) / SOFT2)[..., None]
    if k:
        b = 1 - b
    o = sample(d[other], ys, xs)
    out = frame.astype(np.float32)
    out[:, FX0:FX1] = out[:, FX0:FX1] * (1 - b) + o * b
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def flood(frame, t, dur):
    if not _w:
        _water_prep()
    _, up = _levels(T_FLOOD + t)
    return _with_other(frame, "channel/shape_flood", "S", t, dur, up, True)


def settled(frame, t, dur):
    if not _w:
        _water_prep()
    _, up = _levels(T_SHAPE + t)
    if up <= -SOFT2:
        return frame
    return _with_other(frame, "channel/shape", "F", t, dur, up, False)


# ---- the new window ------------------------------------------------------------------------------------------------------------------------------------------------------
WIN = (948.0, 247.0)
PY0, PY1, PX0, PX1 = 130, 412, 820, 1076
C1 = 0.535             # the camera lifts to here: at its own framing the window would sit against the top edge of the picture


def _lit(t):
    return sm((t - 0.06) / 0.62)


def _newwin_prep():
    from ink_cut import mover
    S, N = plate("channel/shape"), plate("channel/newwin")
    warm = (PAPER * np.exp(-0.5 * absorb("teng"))).astype(np.float32)
    warm = warm * (255.0 if warm.max() <= 2.0 else 1.0)
    P = (slice(PY0, PY1), slice(PX0, PX1))
    s, dl = S[P], (N[P] - S[P])
    yy, xx = _Y[P].astype(np.float32), _X[P].astype(np.float32)
    r = np.hypot(xx - WIN[0], yy - WIN[1])
    L = (dl * LUM).sum(-1)
    wl = (warm * LUM).sum()
    light = np.clip(L / np.maximum(wl - (s * LUM).sum(-1), 20.0), 0, 1)
    light = light * sm((96 - r) / 30)
    upper = light * sm((262.0 - yy) / 3.0)            # the window and its glow
    glow = (0.15 * np.exp(-((r / 44.0) ** 2)) + 0.05 * np.exp(-((r / 80.0) ** 2))) * sm((118.0 - r) / 50.0)
    dark = np.minimum(dl, 0) * (r < 13)[..., None]      # the window's frame, there before there is any light behind it
    # where it can shine back from: the water on this still
    ms = mover(shot("channel/shape")[0])
    G = (S * LUM).sum(-1)
    wet = (G > gaussian_filter(G, 16.0) + 9.0) | (ms.wm > 0.5)
    wet = gaussian_filter(wet.astype(np.float32), 1.0)[P]
    below = np.clip(light * sm((yy - 258.0) / 3.0), 0, 1)
    _n.update(s=s, warm=warm, upper=upper.astype(np.float32), glow=glow.astype(np.float32), dark=dark, wet=wet, below=below.astype(np.float32), xx=xx, yy=yy, S=S, P=P, rows=np.arange(PY0, PY1, dtype=np.float32))
    rng = np.random.default_rng(71)
    _n["phase"] = rng.uniform(0, 6.28, (PY1 - PY0, 3)).astype(np.float32)
    _n["chop"] = rng.uniform(0, 1, (PY1 - PY0, 64)).astype(np.float32)


def _broken(t):
    """The reflection, row by row: each row of water takes the light for a moment and lets it go, a little to one side."""
    d = _n
    rows = d["rows"]
    far = np.clip((rows - 262.0) / 136.0, 0, 1)
    ph = d["phase"]
    shift = (np.sin(t * 17.0 + ph[:, 0]) * 0.7 + np.sin(t * 31.0 + ph[:, 1]) * 0.5) * (0.7 + 3.6 * far)
    col = np.where(rows < 300, 948.0 - np.maximum(rows - 268.0, 0) * 0.6, 945.5)
    i = int(t * 24.0) % 63
    f = t * 24.0 - int(t * 24.0)
    on = d["chop"][:, i] * (1 - f) + d["chop"][:, i + 1] * f
    on = sm((on - 0.26) / 0.3) * (1 - 0.55 * far)
    wide = 1.3 + 3.8 * far
    x = (d["xx"] - (col + shift)[:, None]) / wide[:, None]
    return np.exp(-x * x) * on[:, None] * d["wet"] * ((rows > 259) & (rows < 400))[:, None]


def newwin(frame, t, dur):
    if not _n:
        _newwin_prep()
    d = _n
    p = sm(t / max(dur, 1e-3))
    b = box((1.3, 1.32), (0.5, 0.6 + (C1 - 0.6) * p), min(t, dur + 0.8), dur)
    lit = _lit(t)
    if t <= 0.0:
        return crop8(d["S"], b)
    reach = sm((t - 0.14) / 0.7)
    s = d["s"]
    a = d["upper"] * lit + d["glow"] * lit * lit
    # the panes come up from a dull amber: at first the light is low and red in the glass, not a white lamp being faded in
    amber = d["warm"] * np.array((0.86, 0.66, 0.44), np.float32)
    c = amber + (d["warm"] - amber) * lit
    a = np.clip(a + (d["below"] * 0.55 + 0.8 * _broken(t)) * reach, 0, 1)[..., None]
    out = d["S"].copy()
    out[d["P"]] = s + d["dark"] * sm(t / 0.14) + (c - s) * a
    return crop8(out, b)

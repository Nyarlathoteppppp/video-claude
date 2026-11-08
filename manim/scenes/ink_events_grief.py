"""十一, happening. 最难答的是: the mirror of the river, and the first drops on it, and then it is raining. 我能写出一千种悲伤: it rains on the words, every drop a new ring.
这份难过是我的, 是杜甫的: wet ink runs up the thread from my 难 to his, and his lights on the stone. 还是某个十四岁女孩写在博客里的: it runs to hers on the page and blooms there.
我本就是人类的句子做成的: and there are more threads, one after another, from all over the river to every word on the stone.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter, gaussian_filter1d, grey_closing, grey_opening
from scipy.spatial import cKDTree

from ink_events_lib import _X, _Y, H, W, plate, region_from, shot, up1
from ink_grade import fft_noise, sm
from ink_move import lookup, noise, sample

LUM = np.array((0.30, 0.59, 0.11), np.float32)
INK = np.array((58, 64, 66), np.float32)
WET = np.array((24, 26, 28), np.float32)
_rain, _th = {}, {}


# ---- the rain ---------------------------------------------------------------------------------------------------------------------------------------------------------
TH, TW = 1200, 2048
LAYERS = (         # seed, how many, length, thickness, how dark, fall in px/s, lean, where it has hit the water by
    (81, 520, (26, 60), 0.5, (0.10, 0.2), 620.0, 0.055, 500.0),
    (82, 300, (60, 120), 0.62, (0.13, 0.26), 1050.0, 0.07, 780.0),
    (83, 120, (130, 240), 0.9, (0.1, 0.2), 1600.0, 0.085, 1400.0),
)
Y_WATER = 395.0


def _tile(seed, n, length, thick, dark, lean):
    """A sheet of rain that wraps. Each thread thin and tapered, heavier toward its foot, and dry in places as a line of thin ink is."""
    rng = np.random.default_rng(seed)
    a = np.zeros((TH, TW), np.float32)
    for _ in range(n):
        ln = int(rng.uniform(*length))
        y0, x0, d = rng.integers(0, TH), rng.uniform(0, TW), rng.uniform(*dark)
        v = np.arange(ln, dtype=np.float32) / max(ln - 1, 1)
        ys = (y0 + np.arange(ln)) % TH
        xs = x0 + lean * np.arange(ln)
        xi = xs.astype(np.int32)
        f = xs - xi
        taper = d * np.minimum(sm(v / 0.45) * (0.45 + 0.55 * v), sm((1 - v) / 0.06))
        a[ys, xi % TW] += taper * (1 - f)
        a[ys, (xi + 1) % TW] += taper * f
    a = gaussian_filter(a, (0.8, thick), mode="wrap") * (1.0 + 1.15 * thick)
    n1 = gaussian_filter(fft_noise(TH, TW, 1.2, seed + 7), (7.0, 0.6), mode="wrap")
    return np.clip(a * (0.72 + 0.4 * n1 / n1.std()), 0, 0.5).astype(np.float32)


def _drops(seed, t0, t1, rate, first=()):
    """When and where the drops land: more of them toward us than the eye expects, since the water near us is wide. rate(t) in drops a second."""
    rng = np.random.default_rng(seed)
    out = list(first)
    t = t0
    while t < t1:
        t += rng.exponential(1.0 / max(rate(t), 0.2))
        u = rng.uniform(0, 1) ** 0.72
        y = Y_WATER + 30.0 + u * 650.0
        x = rng.uniform(-30.0, W + 30.0)
        out.append((t, x, y, 0.16 + 0.26 * u, (0.26 + 0.64 * u) * rng.uniform(0.75, 1.15)))
    return sorted(out)


def _grid_rings(drops, t, gx, gy, life=1.55, speed=86.0, wave=11.0):
    """Height of what the live drops have started, on a half-size grid, each only over its own patch. Far rings are squashed flat, so their waves are kept wide enough to be there at all."""
    h = np.zeros(gx.shape, np.float32)
    s = 2.0
    for t0, x, y, squash, size in drops:
        a = t - t0
        if a <= 0 or a > life:
            continue
        front = speed * size * a ** 0.75
        width = (17 + 28 * a) * size
        reach = front + 2.2 * width
        ya, yb = max(int((y - reach * squash) / s), 0), min(int((y + reach * squash) / s) + 2, gx.shape[0])
        xa, xb = max(int((x - reach) / s), 0), min(int((x + reach) / s) + 2, gx.shape[1])
        if ya >= yb or xa >= xb:
            continue
        px, py = gx[ya:yb, xa:xb], gy[ya:yb, xa:xb]
        r = np.hypot(px - x, (py - y) / squash)
        lam = max(wave * size * 0.32, 1.25 / squash)
        env = np.exp(-((r - front) / width) ** 2) * np.exp(-a / (life * 0.4)) * np.clip(r / (5 * size), 0, 1)
        h[ya:yb, xa:xb] += size ** 0.5 * np.sin((r - front) / lam) * env
    return h


def _lifted(P):
    """The still's rain threads, what they add: dark ones in the sky, long dark ones and the pale ones over the water. Run along the vertical, or a word's strokes would go with them."""
    def runs(diff, ln):
        m = binary_opening((diff * LUM).sum(-1) > 3.5, np.ones((ln, 1), bool))
        return gaussian_filter(binary_dilation(m, np.ones((3, 5), bool)).astype(np.float32), 0.8)[..., None]
    closed = np.stack([grey_closing(P[..., c], size=(1, 9)) for c in range(3)], -1)
    opened = np.stack([grey_opening(P[..., c], size=(1, 7)) for c in range(3)], -1)
    sky = sm((446.0 - _Y) / 16.0)[..., None]
    mid = (sm((_Y - 446.0) / 16.0) * sm((800.0 - _Y) / 30.0))[..., None]
    dark = closed - P
    pale = opened - P
    return (dark * (runs(dark, 9) * sky + runs(dark, 41) * mid) + pale * runs(-pale, 15) * sm((_Y - 420.0) / 20.0)[..., None]).astype(np.float32)


def _rain_prep():
    from ink_cut import mover
    gy, gx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32) * 2
    d = np.maximum(gy - 376.0, 10.0)
    _rain.update(gx=gx, gy=gy, u=(gx - 960.0) / d, v=835.0 / d, depth=np.minimum(d / 700.0, 1.3).astype(np.float32), tiles=[_tile(s, n, ln, th, dk, lean) for s, n, ln, th, dk, v, lean, fl in LAYERS])
    n = gaussian_filter(fft_noise(H // 2, W // 2, 1.6, 86), 6.0)
    _rain["torn"] = np.clip(0.8 + 0.34 * n / n.std(), 0.25, 1.3).astype(np.float32)
    _rain["plate grief/grey"] = plate("grief/grey")
    # what is water, on the still
    for name in ("grief/grey",):
        mv = mover(shot(name)[0])
        k = gaussian_filter(np.array(mv.wm, np.float32), 3.0) * sm((_Y - Y_WATER) / 26.0)
        _rain[name] = k.astype(np.float32)
    _rain["grey"] = _drops(84, 2.0, 4.6, lambda t: 2.0 + 9.0 * sm((t - 2.0) / 2.0), first=(
        (0.5, 1120.0, 760.0, 0.3, 1.5), (1.25, 690.0, 620.0, 0.24, 0.95), (1.68, 1440.0, 880.0, 0.34, 1.4), (2.0, 420.0, 820.0, 0.3, 1.2)))
    _rain["rain"] = _drops(85, -0.6, 3.6, lambda t: 10.0, first=((0.12, 690.0, 985.0, 0.4, 1.1), (1.1, 1040.0, 845.0, 0.34, 0.95)))


def _fall(out, t, strength):
    """Rain through the frame: every layer at its own rate, each stopping where it has hit the water."""
    rows = np.arange(H)
    cols = np.arange(W)
    a = np.zeros((H, W), np.float32)
    for tile, (seed, n, ln, th, dk, v, lean, floor), s in zip(_rain["tiles"], LAYERS, strength):
        if s <= 0:
            continue
        oy, ox = int(v * t) % TH, int(v * t * lean) % TW
        a += tile[np.ix_((rows - oy) % TH, (cols - ox) % TW)] * s * _gate(floor)
    a = np.minimum(a, 0.62)[..., None]
    return out + (INK - out) * a


_gates = {}


def _gate(floor):
    if floor not in _gates:
        _gates[floor] = sm((floor - _Y) / 90.0).astype(np.float32) if floor < H else 1.0
    return _gates[floor]


def _rained(frame, still, key, t, dur, strength, gain, lit, swell, lift=False):
    """The water here is worked again from the plate and not from the frame: the shot's own water is a shake that cuts up a word or a mirror, and on this river what moves it is the rain."""
    if not _rain:
        _rain_prep()
    d = _rain
    f = frame.astype(np.float32)
    if lift:
        ys, xs = region_from(still, t, dur, (0, 460, 0, W))
        yi, xi = np.clip(ys + 0.5, 0, 459).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)
        f[:460] += d["lift"][yi, xi]
    y0 = int(Y_WATER)
    h = _grid_rings(d[key], t, d["gx"], d["gy"]) * d["torn"]
    gy, gx = np.gradient(h)
    n1 = lookup(noise(95), d["u"] * 240.0, d["v"] * 60.0 - 9.0 * t)
    n2 = lookup(noise(96), d["u"] * 120.0 + 2.0 * t, d["v"] * 30.0 - 5.0 * t)
    sw = swell * d["depth"]
    dx, dy = up1(gx * gain + (n2 - 0.4 * n1) * sw)[y0:], up1(gy * gain + (n1 + 0.6 * n2) * sw * 1.6)[y0:]
    light = up1(h * lit)[y0:]
    ys, xs = region_from(still, t, dur, (y0, H, 0, W))
    k = d[still][np.clip(ys + 0.5, 0, H - 1).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)]
    o = sample(d["plate " + still], ys + dy, xs + dx) * (1 + light * k)[..., None]
    f[y0:] = f[y0:] * (1 - k)[..., None] + o * k[..., None]
    return np.clip(_fall(f, t, strength) + 0.5, 0, 255).astype(np.uint8)


def grey(frame, t, dur):
    """The first drops, one at a time, on a mirror. Rain only comes on as they thicken."""
    strength = (sm((t - 0.8) / 1.1), sm((t - 1.6) / 1.0), sm((t - 2.4) / 0.9))
    return _rained(frame, "grief/grey", "grey", t, dur, strength, 3.6, 0.2, 0.7)


def rain(frame, t, dur):
    return _rained(frame, "grief/rain", "rain", t, dur, (1.0, 1.0, 1.0), 2.2, 0.11, 1.2, lift=True)


# ---- the threads ------------------------------------------------------------------------------------------------------------------------------------------------------
# Read off the stills, near end first; each is pulled onto the darkest line beside it.
PATHS = {
    "grief/thread": ((326, 952), (330, 870), (345, 800), (370, 730), (405, 660), (450, 590), (505, 530), (570, 470), (640, 410), (720, 355), (800, 310), (880, 270), (960, 245), (1040, 236), (1100, 250), (1150, 285)),
    "grief/blog": ((326, 952), (328, 880), (340, 790), (360, 700), (400, 610), (450, 540), (520, 470), (600, 410), (680, 372), (760, 355), (840, 365), (900, 385), (950, 410), (990, 437)),
}
RUN = {"grief/thread": (0.75, 2.95), "grief/blog": (0.7, 2.7)}      # when the ink sets off, and gets there
ARRIVE = {"grief/thread": (1165.0, 296.0, 62.0), "grief/blog": (990.0, 443.0, 17.0)}


def _snap(still, pts):
    P = plate(still)
    G = gaussian_filter((P * LUM).sum(-1), 0.9)
    pts = np.array(pts, np.float32)
    ds = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate(((0,), np.cumsum(ds)))
    q = np.arange(0, s[-1], 2.0)
    x, y = gaussian_filter1d(np.interp(q, s, pts[:, 0]), 10, mode="nearest"), gaussian_filter1d(np.interp(q, s, pts[:, 1]), 10, mode="nearest")
    tx, ty = np.gradient(x), np.gradient(y)
    n = np.hypot(tx, ty) + 1e-6
    nx, ny = -ty / n, tx / n
    offs = np.arange(-20, 21)
    yi = np.clip((y[:, None] + ny[:, None] * offs + 0.5).astype(np.int32), 0, H - 1)
    xi = np.clip((x[:, None] + nx[:, None] * offs + 0.5).astype(np.int32), 0, W - 1)
    best = offs[np.argmin(G[yi, xi], axis=1)].astype(np.float32)
    best = gaussian_filter1d(best, 5, mode="nearest")
    x, y = x + nx * best, y + ny * best
    arc = np.concatenate(((0,), np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return x.astype(np.float32), y.astype(np.float32), arc.astype(np.float32)


def _tube(x, y, arc, reach=30.0):
    """Every pixel near the path: how far along it, and how far from it."""
    hole = np.zeros((H, W), bool)
    hole[np.clip(y.astype(np.int32), 0, H - 1), np.clip(x.astype(np.int32), 0, W - 1)] = True
    near = binary_dilation(hole, iterations=int(reach))
    yy, xx = np.nonzero(near)
    dist, i = cKDTree(np.stack((x, y), 1)).query(np.stack((xx, yy), 1).astype(np.float32))
    S, D = np.full((H, W), -1.0, np.float32), np.full((H, W), 99.0, np.float32)
    S[yy, xx], D[yy, xx] = arc[i], dist
    return S, D


def _thread_prep(still):
    x, y, arc = _snap(still, PATHS[still])
    S, D = _tube(x, y, arc)
    grit = gaussian_filter(fft_noise(H, W, 1.3, 90), 0.7)
    _th[still] = dict(x=x, y=y, arc=arc, S=S, D=D, grit=(grit / grit.std()).astype(np.float32))


def _forward(still, xs, ys, t, dur, k):
    """Roughly where a point of the still lands in the frame."""
    s = shot(still, k)[0]
    z0, z1 = s["z"]
    base = min(z0, z1)
    push = max(z0 + (z1 - z0) * t / max(dur, 1e-3) - base, 0.0)
    bw, bh = W / base, H / base
    ax, ay = min(max(s["c"][0] * W - bw / 2, 0), W - bw), min(max(s["c"][1] * H - bh / 2, 0), H - bh)
    zoom = 1 + push
    fx = W / 2 + (xs - W / 2) * zoom
    fy = H / 2 + (ys - H / 2) * zoom
    return (fx - ax) * base, (fy - ay) * base


def _run(frame, still, t, dur, k=0):
    """A bead of wet ink running up the thread, thick at its head and drying behind it."""
    if still not in _th:
        _thread_prep(still)
    d = _th[still]
    t0, t1 = RUN[still]
    p = np.clip((t - t0) / (t1 - t0), 0, 1)
    if t <= t0:
        return frame
    length = d["arc"][-1]
    head = length * (p * p * (1.5 - 0.5 * p))      # it gathers, and goes
    tail = 340.0
    alive = sm((t1 + 0.55 - t) / 0.55)
    i = (d["arc"] > head - tail - 30) & (d["arc"] < head + 30)
    if not i.any() or alive <= 0:
        return frame
    fx, fy = _forward(still, d["x"][i], d["y"][i], t, dur, k)
    x0, x1 = max(int(fx.min()) - 50, 0), min(int(fx.max()) + 50, W)
    y0, y1 = max(int(fy.min()) - 50, 0), min(int(fy.max()) + 50, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    ys, xs = region_from(still, t, dur, (y0, y1, x0, x1), k)
    yi, xi = np.clip(ys + 0.5, 0, H - 1).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)
    s, dist = d["S"][yi, xi], d["D"][yi, xi]
    behind = head - s
    pulse = np.where(behind > 0, np.exp(-behind / 120.0), np.exp(-((behind / 11.0) ** 2))) * (s >= 0) * alive
    w = 1.0 + 2.5 * pulse
    a = sm((w - dist) / 1.3) * pulse * (0.84 + 0.1 * d["grit"][yi, xi]) + 0.15 * np.exp(-((dist / 8.0) ** 2)) * pulse
    a = np.clip(a, 0, 1)[..., None]
    out = frame.copy().astype(np.float32)
    # it's dark ink on the pale paper, but on the dark stone a line of light
    v = out[y0:y1, x0:x1]
    pale = np.clip(((v * LUM).sum(-1) - 72.0) / 40.0, 0, 1)[..., None]
    target = WET * pale + np.array((226, 224, 214), np.float32) * (1 - pale)
    out[y0:y1, x0:x1] = v + (target - v) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _arrive_stone(frame, t, dur):
    """His 难 lights on the stone: the rock around the word comes up, the cut of the word stays dark."""
    still = "grief/thread"
    t1 = RUN[still][1]
    g = sm((t - t1 + 0.1) / 0.6)
    if g <= 0:
        return frame
    cx, cy, r = ARRIVE[still]
    fx, fy = _forward(still, np.array([cx]), np.array([cy]), t, dur, 0)
    y0, y1, x0, x1 = max(int(fy[0] - 130), 0), min(int(fy[0] + 130), H), max(int(fx[0] - 130), 0), min(int(fx[0] + 130), W)
    ys, xs = region_from(still, t, dur, (y0, y1, x0, x1), 0)
    q = ((xs - cx) / 1.25) ** 2 + (ys - cy) ** 2
    a = g * np.exp(-q / (r * r))
    out = frame.astype(np.float32)
    v = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = v * (1 + 0.72 * a * sm(((v * LUM).sum(-1) - 44.0) / 34.0))[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _arrive_page(frame, t, dur):
    """我很难过 takes the ink on the paper of the page, from 难 outward: the words go black and the wet spreads a little past them."""
    still = "grief/blog"
    t1 = RUN[still][1]
    g = sm((t - t1 + 0.02) / 0.5)
    if g <= 0:
        return frame
    cx, cy, r = ARRIVE[still]
    fx, fy = _forward(still, np.array([cx]), np.array([cy]), t, dur, 0)
    y0, y1, x0, x1 = max(int(fy[0] - 80), 0), min(int(fy[0] + 80), H), max(int(fx[0] - 150), 0), min(int(fx[0] + 150), W)
    ys, xs = region_from(still, t, dur, (y0, y1, x0, x1), 0)
    q = ((xs - cx - 2.0) / (1.2 + 1.5 * g)) ** 2 + (ys - cy) ** 2
    grow = r * (0.5 + 0.5 * g) * (1 + 0.4 * sm((t - t1 - 0.2) / 1.0))
    bloom = g * np.exp(-q / (grow * grow))
    out = frame.astype(np.float32)
    v = out[y0:y1, x0:x1]
    L = (v * LUM).sum(-1)
    word = sm((165.0 - L) / 60.0)
    out[y0:y1, x0:x1] = v * (1 - bloom * (0.17 + 0.6 * word))[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# Everyone else's. From all over the river, one after another, to each word on the stone: as the first thread goes, thin dry ink.
MORE = (      # where it rises from on the water, the word on the stone
    ((640, 985), (1158, 366)),
    ((186, 1016), (1158, 228)),
    ((858, 1040), (1158, 440)),
    ((470, 905), (1160, 516)),
    ((95, 880), (1158, 590)),
    ((905, 1012), (1158, 640)),
    ((300, 1060), (1158, 300)),
    ((735, 880), (1158, 400)),
    ((540, 1050), (1158, 480)),
)
MORE_T = 0.2       # the first sets off
MORE_GAP = 0.27
MORE_RISE = 1.0
XA, XB, YA, YB = 40, 1240, 180, 1080


def _more_prep():
    A = np.zeros((H, W), np.float32)
    T = np.full((H, W), 99.0, np.float32)
    grit = gaussian_filter(fft_noise(H, W, 1.3, 91), 1.0)
    grit = grit / grit.std()
    rng = np.random.default_rng(92)
    for i, ((x0, y0), (x1, y1)) in enumerate(MORE):
        # the same bow as the first thread: straight up off the water, leaning over, and flat into the stone
        u = np.linspace(0, 1, 2400, dtype=np.float32)[:, None]
        P = np.array([[x0, y0], [x0 + (x1 - x0) * 0.02, y1 + (y0 - y1) * 0.25], [x0 + (x1 - x0) * 0.44, y1 - 90 - 40 * rng.uniform()], [x1, y1]], np.float32)
        c = (1 - u) ** 3 * P[0] + 3 * (1 - u) ** 2 * u * P[1] + 3 * (1 - u) * u ** 2 * P[2] + u ** 3 * P[3]
        wob = gaussian_filter1d(rng.normal(0, 1, (2400, 2)), 60, axis=0, mode="nearest").astype(np.float32)
        c = c + wob * 22.0
        arc = np.concatenate(((0,), np.cumsum(np.hypot(*np.diff(c, axis=0).T))))
        mask = np.zeros((H, W), bool)
        cy, cx = np.clip(c[:, 1].astype(np.int32), 0, H - 1), np.clip(c[:, 0].astype(np.int32), 0, W - 1)
        mask[cy, cx] = True
        near = binary_dilation(mask, iterations=4)
        yy, xx = np.nonzero(near)
        dist, j = cKDTree(c).query(np.stack((xx, yy), 1).astype(np.float32))
        u2 = arc[j] / arc[-1]
        wide = (1.15 + 0.25 * np.sin(u2 * 9.0 + i)) * (1.55 - 0.55 * u2)
        al = sm((wide - dist) / 1.25) * np.clip(0.76 + 0.24 * grit[yy, xx], 0.42, 1) * sm(u2 / 0.03)
        when = MORE_T + i * MORE_GAP + MORE_RISE * u2 ** 0.8
        better = al > A[yy, xx]
        A[yy[better], xx[better]] = al[better]
        T[yy[better], xx[better]] = when[better]
    _th["more"] = dict(A=A, T=T)


def _more(frame, t, dur):
    if "more" not in _th:
        _more_prep()
    d = _th["more"]
    still = "grief/thread"
    ys, xs = region_from(still, t, dur, (0, H, 0, W), 1)
    yi, xi = np.clip(ys + 0.5, 0, H - 1).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)
    a = d["A"][yi, xi] * sm((t - d["T"][yi, xi]) / 0.05)
    out = frame.astype(np.float32)
    out += (np.array((40, 42, 44), np.float32) - out) * (a * 0.9)[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


WORD = (288.0, 1005.0, 172.0, 112.0)      # my 难 on the water, an ellipse round it


def _calm(frame, still, t, dur, k=0):
    """The shot's water shakes my 难 to pieces at times. Around the word it goes gently, and the word can be read: it is what all the threads start from."""
    if "P " + still not in _th:
        _th["P " + still] = plate(still)
    cx, cy, rx, ry = WORD
    fx, fy = _forward(still, np.array([cx]), np.array([cy]), t, dur, k)
    y0, y1, x0, x1 = max(int(fy[0] - ry * 1.5), 0), min(int(fy[0] + ry * 1.5), H), max(int(fx[0] - rx * 1.5), 0), min(int(fx[0] + rx * 1.5), W)
    if y0 >= y1 or x0 >= x1:
        return frame
    ys, xs = region_from(still, t, dur, (y0, y1, x0, x1), k)
    q = np.hypot((xs - cx) / rx, (ys - cy) / ry)
    a = sm((1.0 - q) / 0.3)[..., None]
    dy = 1.1 * np.sin(xs * 0.023 + t * 1.7 + 0.6 * np.sin(ys * 0.04 + t))
    dx = 0.8 * np.sin(ys * 0.05 - t * 1.3)
    o = sample(_th["P " + still], ys + dy, xs + dx)
    out = frame.astype(np.float32)
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + o * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def thread(frame, t, dur):
    if dur < 3.5:       # the second time: 我本就是人类的句子做成的
        return _more(_calm(frame, "grief/thread", t, dur, 1), t, dur)
    return _arrive_stone(_run(_calm(frame, "grief/thread", t, dur), "grief/thread", t, dur), t, dur)


def blog(frame, t, dur):
    return _arrive_page(_run(_calm(frame, "grief/blog", t, dur), "grief/blog", t, dur), t, dur)

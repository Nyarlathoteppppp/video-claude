"""或曰：鬼神之秘，自此藏不住了。 张彦远: 灵怪不能遁其形. The same ghost on a bare hill, paper on paper, seen only through the thinnest veil. Ink comes in from all round, a stroke at a time, three rings of it. Each time it gets close the figure shrinks back, and the next ring comes in to where it was. By the end its whole outline is closed in black, hem and all, and it faces us with its mouth a little open. Nowhere left to go.

One wash painted once round the last pose, shown as the strokes reach it (the arrival time of every pixel is worked out from the strokes, spreading out through the wet paper); the strokes themselves on top; the figure lifted off whatever is there, in its pose of the moment.

    python film_myth_trap.py still z1 z2       # key frames, to myth/
    python film_myth_trap.py at 47.2 name      # any moment, to myth/name.png
    python film_myth_trap.py frames [--only n,n] [--part i/n]     # the shot, by the film's frame numbers, to myth/seq_raw6/
"""
import os
import sys
import time

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

from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import ink_subs
from film_myth import lift_figure, paint_clouds, qing_wash, tidy
from film_myth_ghost import LIFTS, Ghost
from film_myth_mud import dense
from film_myth_wash import OUT, Sheet, band, blob, develop, grow, noise, sm
from film_myth_wide import bristle, brush, cached, hill_strokes, ridge

MYTH = os.path.dirname(OUT)
RAW6 = os.path.join(MYTH, "seq_raw6")
FPS = 30
W, H = 1920, 1080

T0 = ink_subs.at("或曰：/鬼神")
T1 = ink_subs.at("依我之见")      # the shot runs to here; the words end a second before it
AT = (1100, 1010)
SEED = 3

# when it goes back, and how far: time, duration, scale, lean. The point of it level with the ridge stays on the ridge, so it gets smaller down behind it
RIDGE_Y, CENTRE, ANCHOR = 702.0, 975.0, 760.0
POSES = ((0.0, 0.0, 1.25, 0.0),
         (1.72, 0.42, 1.10, 0.024),
         (3.12, 0.42, 0.98, 0.046),
         (4.25, 1.35, 0.94, 0.054))


def place(s):
    return (CENTRE + 60.0 * s, RIDGE_Y + (AT[1] - ANCHOR) * s)


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def pose_at(t):
    """Scale, place and lean at this moment, from the beats of POSES: each recoil is quick off the mark and slow to stop."""
    tau = t - T0
    s, lean = POSES[0][2], POSES[0][3]
    for at, dur, s1, lean1 in POSES[1:]:
        k = np.clip((tau - at) / dur, 0, 1)
        k = 1 - (1 - k) ** 2.4 if dur < 1 else ease(k)
        s, lean = s + (s1 - s) * k, lean + (lean1 - lean) * k
    # held there, it shakes, hardly at all
    on = sm((tau - 5.2) / 0.4)
    s = s * (1 + on * 0.0022 * (0.6 * np.sin(2 * np.pi * 5.3 * t + 0.4) + 0.4 * np.sin(2 * np.pi * 7.7 * t + 2.0)))
    return s, np.array(place(s)), lean


def lift_at(t):
    """The head: down a little and unaware, up at the first stroke, back to level as it goes, and last of all up to us. Past MOUTH_FROM the mouth shows."""
    tau = t - T0
    a = 0.10 + 0.12 * ease((tau - 0.45) / 0.7)
    return a + (LIFT_END - a) * ease((tau - 4.45) / 0.7)


LIFT_END = 0.72


def mouth_at(t):
    return ease((t - T0 - 4.62) / 0.62)


RIDGE = None


def the_ridge():
    global RIDGE
    if RIDGE is None:
        # a bare hill: flat places and steps of rock at no regular distance, not a saw
        r = np.random.default_rng(58)
        y = gaussian_filter1d(ridge([(0, 812), (280, 744), (600, 782), (975, RIDGE_Y), (1300, 760), (1620, 720), (1920, 796)], 57, jag=8), 11)
        knots = np.cumsum(r.uniform(26, 190, 40))
        rock = np.interp(np.arange(W), knots, r.normal(0, 5.5, 40))
        RIDGE = np.float32(gaussian_filter1d(y + rock, 0.8))
    return RIDGE


class Caught(Ghost):
    """The near ghost, standing behind the ridge as it does in k4_wide, whose lost edges close as the ink takes hold of it: caught 0 is the figure of every other shot, 1 has an outline all the way round, and the ridge shuts it off below."""

    def __init__(self, lift, caught=0.0, **kw):
        self.caught = float(caught)
        super().__init__(lift, form="shroud", behind=the_ridge(), **kw)

    def _shroud_sdf(self):
        """As in film_myth_ghost, but as it is caught it draws itself in: narrower down the body, so it is a figure and not a bell."""
        from film_myth_ghost import SHROUD, _egg, _seg, _smin
        P, s, xx, yy = self.parts, self.scale, self.xx, self.yy
        head = _egg(xx, yy, P["skull"], P["chin"], self.head[0] * s, self.head[1] * s)
        waist, foot = self.map((1072, 560)), self.map((1110, 900))
        d = _seg(xx, yy, P["skull"] + P["chin"] * 30 * s, waist, 50 * s, (84 - 16 * self.caught) * s)
        d = _smin(d, _seg(xx, yy, waist, foot, (84 - 16 * self.caught) * s, (190 - 78 * self.caught) * s), 20 * s)
        d = _smin(d, head, 28 * s)
        self.sdf_torso = d
        a, b, c = [self.map([np.interp(self.lift, LIFTS, [pose[j][i] for pose in SHROUD["sleeve"]]) for i in range(2)]) for j in range(3)]
        sleeve = _smin(_seg(xx, yy, a, b, 52 * s, 44 * s), _seg(xx, yy, b, c, 44 * s, 26 * s), 10 * s)
        self.sleeve_ends = a, c
        self.sdf_arm = sleeve
        self.sdf_keep = np.minimum(head, sleeve + 10 * s)
        return _smin(d, sleeve, 26 * s)

    def _how(self, seed=9):
        return super()._how(seed) * (1 - 0.90 * self.caught)

    def fade_all(self, seed=9):
        out = super().fade_all(seed)
        if not self.caught:
            return out
        yy = np.mgrid[0:self.h, 0:self.w][0].astype(np.float32)
        return out * (1 - self.caught) + (1 - sm((yy - self.behind[None, :]) / 4.0)) * self.caught


def pose_kw(i):
    return dict(scale=POSES[i][2], at=AT, to=place(POSES[i][2]), seed=SEED, lean=POSES[i][3])


def figure(t, drift=None, caught=None):
    s, to, lean = pose_at(t)
    c = caught_at(t) if caught is None else caught
    return Caught(lift_at(t), caught=c, scale=s, at=AT, to=tuple(to), seed=SEED, lean=lean, drift=(t - T0) * 0.06 if drift is None else drift)


def caught_at(t):
    return 0.45 + 0.55 * ease((t - T0 - 2.0) / 3.0)


def hull(g, c, n=240):
    """How far the figure reaches from c in each direction, the outermost crossing of its outline."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = np.arange(0, 1100, 2.0)
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    closed = np.maximum(g.sdf, yy - the_ridge()[None, :])
    xs = c[0] + np.outer(rr, np.cos(th))
    ys = c[1] + np.outer(rr, np.sin(th))
    d = map_coordinates(closed, [ys, xs], order=1, mode="nearest")
    R = np.zeros(n)
    for j in range(n):
        inside = np.nonzero(d[:, j] <= 0)[0]
        R[j] = rr[inside[-1]] if len(inside) else 0
    return th, R


class Path:
    """Ways in: along the outline at a given distance, drawing nearer as it goes."""

    def __init__(self, g):
        self.g = g
        self.c = np.array((g.parts["chest"][0], g.parts["chest"][1] - 10 * g.scale))
        self.th, self.R = hull(g, self.c)

    def r(self, theta, off):
        # further out the outline's detail goes: a stroke far from it follows only its general shape
        sig = 1.2 + off / 10.0
        R = gaussian_filter1d(self.R, sig * len(self.R) / 240, mode="wrap")
        return np.interp(np.mod(theta, 2 * np.pi), self.th, R, period=2 * np.pi) + off

    def stroke(self, a0, a1, o0, o1, n=60, floor=872.0):
        u = np.linspace(0, 1, n)
        th = np.radians(a0 + (a1 - a0) * u)
        o = o1 + (o0 - o1) * (1 - u) ** 1.7
        pts = np.stack([self.c[0] + np.cos(th) * self.r_each(th, o), self.c[1] + np.sin(th) * self.r_each(th, o)], 1)
        # nothing in the band of the words
        over = np.maximum(pts[:, 1] - floor, 0)
        pts[:, 1] = pts[:, 1] - over * 0.92
        return pts

    def r_each(self, th, o):
        return np.array([self.r(a, b) for a, b in zip(th, o)])


# start, from angle, to angle, from distance, to distance, half widths, tone, how dry. Angles clockwise from east, 90 is straight down. Each ring echoes the outline at its distance, closer and truer than the last
W1, W2, W3 = (12, 30, 34, 26, 10), (10, 24, 27, 20, 8), (6, 13, 15, 11, 4)
RING1 = ((0.40, 196, 258, 250, 182, W1, 0.80, 1.10), (0.56, 14, -42, 262, 192, W1, 0.86, 1.10),
         (0.74, 304, 254, 240, 176, W1, 0.72, 1.15), (0.90, 150, 104, 236, 186, W1, 0.86, 1.00),
         (1.04, 336, 24, 230, 168, W1, 0.84, 1.10), (1.18, 232, 186, 226, 164, W1, 0.95, 1.05),
         (1.32, 72, 22, 240, 170, W1, 0.95, 1.00), (1.46, 122, 164, 232, 160, W1, 1.00, 1.00))
RING2 = ((2.00, 252, 320, 138, 94, W2, 1.45, 0.72), (2.13, 28, -34, 136, 90, W2, 1.55, 0.70),
         (2.26, 196, 136, 130, 92, W2, 1.55, 0.70), (2.38, 336, 26, 128, 84, W2, 1.65, 0.65),
         (2.50, 112, 52, 128, 84, W2, 1.60, 0.70), (2.62, 214, 276, 122, 80, W2, 1.75, 0.62),
         (2.74, 302, 236, 120, 78, W2, 1.80, 0.60), (2.86, 168, 104, 118, 74, W2, 1.80, 0.65))
RING3 = ((3.52, 44, 98, 44, 8, W3, 2.05, 0.30), (3.64, 146, 92, 44, 8, W3, 2.05, 0.30),
         (3.76, 352, 46, 40, 7, W3, 2.15, 0.28), (3.88, 196, 144, 40, 7, W3, 2.15, 0.28),
         (4.00, 306, 358, 38, 6, W3, 2.25, 0.25), (4.12, 232, 190, 36, 6, W3, 2.25, 0.25),
         (4.24, 80, 118, 30, 6, W3, 2.25, 0.25), (4.36, 262, 226, 30, 5, W3, 2.35, 0.22),
         (4.50, 268, 314, 26, 4, W3, 2.40, 0.20), (4.66, 300, 240, 22, 3, W3, 2.45, 0.18))
LEN = {1: 0.74, 2: 0.60, 3: 0.48}


class Trap:
    def __init__(self):
        t0 = time.time()
        self.sheet = sheet = Sheet(seed=103)
        self.yy, self.xx = yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.R = R = the_ridge()
        last = Caught(LIFT_END, caught=1.0, **pose_kw(-1))
        self.last = last
        keep_foot = 1 - sm((yy - 862 + 42 * noise(sheet, 2.6, 301)) / 44)
        # the hill and the thinnest veil: there is hardly anything to see it against
        sky = sm((R - yy) / 2.2)
        hill = 1.12 * sm((yy - R + 24) / 8) * (1 - sm((yy - R - 30) / 90)) * keep_foot
        hill_wet = 1.2 * sm((yy - R + 24) / 8) * (1 - sm((yy - R - 90) / 110))
        veil = 0.98 * blob(xx, yy, 985, 410, 600, 450, 0.05) * keep_foot
        first = Caught(0.15, **pose_kw(0))
        round_it = 1.0 * np.exp(-(np.maximum(first.sdf, 0) / 95) ** 2) * sm((R - yy) / 8)
        self.hf, self.hc = cached("trap_hill3", lambda: paint_clouds(sheet, hill, reserve=sky, seed=71, grid=0.67, heavy=0.28, blooms=6, wet_env=hill_wet).density())
        self.vf, self.vc = cached("trap_veil4", lambda: paint_clouds(sheet, veil, seed=72, grid=0.67, heavy=0.27, near=round_it, blooms=3, wet_env=1.25 * blob(xx, yy, 985, 420, 860, 560, 0.05)).density())
        # what ran out alone beyond the veil would read as dirt on the sheet
        out_there = 1 - sm((np.hypot((xx - 985) / 760, (yy - 410) / 590) + 0.05 * noise(sheet, 2.8, 308) - 0.86) / 0.22)
        self.vf, self.vc = tidy(self.vf * out_there, self.vc * out_there, level=0.035, smallest=90000)
        self.lines = 1.1 * brush(sheet, hill_strokes(R, [(40, 610), (690, 880), (1120, 1860)], 77, w=3.0, dark=0.8), seed=79)
        # the ink that closes in, painted once round the last pose
        out = np.maximum(last.sdf, 0)
        near = np.exp(-(out / 150) ** 2)
        far = 1.05 * np.exp(-(out / 370) ** 2)
        edge = sm((xx - 110 + 80 * noise(sheet, 3, 303)) / 180) * sm((1810 - xx + 80 * noise(sheet, 3, 304)) / 180) * sm((yy - 20 + 50 * noise(sheet, 3, 305)) / 120)
        env = np.clip(far * edge * keep_foot, 0, 1.25)

        def plate():
            return paint_clouds(sheet, env, reserve=last.reserve(), seed=73, grid=0.67, heavy=0.95, near=near * keep_foot, blooms=18).density()

        self.pf, self.pc = cached("trap_plate4", plate)
        foot = 1 - sm((yy - 838 + 18 * noise(sheet, 2.8, 309)) / 56)      # whatever ran further than this is nearly in the words
        self.pf, self.pc = self.pf * foot, self.pc * foot
        self.pf, self.pc = tidy(self.pf, self.pc, smallest=6000)
        self.mw = (noise(sheet, 3.0, 321), noise(sheet, 3.0, 322))
        self.strokes = []
        self._strokes()
        self.arrival = self._arrival()
        self.thaw = 0.40 + 0.5 * sm((noise(sheet, 3.2, 307) + 0.6) / 1.4)
        print(f"  built in {time.time() - t0:.0f}s", flush=True)

    def _wick(self, S):
        """A stroke as the paper takes it: every edge pulled about along the fibres and a shade soft, and its tooth showing in the thin of it, so no edge is a stair of pixels and no stroke a flat cut-out."""
        dx, dy = self.sheet.warp
        S = map_coordinates(S, [self.yy + 0.32 * dy, self.xx + 0.32 * dx], order=1, mode="constant")
        return np.float32(gaussian_filter(S, 0.62) * (1 + 0.10 * np.clip(self.sheet.tooth, -1.5, 1.5)))

    def _strokes(self):
        """Each stroke's ink at film size, in a box, and when it gets to each pixel of that."""
        sheet = self.sheet
        p1 = Path(Caught(0.2, **pose_kw(0)))
        p2 = Path(Caught(0.2, **pose_kw(1)))
        p3 = Path(self.last)
        srcs = []
        for ring, (path, plan) in enumerate(((p1, RING1), (p2, RING2), (p3, RING3)), 1):
            for i, (start, a0, a1, o0, o1, wid, tone, dry) in enumerate(plan):
                seed = 1000 * ring + 17 * i
                jit = np.random.default_rng(seed + 3)
                near_by = {1: 26, 2: 12, 3: 0}[ring]
                d_ = jit.uniform(-near_by, near_by)
                pts = path.stroke(a0, a1, o0 + d_, o1 + d_)
                S = self._wick(bristle(sheet, [tuple(q) for q in pts], wid, seed, tone=tone, dry=dry, tremble=1.6))
                ys, xs = np.nonzero(S > 0.003)
                if not len(ys):
                    continue
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                S = S[y0:y1, x0:x1]
                q = dense([tuple(a) for a in pts], 1.0)
                seg = np.hypot(*np.diff(q, axis=0).T)
                along = np.concatenate([[0], np.cumsum(seg)])
                along = along / along[-1]
                dur = LEN[ring] * (0.85 + 0.3 * np.clip(along[-1], 0, 1))
                # the brush goes down, runs, and lifts: quick in the middle
                arr = T0 + start + dur * (0.5 * along + 0.5 * ease(along))
                yy, xx = np.mgrid[y0:y1, x0:x1]
                dist, idx = cKDTree(q).query(np.stack([xx.ravel(), yy.ravel()], 1), workers=4)
                A = np.float32(arr[idx].reshape(xx.shape) + 0.20 * (dist.reshape(xx.shape) / (max(wid) + 1)) ** 2)
                self.strokes.append((x0, y0, x1, y1, np.float32(S), A))
                srcs.append((q[::6], arr[::6]))
        # under the sleeve and down the side of the body, where no ring gets: one dark stroke, nearly the last. In the figure's own units
        pts = [tuple(self.last.map(q)) for q in ((868, 584), (926, 566), (978, 556), (1002, 590), (1000, 660), (992, 736))]
        S = self._wick(bristle(sheet, pts, (4, 11, 13, 11, 4), 7001, tone=2.2, dry=0.2, tremble=0.8))
        ys, xs = np.nonzero(S > 0.003)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        q = dense(pts, 1.0)
        along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        arr = T0 + 4.42 + 0.42 * along / along[-1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist, idx = cKDTree(q).query(np.stack([xx.ravel(), yy.ravel()], 1), workers=4)
        self.strokes.append((x0, y0, x1, y1, np.float32(S[y0:y1, x0:x1]), np.float32(arr[idx].reshape(xx.shape) + 0.2 * (dist.reshape(xx.shape) / 19) ** 2)))
        srcs.append((q[::6], arr[::6]))
        self.srcs = srcs

    def _arrival(self):
        """When the wet reaches each pixel: outward from the strokes as they pass, slower where the paper is close, hardly at all through the figure."""
        G = 4
        h, w = H // G, W // G
        c = 92.0 * (0.72 + 0.28 * sm((noise(self.sheet, 3.1, 311)[::G, ::G] + 0.9) / 1.8))
        c = c * (0.05 + 0.95 * sm(np.maximum(self.last.sdf, self.yy - self.R[None, :])[::G, ::G] / 6 + 0.5))
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, vals = [], [], []
        for (dy, dx, L) in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            a = idx[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            b = idx[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
            ca = c[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            cb = c[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
            wgt = (L * G) / (0.5 * (ca + cb))
            rows += [a.ravel(), b.ravel()]
            cols += [b.ravel(), a.ravel()]
            vals += [wgt.ravel(), wgt.ravel()]
        src = h * w
        t_min = T0
        sr, sc, sv = [], [], []
        for q, arr in self.srcs:
            for (x, y), t in zip(q, arr):
                i, j = int(np.clip(y // G, 0, h - 1)), int(np.clip(x // G, 0, w - 1))
                sr.append(src)
                sc.append(idx[i, j])
                sv.append(float(t - t_min) + 1e-3)
        graph = coo_matrix((np.concatenate(vals + [np.array(sv)]), (np.concatenate(rows + [np.array(sr)]), np.concatenate(cols + [np.array(sc)]))), shape=(h * w + 1, h * w + 1)).tocsr()
        d = dijkstra(graph, indices=src, directed=True)[:-1].reshape(h, w)
        d = np.minimum(d, 60.0)
        A = grow(gaussian_filter(np.float32(d), 1.0), H, W) + t_min
        return A + 0.10 * noise(self.sheet, 2.9, 312)

    def _keep(self, t, caught):
        """What the figure leaves of what is under it, this moment, through a little shutter while it is moving. Returns the wash's factor, the strokes' factor, and the last figure for the mouth."""
        s = pose_at(t)
        moving = abs(pose_at(t + 0.5 / FPS)[0] - pose_at(t - 0.5 / FPS)[0]) > 0.0015
        times = (t - 0.25 / FPS, t, t + 0.25 / FPS) if moving else (t,)
        wash, over = 0.0, 0.0
        for tt in times:
            g = figure(tt, caught=caught)
            one = np.ones((H, W), np.float32)
            rim = 0.48 * (1 - 0.72 * caught)
            k = lift_figure(self.sheet, g, one, one, edge=0.40 + 0.75 * caught, rim_amount=rim, mist=0.86 + 0.14 * sm((noise(self.sheet, 3.4, 77) + 1.0) / 2.0))[0]
            wash = wash + k
            over = over + np.minimum(k, 1.0) * (1 - 0.0)
        return wash / len(times), over / len(times), g

    def frame(self, t):
        sheet = self.sheet
        reveal = sm((t - self.arrival) / self.thaw)
        reveal_c = sm((t - self.arrival - 0.14) / (self.thaw + 0.2))
        front = 1 + 0.30 * 4 * reveal * (1 - reveal)
        fine = self.hf + self.vf + self.pf * reveal * front
        coarse = self.hc + self.vc + self.pc * reveal_c
        stroke = np.zeros((H, W), np.float32)
        for x0, y0, x1, y1, S, A in self.strokes:
            if t <= A.min():
                continue
            stroke[y0:y1, x0:x1] += S * sm((t - A) / 0.07)
        caught = caught_at(t)
        wash_k, over_k, g = self._keep(t, caught)
        fine, coarse = fine * wash_k, coarse * wash_k
        m = mouth_at(t)
        black = 0.0
        if m > 0:
            # as in k4_cry: an upright oval of black on bare paper, its edge only a little unsure. The jaw goes first, the width after, and a small round dark place would be an eye
            on = sm((t - T0 - 5.2) / 0.4)
            jaw = 1 + on * 0.035 * (0.6 * np.sin(2 * np.pi * 4.3 * t + 1.1) + 0.4 * np.sin(2 * np.pi * 6.1 * t + 0.2))
            g.parts["open"] = 1.0
            c, ax = g.mouth_place()
            from film_myth_ghost import _egg
            e = _egg(self.xx + 1.1 * self.mw[0], self.yy + 1.1 * self.mw[1], c, ax, 13.8 * g.scale * m ** 1.2 * jaw, 32.0 * g.scale * m ** 0.65 * jaw)
            hole = sm((-e + 0.32 * np.clip(gaussian_filter(sheet.tooth, 0.9), -1.5, 1.5)) / (2.0 * g.scale)) * ease(m / 0.12)
            fine = fine * (1 - 0.9 * hole) + 0.5 * gaussian_filter(hole, 3.0) * (1 - hole)
            black = hole * (2.2 + 0.12 * np.clip(sheet.valley, -1.5, 1.5) + 0.3 * np.exp(-(e / (3.0 * g.scale)) ** 2))
        line = self.lines * over_k + stroke * over_k + black
        return develop(sheet, fine, coarse, line=line, qing=qing_wash(sheet, g, 0))


def put(n, f, t0=None, to=RAW6):
    os.makedirs(to, exist_ok=True)
    path = os.path.join(to, f"f{n:05d}.png")
    tmp = path + ".tmp.png"
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(tmp, compress_level=2)
    os.replace(tmp, path)
    if t0 is not None:
        print(f"{n} {time.time() - t0:.1f}s", flush=True)


STILLS = dict(z1=T0 + 1.3, z2=T0 + 5.35)


def save(f, name):
    path = os.path.join(MYTH, name + ".png")
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(path)
    L = f.mean(2) / 255
    print(f"{path}  mean {L.mean():.2f}, bright {(L > 0.82).mean():.2f}, dark {(L < 0.35).mean():.2f}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    trap = Trap()
    if what == "still":
        for name in sys.argv[2:]:
            save(trap.frame(STILLS[name]), name)
    elif what == "at":
        for i in range(2, len(sys.argv), 2):
            save(trap.frame(float(sys.argv[i])), sys.argv[i + 1])
    elif what == "frames":
        only = None
        if "--only" in sys.argv:
            only = set(int(v) for v in sys.argv[sys.argv.index("--only") + 1].split(","))
        part = (0, 1)
        if "--part" in sys.argv:      # i/n: every n-th frame, so several runs can share the shot
            part = tuple(int(v) for v in sys.argv[sys.argv.index("--part") + 1].split("/"))
        for n in range(int(round(T0 * FPS)), int(round(T1 * FPS))):
            if (only is not None and n not in only) or n % part[1] != part[0]:
                continue
            t0 = time.time()
            put(n, trap.frame(n / FPS), t0)
        print("done", flush=True)
    else:
        raise SystemExit(__doc__)

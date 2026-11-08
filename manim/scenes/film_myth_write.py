"""He writes 山 in the mud: film seconds 15.5 up to 而天雨粟. The same form as on the page, the centre line of the oracle-bone outline taken as the road the twig's point goes down: three peaks and a base, a stroke at a time, a pause between. Nothing comes up by itself: the mud is one height field, every frame it is cut as far as the point has got, pushed up in front of the point, and lit again.

    python film_myth_write.py still s1 s2      # 17.5 on the second peak, 20.8 finished and the twig lifted: myth/s1.png, myth/s2.png
    python film_myth_write.py at 16.3 name     # any moment, to myth/name.png

The push onto the mud (film_myth_world.py) takes its close layer from Writer.film, so the frame it lands on is this one's frame. oracle_glyphs.json stays out of the repo and is read at render time.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from film_myth import CAM_K1, mud_wash
from film_myth_mud import Mud, TRACK_AT, ink_of, twig_shadow, unit
from film_myth_wash import OUT, Sheet, band, blob, develop, noise, sm
from film_myth_wide import cached, painted_twig

W, H = 1920, 1080
MYTH = os.path.dirname(OUT)
CAM = CAM_K1                        # the frame the push lands on: the print where k1_close had it
ORIGIN = np.array((CAM["x0"], CAM["y0"]), np.float64)
SHAN_S = 2.4                        # film pixels to a unit of the 300-square glyph
SHAN_C = (1140.0, 470.0)            # where the middle of the 300 square lies in the frame
GROOVE = 17.0                       # half width of the cut in film pixels
DEPTH = 12.0

# The centre line of the outline, in glyph units, as the thinned glyph has it. A stroke is a list of legs: each leg goes down the centre line through its points, one leg to the next is straight. A peak's point is a leg alone, since the centre line itself forks a little under each peak. In this order: left peak; the tall one; the right one; the base, from the right, so the point ends at the left end of it.
STROKES = (
    (((28.6, 243.8), (60.4, 111.0)), ((64.0, 84.0),), ((76.8, 110.0), (110.0, 152.6), (92.0, 223.5), (88.7, 243.0))),
    (((110.0, 152.6), (150.0, 68.5)), ((162.4, 53.0),), ((166.0, 66.0), (178.8, 150.0), (188.6, 221.8), (189.7, 241.7))),
    (((182.3, 153.7), (232.0, 109.7)), ((247.3, 90.0),), ((248.7, 106.5), (267.2, 220.7), (269.0, 239.6))),
    (((269.0, 239.6), (88.7, 243.0), (28.6, 243.8), (21.6, 251.5)),),
)
BEATS = ((15.85, 16.75), (17.0, 17.95), (18.2, 18.95), (19.15, 19.85))      # one sound of the point in the mud each
PILES = (0.45, 0.45, 0.35, 0.8)
TWIG_IN = (15.25, 15.8)
LIFT = (20.0, 20.75)
LIFT_H = 100.0                      # how high over the mud it ends, film pixels
LIFT_TO = (1090.0, 850.0)           # where on the ground under its point, by then
PEN_UP = 34.0                       # how high it goes between strokes
SOFT = 16.0                         # the cut's front, how many pixels long
M = 90                              # how much more mud than the frame is lit, for the shadows and the slow swells at the edge


# ---------------------------------------------------------------- the road

def glyph_bits(ch, sc=3):
    """The outline in a 300 square, sc pixels to the unit, even-odd."""
    from film_myth_world import GLYPHS, svg_subpaths
    with open(GLYPHS, encoding="utf8") as fh:
        v = json.load(fh)[ch]
    p = v if isinstance(v, str) else (v.get("path") or v.get("d"))
    out = np.zeros((300 * sc, 300 * sc), bool)
    for part in svg_subpaths(p):
        im = Image.new("1", (300 * sc, 300 * sc), 0)
        ImageDraw.Draw(im).polygon([(a * sc, b * sc) for a, b in part], fill=1)
        out ^= np.asarray(im, bool)
    return out


def thinned(bits):
    """Zhang-Suen: the mask down to its one-pixel centre line."""
    b = np.pad(bits.astype(np.uint8), 1)
    moved = True
    while moved:
        moved = False
        for step in (0, 1):
            P = [np.roll(np.roll(b, dy, 0), dx, 1) for dy, dx in ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            n = sum(P)
            a = sum(((P[i] == 0) & (P[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            c = (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0) if step == 0 else (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
            off = (b == 1) & (n >= 2) & (n <= 6) & (a == 1) & c
            if off.any():
                b[off] = 0
                moved = True
    return b[1:-1, 1:-1].astype(bool)


class Medial:
    def __init__(self, ch, sc=3):
        self.sc = sc
        ys, xs = np.nonzero(thinned(glyph_bits(ch, sc)))
        self.pts = np.stack([xs, ys], 1).astype(np.float64) / sc
        self.tree = cKDTree(self.pts)
        a, b = zip(*self.tree.query_pairs(1.5 / sc))
        a, b = np.array(a), np.array(b)
        w = np.hypot(*(self.pts[a] - self.pts[b]).T)
        n = len(self.pts)
        self.g = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()

    def walk(self, stops):
        """Down the centre line through these points."""
        idx = [int(self.tree.query(p)[1]) for p in stops]
        out = [idx[0]]
        for u, v in zip(idx[:-1], idx[1:]):
            _, prev = dijkstra(self.g, indices=v, return_predecessors=True)
            i = u
            while i != v:
                i = int(prev[i])
                out.append(i)
        return self.pts[out]


def strokes_in_glyph():
    """Each stroke as a line in glyph units, one unit to a step or less, its corners rounded a little only."""
    med = Medial("山")
    out = []
    for legs in STROKES:
        q = np.vstack([med.walk(leg) if len(leg) > 1 else np.array(leg, np.float64) for leg in legs])
        seg = np.hypot(*np.diff(q, axis=0).T)
        along = np.r_[0, np.cumsum(seg)]
        s = np.arange(0, along[-1], 0.25)
        q = np.stack([np.interp(s, along, q[:, 0]), np.interp(s, along, q[:, 1])], 1)
        keep = q[[0, -1]].copy()
        q = np.stack([gaussian_filter1d(q[:, 0], 8, mode="nearest"), gaussian_filter1d(q[:, 1], 8, mode="nearest")], 1)
        q[[0, -1]] = keep
        out.append(q)
    return out


def even_steps(q, step=1.0):
    seg = np.hypot(*np.diff(q, axis=0).T)
    along = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, along[-1] + 1e-6, step)
    return np.stack([np.interp(s, along, q[:, 0]), np.interp(s, along, q[:, 1])], 1)


def stroke_profile(n, i):
    """Half width and depth along a stroke: the point goes in narrow, the cut widens as the hand leans on it."""
    u = np.linspace(0, 1, 40)
    press = np.minimum(u / 0.06, 1) ** 0.7
    base = 1.1 if i == len(STROKES) - 1 else 1.0
    return GROOVE * base * (0.70 + 0.30 * press) * (1 + 0.06 * np.sin(u * 7 + i)), DEPTH * (0.6 + 0.4 * press)


def shan_on_bank(m, to_ground, where=(772.0, 955.0), size=1.25):
    """The same character on the bank of the wide shot: cut in the ground plane, so the perspective lays it back. `where` is the film position of the middle of its base."""
    gx, gy = to_ground(*where)
    for i, q in enumerate(strokes_in_glyph()):
        p = even_steps(np.stack([gx + (q[:, 0] - 148) * size, gy + (q[:, 1] - 242) * size], 1))
        wid, dep = stroke_profile(len(p), i)
        # cut as the prints beside it are: at this size, lips and crumbs would only be noise round the strokes
        m.cut(p, wid * size * 0.62, dep * 0.8, lip=0.3, seed=500 + 10 * i, pile=0.1, start=0.0, crumbs=0.05, grainy=0.0, chew_k=0.02)


# ---------------------------------------------------------------- the mud, cut as far as the point has got

def lit(h, light=(-0.36, -0.93), reach=25, tan_e=0.82):
    """The surface in light, as Mud.shade has it, with the march toward the light done a row at a time and the sideways part of each step shared between two columns. Rounding that part to whole pixels strings a slanting groove's shadow into beads."""
    gy, gx = np.gradient(gaussian_filter(h, 0.6))
    n = np.hypot(*light)
    lx, ly = light[0] / n, light[1] / n
    s = (-gx * lx - gy * ly) / np.sqrt(1 + gx * gx + gy * gy)
    occ = np.zeros_like(h)
    ky = abs(ly)
    up = int(np.sign(ly))
    for i in range(1, int(reach * ky) + 1):
        dx = lx / ky * i
        x0 = int(np.floor(dx))
        f = dx - x0
        row = np.roll(h, -up * i, axis=0)
        over = np.roll(row, -x0, axis=1) * (1 - f) + np.roll(row, -(x0 + 1), axis=1) * f
        occ = np.maximum(occ, over - h - i / ky * tan_e)
    shadow = gaussian_filter(np.clip((occ - 0.35) / 1.3, 0, 1), 0.7)
    cav = h - gaussian_filter(h, 9)
    by, bx = np.gradient(gaussian_filter(h, 16))
    broad = gaussian_filter(-bx * ly - by * lx, 6)      # crossed as Mud.shade has them: this is the tone of the swells in k1_close
    return np.float32(s), np.float32(shadow), np.float32(cav), np.float32(broad)


class Stroke:
    """What one stroke does to the mud, as far as it goes, and how far along it each pixel of that is."""

    def __init__(self, m, q_film, i, box):
        self.i = i
        self.q = q_film
        self.L = len(q_film) - 1
        x0, y0, x1, y1 = box
        wid, dep = stroke_profile(len(q_film), i)
        kept = m.H
        m.H = np.zeros_like(kept)
        m.cut(q_film + ORIGIN, wid, dep, lip=0.55, seed=100 + 10 * i, pile=PILES[i], start=0.3, crumbs=0.6, grainy=0.2, chew_k=0.025)      # a chewed edge at this size strings the shadow in the groove into beads
        d = m.H[y0:y1, x0:x1].copy()
        m.H = kept
        ys, xs = np.nonzero(np.abs(d) > 1e-3)
        self.win = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        a, b, c, e = self.win
        d = d[a:b, c:e]
        # between the two arms of a peak the lips of both meet in a crease, which would hold a line of black specks
        self.floor, self.lip = gaussian_filter(np.minimum(d, 0), 0.9), gaussian_filter(np.maximum(d, 0), 1.5)
        gy, gx = np.mgrid[a:b, c:e]
        pts = np.stack([gx.ravel() + x0 - ORIGIN[0], gy.ravel() + y0 - ORIGIN[1]], 1)
        self.along = np.float32(cKDTree(q_film).query(pts, workers=4)[1].reshape(gx.shape))
        tan = np.gradient(q_film, axis=0)
        self.tan = tan / (np.hypot(tan[:, 0], tan[:, 1])[:, None] + 1e-6)
        self.wid = np.interp(np.linspace(0, 1, len(q_film)), np.linspace(0, 1, len(wid)), wid)
        self.dep = np.interp(np.linspace(0, 1, len(q_film)), np.linspace(0, 1, len(dep)), dep)
        # the point slows round a corner and where it goes in and comes out
        k = 14
        t0, t1 = self.tan[np.maximum(np.arange(len(q_film)) - k, 0)], self.tan[np.minimum(np.arange(len(q_film)) + k, self.L)]
        turn = np.arccos(np.clip((t0 * t1).sum(1), -1, 1))
        s = np.arange(len(q_film), dtype=np.float64)
        dt = 1 + 1.5 * sm(turn / 1.0) + 1.3 * np.exp(-s / 16) + 1.3 * np.exp(-(self.L - s) / 16)
        self.T = np.r_[0, np.cumsum(dt[:-1])]
        self.T = self.T / self.T[-1]
        self.a, self.b = BEATS[i]

    def front(self, t):
        u = float(np.clip((t - self.a) / (self.b - self.a), 0, 1))
        if u <= 0:
            return 0.0
        if u >= 1:
            return self.L + SOFT
        return float(np.interp(u, self.T, np.arange(self.L + 1)))


class Writer:
    def __init__(self):
        t0 = time.time()
        self.sheet = sc = Sheet(seed=93)
        self.yy, self.xx = np.mgrid[0:H, 0:W].astype(np.float32)
        m = Mud()
        m.print_(TRACK_AT, np.pi, 300)
        self.box = box = (CAM["x0"] - M, CAM["y0"] - M, CAM["x0"] + W + M, CAM["y0"] + H + M)
        self.H0 = m.H[box[1]:box[3], box[0]:box[2]].copy()
        self.rubble = band(self.H0.shape[0], self.H0.shape[1], 4.0, 12, 171)
        self.strokes = []
        for i, q in enumerate(strokes_in_glyph()):
            film = np.stack([SHAN_C[0] + (q[:, 0] - 150) * SHAN_S, SHAN_C[1] + (q[:, 1] - 150) * SHAN_S], 1)
            self.strokes.append(Stroke(m, even_steps(film), i, box))
        fine, coarse, env = mud_wash(sc, CAM, "k1", (850, 520), (980, 470))
        self.present = sm((env - 0.24 + 0.14 * noise(sc, 3.2, 81)) / 0.42)
        self.D_wash = fine + coarse
        # the water left one small dark oval just where the tall peak's inside is: it looks back at you from inside the character
        m_ = blob(self.xx, self.yy, 1240, 555, 80, 50)
        self.D_wash = self.D_wash * (1 - m_) + gaussian_filter(self.D_wash, 38.0) * m_
        self._twig(sc)
        self._rel = {}
        self.blank = self.relief((0.0,) * len(self.strokes), None)
        print(f"writer in {time.time() - t0:.0f}s", flush=True)

    def _twig(self, sc):
        """One branch, the one in his hand on the page, painted once whole on a canvas of its own and carried: long enough that its far end never comes into the frame."""
        self.C = np.array((70.0, 60.0))
        off = [(0, 0), (84, 92), (214, 160), (318, 284), (482, 368), (604, 508), (830, 670), (1010, 810), (1235, 985)]
        path = [tuple(self.C + o) for o in off]
        radius = [10.5, 15.0, 19.5, 24.0, 29.0, 35.0, 43.0, 49.0, 56.0]
        stick = dict(path=path, radius=radius, stubs=((0.24, -0.9, 74, 0.60), (0.42, 0.82, 92, 0.56), (0.72, -0.8, 90, 0.5)), seed=44)
        ink, keep = cached("xw_twig", lambda: painted_twig(sc, stick))
        along = np.clip((self.xx - 900) / 350, 0, 1)
        self.t_ink = ink * (1 - along) + gaussian_filter(ink, 3.0) * along
        self.t_keep = keep
        self.t_under = gaussian_filter(twig_shadow(path, radius, [0.0, 24, 62, 110, 170, 230, 300], light=(0.30, 0.78)), 3.0)
        self.axis = np.array(off[-1], np.float64)
        self.ref = self.strokes[1].q[len(self.strokes[1].q) // 2]

    def relief(self, fronts, bow):
        key = (tuple(round(f, 1) for f in fronts), None if bow is None else tuple(round(v, 1) for v in bow))
        if key in self._rel:
            return self._rel[key]
        h = self.H0.copy()
        floor, lips = np.zeros_like(h), np.zeros_like(h)
        for st, f in zip(self.strokes, fronts):
            if f <= 0:
                continue
            a, b, c, e = st.win
            rev = np.clip((f - st.along) / SOFT + 0.5, 0, 1)
            floor[a:b, c:e] = np.minimum(floor[a:b, c:e], st.floor * rev)
            lips[a:b, c:e] += st.lip * rev
        h += floor + lips * (1 - 0.85 * np.clip(-floor / 3.0, 0, 1))
        if bow is not None:
            # the mud shoved along in front of the point
            x, y, tx, ty, rad, amp = bow
            x, y = x + M, y + M
            x0, x1, y0, y1 = int(max(x - 3 * rad, 0)), int(min(x + 3 * rad, h.shape[1])), int(max(y - 3 * rad, 0)), int(min(y + 3 * rad, h.shape[0]))
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            da, db = (gx - x) * tx + (gy - y) * ty, -(gx - x) * ty + (gy - y) * tx
            h[y0:y1, x0:x1] += np.float32(amp * np.exp(-(da / (rad * 0.7)) ** 2 - (db / (rad * 1.15)) ** 2) * (1 + 0.35 * self.rubble[y0:y1, x0:x1]))
        s, sh, cav, broad = lit(h)
        cut = lambda a: a[M:M + H, M:M + W]
        rel, lift = ink_of(cut(s), cut(sh), cut(cav))
        out = rel + np.clip(0.16 - cut(broad) * 2.4, 0, 0.5), lift
        # most frames ask for a new state, a pause asks for the last one again
        if bow is None:
            self._rel[key] = out
        return out

    def pose(self, t):
        """Where the point is on the ground, how far above it, how far each stroke has got, and the mud in front of the point."""
        S = self.strokes
        fronts = tuple(st.front(t) for st in S)
        bow = None
        last = S[-1]
        if t < TWIG_IN[0]:
            return dict(ground=None, h=0.0, fronts=fronts, bow=None)
        if t < S[0].a:
            u = sm((t - TWIG_IN[0]) / (TWIG_IN[1] - TWIG_IN[0]))
            first = S[0].q[0]
            g, h = first + np.array((900.0, 720.0)) * (1 - u), PEN_UP * (1 - u)
        elif t >= last.b:
            u = sm((t - LIFT[0]) / (LIFT[1] - LIFT[0]))
            # after that it only hangs there, as a hand does
            g, h = last.q[-1] + (np.array(LIFT_TO) - last.q[-1]) * u, LIFT_H * u + 6 * max(t - LIFT[1], 0)
        else:
            g, h = None, 0.0
            for i, st in enumerate(S):
                if st.a <= t < st.b:
                    f = fronts[i]
                    j = int(min(f, st.L))
                    g = st.q[j]
                    speed = (st.front(t + 1 / 60) - st.front(t - 1 / 60)) * 30
                    wv = st.wid[j]
                    c = g + st.tan[j] * wv * 1.05
                    bow = (float(c[0]), float(c[1]), float(st.tan[j, 0]), float(st.tan[j, 1]), float(wv * 1.5), float(st.dep[j] * 0.6 * min(speed / 400, 1)))
                    break
                if i + 1 < len(S) and st.b <= t < S[i + 1].a:
                    u = (t - st.b) / (S[i + 1].a - st.b)
                    g = st.q[-1] + (S[i + 1].q[0] - st.q[-1]) * sm(u)
                    h = PEN_UP * np.sin(np.pi * u) ** 0.7
                    break
        return dict(ground=np.asarray(g, np.float64), h=float(h), fronts=fronts, bow=bow)

    def twig_at(self, t):
        """The branch at this moment, carried: rotated about its point as the hand follows the point part of the way, and a little larger the nearer it comes to the eye. Three moments of a short shutter: three copies far apart would weave its bark."""
        ink, keep, under = 0, 0, 0
        moments = (-0.08, 0.0, 0.08)
        for dt in moments:
            P = self.pose(t + dt / 30)
            g, h = P["ground"], P["h"]
            tip = g + np.array((0.0, -0.35 * h))
            far = self.axis + 0.45 * (self.ref - g) + 90 * np.array((np.sin(t * 1.7), np.cos(t * 1.3)))
            th = np.arctan2(far[1], far[0]) - np.arctan2(self.axis[1], self.axis[0])
            s = 1 + 0.10 * h / LIFT_H
            c, sn = np.cos(th) / s, np.sin(th) / s

            def tx(p):
                return (c, sn, self.C[0] - c * p[0] - sn * p[1], -sn, c, self.C[1] + sn * p[0] - c * p[1])

            warp = lambda a, p, how: np.asarray(Image.fromarray(np.float32(a), "F").transform((W, H), Image.AFFINE, tx(p), how), np.float32)
            ink = ink + warp(self.t_ink, tip, Image.BICUBIC)
            keep = keep + warp(self.t_keep, tip, Image.BILINEAR)
            under = under + warp(self.t_under, g + np.array((0.30, 0.78)) * h, Image.BILINEAR)
        n = len(moments)
        return np.maximum(ink / n, 0), np.clip(keep / n, 0, 1), np.clip(under / n, 0, 1)

    def film(self, t):
        P = self.pose(t)
        rel, lift = self.relief(P["fronts"], P["bow"])
        D = 0.55 * self.D_wash * (1 - 0.6 * lift) + rel * self.present
        if P["ground"] is not None:
            ink, keep, under = self.twig_at(t)
            D = D * (1 - 0.5 * keep) + 0.28 * under * (0.4 + 0.6 * self.present) * (1 - keep) + ink
        return develop(self.sheet, D, np.zeros_like(D), grain=0)

    def image(self, t):
        return Image.fromarray(np.uint8(np.clip(self.film(t) + 0.5, 0, 255)))


STILLS = dict(s1=17.5, s2=20.8)


if __name__ == "__main__":
    what = sys.argv[1]
    wr = Writer()
    if what == "still":
        for name in sys.argv[2:] or list(STILLS):
            wr.image(STILLS[name]).save(os.path.join(MYTH, name + ".png"))
            print(name, flush=True)
    elif what == "at":
        wr.image(float(sys.argv[2])).save(os.path.join(MYTH, sys.argv[3] + ".png"))

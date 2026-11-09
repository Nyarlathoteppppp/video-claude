"""或曰：文字既出，人始知欺。 A bound run of bamboo slips, close, an account of grain lent: 貸粟一石. Half a beat of nothing, then with no hand anywhere the 一 grows an upright, top to bottom and slowly, in ink wetter and blacker than the old, the surface shaved just ahead of it. It is 十. Then a very slow push.

A slip is thin and long, its sides shaved straight. Its face is long straight fibres and nothing else: no figure as in wood. 竹青, the dense skin side, greener, along one edge; 竹黄 pale and yellow across the rest; once in a while a node. Ink has soaked into it along the fibres. The slips either side are only begun, in thin colour and thin ink, and go back into the paper.

Everything is drawn at 1.1 times film size, so the push never enlarges a pixel. Light as in the rest of the myth: from above and a little left.

    python film_myth_slip.py still y1 y2       # the two key frames, to myth/
    python film_myth_slip.py frames [--only n,n]     # the shot, by the film's frame numbers, to myth/seq_raw7/
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

from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, grey_closing, map_coordinates
from scipy.spatial import cKDTree

import ink_subs
from film_myth_wash import INK_ABSORB, OUT, QING_ABSORB, Sheet, noise, sm
from film_myth_wide import GRAIN_ABSORB

MYTH = os.path.dirname(OUT)
RAW7 = os.path.join(MYTH, "seq_raw7")
FPS = 30
K = 1.1
CW, CH = int(1920 * K), int(1080 * K)
TILT = np.radians(1.2)
HW, GAP = 100.0, 1.3      # half a slip, and what shows between two
LI = os.path.expanduser(os.environ.get("INK_FONTS", "~/render_work/fonts") + "/Libian.ttc")
EM, CHAR_GAP, TOP = 180, 27, 232
TEXT = "貸粟一石"      # 貸, not 贷: the form a Han scribe wrote

T_CUT = ink_subs.at("或曰：/文字")
T_END = ink_subs.at("或曰：/鬼神")
T_GROW0 = T_CUT + 0.6      # half a beat of nothing
T_GROW1 = T_GROW0 + 2.2
T_PUSH0, Z_END = T_GROW1 - 0.35, 1.045
ABOVE, BELOW = 58.0, 68.0      # how far the upright stands above and below the 一

# k, where it ends, its own slight angle, seed, its writing, how far its writing is slid along, how strong its ink, where its node is, which edge is its skin
SLIPS = (dict(k=0, end=906, ang=0.0, seed=11, text=TEXT, dv=0, ink=1.0, node=118, skin=-1),
         dict(k=-1, end=872, ang=0.0011, seed=23, text="貸麥三斗", dv=22, ink=0.34, node=660, skin=-1),
         dict(k=1, end=889, ang=-0.0008, seed=37, text="貸粟五斗", dv=-16, ink=0.30, node=372, skin=-1))
CORDS = (196.0, 852.0)
CORD_R = 3.5


def unit(a):
    a = a - a.mean()
    return a / (a.std() + 1e-9)


class Grain:
    """The fibres of bamboo in a flat periodic field with v along them, to be read through (u, v) of each slip: lines far longer than they are wide, and very fine. Film pixels, three cells to the pixel across the fibres."""

    SU, SV = 3.0, 1.0

    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.h, self.w = int(1400 * self.SV), int(1400 * self.SU)
        shape = (self.h, self.w)
        make = lambda sv, su: np.float32(unit(gaussian_filter(r.normal(0, 1, shape), (sv * self.SV, su * self.SU), mode="wrap")))
        self.fine = make(240, 0.32)      # the single fibre
        self.bundle = make(320, 0.95)    # a few together
        self.wide = make(420, 3.2)       # what a planing leaves
        self.stain = make(150, 34)       # handling, very broad
        self.patch = make(14, 3)         # where the ink wore
        # the vascular bundles, cut along: very short dark flecks lying in the fibres, not many
        im = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(im)
        for _ in range(int(self.w * self.h / 1000)):
            x, y, n = r.uniform(0, self.w), r.uniform(0, self.h), r.uniform(4, 15)
            d.line([(x, y), (x, y + n)], fill=int(r.uniform(90, 255)), width=1)
        self.fleck = np.float32(gaussian_filter(np.asarray(im, np.float32) / 255, (0.8, 0.9), mode="wrap"))

    def read(self, a, u, v, du=0.0, dv=0.0):
        return np.float32(map_coordinates(a, [(v + 200 + dv) * self.SV, (u + 700 + du) * self.SU], order=1, mode="grid-wrap"))


def wander1d(n, seed, sig, amp):
    return amp * unit(gaussian_filter1d(np.random.default_rng(seed).normal(0, 1, n), sig, mode="wrap"))


class Writing:
    """A column of characters as a signed distance in the plane of the slip: the face is clerical, each character a little off from a type-case, then the grain and the hand wander it where it is read."""

    S = 2.6
    U0, V0 = 260, 80      # the field's own margin, film pixels

    def __init__(self, text, seed, dv=0):
        r = np.random.default_rng(seed)
        w, h = int(2 * self.U0 * self.S), int((1100 + self.V0) * self.S)
        im = Image.new("L", (w, h), 0)
        y = TOP + dv
        self.cells = {}
        for ch in text:
            scale = 1 + r.normal(0, 0.018)
            font = ImageFont.truetype(LI, int(EM * self.S * scale))
            x0, y0, x1, y1 = font.getbbox(ch)
            cw, chh = x1 - x0, y1 - y0
            pad = 40
            tile = Image.new("L", (cw + 2 * pad, chh + 2 * pad), 0)
            ImageDraw.Draw(tile).text((pad - x0, pad - y0), ch, font=font, fill=255)
            tile = tile.rotate(float(r.normal(0, 0.8)), resample=Image.BICUBIC, expand=False)
            du = float(r.normal(0, 2))
            need = (ABOVE + BELOW) if ch == "一" else chh / self.S
            top = y + (ABOVE - chh / self.S / 2 if ch == "一" else 0.0)
            px, py = int((self.U0 + du) * self.S - tile.width / 2), int((top + self.V0) * self.S - pad)
            im.paste(255, (px, py), tile)
            self.cells[ch] = (du, top + chh / self.S / 2, cw / self.S, chh / self.S)
            y += need + CHAR_GAP
        m = np.asarray(im, np.float32) > 127
        self.sdf = np.float32((distance_transform_edt(~m) - distance_transform_edt(m)) / self.S)

    def read(self, u, v):
        return np.float32(map_coordinates(self.sdf, [(v + self.V0) * self.S, (u + self.U0) * self.S], order=1, mode="constant", cval=99.0))


def soaked(d, fibre, ink, seep_along=6.0):
    """Ink in a face of fibres, from the distance d to the written stroke: its edge is combed by the fibres, it stands a little fuller where it stopped, and it has run a short way along the fibres, further along some than others."""
    core = sm((-d + 0.42 * fibre + 0.45) / 1.15)
    pool = np.exp(-(d / 1.9) ** 2) * core
    run = gaussian_filter(core, (seep_along * K, 0.24 * K))
    far = gaussian_filter(core, (seep_along * 2.2 * K, 0.3 * K))
    hairs = sm((fibre + 0.05) / 0.55)
    seep = (0.56 * run + 0.22 * far) * hairs * (1 - core)
    return core, ink * (core * (1 + 0.14 * pool)) , ink * seep


class Slips:
    def __init__(self):
        t0 = time.time()
        self.sheet = sheet = Sheet(seed=97, w=CW, h=CH)
        yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
        xf, yf = (xx - CW / 2) / K + 960, (yy - CH / 2) / K + 540
        c, s_ = np.cos(TILT), np.sin(TILT)
        self.u = u = (xf - 960) * c + (yf - 540) * s_
        self.v = v = -(xf - 960) * s_ + (yf - 540) * c + 540
        grain = Grain(5)
        n_lo, n_mid = noise(sheet, 3.4, 501), noise(sheet, 2.5, 502)
        T = np.zeros((CH, CW), np.float32)      # thin ink on the bamboo
        G = np.zeros((CH, CW), np.float32)      # the straw glaze
        Q = np.zeros((CH, CW), np.float32)      # the green of the skin
        INK = np.zeros((CH, CW), np.float32)
        M = np.zeros((CH, CW), np.float32)
        P = np.zeros((CH, CW), np.float32)
        for sl in SLIPS:
            k, seed = sl["k"], sl["seed"]
            uk = u - k * (2 * HW + GAP) - sl["ang"] * (v - 540)
            vi = np.clip(((v + 300) * 2).astype(np.int32), 0, 3199)
            # shaved: straight, a knife's very slight run-out and nothing else
            nl, nr = wander1d(3200, seed, 240, 0.35), wander1d(3200, seed + 2, 240, 0.35)
            side = np.where(uk < 0, -uk - (HW + nl[vi]), uk - (HW + nr[vi]))
            ui = np.clip(((uk + 400) * 2).astype(np.int32), 0, 1599)
            cutj = wander1d(1600, seed + 5, 40, 0.6) + wander1d(1600, seed + 6, 2.5, 0.22)
            end = v - (sl["end"] + 0.03 * uk + cutj[ui])
            sdf = np.maximum(side, end)
            du, dv = 311.0 * (seed % 7), 97.0 * seed
            # near a node the fibres swing a little aside, and close up
            near = np.exp(-((v - sl["node"]) / 26) ** 2)
            ug = uk + 2.0 * near * np.sin(uk / 16 + seed)
            fine, bundle, wide = grain.read(grain.fine, ug, v, du, dv), grain.read(grain.bundle, ug, v, du, dv), grain.read(grain.wide, ug, v, du, dv)
            stain, patch, fleck = grain.read(grain.stain, uk, v, du, dv), grain.read(grain.patch, uk, v, du, dv), grain.read(grain.fleck, ug, v, du, dv)
            inside = sm(-sdf / 0.8 + 0.5)
            across = sl["skin"] * uk / HW      # 1 at the skin's edge
            skin = sm((across - 0.05) / 0.95)
            fibre = 0.72 * fine + 0.38 * bundle
            # the face: thin lines of ink in a very pale wash, a shade more of everything toward the skin, where the fibres lie closest
            tone = 0.030 + 0.008 * stain + (0.019 + 0.020 * skin) * np.maximum(fibre, 0) + 0.006 * np.maximum(wide, 0) + (0.05 + 0.09 * skin) * fleck
            glaze = 0.205 + 0.05 * skin + 0.009 * wide + 0.020 * stain + 0.012 * bundle
            green = 0.150 * skin ** 1.7 * (1 + 0.10 * wide + 0.10 * bundle)
            # the shaved sides: a narrow facet, the one toward the light paler and the other a little darker. No line
            facet = np.exp(-(np.maximum(-sdf, 0) / 2.6) ** 2) * inside
            lit_side = (uk < 0) & (end < side - 1.5) | ((uk < 0) & (side > -5))
            tone = np.where(lit_side, tone * (1 - 0.32 * facet), tone + 0.09 * facet)
            tone = tone + 0.055 * np.exp(-(np.maximum(-end, 0) / 2.2) ** 2) * inside      # the cut end is turned from the light
            tone = tone + 0.05 * sm((v - sl["end"] + 80 + 26 * stain) / 80)              # handled toward the foot
            # the node: a low ridge, planed, a band where the grain closes up
            vn = v - sl["node"] - 0.05 * uk - 0.9 * np.sin(uk / 31 + seed)
            band = np.exp(-(vn / 1.7) ** 2)
            tone = tone + 0.085 * band * (0.72 + 0.28 * bundle) + 0.024 * np.exp(-(vn / 6.0) ** 2) + 0.010 * near - 0.018 * np.exp(-((vn + 4.2) / 2.2) ** 2)
            glaze = glaze + 0.05 * near
            # writing
            w = Writing(sl["text"], seed, sl["dv"])
            uw = uk + 1.1 * noise(sheet, 3.2, seed + 40)
            vw = v + 1.1 * noise(sheet, 3.2, seed + 41)
            if k == 0:
                self.writing, self.uk0 = w, uk
                self.fibre0, self.bundle0, self.patch0, self.wide0 = fibre, bundle, patch, wide
            d = w.read(uw, vw) - 0.7 * noise(sheet, 3.0, seed + 42)
            _, body, seep = soaked(d, fibre, 1.0)
            wear = sm((0.30 * fibre + 0.85 * patch + 0.65 * stain - 1.0) / 1.1)
            old = sl["ink"] * (1.70 * body * (1 - 0.34 * wear) + 0.44 * seep)
            # a neighbour is laid in thin and let go of toward the outside: first its ink, then its colour, its edge last of all lost in bare paper
            pres = np.ones_like(u) if k == 0 else 0.84 * (1 - sm((np.abs(u) - 112 + 44 * n_lo + 15 * n_mid) / 178)) * (1 - sm((np.abs(u) - 252) / 42))
            take = inside > M
            T, G, Q, INK = np.where(take, tone, T), np.where(take, glaze, G), np.where(take, green, Q), np.where(take, old, INK)
            P = np.where(take, pres, P)
            M = np.maximum(M, inside)
        # across a gap the bundle carries straight on, however wide the knife left it, so a gap can only be a dark line and never bare paper. Closing across the slips fills only what has a slip on both sides
        closed = grey_closing(M, size=(1, 13))
        self.M, self.pres = M, grey_closing(P, size=(1, 13)) * closed
        # the cords: two thin strands twisted, lying over everything, and a small dark under them
        CordT = T.copy()
        for cv in CORDS:
            wob = 1.6 * unit(gaussian_filter1d(np.random.default_rng(int(cv)).normal(0, 1, 2400), 90))
            ui = np.clip(u + 1200, 0, 2399).astype(np.int32)
            dv_ = v - cv - wob[ui]
            rr = np.random.default_rng(int(cv) + 1)
            fat = 1 + 0.11 * unit(gaussian_filter1d(rr.normal(0, 1, 2400), 22))[ui]
            drift = 0.75 * unit(gaussian_filter1d(rr.normal(0, 1, 2400), 30))[ui]
            on = sm((CORD_R * fat - np.abs(dv_)) / 0.8)
            tw = np.sin(2 * np.pi * (u / 8.3 + dv_ / 6.2) + 2.2 * drift)
            pres_c = 1 - sm((np.abs(u) - 150 + 44 * n_lo + 15 * n_mid) / 130)
            under = np.exp(-((dv_ - CORD_R - 2.0) / 2.6) ** 2) * (1 - on)
            T = T + pres_c * under * 0.20 * M
            T = T * (1 - pres_c * on) + pres_c * on * (0.30 + 0.115 * tw + 0.05 * self.fibre0)
            G = G * (1 - pres_c * on) + pres_c * on * 0.33
            Q = Q * (1 - pres_c * on)
            INK = INK * (1 - pres_c * on)
        self.T, self.G, self.Q, self.INK = T, G, Q, INK
        # what is between two slips: a warm hairline, never paler than the slips, gone as they go
        between = closed * gaussian_filter(self.pres, 2.2 * K) * (1 - M)
        self.gap = 0.60 * between
        self.gap_glaze = 0.42 * between
        # the bundle lying on the paper: a narrow dark where it touches and a wide pale below and to the right of it, the light being above and left. No wash: on the paper it read as smoke
        pm = self.pres
        near_ = np.roll(np.roll(gaussian_filter(pm, 1.8 * K), int(2 * K), axis=0), int(1 * K), axis=1)
        wide_ = np.roll(np.roll(gaussian_filter(pm, 9.0 * K), int(9 * K), axis=0), int(5 * K), axis=1)
        self.ground = np.float32((0.10 * near_ + 0.075 * wide_) * (1 + 0.16 * noise(sheet, 2.8, 503)) * (1 - sm((M - 0.2) / 0.5)))
        self._stroke()
        self.base = None
        print(f"  slips built in {time.time() - t0:.0f}s", flush=True)

    def _stroke(self):
        """The upright and its timing, in the plane of the slip, and a box round it where each frame has work to do."""
        du, cy, cw, chh = self.writing.cells["一"]
        self.cross = (du + 3.0, cy)
        n = 300
        a = np.linspace(0, 1, n)
        ua = du + 3.0 + 2.2 * np.sin(np.pi * a) + 0.8 * np.sin(2 * np.pi * a * 1.3 + 0.4)
        va = cy - ABOVE + (ABOVE + BELOW) * a
        self.path = np.stack([ua, va], 1)
        # thick and round where the hair went down, a shade thicker through the middle, thinning into the tail
        r = 5.9 + 0.5 * np.sin(np.pi * a) ** 0.8 + 0.8 * np.exp(-((a - 0.02) / 0.04) ** 2) - 2.4 * sm((a - 0.86) / 0.14)
        r = r + 0.34 * unit(gaussian_filter1d(np.random.default_rng(31).normal(0, 1, n), 9))
        self.r = r
        # slow going in, slow over the 一 where it catches, slow coming out, and never quite even
        speed = 1.0 - 0.55 * np.exp(-(a / 0.06) ** 2) - 0.38 * np.exp(-((a - ABOVE / (ABOVE + BELOW)) / 0.05) ** 2) - 0.45 * np.exp(-((a - 1) / 0.07) ** 2)
        speed = speed * (1 + 0.16 * unit(gaussian_filter1d(np.random.default_rng(8).normal(0, 1, n), 18)))
        speed = np.maximum(speed, 0.2)
        time_ = np.concatenate([[0], np.cumsum(1 / speed[:-1])])
        self.at = T_GROW0 + (T_GROW1 - T_GROW0) * time_ / time_[-1]
        # a box in canvas pixels round it
        c, s_ = np.cos(TILT), np.sin(TILT)
        X = lambda uu, vv: ((uu * c - (vv - 540) * s_) * K + CW / 2, (uu * s_ + (vv - 540) * c) * K + CH / 2)
        xs, ys = zip(*[X(uu, vv) for uu in (du - 70, du + 70) for vv in (cy - ABOVE - 34, cy + BELOW + 34)])
        self.box = int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1
        x0, y0, x1, y1 = self.box
        u, v, uk = self.u[y0:y1, x0:x1], self.v[y0:y1, x0:x1], self.uk0[y0:y1, x0:x1]
        dist, idx = cKDTree(self.path).query(np.stack([uk.ravel(), v.ravel()], 1), workers=4)
        self.b_dist = np.float32(dist.reshape(u.shape))
        self.b_idx = idx.reshape(u.shape)
        self.b_r = np.float32(r[self.b_idx])
        self.b_side = np.float32(uk - ua[self.b_idx])
        self.b_A = np.float32(self.at[self.b_idx] + 0.05 * np.clip(self.b_dist / r[self.b_idx], 0, 2) ** 2)
        # where it was shaved first: a narrow pale place round the crossing, ragged, the fibres cut through. Only a little: a trace of a knife and no more
        cu, cv = self.cross
        fibre, patch = self.fibre0[y0:y1, x0:x1], self.patch0[y0:y1, x0:x1]
        wob = lambda sd, sig, amp: amp * unit(gaussian_filter(np.random.default_rng(sd).normal(0, 1, v.shape), sig))
        ragged = 1.8 * fibre + wob(61, 5, 2.2)
        shave = sm((uk - (cu - 21) + ragged) / 1.2) * sm(((cu + 17) - uk + ragged) / 1.2) * sm((v - (cv - 42) + 2.5 * patch) / 2.2) * sm(((cv + 48) - v + 2.5 * patch) / 2.2)
        self.b_rub = np.float32(shave * (0.40 + 0.60 * sm((fibre + 0.5 * patch + 0.7) / 1.1)) * (1 - 0.45 * sm((v - cv - 14) / 34)))
        self.b_As = np.float32(self.at[np.clip(self.b_idx, 0, n - 1)] - 0.14)

    def still_base(self):
        if self.base is None:
            self.D0 = self.ground + self.gap + self.pres * (self.T + self.INK)
            self.base = self.colour(self.sheet.rgb, self.D0, self.pres * self.G + self.gap_glaze, self.pres * self.Q)
        return self.base

    @staticmethod
    def colour(rgb, D, G, Q):
        return rgb * np.exp(-D[..., None] * INK_ABSORB - G[..., None] * GRAIN_ABSORB - Q[..., None] * QING_ABSORB)

    def canvas(self, t):
        f = self.still_base()
        if t < T_GROW0 - 0.3:
            return f
        f = f.copy()
        x0, y0, x1, y1 = self.box
        pres = self.pres[y0:y1, x0:x1]
        T_, INK_, G_, Q_ = self.T[y0:y1, x0:x1], self.INK[y0:y1, x0:x1], self.G[y0:y1, x0:x1], self.Q[y0:y1, x0:x1]
        fibre = self.fibre0[y0:y1, x0:x1]
        # the knife goes first
        rub = self.b_rub * sm((t - self.b_As) / 0.22)
        T2 = T_ * (1 - 0.45 * rub)
        INK2 = INK_ * (1 - 0.44 * rub)
        D = self.ground[y0:y1, x0:x1] + self.gap[y0:y1, x0:x1] + pres * (T2 + INK2)
        G = pres * G_ * (1 - 0.30 * rub)
        Q = pres * Q_ * (1 - 0.30 * rub)
        # then the ink: it goes on narrow and takes up its width, black, and it soaks in along the fibres as it dries back a little
        age = t - self.b_A
        grow = 0.84 + 0.16 * sm(age / 0.5)
        edge = self.b_r * grow - self.b_dist
        core = sm((edge + 0.36 * fibre + 0.45) / 1.15) * sm(age / 0.09)
        pool = np.exp(-(edge / 2.2) ** 2)
        fresh = 2.85 - 0.62 * sm((age - 0.25) / 1.5) * (1 - 0.9 * pool) + 0.14 * sm(age / 1.2) * fibre
        ink = core * fresh
        soak = sm(age / 1.1)
        hairs = sm((fibre + 0.05) / 0.55)
        wick = soak * hairs * (0.58 * gaussian_filter(core, (6.0 * K, 0.24 * K)) + 0.22 * gaussian_filter(core, (13.0 * K, 0.3 * K)))
        D = np.maximum(D, ink) + 0.22 * np.minimum(D, ink) * core + wick * (1 - core)
        out = self.colour(self.sheet.rgb[y0:y1, x0:x1], D, G, Q)
        # wet: a thin line of sky down the side toward the light, gone as it dries
        wet = np.exp(-((self.b_side + self.b_r * 0.45) / 1.3) ** 2) * core * (1 - sm((age - 0.12) / 0.7))
        out = out + wet[..., None] * 58.0
        f[y0:y1, x0:x1] = out
        return f

    def frame(self, t):
        f = self.canvas(t)
        z = 1 + (Z_END - 1) * sm((t - T_PUSH0) / (T_END - T_PUSH0)) ** 1.6
        # the push is about the crossing, as near as the composition lets it
        c, s_ = np.cos(TILT), np.sin(TILT)
        cu, cv = self.cross
        xc, yc = 960 + cu * c - (cv - 540) * s_, 540 + cu * s_ + (cv - 540) * c
        k = K / z
        cx, cy = CW / 2 + (xc - 960) * K, CH / 2 + (yc - 540) * K
        tx = (k, 0.0, cx - k * xc, 0.0, k, cy - k * yc)
        chans = [np.asarray(Image.fromarray(np.float32(f[..., i]), "F").transform((1920, 1080), Image.AFFINE, tx, Image.BICUBIC), np.float32) for i in range(3)]
        return np.clip(np.stack(chans, 2), 0, 255)


def put(n, f, t0=None, to=RAW7):
    os.makedirs(to, exist_ok=True)
    path = os.path.join(to, f"f{n:05d}.png")
    tmp = path + ".tmp.png"
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(tmp, compress_level=2)
    os.replace(tmp, path)
    if t0 is not None:
        print(f"{n} {time.time() - t0:.1f}s", flush=True)


STILLS = dict(y1=T_CUT + 0.35, y2=T_GROW1 + 0.5)

if __name__ == "__main__":
    what = sys.argv[1]
    slips = Slips()
    if what == "still":
        for name in sys.argv[2:]:
            f = slips.frame(STILLS[name])
            path = os.path.join(MYTH, name + ".png")
            Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(path)
            print(path, flush=True)
    elif what == "at":
        f = slips.frame(float(sys.argv[2]))
        Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(os.path.join(MYTH, sys.argv[3] + ".png"))
    elif what == "times":
        at = np.interp([0.0, ABOVE / (ABOVE + BELOW), 1.0], np.linspace(0, 1, len(slips.at)), slips.at)
        print(f"knife first touches {slips.b_As.min():.2f}; upright starts {at[0]:.2f}, crosses the 一 {at[1]:.2f}, ends {at[2]:.2f}; push from {T_PUSH0:.2f}", flush=True)
    elif what == "frames":
        only = None
        if "--only" in sys.argv:
            only = set(int(v) for v in sys.argv[sys.argv.index("--only") + 1].split(","))
        for n in range(int(round(T_CUT * FPS)), int(round(T_END * FPS))):
            if only is not None and n not in only:
                continue
            t0 = time.time()
            put(n, slips.frame(n / FPS), t0)
        print("done", flush=True)
    else:
        raise SystemExit(__doc__)

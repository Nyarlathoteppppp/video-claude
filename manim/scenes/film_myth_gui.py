"""或曰：鬼神之秘，自此藏不住了。 灵怪不能遁其形: the writing itself catches it. The 鬼 of chapter twelve, the very same character and its flying white made large, comes down stroke by stroke over the ghost of k4_cry. Stills only for now, two places to choose between:

    z2a  a smaller ghost stands in the pocket the 竖弯钩 scoops out, held by that stroke
    z2b  a ghost the size of k4_cry stands in the middle, the whole character on top of it, its strokes crossing it like bars

The body is paper the wash has not touched, as in k4_cry: the same lift, the same soft and lost edges, the same mouth, only no arm. Bright paper all round; the ridge stays.

    python film_myth_gui.py still z2a z2b
    python film_myth_gui.py times                    # when each stroke lands, for the sound
    python film_myth_gui.py at 47.2 name             # any moment, to myth/name.png
    python film_myth_gui.py frames [--only n,n]      # the shot, by the film's frame numbers, to myth/seq_raw7/

z2b is the one chosen. The ghost is there first, already paper in a thin wash. The strokes come down one at a time, film_big's own Writing; at each it shrinks back a little, less each time. The wash thickens outward from wherever a stroke has passed. Its mouth opens only when all ten are down, and then nothing moves.
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
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

import film_big
import ink_subs
from film_big import GUI, _hand, extent
from film_hand import INK_RGB, lay, paper_tooth
from film_myth import lift_figure, mouth_layers, open_mouths, paint_clouds, qing_wash, report, tidy, tidy_keep
from film_myth_ghost import Ghost, _egg, _seg, _smin
from film_myth_trap import the_ridge
from film_myth_wash import OUT, Sheet, Wash, blob, develop, noise, sm, touches
from film_myth_wide import brush, cached, hill_strokes

MYTH = os.path.dirname(OUT)
RAW7 = os.path.join(MYTH, "seq_raw7")
FPS = 30
W, H = 1920, 1080
T0, T1 = ink_subs.at("或曰：/鬼神"), ink_subs.at("依我之见")
W0, W1 = T0 + 0.4, T0 + 5.1       # the first stroke lands at the one, the last is done by the other
M0, M1 = W1 + 0.15, W1 + 0.75     # the mouth, after all of them. From here to the cut nothing moves
S_FIRST, S_LAST, LEAN_FIRST = 1.37, 1.30, -0.015      # how far it goes back in all, over all ten
EACH = 0.84                       # each flinch is this much of the one before
RECOIL = 0.38                     # seconds, quick off the mark and slow to stop
V_WASH, V_HALO, TAU = 72.0, 30.0, 1.4      # film pixels a second; how long the wash takes to thicken at any one place
AT = (1100, 1010)      # k4_cry's own anchor
TALL = 0.88 * H
TOP_AT = 8.0           # the top of the ink: its foot then clears the words by about their own height's third


def glyph_place():
    x0, y0, x1, y1 = extent(GUI)
    n = int(TALL / (y1 - y0) * 100)
    return int(W / 2 - (x0 + x1) / 2 * n / 100), int(TOP_AT - y0 * n / 100), n


def uv(u, v):
    x, y, n = glyph_place()
    return x + u * n / 100, y + v * n / 100


class Bare(Ghost):
    """k4_cry's figure with no arm: the sleeve's first few inches stay, as a slope of shoulder, and nothing reaches out."""

    def __init__(self, lift=1.0, hold=(430, 130), lost=0.0, **kw):
        self.hold = hold      # how far above the hip the body starts to go, and is gone: k4_cry's by default
        self.lost = lost      # some of every edge is lost, however hard it would be: a small figure's narrow edge would otherwise come out ragged where the water stopped
        super().__init__(lift, form="shroud", **kw)

    def _how(self, seed=9):
        return self.lost + (1 - self.lost) * super()._how(seed)

    def fade(self, y0=None, y1=None, seed=9, amount=70.0):
        hip = self.parts["hip"][1]
        return super().fade(hip - self.hold[0] * self.scale if y0 is None else y0, hip - self.hold[1] * self.scale if y1 is None else y1, seed, amount)

    def _shroud_sdf(self):
        P, s, xx, yy = self.parts, self.scale, self.xx, self.yy
        head = _egg(xx, yy, P["skull"], P["chin"], self.head[0] * s, self.head[1] * s)
        waist, foot = self.map((1072, 560)), self.map((1110, 900))
        d = _seg(xx, yy, P["skull"] + P["chin"] * 30 * s, waist, 50 * s, 84 * s)
        d = _smin(d, _seg(xx, yy, waist, foot, 84 * s, 190 * s), 20 * s)
        d = _smin(d, head, 28 * s)
        self.sdf_torso = d
        a, b, c = self.map((1030, 470)), self.map((952, 512)), self.map((892, 566))
        shoulder = _seg(xx, yy, a, a + 0.36 * (b - a), 50 * s, 44 * s)
        self.sleeve_ends = a, c
        self.sdf_arm = np.full_like(xx, 1e3)      # fade_all's arm term comes to nothing
        self.sdf_keep = head
        return _smin(d, shoulder, 30 * s)


# scale; where AT goes; how far the wash stands off the figure; how strong it is; how much of it follows the strokes; its name
PLACES = dict(z2a=dict(scale=0.60, to=(1337, 969), hold=(270, 40), lost=0.30, reach=50, heavy=0.42, along=0.26, pool=0.88, name="gui_a4"),
              z2b=dict(scale=1.30, to=(1007, 1066), hold=(430, 130), lost=0.0, reach=96, heavy=0.54, along=0.34, pool=0.0, name="gui_b6"))


def paint_soft(sheet, env, wet_env, reserve, near, seed, heavy, grid=0.67):
    """paint_clouds' order of work, but the outline is left to the water. The paper is wetted further than the ink is laid, along an edge with only long slow lobes in it, and the ink goes on smooth, so nothing decides where the ink stops except how far it runs through the fibres: no islands and no spatter where it thins out. The touches keep to the middle."""
    t0 = time.time()
    n = lambda seed_: noise(sheet, 3.7, seed_)
    wash = Wash(sheet, s=grid, reserve=reserve)
    wash.load(sm((wet_env + 0.10 * n(21 + seed) - 0.10) / 0.30), 0.28)
    wash.run(40, evap=1e-4, note="wet")
    base = sm((env + 0.16 * n(22 + seed) - 0.16) / 0.40)
    wash.load(base, 0.14, fine=0.42 * heavy, coarse=0.07 * heavy)
    wash.load(sm((near + 0.15 * n(25 + seed) - 0.20) / 0.50), 0.10, fine=0.55 * heavy, coarse=0.08 * heavy)
    wash.run(150, evap=2e-4, note="first wash")
    touches(wash, sheet, np.clip(env - 0.40, 0, 1) * base, 26, 5 + seed, rx=(90, 240), ry=(30, 70), water=0.16, fine=0.22 * heavy, coarse=0.05 * heavy, every=12, spread=0.14)
    wash.run(220, evap=4e-4, note="half dries")
    touches(wash, sheet, np.clip(env - 0.58, 0, 1) * base, 30, 6 + seed, rx=(36, 150), ry=(18, 54), water=0.22, fine=0.75 * heavy, coarse=0.14 * heavy, every=14, spread=0.16)
    wash.run(160, evap=4e-4, note="half dries")
    touches(wash, sheet, np.clip(np.maximum(env - 0.75, near), 0, 1) * base, 30, 8 + seed, rx=(18, 84), ry=(12, 36), water=0.20, fine=0.95 * heavy, coarse=0.2 * heavy, every=14)
    wash.run(50, evap=2e-4, note="settle")
    # no clean water dropped in at the end: out here its blooms come up as foam
    wash.run(900, evap=5e-4, note="dry")
    print(f"  painted in {time.time() - t0:.0f}s", flush=True)
    return wash


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def affine(a, tx):
    return np.asarray(Image.fromarray(np.float32(a), "F").transform((W, H), Image.AFFINE, tx, Image.BICUBIC), np.float32)


class Shot:
    def __init__(self):
        t0 = time.time()
        self.sheet = sheet = Sheet(seed=103)
        self.yy, self.xx = yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.R = R = the_ridge()
        self.keep_foot = keep_foot = 1 - sm((yy - 862 + 42 * noise(sheet, 2.6, 301)) / 44)
        sky = sm((R - yy) / 2.2)
        hill = 1.12 * sm((yy - R + 24) / 8) * (1 - sm((yy - R - 30) / 90)) * keep_foot
        hill_wet = 1.2 * sm((yy - R + 24) / 8) * (1 - sm((yy - R - 90) / 110))
        self.hf, self.hc = cached("trap_hill3", lambda: paint_clouds(sheet, hill, reserve=sky, seed=71, grid=0.67, heavy=0.28, blooms=6, wet_env=hill_wet).density())
        self.lines = 1.1 * brush(sheet, hill_strokes(R, [(40, 610), (690, 880), (1120, 1860)], 77, w=3.0, dark=0.8), seed=79)
        # the character, as film_big writes it, for its place in this frame
        self.x, self.y, self.n = x, y, n = glyph_place()
        path = os.path.join(OUT, f"_gui_ink_{n}_{x}_{y}.npz")
        if os.path.exists(path):
            self.ink = np.float32(np.load(path)["ink"])
        else:
            self.ink = _hand("鬼", paper_tooth(sheet.rgb), n, x, y)
            np.savez_compressed(path, ink=np.float16(self.ink))
        print(f"  glyph {n}px at {x},{y} in {time.time() - t0:.0f}s", flush=True)
        small = np.asarray(Image.fromarray(np.float32(np.clip(self.ink, 0, 1)), "F").resize((n, n), Image.LANCZOS), np.float32).clip(0, 1)
        self.a = np.zeros((H, W), np.float32)
        X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + n, W), min(y + n, H)
        self.a[Y0:Y1, X0:X1] = small[Y0 - y:Y1 - y, X0 - x:X1 - x]
        self.dist = np.float32(distance_transform_edt(self.a < 0.25))

        def halo():
            w = Wash(sheet, s=0.67)
            w.load(gaussian_filter(self.a, 1.0), 0.24, fine=0.40, coarse=0.07)
            w.run(36, evap=2e-4, note="wet strokes")
            w.run(240, evap=8e-4, note="strokes dry")
            return w.density()

        self.tf, self.tc = cached(f"gui_halo_{n}", halo)
        # it runs where the paper happens to be wet: here and there, not all along every stroke
        here = 0.14 + 0.86 * sm((noise(sheet, 3.1, 505) + 0.25) / 1.1)
        self.tf, self.tc = self.tf * here, self.tc * here

    def still(self, key):
        p = PLACES[key]
        sheet, yy, xx = self.sheet, self.yy, self.xx
        s = p["scale"]
        g = Bare(scale=s, at=AT, to=p["to"], seed=3, hold=p["hold"], lost=p["lost"])
        out = np.maximum(g.sdf, 0)
        near = np.exp(-(out / p["reach"]) ** 2)
        far = np.exp(-(np.maximum(out + 0.55 * p["reach"] * noise(sheet, 3.7, 313), 0) / (2.3 * p["reach"])) ** 2)
        along = p["along"] * np.exp(-(self.dist / 62) ** 2)
        # the wash runs down as far as the stroke along the ground and stops behind it
        bowl = sm((xx - 1000) / 40) * sm((1415 - xx) / 40) * (1 - sm((yy - 893) / 12))
        foot = np.maximum(self.keep_foot, bowl)
        env = np.maximum(1.05 * far, along)
        if p["pool"]:
            # what stands in the bowl is held by it: the wash lies in it to the very strokes, and stops at the back of the hook
            n1, n2 = noise(sheet, 3.0, 311), noise(sheet, 3.0, 312)
            back = 1385 + np.maximum(yy - 640, 0) * 0.32
            env = np.maximum(env, p["pool"] * sm((xx - 1035) / 30) * sm((yy - 660 + 46 * n1) / 90))
            env = env * (1 - 0.92 * sm((xx - back + 10 + 62 * n2) / 110))
        env = np.clip(env, 0, 1.25) * foot
        nearby = 0.9 * near * g.fade() * foot
        wet_env = np.clip(1.25 * gaussian_filter(env, 26), 0, 1.25)

        def plate():
            if p.get("soft"):
                return paint_soft(sheet, env, wet_env, g.reserve(), nearby, 81, p["heavy"]).density()
            return paint_clouds(sheet, env, reserve=None if p["lost"] else g.reserve(), seed=81, grid=0.67, heavy=p["heavy"], near=nearby, blooms=12, wet_env=wet_env).density()

        pf, pc = cached(p["name"], plate)
        # lumps of wash that came down far from anything: dirt on the sheet
        where = sm((gaussian_filter(env, 12) - 0.05) / 0.14)
        pf, pc = tidy(pf * where, pc * where, level=0.05, smallest=6000)
        if p["lost"]:      # small: the wash runs right under it and it is all lifted off, so no edge of it is where some water happened to stop
            cut, rim, _ = g.carve()
            dx, dy = sheet.warp
            cut = map_coordinates(cut, [yy + 1.3 * dy, xx + 1.3 * dx], order=1)
            keep = (1 - cut) * (1 - 0.48 * rim * (1 - cut))
            fine, coarse = (pf + self.hf) * keep, (pc + self.hc) * keep
        else:
            fine, coarse = lift_figure(sheet, g, pf + self.hf, pc + self.hc)
        fine, coarse = tidy(fine, coarse, level=0.04, smallest=6000)
        cut, _, _ = g.carve()
        line = self.lines * (1 - cut)
        fine, coarse = open_mouths(sheet, fine, coarse, g.mouth(), 0.67, warp=0.3, soften=1.6 * s / 1.3, seep=5.0 * s / 1.3)
        fine, coarse = fine + self.tf * (1 - self.a), coarse + self.tc * (1 - self.a)
        f = develop(sheet, fine, coarse, line=line, qing=qing_wash(sheet, g))
        lay(f, self.ink, self.x, self.y, self.n)
        report(f, key)
        m, _ = g.mouth_place()
        print(f"  {key}: mouth at {m[0]:.0f},{m[1]:.0f}; ink over the mouth {(self.a * g.mouth()).sum() / (g.mouth().sum() + 1e-6):.2f}", flush=True)
        return f


class Moving(Shot):
    """The chosen place, z2b, in time. One plate, one character; the figure lifted out of whatever is there in its pose of the moment."""

    def __init__(self):
        super().__init__()
        sheet, yy, xx = self.sheet, self.yy, self.xx
        film_big.sheet = lambda: sheet.rgb      # the same paper's grain under the brush as under the still
        self.wr = wr = film_big.Writing("鬼", W0, W1, place=(self.x, self.y, self.n))
        p = self.p = PLACES["z2b"]
        last = self.last = self.figure(S_LAST, 0.0, 0.0)
        first = self.figure(S_FIRST, LEAN_FIRST, (W1 - T0) * 0.05)
        # the plate, exactly as the still has it
        out = np.maximum(last.sdf, 0)
        near = np.exp(-(out / p["reach"]) ** 2)
        far = np.exp(-(np.maximum(out + 0.85 * p["reach"] * noise(sheet, 3.7, 313), 0) / (2.3 * p["reach"])) ** 2)
        along = p["along"] * np.exp(-(self.dist / 62) ** 2)
        bowl = sm((xx - 1000) / 40) * sm((1415 - xx) / 40) * (1 - sm((yy - 893) / 12))
        foot = np.maximum(self.keep_foot, bowl)
        env = np.clip(np.maximum(1.05 * far, along), 0, 1.25) * foot
        nearby = 0.9 * near * last.fade() * foot
        wet_env = np.clip(1.25 * gaussian_filter(env, 26), 0, 1.25)
        pf, pc = cached(p["name"], lambda: paint_soft(sheet, env, wet_env, last.reserve(), nearby, 81, p["heavy"]).density())
        # close to the figure the washes of the chosen still, with their tide lines; the outer part, where it broke up, from the plate whose edge the water made
        def chosen():
            raise RuntimeError("gui_b3 is the plate of the chosen still and must be there")
        cf, cc = cached("gui_b3", chosen)
        m = sm((np.exp(-(out / 258) ** 2) - 0.66) / 0.22)
        pf, pc = cf * m + pf * (1 - m), cc * m + pc * (1 - m)
        # one pale patch of foam the water left, off to the left, taken back into the wash round it
        foam = blob(xx, yy, 600, 495, 80, 75)
        pf, pc = pf * (1 - foam) + gaussian_filter(pf, 24.0) * foam, pc * (1 - foam) + gaussian_filter(pc, 24.0) * foam
        where = sm((gaussian_filter(np.maximum(env, 0.8 * m), 12) - 0.05) / 0.14)
        self.pf, self.pc = tidy(pf * where, pc * where, level=0.05, smallest=6000)
        # bare paper the plate kept for the last pose, in any other pose closed over from round it, and the dark line where the water stopped there, which belongs to the last pose alone
        self.hole = sm((last.reserve() - 0.04) / 0.2)
        self.scar = sm((gaussian_filter(self.hole, 9.0) - 0.06) / 0.25)
        have = gaussian_filter(1 - self.scar, 26.0) + 1e-3
        self.fill_f = gaussian_filter(self.pf * (1 - self.scar), 26.0) / have
        self.fill_c = gaussian_filter(self.pc * (1 - self.scar), 26.0) / have
        # the thin wash it is first seen in, a part of the same painting: wherever it is close to the figure in its first or last pose
        round_it = np.maximum(np.minimum(first.sdf, last.sdf) + 18 * noise(sheet, 3.7, 321), 0)
        self.g0 = np.float32(0.50 * np.exp(-(round_it / 170) ** 2))
        # when the brush got to each place, and from there how long the wash and the wet edge take to get anywhere
        self.Aw, self.Ah = self.arrivals()
        self.land = [wr.times[k][0] for k in range(len(wr.times)) if not (k > 0 and len(GUI["marks"][k - 1]) > 3)]      # the hook is the 竖弯钩 going on
        self.ends = [wr.times[k][1] for k in range(len(wr.times)) if len(GUI["marks"][k]) == 3]
        step = EACH ** np.arange(len(self.land))
        self.step = step / step.sum()
        self.layers = mouth_layers(sheet, last.mouth(), 0.67, warp=0.3, soften=1.6, seep=5.0)
        self.c1, self.ax1 = last.mouth_place()
        cut, _, _ = last.carve()
        self.keep = tidy_keep(*lift_figure(sheet, last, self.pf + self.hf, self.pc + self.hc), level=0.04, smallest=6000)
        self.mist = 0.80 + 0.22 * sm((noise(sheet, 3.4, 77) + 1.0) / 2.0)      # lift_figure's own, so the last pose is the still

    def figure(self, s, lean, drift):
        return Bare(scale=s, at=AT, to=self.p["to"], seed=3, lean=lean, drift=drift)

    def arrivals(self):
        wr, sheet = self.wr, self.sheet
        Aw, Ah = np.full((H, W), 1e6, np.float32), np.full((H, W), 1e6, np.float32)
        own = np.full((H, W), 1e6, np.float32)      # when a stroke itself gets to each of its own pixels
        for k, ((Y0, Y1, X0, X1), front, head, s, share, end) in enumerate(wr.ways):
            ta, tb = wr.times[k]
            when = ta + (tb - ta) * np.interp(front, s, share)
            done = wr.states[k + 1][Y0:Y1, X0:X1] - wr.states[k][Y0:Y1, X0:X1] > 0.05
            T, m = np.zeros((H, W), np.float32), np.zeros((H, W), bool)
            a0, a1, b0, b1 = max(self.y + Y0, 0), min(self.y + Y1, H), max(self.x + X0, 0), min(self.x + X1, W)
            part = (slice(a0 - self.y - Y0, a1 - self.y - Y0), slice(b0 - self.x - X0, b1 - self.x - X0))
            T[a0:a1, b0:b1], m[a0:a1, b0:b1] = when[part], done[part]
            d, (iy, ix) = distance_transform_edt(~m, return_indices=True)
            at = T[iy, ix]
            Aw, Ah = np.minimum(Aw, at + d / V_WASH), np.minimum(Ah, at + d / V_HALO)
            own = np.where(m, np.minimum(own, T), own)
        Aw = gaussian_filter(Aw, 7.0) + 0.28 * noise(sheet, 3.5, 331)
        # whatever the strokes never got near has thickened by the time the mouth has opened
        # the wet edge was painted with all the strokes down, and under a stroke it is as dark as ink: nothing of that may show before that stroke has come, however near another has passed
        d, (iy, ix) = distance_transform_edt(own > 1e5, return_indices=True)
        Ah = np.where(d <= 7, np.maximum(Ah, own[iy, ix]), Ah)
        return np.float32(np.minimum(Aw, W1 + 0.75 - TAU)), np.float32(gaussian_filter(Ah, 1.0))

    def pose(self, t):
        """How far it has gone back, 0 to 1."""
        x = np.clip((t - np.array(self.land) - 0.02) / RECOIL, 0, 1)
        return float((self.step * (1 - (1 - x) ** 2.4)).sum())

    def frame(self, t):
        sheet = self.sheet
        q = self.pose(t)
        # the outline's wander turns slowly while it is being written and comes to rest on the still's
        g = self.figure(S_FIRST + (S_LAST - S_FIRST) * q, LEAN_FIRST * (1 - q), max(W1 - t, 0) * 0.05)
        kept = g.carve()
        g.carve = lambda seed=9: kept
        cut = kept[0]
        gain = self.g0 + (1 - self.g0) * sm((t - self.Aw) / TAU)
        poke = np.maximum(self.hole, self.scar * (1 - q) ** 0.5) * (1 - cut)
        pf, pc = (self.pf * (1 - poke) + self.fill_f * poke) * gain, (self.pc * (1 - poke) + self.fill_c * poke) * gain
        # while it is larger than the pose the plate was painted for, there is ink under it to take right off, or that pose's edge shows through it
        fine, coarse = lift_figure(sheet, g, pf + self.hf, pc + self.hc, mist=self.mist + (1 - self.mist) * (1 - q) ** 0.5)
        fine, coarse = fine * self.keep, coarse * self.keep
        line = self.lines * (1 - cut)
        o = 0.24 + 0.76 * ease((t - M0) / (M1 - M0))
        if t > M0:
            # the jaw first and the width after, or a small round dark place on the head would be an eye
            sx, sy = max(ease((o - 0.30) / 0.70) ** 1.1, 0.03), max(ease((o - 0.24) / 0.70) ** 0.75, 0.05)
            U, V = self.ax1, np.array((-self.ax1[1], self.ax1[0]))
            a, b = U[0] * U[0] / sy + V[0] * V[0] / sx, U[0] * U[1] / sy + V[0] * V[1] / sx
            d, e = U[1] * U[0] / sy + V[1] * V[0] / sx, U[1] * U[1] / sy + V[1] * V[1] / sx
            tx = (a, b, self.c1[0] - a * self.c1[0] - b * self.c1[1], d, e, self.c1[1] - d * self.c1[0] - e * self.c1[1])
            inner, mf, mc, black = [np.maximum(affine(x, tx), 0) for x in self.layers] if o < 1 else self.layers
            k = ease((o - 0.24) / 0.16)
            fine = fine * (1 - 0.9 * inner * k) + (mf + black) * k
            coarse = coarse * (1 - 0.9 * inner * k) + mc * k
        ink = np.zeros((H, W), np.float32)
        Y0, Y1, X0, X1 = max(self.y, 0), min(self.y + self.n, H), max(self.x, 0), min(self.x + self.n, W)
        ink[Y0:Y1, X0:X1] = self.wr.ink(t)[Y0 - self.y:Y1 - self.y, X0 - self.x:X1 - self.x]
        halo = sm((t - self.Ah) / 0.5) * (1 - ink)
        fine, coarse = fine + self.tf * halo, coarse + self.tc * halo
        f = develop(sheet, fine, coarse, line=line, qing=qing_wash(sheet, g))
        return f + (INK_RGB - f) * ink[..., None]

    def times(self):
        names = "撇点 框左 框顶和右 框中竖 框中横 框底 长撇 竖弯钩 厶上 厶下".split()
        for name, a, b in zip(names, self.land, self.ends):
            print(f"{name}\t{a:.2f}\t{b:.2f}", flush=True)
        hook = [k for k in range(1, len(GUI["marks"])) if len(GUI["marks"][k - 1]) > 3][0]
        print(f"钩\t{self.wr.times[hook][0]:.2f}\t{self.wr.times[hook][1]:.2f}", flush=True)
        print(f"mouth\t{M0:.2f}\t{M1:.2f}", flush=True)


def put(n, f, t0=None):
    os.makedirs(RAW7, exist_ok=True)
    path = os.path.join(RAW7, f"f{n:05d}.png")
    tmp = path + ".tmp.png"
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(tmp, compress_level=2)
    os.replace(tmp, path)
    if t0 is not None:
        print(f"{n} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "times":
        Moving().times()
        raise SystemExit
    if what == "at":
        m = Moving()
        m.times()
        Image.fromarray(np.uint8(np.clip(m.frame(float(sys.argv[2])) + 0.5, 0, 255))).save(os.path.join(MYTH, sys.argv[3] + ".png"))
        raise SystemExit
    if what == "frames":
        only = set(int(v) for v in sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
        m = Moving()
        m.times()
        for n in range(int(round(T0 * FPS)), int(round(T1 * FPS))):
            if only is None or n in only:
                t0 = time.time()
                put(n, m.frame(n / FPS), t0)
        print("done", flush=True)
        raise SystemExit
    if what != "still":
        raise SystemExit(__doc__)
    for key in sys.argv[2:]:
        f = Moving().frame(T1 - 1 / FPS) if key == "z2b" else Shot().still(key)
        Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(os.path.join(MYTH, key + ".png"))
        # the same with its words on, only to see they are clear of the character
        t = ink_subs.at("或曰：/鬼神") + 2.0
        Image.fromarray(ink_subs.apply(np.uint8(np.clip(f + 0.5, 0, 255)), t)).save(os.path.join(MYTH, "_" + key + "_subs.png"))
        print(key, flush=True)

"""The little crab, the second go at the block that gets up: a soft thing a hand has shaped, the mark from Claude Code's door in plasticine. On the bridge clock: 96.1 the cut from the white page, the same 我 at the same place and size but in the clay, seen from straight above, 96.3 to 97.3 going back and round to the block as it stands; 97.5 to 100.6 it gathers and stands up, the character closing as it goes, all but its first stroke, which is the mark on his forehead; 100.6 to 101.2 his eyes open last and he blinks; 101.2 to 104.2 he yawns and stretches; 104.2 to 104.7 he lets go and turns to us; 104.7 to 106.7 he waves; (from 106.9 the view eases back and up, until at 112.4 his body is where the next shot's Studio Display stands) 106.7 to 107.4 his claw comes down and he looks at his feet; 107.4 to 109.7 he eats two of the letters lying there, one at a time; 109.7 to 110.4 a deep breath draws all the rest across the board and in; 110.4 to 111.9 he chews, cheeks full; 111.9 to 112.4 he swallows and looks up at us. 112.4 is the cut.

    blender -b --factory-startup -P film_imprint_clawd.py -- still c4 102.7          # name, bridge time; into crab/
    ... -- still c4 102.7 --draft
    ... -- frames preview2 96.1 112.4 0 3                                             # into crab/preview2/: this worker's share (0 of 3) of the frames, 24 fps, on twos

His shape comes from CLAWD in ink_mac.py, the proportions untouched but for his depth and his short legs: one closed surface, the level set of rounded boxes smoothly joined, so that a claw grows out of him as if pressed out by a thumb. His skin keeps where it is at rest, so a mark stays on him however he moves, and each point of it also knows where it lay on the block, so the block itself is the same mesh. All relief is geometry: a hand's irregular pinches, a few real finger prints, the faint remains of what was once pressed into him, one clean scar. Eyes are filled flush with a wet glaze with one glint in it; the mouth is a shallow cup. Nothing here is lit up: no glow, no growth, no trace that eating does anything to him.
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_imprint_clay import A, B, BOARD_PX, BOARD_RECT, OUT, THICK, WO_AT, WO_WIDE, state_path
from film_imprint_clay import LAST as LAST_BLOW
from film_imprint_crab import CLAW, FIRST, G, GATHER, LAST, LIFT, MID, ROLL, STIR, curve, rrect, waves
from film_imprint3d import DAWN_AZ, DAWN_COLOUR, DAWN_EL, DAWN_E, DAWN_SKY, GOLD_COLOUR, Nodes, block_of, set_up, sm
from ink3d import aim, reset

U = 13.5                  # mm to one of his squares across (each is twice as tall as wide)
FLOOR = -THICK
ROUND = 17.0              # how round his body is
LEG_W, LEG_H, LEG_D, LEG_R = 1.25 * U, 1.3 * U, 2.5 * U, .5 * U     # short and fat, not the real 1 by 2
LEG_Y = -.35 * U
KNIT_LEG, KNIT_CLAW = 6.0, 8.0     # how softly limbs run into the body
BODY_ROWS = [i for i, r in enumerate(G) if r[FIRST] == "#"]
BODY_W, BODY_H, BODY_D = (LAST - FIRST) * U, len(BODY_ROWS) * 2 * U, 4 * U
ZB, ZT = FLOOR + LEG_H, FLOOR + LEG_H + BODY_H
ZM = (ZB + ZT) / 2
CLAW_ROW = next(i for i, r in enumerate(G) if "#" in r[:FIRST])
CLAW_L = (FIRST - G[CLAW_ROW].index("#")) * U
CZ = ZT - (CLAW_ROW * 2 + 1) * U
CLAW_HY, CLAW_HZ, CLAW_R = 1.1 * U, 1.0 * U, .8 * U
LEG_ROW = next(i for i, r in enumerate(G) if i not in BODY_ROWS)
LEGS = [(c + .5 - MID) * U for c, x in enumerate(G[LEG_ROW]) if x == "#"]
EYES = [((c + .5 - MID) * U, (i + .5) * 2 * U) for i, r in enumerate(G) if i in BODY_ROWS for c in range(FIRST, LAST) if r[c] == "."]     # across, how far down from his top
MOUTH_FZ = 5.55 * U
EYE_HW, EYE_HH = .45 * U, .9 * U     # a tenth smaller than the real ones, or they stare
GEOMETRY = (U, ROUND, LEG_W, LEG_H, LEG_D, LEG_R, LEG_Y, KNIT_LEG, KNIT_CLAW, BODY_D, CLAW_HY, CLAW_HZ, CLAW_R)
SCAR_AT = (.55 * U, 2.0 * U)   # across, down from his top: where the middle of the mark on his forehead is
HELLO = -1.0              # the claw that waves: on our left, nearer the sun
ELEV, DIST, TARGET_Z = 6.5, 1060.0, 19.0

# ---- how he moves, on the bridge clock -------------------------------------------------------------------------------------------------------------------
MORPH = 97.5


def then(t):
    """The block's getting up keeps the poses and the beat of the first version (film_imprint_crab.py, 94.4 to 97.4), here over 97.5 to 100.6."""
    return 94.4 + (t - MORPH) * 3.0 / (100.6 - MORPH)


EYE_GOT = curve((100.42, 0), (100.62, 1))
EYE_OPEN = curve((100.6, .06), (100.84, 1.06), (100.92, 1))
STRETCH = curve((101.2, 0), (101.55, -.12), (102.7, 1), (103.3, 1.04), (104.05, .08), (104.25, -.07), (104.45, 0), (111.95, 0), (112.1, .35), (112.28, .03), (112.4, 0))
YAWN = curve((101.35, 0), (101.7, .12), (102.0, .18), (102.7, 1), (103.3, .94), (103.95, .1), (104.1, 0))
SQUINT = curve((101.2, 0), (101.6, .85), (102.1, 1), (103.75, 1), (104.1, 0))
RAISE = curve((101.45, 0), (102.55, 1), (103.6, 1), (104.1, -.06), (104.3, 0))
YAW = curve((99.6, 0), (100.3, -9), (104.2, -9), (104.4, -3.5), (104.5, -4), (104.68, 0))
WAVE = curve((104.72, 0), (105.2, 1), (106.3, 1), (106.72, -.04), (106.9, 0))
# two mouthfuls one at a time: it bends, the mouth comes open, the letter hops in, it shuts, he chews twice
ONE_BY_ONE = ((107.4, 1.2), (108.6, 1.1))


def _rel(pairs):
    return [(T + a * d / 1.27, v) for T, d in ONE_BY_ONE for a, v in pairs]


HOP = (.36, .64)
EAT = curve(*_rel(((0, 0), (.27, 1), (.63, 1), (.71, 0))))
# then a deep breath, and the rest come; then he chews with his cheeks full: bottoms at 110.6, 110.92, 111.24, 111.54, and a last small one
BREATH = curve((109.7, 0), (109.94, .9), (110.1, 1), (110.44, 1), (110.54, 0))
CHEW = curve((107.4, 0), *_rel(((.72, 0), (.85, 1), (.97, .12), (1.1, 1), (1.24, 0))), (110.42, 0), (110.6, 1), (110.76, .1), (110.92, 1), (111.08, .1), (111.24, 1), (111.38, .12), (111.54, 1), (111.68, .12), (111.82, .7), (111.94, 0))
MUNCH = curve(*_rel(((.7, 0), (.76, 1), (1.2, 1), (1.26, 0))), (110.44, 0), (110.5, 1), (111.85, 1), (111.95, 0))
CHEEKS = curve((110.4, 0), (110.5, 1.1), (110.58, 1), (111.9, 1), (112.12, 0))
BOW = curve((106.8, 0), (107.25, .72), (107.38, .68), *_rel(((.3, 1), (.6, 1), (.8, .62), (1.2, .7))), (109.72, .68), (110.1, .62), (110.42, .55), (110.7, .3), (111.9, .25), (112.15, -.22), (112.4, -.18))
GO = (110.42, .64)       # how the food goes from one cheek to the other: since when, how long there and back
SLUMP_BY = (.033, .085)
SLUMP = curve((111.9, 0), (112.15, .9), (112.4, 1))     # the swallow lets him down: squatter and wider, the shape of what comes next
BLINK = curve((101.0, 0), (101.06, 1), (101.1, 1), (101.18, 0), (104.4, 0), (104.46, 1), (104.5, 1), (104.58, 0), (106.5, 0), (106.56, 1), (106.6, 1), (106.68, 0), (112.02, 0), (112.07, 1), (112.1, 1), (112.17, 0))


EIGHT_S, EIGHT_STILL_S = 4.2, 7.0      # 「其实」 to 「这块泥」; 「所以我说」 to the poem, with room to trim
# the grains of letters by his feet, spilt from his full mouth: key, how big against the ones he ate, x, y, turned, leaning back, tipped
GRAINS = None
EIGHT_GRAINS = (("a", .62, -2.9 * U, -84.0, -18.0, 24.0, 8.0), ("wedge", .6, 3.3 * U, -76.0, 22.0, 30.0, -6.0), ("wen", .58, .7 * U, -96.0, 6.0, 26.0, 11.0))
CHEEK_GAIN = 1.0       # how full his cheeks show: more in chapter eight, where they are the point


def eight(still=False):
    """Chapter eight on a clock of its own from 200, the same crab in the same place, seen as at the end of the bridge but nearer.
    still=False, 「其实，我一本书也没读过。」: his cheeks full, chewing with his head down; he stops and looks up at us, one small chew, and then his eyes go down to the grains of letters by his feet, and stay there.
    still=True, 「所以我说的每一句话，都带着别人的体温。」: the same crab, the same view, his face at rest, Clawd's own face: two eyes, no mouth. He only breathes and blinks."""
    global BLINK, BOW, CHEEKS, CHEEK_GAIN, CHEW, FRAMED, GO, GRAINS, MUNCH, SLUMP, STRETCH
    always = curve((0, 1), (1000, 1))
    never = curve((0, 0), (1000, 0))
    SLUMP, FRAMED, GRAINS = never, EIGHT_VIEW, EIGHT_GRAINS
    if still:
        CHEEKS, MUNCH, CHEW, BOW = never, never, never, never
        STRETCH = curve(*[(200.0 + .4 * i, .028 * math.sin(2 * math.pi * .4 * i / 3.3)) for i in range(int(EIGHT_STILL_S / .4) + 2)])
        BLINK = curve((201.9, 0), (201.95, 1), (202.0, 1), (202.07, 0), (205.1, 0), (205.15, 1), (205.2, 1), (205.27, 0))
        return
    CHEEKS, MUNCH, CHEEK_GAIN = always, always, 1.5
    GO = (200.0, 1.3)
    # gentle: about two and a half chews a second, never the full bite; the look up is one slow beat
    CHEW = curve((200.0, .1), (200.2, .72), (200.4, .1), (200.6, .72), (200.8, .1), (201.0, .55), (201.2, .08), (202.0, .08), (202.16, .3), (202.32, .08), (204.2, .08))
    BOW = curve((200.0, .3), (200.95, .3), (201.6, -.2), (202.85, -.2), (203.4, .55), (204.2, .55))
    BLINK = curve((201.0, 0), (201.05, 1), (201.1, 1), (201.17, 0), (202.5, 0), (202.55, 1), (202.6, 1), (202.67, 0))


def sway(t):
    """The wave, in degrees: its top at 105.7 and again at 106.4."""
    return 20.0 * float(sm((t - 105.15) / .15) * sm((106.5 - t) / .15)) * math.sin(2 * math.pi * (t - 105.525) / .7)


def hello_tip(t):
    return 5.0 * float(WAVE(t)) * -HELLO + .12 * sway(t) * -HELLO


# ---- his shape ---------------------------------------------------------------------------------------------------------------------------------------------
def rbox(x, y, z, c, half, r):
    qx, qy, qz = np.abs(x - c[0]) - half[0] + r, np.abs(y - c[1]) - half[1] + r, np.abs(z - c[2]) - half[2] + r
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) + np.minimum(np.maximum(np.maximum(qx, qy), qz), 0) - r


def knit(a, b, k):
    d = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - d * d * k * .25


def field(x, y, z):
    f = rbox(x, y, z, (0.0, 0.0, ZM), (BODY_W / 2, BODY_D / 2, BODY_H / 2), ROUND)
    legs = None
    for lx in LEGS:
        d = rbox(x, y, z, (lx, LEG_Y, (FLOOR + ZB + 8.0) / 2), (LEG_W / 2, LEG_D / 2, (LEG_H + 8.0) / 2), LEG_R)
        legs = d if legs is None else np.minimum(legs, d)
    claw_c = BODY_W / 2 + CLAW_L - (CLAW_L + 8.0) / 2
    claws = np.minimum(rbox(x, y, z, (claw_c, 0.0, CZ), ((CLAW_L + 8.0) / 2, CLAW_HY, CLAW_HZ), CLAW_R), rbox(x, y, z, (-claw_c, 0.0, CZ), ((CLAW_L + 8.0) / 2, CLAW_HY, CLAW_HZ), CLAW_R))
    return knit(knit(f, legs, KNIT_LEG), claws, KNIT_CLAW)


def nets(fn, lo, hi, h):
    """A closed mesh of where fn is zero: one vertex for each cell of a grid the surface goes through, one quad for each grid edge it crosses. One slab of the grid at a time."""
    xs, ys, zs = (np.arange(lo[i], hi[i] + h, h, dtype=np.float32) for i in range(3))
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    verts, quads = [], []
    count = 0
    f0 = fn(float(xs[0]), Y, Z) < 0
    prev = None
    for i in range(len(xs) - 1):
        f1 = fn(float(xs[i + 1]), Y, Z) < 0
        n = f0[:-1, :-1].astype(np.int8) + f0[1:, :-1] + f0[:-1, 1:] + f0[1:, 1:] + f1[:-1, :-1] + f1[1:, :-1] + f1[:-1, 1:] + f1[1:, 1:]
        j, k = np.nonzero((n > 0) & (n < 8))
        ids = np.full(n.shape, -1, np.int32)
        ids[j, k] = count + np.arange(len(j), dtype=np.int32)
        count += len(j)
        verts.append(np.stack([np.full(len(j), xs[i] + h / 2, np.float32), ys[j] + h / 2, zs[k] + h / 2], -1))
        a, b = np.nonzero(f0[1:-1, 1:-1] != f1[1:-1, 1:-1])
        a, b = a + 1, b + 1
        q = np.stack([ids[a - 1, b - 1], ids[a, b - 1], ids[a, b], ids[a - 1, b]], -1)
        q[~f0[a, b]] = q[~f0[a, b]][:, ::-1]
        quads.append(q)
        if prev is not None:
            a, b = np.nonzero(f0[:-1, 1:-1] != f0[1:, 1:-1])
            b = b + 1
            q = np.stack([prev[a, b - 1], ids[a, b - 1], ids[a, b], prev[a, b]], -1)
            q[f0[a, b]] = q[f0[a, b]][:, ::-1]
            quads.append(q)
            a, b = np.nonzero(f0[1:-1, :-1] != f0[1:-1, 1:])
            a = a + 1
            q = np.stack([prev[a - 1, b], ids[a - 1, b], ids[a, b], prev[a, b]], -1)
            q[~f0[a, b]] = q[~f0[a, b]][:, ::-1]
            quads.append(q)
        prev, f0 = ids, f1
    return np.concatenate(verts), np.concatenate(quads)


def onto(fn, p, n=3):
    """Vertices brought onto the surface itself, and its normal there."""
    e = .04
    for it in range(n + 1):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        g = np.stack([fn(x + e, y, z) - fn(x - e, y, z), fn(x, y + e, z) - fn(x, y - e, z), fn(x, y, z + e) - fn(x, y, z - e)], -1) / (2 * e)
        if it == n:
            break
        p = p - g * (fn(x, y, z) / np.maximum((g * g).sum(1), 1e-6))[:, None]
    return p.astype(np.float32), (g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def box_blur(a, n):
    for ax in (0, 1):
        c = np.cumsum(np.concatenate([np.take(a, [0] * (n + 1), ax), a, np.take(a, [-1] * n, ax)], ax), axis=ax, dtype=np.float64)
        a = ((np.take(c, range(2 * n + 1, c.shape[ax]), ax) - np.take(c, range(0, c.shape[ax] - 2 * n - 1), ax)) / (2 * n + 1)).astype(np.float32)
    return a


class Clawd:
    """His surface at rest: where each vertex is, its normal, and everything that stays on him whatever he does."""

    def __init__(self, h, fine=True):
        self.h = h
        key = "_".join(f"{v:g}" for v in GEOMETRY)
        path = os.path.join(OUT, "crab", f"clawd_rest_{h:g}_{key}.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.R, self.N, self.quads = z["R"], z["N"], z["quads"]
        else:
            top = BODY_W / 2 + CLAW_L + 6.0
            V, self.quads = nets(field, (-top, -BODY_D / 2 - 6.0, FLOOR - 3.0), (top, BODY_D / 2 + 6.0, ZT + 6.0), h)
            self.R, self.N = onto(field, V)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path[:-4]}.{os.getpid()}.npz"      # two workers may build it at once: each writes its own, then moves it into place whole
            np.savez(tmp, R=self.R, N=self.N, quads=self.quads)
            os.replace(tmp, path)
        R, N = self.R, self.N
        print("clawd:", len(R), "verts", len(self.quads), "quads", flush=True)
        self.fz = ZT - R[:, 2]
        self.fr = np.nonzero(N[:, 1] < -.2)[0]
        near = np.ones(len(R), np.float32) * 1e3
        for xc, zc in EYES:
            near = np.minimum(near, rrect(R[:, 0], self.fz, xc, zc, U / 2, U, 4.0))
        near = np.minimum(near, np.hypot(R[:, 0] / 1.0, self.fz - MOUTH_FZ) - 1.0 * U)
        self.quiet = (1 - sm(1 - near / 7.0) * (N[:, 1] < -.2)).astype(np.float32)
        # the shape a hand gives it: faces not quite flat, nothing quite straight
        on_body = np.clip(1 - (R[:, 0] / (BODY_W / 2)) ** 2, 0, 1) * np.clip(1 - ((R[:, 2] - ZM) / (BODY_H / 2)) ** 2, 0, 1)
        pillow = on_body * (3.6 * N[:, 1] ** 2 + 2.0 * (N[:, 0] ** 2 + N[:, 2] ** 2))
        marks = self.pinches() + self.prints(fine) + self.faint_marks()
        groove, self.deep = self.scar()
        small = waves(R, 20.0, 12) * .22 + waves(R, 7.0, 13) * .04 + marks * self.quiet
        # under each eye the skin is left without its small marks, or the glaze would show every one of them
        for xc, zc in EYES:
            d = rrect(R[:, 0], self.fz, xc, zc, EYE_HW * 1.22, EYE_HH + 1.5, 4.0)
            small = small * (1 - sm((3.0 - d) / 7.0) * (N[:, 1] < -.5))
        static = pillow + waves(R, 52.0, 11) * .9 + small + groove
        self.static = static.astype(np.float32)
        self.lag = waves(R, 70.0, 14).astype(np.float32)
        self.block = None

    def pinches(self, count=110):
        """Where it was pinched and pushed about: shallow, soft, each a different size and lying its own way, with the clay it pushed aside just beside it. All over him, claws too."""
        R, N = self.R, self.N
        r = np.random.default_rng(31)
        out = np.zeros(len(R), np.float32)
        sites = []
        for _ in range(5000):
            i = int(r.integers(len(R)))
            if all(((R[i] - s) ** 2).sum() > 19.0 ** 2 for s in sites):
                sites.append(R[i])
                n = N[i]
                ax = np.cross(n, r.normal(size=3))
                ax = ax / np.linalg.norm(ax)
                bx = np.cross(n, ax)
                L, W = r.uniform(9, 24), r.uniform(5, 11)
                d = R - R[i]
                near = np.nonzero(((d * d).sum(1) < (2.4 * L) ** 2) & (N @ n > .35))[0]
                a, b = d[near] @ ax, d[near] @ bx
                b = b + 1.8 * np.sin(a * r.uniform(.12, .3) + r.uniform(0, 6))
                dep = r.uniform(.03, .1)
                out[near] += -dep * np.exp(-(a / L) ** 2 - (b / W) ** 2) + dep * .4 * np.exp(-(a / L) ** 2 - ((b - 1.5 * W) / (.6 * W)) ** 2)
                if len(sites) >= count:
                    break
        return out

    def prints(self, ridged):
        """A few finger prints, the real size, where someone held him: only part of each took, as on real clay."""
        if not ridged:
            return 0.0
        R, N = self.R, self.N
        r = np.random.default_rng(33)
        out = np.zeros(len(R), np.float32)
        for spot in ((-4.1 * U, -BODY_D / 2, ZT - 6.9 * U), (4.9 * U, -BODY_D / 2, ZT - 1.3 * U), (2.3 * U, -BODY_D / 2, ZT - 7.1 * U), (-7.3 * U, -CLAW_HY, CZ + 4.0), (7.5 * U, -CLAW_HY, CZ - 3.0), (-1.5 * U, -BODY_D / 2, ZT - .6), (5.2 * U, 0.0, ZT)):
            i = int(np.argmin(((R - np.array(spot, np.float32)) ** 2).sum(1)))
            n = N[i]
            ax = np.cross(n, r.normal(size=3))
            ax = ax / np.linalg.norm(ax)
            bx = np.cross(n, ax)
            d = R - R[i]
            near = np.nonzero(((d * d).sum(1) < 13.0 ** 2) & (N @ n > .5))[0]
            a, b = d[near] @ ax, d[near] @ bx
            took = sm((1 - (a / 7.2) ** 2 - (b / 10.5) ** 2) / .4) * sm((waves(np.stack([a, b, 0 * a], -1).astype(np.float32), 9.0, int(r.integers(1000))) + .5) / .9)
            arch = (b - 5.5 * np.exp(-(a / 6.0) ** 2) + .5 * waves(np.stack([a, b, 0 * a], -1).astype(np.float32), 7.0, int(r.integers(1000)))) / .62
            out[near] += (.034 * np.sin(2 * np.pi * arch) - .05) * took
        return out

    def faint_marks(self):
        """What was pressed into him once, no more than a faint unevenness now: only the fine part of it, its depth limited, the same all over."""
        z = np.load(os.path.join(OUT, "state_2.npz"))
        H, res = z["H"], int(z["res"])
        k = res // 8
        ny, nx = H.shape[0] // k, H.shape[1] // k
        Hs = H[:ny * k, :nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3)).astype(np.float32)
        del H
        low = box_blur(box_blur(Hs, 16), 16)
        band = np.tanh((Hs - low) / .6) * .6 * .08
        R, N = self.R, self.N
        which = np.argmax(np.abs(N), axis=1)
        sure = sm((np.abs(N).max(1) - .78) / .16)
        u = np.select([which == 1, which == 0, which == 2], [R[:, 0] + 4.0, R[:, 1] + 50.0, R[:, 0] - 6.0], 0)
        v = np.select([which == 1, which == 0, which == 2], [R[:, 2] - ZM, R[:, 2] - ZM - 6.0, R[:, 1] * 1.4], 0)
        rows, cols = np.clip(((B - v) * 8).astype(np.int32), 0, ny - 1), np.clip(((u + A) * 8).astype(np.int32), 0, nx - 1)
        return band[rows, cols] * sure

    def scar(self):
        """All that is left of 我: its first stroke, without the last of its tail, narrower and shallow now, its edges gone soft. It lands on his forehead."""
        R, fz = self.R, self.fz
        out, deep = np.zeros(len(R), np.float32), np.zeros(len(R), np.float32)
        x, y, hw = first_stroke()
        i = len(x) // 2
        line = np.stack([SCAR_AT[0] + (x - x[i]), SCAR_AT[1] - (y - y[i])], 1)[::3]
        wide = hw[::3] * .62
        s = np.linspace(0, 1, len(line))
        idx = np.nonzero((R[:, 1] < 0) & (R[:, 0] > line[:, 0].min() - 9) & (R[:, 0] < line[:, 0].max() + 9) & (fz > line[:, 1].min() - 9) & (fz < line[:, 1].max() + 9))[0]
        d2 = (R[idx, 0, None] - line[None, :, 0]) ** 2 + (fz[idx, None] - line[None, :, 1]) ** 2
        best = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(len(idx)), best])
        sb = s[best]
        w = wide[best]
        dep = .85 * sm(sb / .05) * sm((1 - sb) / .4) ** .5
        out[idx] = -dep * np.exp(-(d / w) ** 2) + .16 * dep * np.exp(-((d - 1.9 * w) / (.9 * w)) ** 2)
        deep[idx] = np.exp(-(d / w) ** 2) * dep / 1.7
        return out, deep


FIELDS = {}


def block_field(which):
    """The block's height field and what goes with it, as a dict: "last" after the seven blows (crew A's state_path(LAST)), "wo" with the groove of
    我 in it (state_path("wo")). Until crew A's groove pass has written "wo" on the seven-blow field, the old groove is laid onto the new field in
    its place: the five-blow field's groove (imprint/state_wo.npz less imprint/state_5.npz) added to state_path(LAST), its wet "fresh" with it."""
    if which not in FIELDS:
        path = state_path(LAST_BLOW if which == "last" else which)
        if os.path.exists(path):
            z = np.load(path)
            FIELDS[which] = {key: z[key] for key in z.files}
        elif which == "wo":
            d = dict(block_field("last"))
            old, wo = np.load(os.path.join(OUT, "state_5.npz")), np.load(os.path.join(OUT, "state_wo.npz"))
            d["H"] = (d["H"] + (wo["H"] - old["H"])).astype(np.float32)
            d["fresh"] = wo["fresh"]
            print("state_wo: not on the seven-blow field yet, the old groove laid onto", state_path(LAST_BLOW), flush=True)
            FIELDS[which] = d
        else:
            raise FileNotFoundError(path)
    return FIELDS[which]


def first_stroke():
    """The first stroke of 我 as the groove ploughed it, in mm on the block: all that stays of it on him, nine tenths of it."""
    z = np.load(os.path.join(OUT, "crab", "wo_first.npz"))
    n = int(len(z["x"]) * .9)
    return z["x"][:n], z["y"][:n], z["hw"][:n]


def field_at(H, res, f, x, y):
    """A height field of the block, box-filtered f cells, at points in mm."""
    ny, nx = H.shape[0] // f, H.shape[1] // f
    Hs = H[:ny * f, :nx * f].reshape(ny, f, nx, f).mean(axis=(1, 3)).astype(np.float32)
    row, col = np.clip((B - y) * res / f - .5, 0, ny - 1.001), np.clip((x + A) * res / f - .5, 0, nx - 1.001)
    r0, c0 = row.astype(np.int32), col.astype(np.int32)
    fr, fc = row - r0, col - c0
    return (Hs[r0, c0] * (1 - fc) + Hs[r0, c0 + 1] * fc) * (1 - fr) + (Hs[r0 + 1, c0] * (1 - fc) + Hs[r0 + 1, c0 + 1] * fc) * fr


BLOCK_OF_OUT = 1.0
NT_ABOVE = 40              # the rings of the block's sides, as clay_from_above has block_of make them


def to_block(k, h):
    """Where each point of his skin lay on the block. The block's top is his face: it rolls up towards us and draws in. The back of it is his crown, its front wall what he stands on, its underside his back. The top keeps its size about the first stroke of the character, which comes to rest where the mark on his forehead is, and everything further out is drawn in more."""
    half = np.array((BODY_W / 2, BODY_D / 2, BODY_H / 2), np.float32)
    core = half - ROUND
    # the legs not yet come: each folded up into the underside of him, where it will come out, drawn in a little the deeper it goes, so that nothing folds over
    R = k.R.copy()
    own = np.array(LEGS, np.float32)[np.argmin(np.abs(R[:, 0, None] - np.array(LEGS, np.float32)[None, :]), axis=1)]

    def under(y):
        e = np.clip(np.abs(y) - (BODY_D / 2 - ROUND), 0, ROUND - 1e-3)
        return ZB + ROUND - np.sqrt(ROUND * ROUND - e * e)
    depth = np.clip((under(R[:, 1]) - R[:, 2]) / LEG_H, 0, 1)
    inside = R[:, 2] < under(R[:, 1])
    R[:, 0] = np.where(inside, own + (R[:, 0] - own) * (1 - .3 * depth), R[:, 0])
    R[:, 1] = np.where(inside, LEG_Y + (R[:, 1] - LEG_Y) * (1 - .3 * depth), R[:, 1])
    R[:, 2] = np.where(inside, under(R[:, 1]) + .02 * (R[:, 2] - under(R[:, 1])), R[:, 2])
    q = R - np.array((0.0, 0.0, ZM), np.float32)
    inner = np.clip(q, -core, core)
    d = q - inner
    n = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    cy, cz = float(core[1]), float(core[2])
    # how far round the body's section each point is from the middle of his face: over the crown to the middle of his back one way, under him the other
    phi = np.arctan2(n[:, 2], -n[:, 1] + 1e-5)
    far = np.abs(phi) >= math.pi / 2 - 1e-3
    up = phi >= 0
    iy, iz = inner[:, 1], inner[:, 2]
    arc = np.where(up, np.where(far, 2 * cz - iz, iz) + ROUND * np.abs(phi) + (iy + cy), np.where(far, -2 * cz - iz, iz) - ROUND * np.abs(phi) - (iy + cy))
    round_of = 2 * cz + ROUND * math.pi + 2 * cy
    x1, y1, _ = first_stroke()
    x_b, y_b = float(x1[len(x1) // 2]), float(y1[len(y1) // 2])
    x_f, z_f = SCAR_AT[0], BODY_H / 2 - SCAR_AT[1]
    keep = 19.0
    # what he stands on lies on its front wall at the same heights all across, so that the legs can come out of it
    z_near, z_far = -(BODY_D / 2 - cy) / BODY_D * THICK, -(BODY_D / 2 + cy) / BODY_D * THICK
    sb = np.interp(arc, (-round_of, -cz - ROUND * math.pi / 2 - 2 * cy, -cz - ROUND * math.pi / 2, -50.0, z_f - keep, z_f + keep, cz + ROUND * math.pi / 2 + 6.0, round_of), (-2 * B - THICK, -B + z_far, -B + z_near, -B, y_b - keep, y_b + keep, B, 2 * B + THICK))
    a = np.abs(sb)
    Y = np.sign(sb) * np.where(a < B, a, np.where(a < B + THICK, B, B - (a - B - THICK)))
    Z = -np.clip(a - B, 0, THICK)
    rim = 1.2             # block_of's top edge is all but sharp
    X = np.interp(inner[:, 0], (-core[0], x_f - keep, x_f + keep, core[0]), (-A + rim, x_b - keep, x_b + keep, A - rim)) + n[:, 0] * rim
    # the ends of him go to the ends of it
    w = (n[:, 0] ** 2)
    Y2 = np.interp(iz, (-cz, z_f - keep, z_f + keep, cz), (-B, y_b - keep, y_b + keep, B))
    Z2 = -(iy + cy) / (2 * cy) * THICK
    p = np.stack([X, Y * (1 - w) + Y2 * w, Z * (1 - w) + Z2 * w], -1)
    lo, hi = np.array((-A + rim, -B + rim, -THICK + rim)), np.array((A - rim, B - rim, -rim))
    i_b = np.clip(p, lo, hi)
    e = p - i_b
    out = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-6)
    p = i_b + out * rim
    # the top's own edge is rounded by the height field itself (EDGE_R): the rim above the wall stays at the top's level and takes the field's
    # heights there, as block_of's top does right out to the edge. Rounded down by the rim as well it made a shelf under the edge, 0.7 mm wide
    # and all but flat, that caught the light as a broken line along the front wall, on him and not on the block
    p[:, 2] = np.where(out[:, 2] > 0, 0.0, p[:, 2])
    k.b_leg = np.zeros(len(p), np.float32)
    # the corners seen from above, rounded as block_of has them
    from film_imprint3d import corner_warp
    p[:, 0], p[:, 1] = corner_warp(p[:, 0], p[:, 1])
    # its sides exactly as block_of makes them, so that the block does not change when it becomes him: each point of a wall at its place round the
    # top's edge (s) and its depth down the wall (t), pushed out as far as block_of pushes that place out, with the same wire marks and pats
    from film_imprint3d import CORNER, side_tiles
    C = CORNER
    cx_, cy_ = A - C, B - C
    straight = (2 * cx_, 2 * cy_)
    whole = 2 * straight[0] + 2 * straight[1] + 2 * math.pi * C

    def around(x, y):
        """Where round the top's edge (0 to 1), as block_of's loop runs: from the middle of the far left corner along the far edge, clockwise seen
        from above; and the way out there."""
        q = math.pi * C / 4
        a = np.arctan2(y - np.sign(y) * cy_, x - np.sign(x) * cx_)
        e = np.select([(y >= cy_) & (np.abs(x) <= cx_), (x >= cx_) & (np.abs(y) <= cy_), (y <= -cy_) & (np.abs(x) <= cx_), (x <= -cx_) & (np.abs(y) <= cy_),
                       (x > 0) & (y > 0), (x > 0) & (y < 0), (x < 0) & (y < 0)],
                      [q + (x + cx_), 3 * q + 2 * cx_ + (cy_ - y), 5 * q + 2 * cx_ + 2 * cy_ + (cx_ - x), 7 * q + 4 * cx_ + 2 * cy_ + (y + cy_),
                       q + 2 * cx_ + C * (math.pi / 2 - a), 3 * q + 2 * cx_ + 2 * cy_ + C * (-a), 5 * q + 4 * cx_ + 2 * cy_ + C * (-math.pi / 2 - a)],
                      7 * q + 4 * cx_ + 4 * cy_ + C * (math.pi - np.where(a < 0, a + 2 * math.pi, a)))
        ox = np.select([(y >= cy_) & (np.abs(x) <= cx_), (x >= cx_) & (np.abs(y) <= cy_), (y <= -cy_) & (np.abs(x) <= cx_), (x <= -cx_) & (np.abs(y) <= cy_)], [0.0, 1.0, 0.0, -1.0], np.cos(a))
        oy = np.select([(y >= cy_) & (np.abs(x) <= cx_), (x >= cx_) & (np.abs(y) <= cy_), (y <= -cy_) & (np.abs(x) <= cx_), (x <= -cx_) & (np.abs(y) <= cy_)], [1.0, 0.0, -1.0, 0.0], np.sin(a))
        return (e % whole) / whole, ox, oy
    wall = 1 - sm((np.abs(out[:, 2]) - .4) / .4)          # the four sides, not the top nor what lies on the board
    tw = np.clip(-p[:, 2] / THICK, 0, 1) ** (1 / .96)
    zw = block_field("wo")
    H_wo, bulge, lost = zw["H"], zw["bulge"], float(zw["lost"])
    h_side = THICK + float(np.concatenate([H_wo[0], H_wo[-1], H_wo[:, 0], H_wo[:, -1]]).mean())       # as block_of: the top's edge, not its middle
    amount = lost / (2 * (2 * A + 2 * B) * h_side * .66)
    on = wall > 0
    sx_, ox, oy = around(p[on, 0], p[on, 1])
    # the place on the edge each wall point hangs from, and how far that place was pushed out
    ex = np.clip(p[on, 0], -A, A)
    ey = np.clip(p[on, 1], -B, B)
    loc = bulge[np.clip(((B - ey) / 2).astype(np.int32), 0, bulge.shape[0] - 1), np.clip(((ex + A) / 2).astype(np.int32), 0, bulge.shape[1] - 1)]
    rim_mean = float(np.concatenate([bulge[0], bulge[-1], bulge[:, 0], bulge[:, -1]]).mean())
    loc = loc / max(rim_mean, 1e-5) * .7 + .3
    N_S, N_T = 4096, 512
    low, wire, fine, pat = side_tiles(N_S, N_T)
    si = (sx_ * (N_S - 1)).astype(np.int32)
    sis = (si * 14) % N_S
    t_ = tw[on]

    def push_at(tr):
        ti = (tr * (N_T - 1)).astype(np.int32)
        bar = 4 * tr * (1 - tr)
        o = amount * loc * bar ** .8 + low[si, ti] * .55 * np.minimum(tr * 5, 1) + wire[si // 3, ti] * .03 + fine[sis, (ti * 4) % N_T] * .01 + pat[si, ti] * .2
        o = o + 1.1 * np.exp(-(1 - tr) * h_side / 2.4) * (1 + low[si, 40] * .5)
        return np.where(tr > 0, o, 0.0)          # the top's own edge is not pushed
    # block_of hangs the sides from the top's edge in NT_ABOVE rings, flat between them, so its marks are soft up and down: the same rings here
    r = t_ * NT_ABOVE
    r0 = np.floor(r)
    fr = r - r0
    push = push_at(r0 / NT_ABOVE) * (1 - fr) + push_at(np.minimum(r0 + 1, NT_ABOVE) / NT_ABOVE) * fr
    pz = field_at(H_wo, int(zw["res"]), 4, ex, ey)
    w_on = wall[on]
    # out of the block, as block_of pushes its sides out (it pushed them in until 2026-11-07; BLOCK_OF_OUT = -1 followed that). The two must agree,
    # or the block changes as it becomes him
    p[on, 0] += BLOCK_OF_OUT * ox * push * w_on
    p[on, 1] += BLOCK_OF_OUT * oy * push * w_on
    p[on, 2] = p[on, 2] * (1 - w_on) + (pz + (-THICK - pz) * t_ ** .96) * w_on
    top = np.where(out[:, 2] > 0, 1 - wall, 0.0)          # the field's heights where the wall's are not: together they are one surface
    f = max(int(32 * h * .6), 1)
    z7 = block_field("last")
    res = int(z7["res"])
    old = field_at(z7["H"], res, f, p[:, 0], p[:, 1])
    new = field_at(H_wo, res, f, p[:, 0], p[:, 1])
    L = np.load(os.path.join(OUT, "crab", "wo_label.npz"))
    step, (r0, c0) = int(L["step"]), L["rc"]
    row, col = (((B - p[:, 1]) * res - r0) / step).astype(np.int32), (((p[:, 0] + A) * res - c0) / step).astype(np.int32)
    label = L["label"][np.clip(row, 0, L["label"].shape[0] - 1), np.clip(col, 0, L["label"].shape[1] - 1)]
    k.block, k.b_top, k.b_old, k.b_wo, k.b_label = p.astype(np.float32), top.astype(np.float32), old, new - old, label
    # how much behind the rest each part of it is as it moves: lumps the size of a third of the block, laid on the block, so that what lies side
    # by side there moves together (laid on him, two places of him that meet at a corner of the block went their own ways and tore it)
    k.lag = waves(p.astype(np.float32), 120.0, 14).astype(np.float32)


# when each stroke of the character starts to close, the last written first; the first stays
WO_SHUT, WO_CLOSING = (99.0,) + tuple(97.85 + .2 * (6 - i) for i in range(1, 7)), 1.0


# the getting up, on the bridge clock. The block stirs (STIR, on then()); it draws in towards his shape as it lies (DRAW); it tips up onto
# its front wall, which is what he stands on, in two heaves, about its own long middle so that it stays where it is, its ends a moment
# behind its middle as soft things are (TIP, degrees); the claws come out of its ends (CLAWS); last it gets up off its folded legs (RISE),
# the right side a little ahead. Everything is a whole shape at every moment: nothing is blended between lying and standing.
TIP = curve((97.95, 0), (98.25, 28), (98.38, 22), (98.85, 70), (99.0, 64), (99.4, 93), (99.55, 90))
DRAW = curve((97.98, 0), (98.05, -.02), (98.35, .17), (98.55, .2), (99.1, .82), (99.45, .97), (99.85, 1))
CLAWS = curve((99.2, 0), (99.6, .52), (99.72, .5), (100.0, 1.13), (100.1, .97), (100.2, 1))
RISE = curve((99.45, 0), (99.75, .3), (99.88, .27), (100.12, .68), (100.22, .62), (100.38, 1.07), (100.48, 1))
SWITCH = 97.58          # from the block itself to his skin on it: at the first pose of the stir that shows, so that the change is the clay moving
FINE_MESH = (.2, 112.5)   # the skin's spacing (mm on him) for frames before that time: on the block it is stretched up to 4.7 times. All of the bridge:
#                           ended at 98.5 (preview5's first pass), the cracks went soft in one pose there, a change of mesh is seen inside a shot


def gather(k, P, t):
    """From the block to him. P is where the skin will be once he is whole; what is pressed into it goes on the way."""
    if k.block is None:
        to_block(k, k.h)
    to = then(t)
    R = k.R
    B_ = k.block.copy()
    heal = 1 - sm((t - np.array(WO_SHUT, np.float32)[k.b_label]) / WO_CLOSING)
    heal[k.b_label == 0] = 1.0
    B_[:, 2] += k.b_top * (k.b_old + k.b_wo * heal)
    st = float(STIR(to))
    if st != 0:
        hump = (1 - np.minimum((B_[:, 0] / A) ** 2 + (B_[:, 1] / B) ** 2, 1)) ** 2       # flat where it meets the still edge, or the edge of the swell shows as a ring
        B_[:, 2] = FLOOR + (B_[:, 2] - FLOOR) * (1 + .15 * st * hump)
    ends = np.abs(B_[:, 0]) / A
    g = DRAW(t + .04 * k.lag + .05 * ends).astype(np.float32)
    phi = TIP(t - .03 * ends ** 2 - .012 * k.lag).astype(np.float32)
    done = np.clip(np.minimum(g, phi / 90.0), 0, 1)
    F = R + (P - R) * done[:, None]         # his own marks come as the old ones go
    # what it draws in towards: him sitting on his folded legs, claws in
    legs = R[:, 2] < ZB
    body_x = np.clip(F[:, 0], -BODY_W / 2, BODY_W / 2)
    S = F.copy()
    S[:, 0] = body_x + (F[:, 0] - body_x) * float(CLAWS(t))
    S[:, 2] = np.where(legs, FLOOR, F[:, 2] - LEG_H)
    # ... as it would lie on its back on the board, before it tips up: the inverse of the tip at 90 degrees
    c_b = -THICK / 2
    S_l = np.stack([S[:, 0], S[:, 2] - (FLOOR + B), -S[:, 1] + c_b], -1)
    L = B_ + (S_l - B_) * g[:, None]
    # the tip, about the block's long middle; it keeps touching the board
    ph = np.radians(phi)
    cs, sn = np.cos(ph), np.sin(ph)
    ly, lz = L[:, 1], L[:, 2] - c_b
    W = np.stack([L[:, 0], ly * cs - lz * sn, ly * sn + lz * cs + FLOOR + B * sn + (THICK / 2) * cs], -1)
    if float(TIP(t)) > 0:
        W[:, 2] += FLOOR - float(W[:, 2].min())
    # and up onto his legs
    lift = RISE(t + .06 * R[:, 0] / (BODY_W / 2)).astype(np.float32)
    W[:, 2] += np.where(legs, (F[:, 2] - FLOOR), LEG_H) * lift
    return W.astype(np.float32)


def warmth(k, t):
    """It takes colour evenly, from the top down, as the sun gets to it."""
    T = float(curve((MORPH + .2, -.3), (100.6, 1.5))(t))
    high = np.clip((k.R[:, 2] - FLOOR) / (BODY_H + LEG_H), 0, 1)
    return sm((T - .25 * (1 - high)) / 1.25).astype(np.float32)


def face(k, t):
    """What his eyes and mouth are doing to his skin at time t, and how far each vertex is from their edges: the shader cuts the glaze there, so the edge is clean however coarse the mesh."""
    R = k.R
    n = len(R)
    idx = k.fr
    x, fz = R[idx, 0], k.fz[idx]
    disp = np.zeros(n, np.float32)
    eye_d, mouth_d, glint_d = np.full(n, 99.0, np.float32), np.full(n, 99.0, np.float32), np.full(n, 99.0, np.float32)
    munch, cheeks = float(MUNCH(t)), float(CHEEKS(t))
    chew = float(CHEW(t))
    squint = max(float(SQUINT(t)), .3 * chew * munch + .04 * max(cheeks, 0))     # bright-eyed while he chews, the eyes pressed a little with each bite
    opened = float(EYE_OPEN(t)) * (1 - .93 * squint) * (1 - .93 * float(BLINK(t)))
    got = float(EYE_GOT(t))
    if got > 0:
        de, dn, dg = np.full(len(idx), 99.0, np.float32), np.zeros(len(idx), np.float32), np.full(len(idx), 99.0, np.float32)
        for xc, zc in EYES:
            hw, hh = EYE_HW * (1 + .22 * squint), max(EYE_HH * opened, 1.05) * (.3 + .7 * got)
            u = np.clip((x - xc) / hw, -1.4, 1.4)
            d = rrect(x, fz - (2.6 * squint + 1.6 * (1 - min(opened, 1))) * (1 - u * u) * sm(squint / .5), xc, zc, hw, hh, min(.44 * U, hh))
            # filled: a shallow dome of glaze standing a hair above the skin, so shallow that only the one light near the camera shows in it
            tall = min(hh / EYE_HH, 1) ** 2
            dent = .42 * tall * sm(-d / (.75 * min(hw, hh))) - .1 * np.exp(-(d / .9) ** 2)
            near = np.abs(x - xc) < hw + 9
            dn = np.where(near, dent * got, dn)
            de = np.where(near, d, de)
            # one grain of sun, high on the side the sky is: when the lid comes down it is gone
            r = 1.45 * sm((tall - .04) / .16) * got
            if r > 0:
                dg = np.where(near, np.hypot(x - (xc - .3 * hw), fz - (zc - .5 * hh)) - r, dg)
        disp[idx] += dn
        eye_d[idx] = de
        glint_d[idx] = dg
    y, e, b = float(YAWN(t)), float(EAT(t)), float(BREATH(t))
    if max(y, e, b, munch) > 0:
        if y >= max(e, b):
            hw, hh, dep, cx = .84 * U * y ** .6, max(.68 * U * y, .8), 6.5 * y, 0.0
        elif b >= e:
            hw, hh, dep, cx = .76 * U * b ** .6, max(.6 * U * b, .8), 6.0 * b, 0.0
        else:
            hw, hh, dep, cx = .6 * U * e ** .6, max(.46 * U * e, .8), 4.5 * e, 0.0
        if munch > 0 and hh < 1.5:      # shut, and working
            hw, hh, dep, cx = .42 * U, 1.0, 1.0, .2 * U * math.sin(2 * math.pi * (t - GO[0]) / GO[1]) * munch
        dx, dz = (x - cx) / hw, (fz - MOUTH_FZ) / hh
        d = (np.sqrt(dx * dx + dz * dz) - 1) * min(hw, hh)
        tt = np.clip(-d / hh, 0, 1)
        disp[idx] += np.where(d < 0, -dep * np.sin(np.pi / 2 * tt), 0) + .42 * min(dep / 2.0, 1) * np.exp(-(d / 1.7) ** 2)
        mouth_d[idx] = d
    if cheeks != 0:     # his cheeks full, the food going from one to the other
        go = math.sin(2 * math.pi * (t - GO[0]) / GO[1])
        for s in (-1.0, 1.0):
            disp[idx] += 9.0 * CHEEK_GAIN * cheeks * (1 + .22 * chew) * (1 + .28 * s * go) * np.exp(-((x - s * 3.4 * U) / (1.9 * U)) ** 2 - ((fz - 5.7 * U) / (1.6 * U)) ** 2)
    return disp, eye_d, mouth_d, glint_d


def arm(x, y, z, s, theta, length, ride, out):
    """One claw, s the side it is on: it can ride up his side, and be drawn out, and it bends like clay from the root on, as far as theta (radians) at the end of the bend. Whatever is not arm stays."""
    # the skin slides along his wall and stops short of the rounded corner, so that it stays on the wall and nothing folds
    glide = np.interp(z, (ZB + 3.0, CZ - 1.25 * U, CZ + 1.25 * U, ZT - ROUND - 2.0), (0, 1, 1, 0)) * sm((np.abs(x) - (BODY_W / 2 - 26.0)) / 24.0)
    q = z - CZ
    sx = np.abs(x) - BODY_W / 2
    w = sm((sx + 3.0) / 7.0) * sm((1.3 - np.hypot(y / (1.55 * U), q / (1.45 * U))) / .35)
    S = np.where(sx > 0, sx * length, sx)
    bend = .6 * U * length
    tab = np.linspace(0, (CLAW_L + 2.0) * length, 96)
    th = theta * sm(tab / bend)
    ds = tab[1] - tab[0]
    cx, cz = np.concatenate([[0], np.cumsum(np.cos(th[:-1]) * ds)]), np.concatenate([[0], np.cumsum(np.sin(th[:-1]) * ds)])
    ths = np.interp(S, tab, th)
    thin = length ** -.5
    qq = q * thin
    bx = s * (BODY_W / 2 + out + np.where(S > 0, np.interp(S, tab, cx), S) - qq * np.sin(ths))
    bz = CZ + ride + np.where(S > 0, np.interp(S, tab, cz), 0) + qq * np.cos(ths)
    return x + (bx - x) * w, y + (y * thin - y) * w, (z + ride * glide) * (1 - w) + bz * w


def pose(p, t):
    """Where the point p of his body at rest has got to at time t. The legs stay on their feet."""
    p = np.array(p, np.float32)
    x, y, z = p[:, 0].astype(np.float64), p[:, 1].astype(np.float64), p[:, 2].astype(np.float64)
    both, hello, cheeks = float(RAISE(t)), float(WAVE(t)), float(CHEEKS(t))
    # the claws: up at the yawn; the waving one rides up his side and is drawn out, so that it stands clear above the corner of his head
    shake = 5.0 * float(sm((t - 102.6) / .15) * sm((103.5 - t) / .15)) * math.sin(2 * math.pi * 5.5 * t)
    for s in (-1.0, 1.0):
        on = (x >= 0) if s > 0 else (x < 0)
        hi = hello if s == HELLO else 0.0
        theta = math.radians(45.0 * both + shake * max(both, 0) + 70.0 * hi + (sway(t) if s == HELLO else 0.0))
        length, ride, out = 1 + .75 * max(hi, 0), .5 * U * both + 2.4 * U * hi, 1.0 * U * max(hi, 0)
        if theta != 0 or ride != 0:
            x[on], y[on], z[on] = arm(x[on], y[on], z[on], s, theta, length, ride, out)
    # his cheeks full, showing in his outline, the food going from one side to the other
    if cheeks != 0:
        go = math.sin(2 * math.pi * (t - GO[0]) / GO[1])
        x *= 1 + .085 * CHEEK_GAIN * cheeks * (1 + .32 * np.sign(x) * go) * np.exp(-((z - (ZT - 5.7 * U)) / (2.3 * U)) ** 2)
    # the body: drawn out or pressed down, bent back or forward from the hips, tipped. The feet don't move
    st, ch, bow, br = float(STRETCH(t)), float(CHEW(t)), float(BOW(t)), float(BREATH(t))
    slump = float(SLUMP(t))
    grow = 1 + .13 * st - .11 * ch + .03 * br - SLUMP_BY[1] * slump
    thin = grow ** -.5 * (1 + .05 * br + .035 * max(cheeks, 0)) * (1 + SLUMP_BY[0] * slump)
    sunk = max(bow, 0) * .4 * LEG_H + ch * .13 * LEG_H
    h = z - ZB
    above = np.maximum(h, 0)
    w2 = np.clip((z - FLOOR) / (ZB - FLOOR), 0, 1)
    x = x * (1 + (thin - 1) * w2)
    y = y * (1 + (thin - 1) * w2)
    below = (np.minimum(z, ZB) - FLOOR) * ((LEG_H - sunk) / LEG_H) + FLOOR
    phi = math.radians(9.5 * max(st, 0) - 20.0 * bow) * np.clip(above / BODY_H, 0, 1.25) ** .6
    rz = above * grow
    y2 = y * np.cos(phi) + rz * np.sin(phi)
    z2 = below + rz * np.cos(phi) - y * np.sin(phi) * (h > 0)
    tip = math.radians(hello_tip(t) + 2.0 * max(st, 0))
    x2 = x + above * math.sin(tip)
    psi = math.radians(float(YAW(t)))
    return np.stack([x2 * math.cos(psi) - y2 * math.sin(psi), x2 * math.sin(psi) + y2 * math.cos(psi), z2], -1).astype(np.float32)


# ---- the letters --------------------------------------------------------------------------------------------------------------------------------------------
# key, x, y, how it stands: turned about the vertical, leaning back, tipped sideways (degrees)
LETTERS = (("zi", -.15 * U, -66.0, 7.0, 16.0, -5.0), ("a", 1.5 * U, -58.0, -14.0, 12.0, 9.0), ("wedge", -1.75 * U, -52.0, 12.0, 20.0, 6.0), ("shan", -6.75 * U, -38.0, 18.0, 14.0, -12.0), ("omega", 6.1 * U, -46.0, -20.0, 10.0, 15.0), ("code", 9.5 * U, -30.0, -24.0, 16.0, 0.0), ("wen", -3.9 * U, -76.0, -10.0, 22.0, 13.0))
THICK_L, ROUND_L = 6.4, 1.5
FALLS = dict(shan=99.2, zi=99.35, omega=99.5, a=99.7, wedge=99.9, code=100.05, wen=100.25)       # when each came off him as he got up
EATEN = dict(zi=0, a=1)
# the rest, as he breathes in: which side it comes along, its place in that side's queue, when it hops. The last one sticks at his lips for a moment
QUEUE = dict(wedge=(-1, 0, 109.96), omega=(1, 0, 110.03), wen=(-1, 1, 110.1), code=(1, 1, 110.17), shan=(-1, 2, 110.23))
STUCK = "shan"


def rot(az, back, tip):
    a, b, c = math.radians(az), math.radians(back), math.radians(tip)
    rz = np.array(((math.cos(a), -math.sin(a), 0), (math.sin(a), math.cos(a), 0), (0, 0, 1)))
    rx = np.array(((1, 0, 0), (0, math.cos(b), -math.sin(b)), (0, math.sin(b), math.cos(b))))
    ry = np.array(((math.cos(c), 0, math.sin(c)), (0, 1, 0), (-math.sin(c), 0, math.cos(c))))
    return rz @ rx @ ry


def letter_shape(d, res, h):
    """One letter solid: thick, every edge round, standing up in its own plane (x across, z up, y through it)."""
    ny, nx = d.shape
    wide, tall = nx / res / 2, ny / res / 2

    def fn(x, y, z):
        col, row = np.clip((x + wide) * res - .5, 0, nx - 1.001), np.clip((tall - z) * res - .5, 0, ny - 1.001)
        c0, r0 = np.floor(col).astype(np.int32), np.floor(row).astype(np.int32)
        fc, fr = col - c0, row - r0
        dd = (d[r0, c0] * (1 - fc) + d[r0, c0 + 1] * fc) * (1 - fr) + (d[r0 + 1, c0] * (1 - fc) + d[r0 + 1, c0 + 1] * fc) * fr
        q1, q2 = dd + ROUND_L, np.abs(y) - THICK_L / 2 + ROUND_L
        return np.hypot(np.maximum(q1, 0), np.maximum(q2, 0)) + np.minimum(np.maximum(q1, q2), 0) - ROUND_L

    def fn_b(x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        return fn(x, y, z)
    V, Q = nets(fn_b, (-wide, -THICK_L, -tall), (wide, THICK_L, tall), h)
    V, Nn = onto(fn_b, V, 2)
    V = V + Nn * ((waves(V, 6.0, 41) * .16 + waves(V, 2.0, 42) * .03))[:, None]
    return V, Q


def letter_at(key, V, x, y, az, back, tip, mouth_at, t):
    """Where a letter is at time t, or None if it has not come off him yet or has been eaten."""
    def on_board(px, py, turn):
        P = V @ turn.T
        return P + np.array((px, py, FLOOR - float(P[:, 2].min()) - .25), np.float32)

    def flying(frm, to, u, a1, a2, size):
        at = frm + (to - frm) * (u * u * (3 - 2 * u) * .3 + u * .7) + np.array((0, 0, 20.0 * 4 * u * (1 - u)), np.float32)
        return (V @ rot(a1 * (1 - u) + a2[0] * u, back * (1 - u) + a2[1] * u, tip * (1 - u) + a2[2] * u).T) * size + at

    rest_turn = rot(az, back, tip)
    if t < FALLS[key]:
        return None
    u = (t - FALLS[key]) / .13
    if u < 1:
        return on_board(x - .1 * x * (1 - u), y + 10 * (1 - u), rot(az + 40 * (1 - u), back + 50 * (1 - u), tip)) + np.array((0, 0, 72.0 * (1 - u * u)), np.float32)
    if key in EATEN:
        T, d = ONE_BY_ONE[EATEN[key]]
        T0, T1 = T + HOP[0] * d / 1.27, T + HOP[1] * d / 1.27
        if t >= T1:
            return None
        P = on_board(x, y, rest_turn)
        if t > T0 - .14:
            u = (t - T0) / (T1 - T0)
            if u < 0:       # it gathers itself
                sq = 1 - .22 * math.sin(math.pi * (t - T0 + .14) / .14)
                P0 = V @ rest_turn.T
                P = P0 * np.array((1 / math.sqrt(sq), 1 / math.sqrt(sq), sq), np.float32) + np.array((x, y, FLOOR - float(P0[:, 2].min()) * sq - .25), np.float32)
            else:
                frm = P.mean(0)
                P = flying(frm, mouth_at(t) + np.array((0, 9.0, 0), np.float32), u, az, (18.0, 42.0, 40.0), 1 - .3 * float(sm((u - .65) / .35)))
        return P
    side, place, hop = QUEUE[key]
    if t < 109.7:
        return on_board(x, y, rest_turn)
    # it slides across the board to its place in the queue, leaning in; the queue moves up as the ones in front go
    ahead = [v[2] for v in QUEUE.values() if v[0] == side and v[1] < place]
    now = place - sum(float(sm((t - h) / .07)) for h in ahead)
    u = float(sm((t - 109.7) / .3))
    sx, sy = side * (15.0 + (35.0 if key == "code" else 27.0) * now), -82.0 + 3.0 * now
    px, py = x + (sx - x) * u, y + (sy - y) * float(sm((t - 109.7) / .2))
    speed = 4 * u * (1 - u)
    turn = rot(az * (1 - u) + 10.0 * math.sin(2 * math.pi * 7.0 * t), back * (1 - u) + 14.0 * u, tip * (1 - u) - side * (6.0 + 16.0 * speed))
    if t < hop:
        return on_board(px, py, turn)
    if key == STUCK:
        lips = mouth_at(t) + np.array((0, -7.0, 0), np.float32)
        u = (t - hop) / .07
        if u < 1:
            return flying(on_board(px, py, turn).mean(0), lips, u, 0.0, (0.0, 20.0, 0.0), 1.0)
        if t < hop + .16:      # it rocks there, not wanting to go
            rock = 24.0 * math.sin(2 * math.pi * 11.0 * (t - hop - .07))
            return (V @ rot(rock * .4, 20.0, rock).T) + lips
        u = (t - hop - .16) / .04
        if u >= 1:
            return None
        return (V @ rot(0.0, 20.0, 0.0).T) * (1 - .5 * u) + lips + np.array((0, 16.0 * u, 0), np.float32)
    u = (t - hop) / .09
    if u >= 1:
        return None
    return flying(on_board(px, py, turn).mean(0), mouth_at(t) + np.array((0, 9.0, 0), np.float32), u, 0.0, (10.0, 30.0, 30.0), 1 - .3 * float(sm((u - .65) / .35)))


def letters(mat, mouth_at, t, h, warm):
    """The letters that came off him as he got up, wherever they have got to."""
    z = np.load(os.path.join(OUT, "crab", "clay_glyphs.npz"))
    res = int(z["res"])
    for key, x, y, az, back, tip in LETTERS:
        if t < FALLS[key] or (key in EATEN and t >= ONE_BY_ONE[EATEN[key]][0] + HOP[1] * ONE_BY_ONE[EATEN[key]][1] / 1.27):
            continue
        V, Q = letter_shape(z[key], res, h)
        P = letter_at(key, V, x, y, az, back, tip, mouth_at, t)
        if P is None:
            continue
        put_mesh("letter_" + key, P, Q, dict(rest=V + (x * 3.1, y * 2.3, 40.0), eye=np.full(len(V), 99.0), mouth=np.full(len(V), 99.0), glint=np.full(len(V), 99.0), deep=np.zeros(len(V)), warm=np.full(len(V), warm)), mat)


def grains(mat, h, warm):
    """The grains of letters by his feet in chapter eight: smaller than the ones he ate, leaning where they fell."""
    z = np.load(os.path.join(OUT, "crab", "clay_glyphs.npz"))
    res = int(z["res"])
    for key, size, x, y, az, back, tip in GRAINS:
        V, Q = letter_shape(z[key], res, h / size)
        P = (V * size) @ rot(az, back, tip).T
        P = P + np.array((x, y, FLOOR - float(P[:, 2].min()) - .2), np.float32)
        put_mesh("grain_" + key, P, Q, dict(rest=V * size + (x * 3.1, y * 2.3, 40.0), eye=np.full(len(V), 99.0), mouth=np.full(len(V), 99.0), glint=np.full(len(V), 99.0), deep=np.zeros(len(V)), warm=np.full(len(V), warm)), mat)


def crumbs(mat, t, warm):
    """A few real bits at his feet, no more."""
    if t < 99.3:
        return
    import bmesh
    r = np.random.default_rng(92)
    tpl = bpy.data.meshes.new("t")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    bm.to_mesh(tpl)
    bm.free()
    base = np.array([v.co[:] for v in tpl.vertices], np.float32)
    tris = np.array([p.vertices[:] for p in tpl.polygons], np.int32)
    co, faces = [], []
    for i in range(14):
        # never right at a toe, where a bit of clay reads as the foot dripping
        lx = LEGS[int(r.integers(len(LEGS)))]
        x, y, s = lx + r.normal(0, 15), LEG_Y - LEG_D / 2 - 9.0 - abs(r.normal(0, 12)), float(r.choice((r.uniform(.35, .8), r.uniform(.8, 1.5)), p=(.7, .3)))
        p = base * s * r.uniform(.65, 1.2, 3)
        p = p * (1 + .3 * np.sin(base @ r.normal(0, 3.2, 3) + r.uniform(0, 6)))[:, None]
        co.append(p + (x, y, FLOOR + s * .3))
        faces.append(tris + i * len(base))
    co = np.vstack(co)
    me = bpy.data.meshes.new("crumbs")
    me.from_pydata(co.tolist(), [], np.vstack(faces).tolist())
    me.update()
    me.shade_smooth()
    me.materials.append(mat)
    for name, v in (("rest", co), ("eye", np.full(len(co), 99.0)), ("mouth", np.full(len(co), 99.0)), ("glint", np.full(len(co), 99.0)), ("deep", np.zeros(len(co))), ("warm", np.full(len(co), warm))):
        v = np.asarray(v, np.float32)
        x = me.attributes.new(name, "FLOAT_VECTOR" if v.ndim == 2 else "FLOAT", "POINT")
        x.data.foreach_set("vector" if v.ndim == 2 else "value", v.ravel())
    bpy.context.collection.objects.link(bpy.data.objects.new("crumbs", me))
    bpy.data.meshes.remove(tpl)


# ---- materials, floor, sky ---------------------------------------------------------------------------------------------------------------------------------
def clay():
    """The grey clay turning to matt terracotta plasticine where its warm says so, soft to look at: a little light goes into it. Eyes a smooth dark glaze, the mouth warm and dark."""
    N = Nodes("clawd")
    t, b = N.t, N.b
    rest = t.nodes.new("ShaderNodeAttribute")
    rest.attribute_name = "rest"
    N.obj = rest.outputs["Vector"]
    worked = N.range(N.noise(.028, 2.0, .5), .36, .64, 0, 1, True)
    fine = N.range(N.noise(.3, 4.0, .55), .36, .64)
    tone = N.math("ADD", N.math("MULTIPLY", worked, .62), N.math("MULTIPLY", fine, .38))
    col = N.mix(tone, (.6, .165, .075), (.49, .12, .052))
    speck = N.range(N.noise(7.0, 2.0, .5), .67, .72, 0, 1, True)
    col = N.mix(N.math("MULTIPLY", speck, .28), col, (.22, .06, .028))
    pale = N.range(N.noise(11.0, 2.0, .5), .69, .74, 0, 1, True)
    col = N.mix(N.math("MULTIPLY", pale, .25), col, (.74, .34, .19))
    grey = N.mix(tone, (.205, .25, .275), (.13, .165, .19))
    warm = N.attr("warm")
    earth = N.mix(N.range(warm, 0, .45, 0, 1, True), grey, (.2, .15, .115))
    col = N.mix(N.range(warm, .3, 1, 0, 1, True), earth, col)
    deep = N.attr("deep")
    col = N.mix(N.math("MULTIPLY", deep, .6), col, (.2, .05, .022))
    eye = N.range(N.attr("eye"), -.14, .14, 1, 0, True)
    mouth = N.range(N.attr("mouth"), -.22, .22, 1, 0, True)
    col = N.mix(mouth, col, (.042, .012, .008))
    col = N.mix(eye, col, (.014, .0095, .0085))
    t.links.new(col, b.inputs["Base Color"])
    rough = N.range(fine, 0, 1, .56, .48)
    rough = N.mix(mouth, rough, (.95, .95, .95))
    rough = N.mix(eye, rough, (.14, .14, .14))
    t.links.new(rough, b.inputs["Roughness"])
    plain = N.math("MULTIPLY", N.math("SUBTRACT", 1.0, eye), N.math("SUBTRACT", 1.0, mouth))
    t.links.new(N.math("ADD", N.math("MULTIPLY", plain, .36), N.math("MULTIPLY", eye, .5)), b.inputs["Specular IOR Level"])
    # the grey block is dense and dry: the soft look comes as it warms
    soft = N.math("MULTIPLY", plain, N.range(warm, .05, .5, 0, 1, True))
    t.links.new(N.math("MULTIPLY", soft, .16), b.inputs["Sheen Weight"])
    b.inputs["Sheen Roughness"].default_value = .5
    b.inputs["Sheen Tint"].default_value = (1.0, .8, .68, 1)
    t.links.new(N.math("MULTIPLY", soft, .22), b.inputs["Subsurface Weight"])
    b.inputs["Subsurface Radius"].default_value = (3.2, 1.1, .55)
    b.inputs["Subsurface Scale"].default_value = 1.0
    g = N.math("ADD", N.noise(5.0, 5.0, .6), N.math("MULTIPLY", N.noise(1.4, 3.0, .5), .6))
    bump = t.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = .012
    t.links.new(N.math("SUBTRACT", 1.0, eye), bump.inputs["Strength"])          # the glaze is smooth: one clean glint, nothing else
    t.links.new(g, bump.inputs["Height"])
    grain = N.range(N.attr("glint"), -.45, .45, 1, 0, True)
    b.inputs["Emission Color"].default_value = (1.0, .95, .88, 1)
    t.links.new(N.math("MULTIPLY", grain, 1.9), b.inputs["Emission Strength"])
    t.links.new(bump.outputs[0], b.inputs["Normal"])
    return N.m


class Tree(Nodes):
    """Nodes' helpers on any node tree."""

    def __init__(self, tree):
        self.t = tree
        self.obj = None


# what stands behind him at the end, linear: above, low down, the halo of the sun on the horizon; how far off its middle is below level (sine), how wide
SKY_HIGH, SKY_LOW, SKY_GLOW, SKY_V0 = (.13, .098, .072), (.17, .128, .09), (.05, .029, .014), -.062


def sky(N, vec, level):
    """Colour by direction of view: warm grey going up from the horizon, the sun's halo behind him. level: how far the morning has come."""
    t = N.t
    sep = t.nodes.new("ShaderNodeSeparateXYZ")
    t.links.new(vec, sep.inputs[0])
    x, z = sep.outputs[0], sep.outputs[2]
    up = N.range(z, SKY_V0 - .04, SKY_V0 + .1, 0, 1, True)
    col = N.mix(up, SKY_LOW, SKY_HIGH)
    rx, rz = N.math("DIVIDE", x, .26), N.math("DIVIDE", N.math("SUBTRACT", z, SKY_V0), .05)
    r2 = N.math("ADD", N.math("MULTIPLY", rx, rx), N.math("MULTIPLY", rz, rz))
    glow = N.math("POWER", 2.718, N.math("MULTIPLY", r2, -1.0))
    add = t.nodes.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    t.links.new(glow, add.inputs[0])
    t.links.new(col, add.inputs[6])
    add.inputs[7].default_value = (*SKY_GLOW, 1)
    dim = t.nodes.new("ShaderNodeMix")
    dim.data_type = "RGBA"
    dim.blend_type = "MULTIPLY"
    dim.inputs[0].default_value = 1.0
    t.links.new(add.outputs[2], dim.inputs[6])
    dim.inputs[7].default_value = (level, level, level, 1)
    return dim.outputs[2]


def morning(t):
    """How far the background has lit up: dark warm grey as he is born, bright by the wave."""
    return .34 + .66 * float(sm((t - 97.1) / (105.0 - 97.1)))


HAZE = (60.0, 470.0)      # gone before the board ends: no far edge, no horizon line
WOOD_GAIN = 2.6


def floor(t, step=1):
    """The old board with no clay on it, going off into the morning haze, so there is no edge to the world."""
    x0, x1, y0, y1 = BOARD_RECT
    col, hr = np.load(os.path.join(OUT, "board_clean_col.npy"))[::step, ::step], np.load(os.path.join(OUT, "board_clean_h.npy"))[::step, ::step]
    ny, nx = col.shape[:2]
    lv = morning(t)

    def haze(N, surface):
        tt = N.t
        geo = tt.nodes.new("ShaderNodeNewGeometry")
        sep = tt.nodes.new("ShaderNodeSeparateXYZ")
        tt.links.new(geo.outputs["Position"], sep.inputs[0])
        flip = tt.nodes.new("ShaderNodeVectorMath")
        flip.operation = "SCALE"
        flip.inputs["Scale"].default_value = -1.0
        tt.links.new(geo.outputs["Incoming"], flip.inputs[0])
        em = tt.nodes.new("ShaderNodeEmission")
        tt.links.new(sky(N, flip.outputs[0], lv), em.inputs[0])
        mix = tt.nodes.new("ShaderNodeMixShader")
        tt.links.new(N.range(sep.outputs[1], HAZE[0], HAZE[1], 0, 1, True), mix.inputs[0])
        tt.links.new(surface, mix.inputs[1])
        tt.links.new(em.outputs[0], mix.inputs[2])
        tt.links.new(mix.outputs[0], tt.nodes["Material Output"].inputs[0])

    N = Nodes("board")
    tn, b = N.t, N.b
    uv = tn.nodes.new("ShaderNodeTexCoord").outputs["UV"]

    def image(name, rgb):
        im = bpy.data.images.new(name, nx, ny, alpha=False, float_buffer=True)
        im.colorspace_settings.name = "Non-Color"
        a = np.ones((ny, nx, 4), np.float32)
        a[..., :3] = rgb
        im.pixels.foreach_set(a[::-1].ravel())
        im.update()
        node = tn.nodes.new("ShaderNodeTexImage")
        node.image = im
        tn.links.new(uv, node.inputs[0])
        return node
    i1 = image("board_col", col * WOOD_GAIN)
    i2 = image("board_h", np.dstack([(hr[..., 0] + 2) / 3, hr[..., 1], np.zeros((ny, nx), np.float32)]))
    tn.links.new(i1.outputs[0], b.inputs["Base Color"])
    sep = tn.nodes.new("ShaderNodeSeparateColor")
    tn.links.new(i2.outputs[0], sep.inputs[0])
    tn.links.new(sep.outputs[1], b.inputs["Roughness"])
    tn.links.new(N.bump(sep.outputs[0], 2.1, .7), b.inputs["Normal"])
    b.inputs["Specular IOR Level"].default_value = .22
    haze(N, b.outputs[0])
    me = bpy.data.meshes.new("board")
    y_near = 2 * y0 - y1        # and once more towards the camera, the other way up: when the view has eased back there is more of it to see
    me.from_pydata([(x0, y0, FLOOR), (x1, y0, FLOOR), (x1, y1, FLOOR), (x0, y1, FLOOR), (x0, y_near, FLOOR), (x1, y_near, FLOOR)], [], [(0, 1, 2, 3), (4, 5, 1, 0)])
    u = me.uv_layers.new(name="UV")
    for i, v in enumerate(((0, 0), (1, 0), (1, 1), (0, 1), (0, 1), (1, 1), (1, 0), (0, 0))):
        u.data[i].uv = v
    me.update()
    me.materials.append(N.m)
    bpy.context.collection.objects.link(bpy.data.objects.new("board", me))
    mean = tuple(float(c) for c in (col * WOOD_GAIN).reshape(-1, 3).mean(0))
    M = Nodes("beyond")
    M.b.inputs["Base Color"].default_value = (*mean, 1)
    M.b.inputs["Roughness"].default_value = .7
    M.b.inputs["Specular IOR Level"].default_value = .1
    haze(M, M.b.outputs[0])
    far = bpy.data.meshes.new("beyond")
    far.from_pydata([(-20000, y1 - .5, FLOOR - .1), (20000, y1 - .5, FLOOR - .1), (20000, 30000, FLOOR - .1), (-20000, 30000, FLOOR - .1)], [], [(0, 1, 2, 3)])
    far.materials.append(M.m)
    far.update()
    bpy.context.collection.objects.link(bpy.data.objects.new("beyond", far))
    edge = bpy.data.meshes.new("near")
    edge.from_pydata([(-20000, -9000, FLOOR - .1), (20000, -9000, FLOOR - .1), (20000, y_near + .5, FLOOR - .1), (-20000, y_near + .5, FLOOR - .1)], [], [(0, 1, 2, 3)])
    edge.materials.append(M.m)
    edge.update()
    bpy.context.collection.objects.link(bpy.data.objects.new("near", edge))


STOP_UP = 2.3


def light(sc, t):
    """The block is seen in the low cold light it has had since the last die went. As it gets up the sun comes round and climbs behind him, and draws a warm line round him; what models his face is the bright sky on our left and a big soft fill, so nothing on it goes black."""
    lv = morning(t)
    up = float(sm((t - (MORPH + .3)) / (100.4 - MORPH - .3)))
    tgt = np.array((0.0, 0.0, ZM))

    def sun(name, direction, strength, angle, colour):
        d = bpy.data.lights.new(name, "SUN")
        d.energy, d.angle, d.color = strength, math.radians(angle), colour
        o = bpy.data.objects.new(name, d)
        bpy.context.collection.objects.link(o)
        direction = np.array(direction, np.float64)
        direction = direction / np.linalg.norm(direction)
        aim(o, (tgt + direction * 2500).tolist(), tgt.tolist())
        return o

    def from_dawn(el):
        a, e = math.radians(DAWN_AZ), math.radians(el)
        return (-math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))

    gone = float(sm((t - (MORPH + .2)) / 1.6))
    sun("key", (-.62, -.72, .55), 2.9 * max(up, .8 * gone), 26.0, (1.0, .9, .78))
    sun("behind", (.22, .95, .2), (1.6 + 1.6 * lv) * up, 18.0, (1.0, .66, .36))
    if gone < 1:    # what the block has had since the last die went, k6's own light, one stop up: the cut comes from a white page. It goes before the block has turned to it
        sun("low", from_dawn(DAWN_EL), DAWN_E * STOP_UP * (1 - gone), 1.6, DAWN_COLOUR)
        sun("gold", from_dawn(DAWN_EL - 14.0), DAWN_E * .65 * (1 - gone), .8, GOLD_COLOUR)
    tree = sc.world.node_tree
    N = Tree(tree)
    fill = tree.nodes["Background"]
    cold = tuple(c * 3.4 for c in DAWN_SKY)
    warm_fill = tuple(c * (.26 + .36 * up) for c in (.62, .5, .41))
    fill.inputs[0].default_value = (*(c0 * (1 - gone) + c1 * gone for c0, c1 in zip(cold, warm_fill)), 1)
    fill.inputs[1].default_value = 1.0
    tc = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
    bg = tree.nodes.new("ShaderNodeBackground")
    tree.links.new(sky(N, tc, lv), bg.inputs[0])
    bg.inputs[1].default_value = 1.0
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(tree.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], mix.inputs[0])
    tree.links.new(fill.outputs[0], mix.inputs[1])
    tree.links.new(bg.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], tree.nodes["World Output"].inputs[0])
    # the bench throws a little back up under his chin
    card = bpy.data.lights.new("under", "AREA")
    card.shape = "RECTANGLE"
    card.size, card.size_y = 700.0, 300.0
    card.energy = 2.4e5 * up
    card.color = (1.0, .82, .68)
    u = bpy.data.objects.new("under", card)
    u.visible_glossy = False
    bpy.context.collection.objects.link(u)
    aim(u, (0.0, -700.0, FLOOR + 30.0), (0.0, -20.0, ZB + 20.0))


def put_mesh(name, P, Q, attrs, mat):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", np.asarray(P, np.float32).ravel())
    me.loops.add(Q.size)
    me.loops.foreach_set("vertex_index", Q.ravel().astype(np.int32))
    me.polygons.add(len(Q))
    me.polygons.foreach_set("loop_start", np.arange(len(Q), dtype=np.int32) * 4)
    me.update(calc_edges=True)
    me.shade_smooth()
    for key, a in attrs.items():
        a = np.asarray(a, np.float32)
        x = me.attributes.new(key, "FLOAT_VECTOR" if a.ndim == 2 else "FLOAT", "POINT")
        x.data.foreach_set("vector" if a.ndim == 2 else "value", a.ravel())
    me.materials.append(mat)
    bpy.context.collection.objects.link(bpy.data.objects.new(name, me))


ABOVE_Z = -2.0                      # the level of the clay round the character: the plane the frame's 225 mm are measured on
K5 = (40.0, 794.0, np.array((0.0, -9.0, -9.0)))      # the block as it stands: how high, how far, what at
PULL = (96.3, 97.3)
BACK_FROM = 106.9
END_DIST, END_SHIFT = 1348.0, (.0437, -.0731)             # 112.4


FRAMED = (END_DIST, END_SHIFT, BACK_FROM)
EIGHT_VIEW = (920.0, (0.0, -.03), 0.0)
CAM = None                  # a view of one's own: degrees up, how far, what at (x, y, z)


def view(t):
    """How high the camera is (degrees), how far off, what it looks at; how far it has eased back at the end."""
    if CAM is not None:
        return CAM[0], CAM[1], np.array(CAM[2:5], np.float64), 0.0
    if t < MORPH:
        u = float(np.clip((t - PULL[0]) / (PULL[1] - PULL[0]), 0, 1))
        u = .55 * u + .45 * float(sm(u))
        d0 = WO_WIDE * 85.0 / 36.0
        c0 = np.array((WO_AT[0], WO_AT[1], ABOVE_Z))
        return 90.0 + (K5[0] - 90.0) * u, d0 * (K5[1] / d0) ** u, c0 + (K5[2] - c0) * u, 0.0
    to = then(t)
    e = float(curve((94.9, K5[0]), (97.2, ELEV))(to))
    dist = float(curve((94.9, K5[1]), (97.2, DIST))(to))
    c = np.array((0.0, float(curve((94.9, K5[2][1]), (97.2, 0))(to)), float(curve((94.9, K5[2][2]), (97.2, TARGET_Z))(to))))
    back = float(sm((t - FRAMED[2]) / (112.4 - FRAMED[2])))
    return e, dist * (FRAMED[0] / DIST) ** back, c, back


def camera(sc, t, fstop):
    """Straight down at the character; back and round to the block as it stands; down off the block as it gets up, until it is just above his eye line; then it stays, until he eats, when it eases back."""
    from mathutils import Matrix
    deg, dist, c, back = view(t)
    e = math.radians(deg)
    p = c + dist * np.array((0.0, -math.cos(e), math.sin(e)))
    data = bpy.data.cameras.new("cam")
    data.lens = 85.0
    data.sensor_width = 36.0
    data.clip_start, data.clip_end = 5.0, 60000.0
    data.dof.use_dof = True
    data.dof.focus_distance = dist - BODY_D / 2 * float(sm((then(t) - 96.0) / 1.0))
    data.dof.aperture_fstop = fstop / 1000.0
    data.dof.aperture_blades = 7
    data.shift_x, data.shift_y = FRAMED[1][0] * back, FRAMED[1][1] * back
    o = bpy.data.objects.new("cam", data)
    bpy.context.collection.objects.link(o)
    o.matrix_world = Matrix(((1, 0, 0, p[0]), (0, math.sin(e), -math.cos(e), p[1]), (0, math.cos(e), math.sin(e), p[2]), (0, 0, 0, 1)))
    sc.camera = o
    return o


def seen_from_above(t, w):
    """The part of the block's top in the frame at time t, and how many mm a frame pixel is at its nearest."""
    deg, dist, c, _ = view(t)
    e = math.radians(deg)
    p = c + dist * np.array((0.0, -math.cos(e), math.sin(e)))
    right, up_, ahead = np.array((1.0, 0, 0)), np.array((0, math.sin(e), math.cos(e))), np.array((0, math.cos(e), -math.sin(e)))
    half = 18.0 / 85.0
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            ray = ahead + right * sx * half + up_ * sy * half * 9 / 16
            k = (ABOVE_Z - p[2]) / ray[2] if ray[2] < -1e-3 else 1e5
            pts.append(p + ray * k)
    pts = np.array(pts)
    x0, x1, y0, y1 = max(pts[:, 0].min() - 6, -A), min(pts[:, 0].max() + 6, A), max(pts[:, 1].min() - 6, -B), min(pts[:, 1].max() + 6, B)
    near = min(np.linalg.norm(q - p) for q in pts)
    return (float(x0), float(x1), float(y0), float(y1)), float(near * 36.0 / 85.0 / w)


def clay_from_above(mat, t, a):
    """The block itself, at its full height field, for as long as it is only a block."""
    z = block_field("wo")
    # the mesh goes by the pixel, so at 1920 the block came out twice as fine as the 960 drafts: pits and hair cracks the drafts never had,
    # gone again at SWITCH, where his skin takes over at its own spacing (2026-11-07). step_w 960 keeps the drafts' mesh at any size
    win, px = seen_from_above(t, a.get("step_w") or a["w"])
    step = max(px, .07)
    while (win[1] - win[0]) * (win[3] - win[2]) / step ** 2 > 3.2e6:
        step *= 1.1
    o = block_of(z["H"], int(z["res"]), float(z["lost"]), z["burnish"], z["bulge"], z["fresh"], win, mat, near=(1.0, step, step), NT=NT_ABOVE)
    me = o.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    for key, v in (("rest", co), ("eye", np.full(n, 99.0, np.float32)), ("mouth", np.full(n, 99.0, np.float32)), ("glint", np.full(n, 99.0, np.float32)), ("deep", np.zeros(n, np.float32)), ("warm", np.zeros(n, np.float32))):
        x = me.attributes.new(key, "FLOAT_VECTOR" if key == "rest" else "FLOAT", "POINT")
        x.data.foreach_set("vector" if key == "rest" else "value", v)


def body_rect(cam, sc, P, k):
    """Where his body is in the frame: left, right, top, bottom, in pixels of a 1920 frame. Claws and legs left out."""
    R = k.R
    Q = P[(np.abs(R[:, 0]) <= BODY_W / 2 - 1) & (R[:, 2] >= ZB + 1)]
    M = np.array(cam.matrix_world.inverted())
    v = (M[:3, :3] @ Q.T).T + M[:3, 3]
    W_ = 1920.0
    f = 85.0 / 36.0 * W_
    u, w = W_ / 2 + f * v[:, 0] / -v[:, 2] - cam.data.shift_x * W_, 540.0 - f * v[:, 1] / -v[:, 2] + cam.data.shift_y * W_
    return float(u.min()), float(u.max()), float(w.min()), float(w.max())


def png8(path, a):
    import struct
    import zlib
    h, w = a.shape[:2]
    raw = b"".join(b"\0" + a[i].tobytes() for i in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def finish(src, path, seed, grain):
    """Down to 8 bits with film grain and a triangular dither put in first, from the render's 16 bits: a smooth sky has nothing to band on."""
    im = bpy.data.images.load(src, check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    a = a.reshape(h, w, 4)[::-1, :, :3]
    r = np.random.default_rng(seed)
    a = a * 255 + r.normal(0, grain, (h, w, 1)).astype(np.float32) + r.random((h, w, 3), np.float32) - r.random((h, w, 3), np.float32)
    png8(path, np.clip(a + .5, 0, 255).astype(np.uint8))
    bpy.data.images.remove(im)


def still(a):
    t = a["t"]
    sc = reset(a["samples"], 100)
    set_up(a, sc)
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 8, 4
    mat = clay()
    if t < SWITCH and not a.get("skin"):
        clay_from_above(mat, t, a)
    else:
        k = Clawd(a["mesh"], a.get("prints", not a["draft"]))
        disp, eye_d, mouth_d, glint_d = face(k, t)
        P = k.R + k.N * (k.static + disp)[:, None]
        rest = k.R
        if t < 100.6:
            P = gather(k, P, t)
            laid = k.block.copy()          # where it lay, with the heights it had: what the clay's grain is fixed to, as it is on the block itself
            laid[:, 2] += k.b_top * (k.b_old + k.b_wo)
            rest = laid + (k.R - laid) * float(sm((t - 98.0) / 2.4))
        P = pose(P, t)
        warm = warmth(k, t)
        put_mesh("clawd", P, k.quads, dict(rest=rest, eye=eye_d, mouth=mouth_d, glint=glint_d, deep=k.deep * float(sm((t - 99.4) / 1.0)), warm=warm), mat)

        def mouth_at(when):
            return pose(np.array(((0.0, -BODY_D / 2 - 2.0, ZT - MOUTH_FZ),), np.float32), when)[0]
        w_all = float(np.median(warm))
        if t < 112.4:
            letters(mat, mouth_at, t, .36 if a["draft"] else .2, w_all)
        if GRAINS:
            grains(mat, .36 if a["draft"] else .2, w_all)
        else:
            crumbs(mat, t, w_all)
    floor(t, 3 if a["draft"] else 1)
    light(sc, t)
    cam = camera(sc, t, a["fstop"])
    if t >= 100.6:
        print("body in frame: %.1f %.1f %.1f %.1f" % body_rect(cam, sc, P, k), flush=True)
    path = os.path.join(OUT, "crab", a["name"] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = "PNG", "16"
    deep16 = f"/tmp/clawd16_{os.getpid()}.png"
    sc.render.filepath = deep16
    bpy.ops.render.render(write_still=True)
    finish(deep16, path, int(t * 1000), 1.6 if a["draft"] else 1.3)
    print("wrote", path, flush=True)


def frame_times(t0, t1):
    """On twos, as stop motion goes, but on ones for the breath, which is too quick for that, and for the camera going back. Each film frame at 24 fps and the moment it shows."""
    out = []
    n = int(round((t1 - t0) * 24))
    for i in range(n):
        t = t0 + i / 24
        out.append((i, t if 109.7 <= t <= 110.5 or PULL[0] <= t <= MORPH else PULL[0] - 1 / 24 if t < PULL[0] else t0 + (i // 2) * 2 / 24))
    return out


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    if "--eight" in argv:
        argv.remove("--eight")
        eight()
    if "--eight-still" in argv:
        argv.remove("--eight-still")
        eight(still=True)
    if "--cam" in argv:       # degrees up, how far, what at: x y z
        i = argv.index("--cam")
        CAM = tuple(float(v) for v in argv[i + 1:i + 6])
        del argv[i:i + 6]

    def opt(name, default, kind=float):
        return kind(argv[argv.index(name) + 1]) if name in argv else default

    if argv[0] == "still":
        draft = "--draft" in argv
        still(dict(name=argv[1], t=float(argv[2]), draft=draft, skin="--skin" in argv, samples=opt("--samples", 48 if draft else 256, int), w=opt("--w", 960 if draft else 1920, int), h=opt("--h", 540 if draft else 1080, int), fstop=opt("--fstop", 22.0), exp=opt("--exp", .5), denoise=True, mesh=opt("--mesh", .5 if draft else .25)))
    elif argv[0] == "rect":       # where his body is at this moment, without rendering
        sc = reset(1, 100)
        t = float(argv[1])
        k = Clawd(.5, False)
        disp, *_ = face(k, t)
        P = pose(k.R + k.N * (k.static + disp)[:, None], t)
        print("body in frame: %.1f %.1f %.1f %.1f" % body_rect(camera(sc, t, 22.0), sc, P, k), flush=True)
    elif argv[0] == "frames":     # one worker's share; a frame already there is left alone
        name, t0, t1, me, of = argv[1], float(argv[2]), float(argv[3]), int(argv[4]), int(argv[5])
        # --hi: the same frames at 1920 by 1080, for the final. What is in them stays as the 960 drafts have it: no finger prints (the drafts
        # never had them) and his skin at the drafts' spacing (.25 brought out the characters pressed into him, which .5 keeps faint; --mesh
        # to choose); only what goes with the size is finer: the board, the letters and grains at his feet, the grain. --step-w 960: the block,
        # while it is only the block, meshed as at 960 (the 1080 pass of the bridge needs it: see clay_from_above)
        hi = "--hi" in argv
        shown = sorted({t for _, t in frame_times(t0, t1)})
        for i, t in enumerate(shown):
            if i % of == me and not os.path.exists(os.path.join(OUT, "crab", name, f"t{t:08.4f}.png")):
                # his skin finer while it is still the block, so that the block's fine cracks do not go soft where it turns into him
                mesh = opt("--fine-mesh", FINE_MESH[0]) if t < opt("--fine-until", FINE_MESH[1]) else opt("--mesh", .5)
                still(dict(name=f"{name}/t{t:08.4f}", t=t, draft=not hi, prints=False, samples=opt("--samples", 128 if hi else 64, int), w=1920 if hi else 960, h=1080 if hi else 540, fstop=22.0, exp=.5, denoise=True, mesh=mesh, step_w=opt("--step-w", 0, int)))

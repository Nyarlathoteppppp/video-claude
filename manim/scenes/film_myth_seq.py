"""鬼夜哭 moving, from 而天雨粟 on, and with the opening from film_myth_world.py the whole low sample: 960x540, 30 fps. Each shot renders in its own process at film size, and burn then puts the subtitles on and encodes. Moments are read from ink_subs.at.

seq_raw4/ holds what this version made, by the new film's frame numbers. seq_raw/ still holds the older frames this one uses unchanged: from the pull back on they are only shifted (see source).

    python film_myth_seq.py k2                 # 而天雨粟, five seconds: the grain falls, the camera goes up from the bank to the bare hills, the ink seeps in round the figures. 山 lies on the bank
    python film_myth_seq.py near               # the near ghost: it raises its head, opens its mouth round 鬼夜哭, and holds the howl until the camera goes back
    python film_myth_seq.py pull               # away to the ridge, and the others answer, old clock, into seq_raw/
    python film_myth_seq.py burn [mp4]         # the subtitles, the small frames, the film, a sheet of frames
    python film_myth_seq.py k2 --only 660,700  # only these frames, by the new film's numbers, are written; whatever has to run before them still runs

The ghost's vision was taken out of the film on 2026-11-01; shot_vision stays here unused, its frames in seq_raw/.
"""
import json
import os
import sys
import time

import numpy as np

import ink_grade

_fft0, _fields = ink_grade.fft_noise, {}


def _fft(h, w, beta, seed):
    """The same few fields of noise are asked for on every frame and take a good part of a second each."""
    key = (h, w, float(beta), seed)
    if key not in _fields:
        a = _fft0(h, w, beta, seed)
        a.flags.writeable = False
        _fields[key] = a
    return _fields[key]


ink_grade.fft_noise = _fft      # before anything else imports it

from types import SimpleNamespace

from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from film_myth import CAM_K3, STICK_DOWN, lift_figure, lumps, mend_plate, mouth_layers, mud_wash, near_ghost, qing_wash, relief_film, shroud_plate, tidy, tidy_keep, vision_fields, vision_hair
from film_myth_ghost import Ghost
from film_myth_mud import Mud, ink_of, twig_shadow
from film_myth_wash import OUT, Sheet, Wash, band, develop, grow, noise, sm
from film_myth_wide import K2_MIST, grain, grain_plan, k1_parts, k2_mist, k2_parts, k4_parts, painted_twig
from film_myth_write import shan_on_bank
import ink_subs

W, H = 1920, 1080
FPS = 30
MYTH = os.path.dirname(OUT)
RAW = os.path.join(MYTH, "seq_raw")
RAW4 = os.path.join(MYTH, "seq_raw4")
SEQ = os.path.join(MYTH, "seq")
ONLY = None
_times = []


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def wanted(n):
    return ONLY is None or n in ONLY


def put(n, f, t0=None, to=RAW):
    os.makedirs(to, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(os.path.join(to, f"f{n:05d}.png"), compress_level=1)
    if t0 is not None:
        _times.append((n, time.time() - t0))
        if len(_times) % 15 == 1:
            print(f"  frame {n} (t {n / FPS:.2f}) {_times[-1][1]:.2f}s", flush=True)


def report(name, setup):
    a = np.array([v for _, v in _times]) if _times else np.zeros(1)
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, f"_time_{name}.json"), "w") as out:
        json.dump(dict(setup=setup, frames=len(_times), mean=float(a.mean()), max=float(a.max()), each=[(int(n), round(float(v), 3)) for n, v in _times]), out)
    print(f"{name}: setup {setup:.0f}s, {len(_times)} frames, {a.mean():.2f}s mean, {a.max():.2f}s worst", flush=True)


def warp_film(sheet, a):
    """A field off a coarser grid onto the film, its edges pulled about along the sheet's own fibres, as Wash.density does."""
    yy, xx = _grid()
    dx, dy = sheet.warp
    return map_coordinates(grow(a, H, W), [yy + dy, xx + dx], order=1, mode="nearest")


_g = None


def _grid():
    global _g
    if _g is None:
        _g = np.mgrid[0:H, 0:W].astype(np.float32)
    return _g


def affine(a, tx):
    return np.asarray(Image.fromarray(np.float32(a), "F").transform((a.shape[1], a.shape[0]), Image.AFFINE, tx, Image.BICUBIC), np.float32)


# ---------------------------------------------------------------- 12-17: grain, bank to hills, the ink seeping in

T_SEEP0, T_SEEP1, T_RIPE = 14.0, 15.95, 16.98      # sources start from the first, the last has started by the second, all the ink is what it will stay by the third
S_MAX, S_BLEND = 760, 320                           # how far the camera goes up: the hills lie this far above the bank, overlapping it by this much of bare paper
T_TILT0, T_TILT1, RAMP = 13.5, 14.7, 0.25      # a second and a half on the bank, so 山 can be read on the ground, then quicker than the rest of the shot


def tilt(t):
    """Even speed, a quarter second to take it up and to give it up."""
    u = t - T_TILT0
    T = T_TILT1 - T_TILT0
    if u <= 0:
        return 0.0
    if u >= T:
        return float(S_MAX)
    v = 1.0 / (T - RAMP)
    if u < RAMP:
        s = 0.5 * v * u * u / RAMP
    elif u > T - RAMP:
        s = 1 - 0.5 * v * (T - u) ** 2 / RAMP
    else:
        s = v * (u - RAMP / 2)
    return S_MAX * s


SEEP_STEP = 4      # film pixels to a node of the grid the ink's way is worked out on


class Arrival:
    """One layer of the hills through time. The ink starts in a few places under the ridge, not along it, and finds its own way from there: faster down the hill than up it, faster where the paper happens to drink, round whatever is reserved. So what it has covered at any moment is a few lobes with a broken edge, which nothing about the ridge's line sets parallel. Behind that edge it is thin and running, and it settles over the next second into exactly what the plate holds."""

    def __init__(self, sheet, fine, coarse, wall, R, xs, t0, seed):
        h, w = H // SEEP_STEP, W // SEEP_STEP
        low = lambda a: np.asarray(Image.fromarray(np.float32(a), "F").resize((w, h), Image.BOX), np.float32)
        drink = np.exp(1.3 * low(noise(sheet, 3.2, seed)) + 0.9 * low(noise(sheet, 2.6, seed + 1)) + 0.5 * low(noise(sheet, 2.0, seed + 2)))
        drink = drink * (1 - 0.97 * np.clip(low(wall), 0, 1))
        # the graph: each node to its eight neighbours; the cost of a step is its length over how well the paper drinks there
        src, dst, cost = [], [], []
        ids = np.arange(h * w).reshape(h, w)
        for dy, dx, way in ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 0.62), (-1, 0, 1.5), (1, 1, 0.78), (1, -1, 0.78), (-1, 1, 1.25), (-1, -1, 1.25)):
            ya, yb = (0, h - dy) if dy >= 0 else (-dy, h)
            xa, xb = (0, w - dx) if dx >= 0 else (-dx, w)
            here, there = (slice(ya, yb), slice(xa, xb)), (slice(ya + dy, yb + dy), slice(xa + dx, xb + dx))
            src.append(ids[here].ravel())
            dst.append(ids[there].ravel())
            cost.append((np.hypot(dx, dy) * way * 2 / (drink[here] + drink[there])).ravel())
        r = np.random.default_rng(seed + 5)
        late = r.uniform(0, 1, len(xs))
        src.append(np.full(len(xs), h * w))
        dst.append(np.array([int((R[x] + 34) // SEEP_STEP) * w + x // SEEP_STEP for x in xs]))
        cost.append(1e-3 + 26.0 * late)
        g = coo_matrix((np.concatenate(cost), (np.concatenate(src), np.concatenate(dst))), shape=(h * w + 1, h * w + 1)).tocsr()
        dist = np.float32(dijkstra(g, indices=h * w)[:-1].reshape(h, w))
        has = low(gaussian_filter(fine + coarse, 4.0)) > 0.03
        far = float(np.percentile(dist[has], 99.8))
        order = np.sort(dist[has])
        rank = np.searchsorted(order, dist) / len(order)      # half by how far, half by how much is left: it should not have nearly filled in the first second and then wait
        A = t0 + (T_SEEP1 - t0) * (0.45 * np.clip(dist / far, 0, 1) ** 1.1 + 0.55 * rank)
        self.A = np.minimum(warp_film(sheet, A) + 0.10 * noise(sheet, 3.2, seed + 3) + 0.03 * noise(sheet, 2.7, seed + 4), T_SEEP1 + 0.06)
        self.edge = 0.04 + 0.14 * sm((noise(sheet, 3.2, seed + 6) + 0.1) / 0.9)      # here it stops on a line, there it goes off soft
        self.fine, self.coarse = fine, coarse
        there_ = sm((gaussian_filter(fine + coarse, 2.0) - 0.015) / 0.05)
        self.fine_b, self.coarse_b = gaussian_filter(fine, 11.0) * there_, gaussian_filter(coarse, 11.0) * there_
        self.there = there_

    def at(self, t):
        u = t - self.A
        come = sm(u / self.edge)
        settle = ease(u / 0.95)
        thin = come * (1 - settle)
        tide = 0.30 * self.there * np.exp(-((u - 0.06) / 0.07) ** 2) * (u > -0.1)      # the wet edge, darker for the moment it is passing
        return thin * (0.50 * self.fine_b + 0.10 * self.fine) + settle * self.fine + tide, thin * (0.50 * self.coarse_b + 0.10 * self.coarse) + settle * self.coarse


def landers():
    """The grain that comes down on the bank near the man: where it comes to rest, how big from there, when it lands, how it bounces."""
    r = np.random.default_rng(77)
    out = []
    while len(out) < 44:
        x, y = r.uniform(40, 1880), r.uniform(690, 1050)
        if 960 < x < 1520 and y < 960:      # not on him
            continue
        L = 9 + 30 * (y - 650) / 400
        h1 = L * r.uniform(0.9, 1.7)
        g = 2600 * L / 25
        T1 = 2 * np.sqrt(2 * h1 / g)
        out.append(dict(x=x, y=y, L=L, tl=r.uniform(12.05, 13.75), h1=h1, g=g, T1=T1, v=640 * L / 25, sx=r.uniform(-1.1, 1.1) * L, spin=r.choice((-1, 1)) * r.uniform(7, 14), ang=r.uniform(0, np.pi), seed=int(r.integers(1 << 30))))
    return sorted(out, key=lambda a: a["L"])


def lander_at(a, t):
    """x, y, height above where it rests, angle: None before it comes into the picture."""
    u = t - a["tl"]
    if u < 0:
        fall = -u
        y = a["y"] - a["v"] * fall
        if y < -a["L"] * 2 - S_MAX:
            return None
        return a["x"] - 0.15 * a["sx"] * fall, y, a["y"] - y, a["ang"] + a["spin"] * u
    T1, h1 = a["T1"], a["h1"]
    T2, h2 = T1 * 0.56, h1 * 0.31
    if u < T1:
        v = u / T1
        return a["x"] + a["sx"] * 0.75 * v, a["y"] - 4 * h1 * v * (1 - v), 4 * h1 * v * (1 - v), a["ang"] + a["spin"] * u
    if u < T1 + T2:
        v = (u - T1) / T2
        return a["x"] + a["sx"] * (0.75 + 0.25 * v), a["y"] - 4 * h2 * v * (1 - v), 4 * h2 * v * (1 - v), a["ang"] + a["spin"] * u
    return a["x"] + a["sx"], a["y"], 0.0, a["ang"] + a["spin"] * (T1 + T2)


def contact(f, x, y, rx, ry, much):
    x0, x1, y0, y1 = int(max(x - 2 * rx, 0)), int(min(x + 2 * rx + 1, W)), int(max(y - 2 * ry, 0)), int(min(y + 2 * ry + 1, H))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    f[y0:y1, x0:x1] *= 1 - much * np.exp(-((xx - x) / rx) ** 2 - ((yy - y) / ry) ** 2)[..., None]


def rain(heads):
    r = np.random.default_rng(78)
    return [dict(x=x, y=y, L=L, ang=ang, seed=seed, v=190 + 6.5 * L, wind=-(8 + 0.4 * L), spin=r.choice((-1, 1)) * (1.2 + 60 / L)) for x, y, L, ang, seed in grain_plan(heads)]


def draw_grain(f, sheet, rs, lands, t, s):
    for a in rs:
        m = a["L"] + 24
        y = a["y"] + a["v"] * (t - 15.5) + (s - S_MAX) * (0.35 + a["L"] / 90)
        y = (y + m) % (H + 2 * m) - m
        x = (a["x"] + a["wind"] * (t - 15.5) + m) % (W + 2 * m) - m
        grain(f, sheet, x, y, a["L"], a["ang"] + a["spin"] * (t - 15.5), a["seed"])
    for a in lands:
        p = lander_at(a, t)
        if p is None:
            continue
        x, y, up, ang = p
        if a["y"] + s > -40 and a["y"] + s < H + 40:
            contact(f, x + 0.25 * up, a["y"] + s + 0.28 * a["L"], a["L"] * 0.62 * (1 + up / 120), a["L"] * 0.2, 0.42 * np.exp(-up / (1.6 * a["L"])))
        grain(f, sheet, x, y + s, a["L"], ang, a["seed"])


def shot_k2():
    t_s = time.time()
    # the bank: k1_wide, his hand and the twig in it, the same man the shot before left
    sheet1, D, coarse, water_lines, ink, twig, fill = k1_parts(0.67, hands=True, marks=shan_on_bank)
    bank = develop(sheet1, D, coarse, line=1.5 * water_lines + np.minimum(ink + twig, 2.7))
    # the hills
    P = k2_parts(0.67)
    sheet = P["sheet"]
    one = np.ones((H, W), np.float32)
    k_g = one
    for g, mist in zip(P["people"], K2_MIST):
        k_g = lift_figure(sheet, g, k_g, k_g, edge=0.0, rim_amount=0.14, mist=k2_mist(sheet, mist))[0]
    keep = tidy_keep(P["fa"] + P["fb"], P["ca"] + P["cb"]) * k_g
    cut = np.maximum.reduce([g.carve()[0] for g in P["people"]])
    guard = sm((cut - 0.35) / 0.35)
    far = Arrival(sheet, P["fa"], P["ca"], guard, P["Ra"], (150, 610, 960, 1160, 1530), T_SEEP0, 61)
    nearer = Arrival(sheet, P["fb"], P["cb"], guard, P["Rb"], (70, 300, 760, 1240, 1760), T_SEEP0 + 0.25, 62)
    first = np.minimum(far.A, nearer.A)
    rs, lands = rain(P["heads"]), landers()
    still = None
    setup = time.time() - t_s
    print(f"setup {setup:.0f}s", flush=True)
    # its own clock is the old one, where the shot started at 12.0: everything in it keeps its timing
    shift = int(round((ink_subs.at("而天雨粟") - 12.0) * FPS))
    for n in range(360, 510):
        t = n / FPS
        if not wanted(n + shift):
            continue
        t0 = time.time()
        s = tilt(t)
        if s >= 1 or t > T_TILT1:
            if t >= T_RIPE and still is not None:
                hills = still
            else:
                fa, ca = far.at(t)
                fb, cb = nearer.at(t)
                # no line stands on the bare paper waiting for the ink: it comes when the ink does
                hills = develop(sheet, (fa + fb) * keep, (ca + cb) * keep, line=P["lines"] * ease((t - first - 0.08) / 0.6))
                if t >= T_RIPE:
                    still = hills
        if s <= 0:
            f = np.float32(bank).copy()
        elif s >= S_MAX:
            f = np.float32(hills).copy()
        else:
            s_i = int(round(s))
            rows = np.arange(H) - s_i      # the bank's row for each row of the frame; negative: above the bank
            f = np.zeros((H, W, 3), np.float32)
            on_bank = rows >= 0
            f[on_bank] = bank[rows[on_bank]]
            on_hills = rows + S_MAX < H
            w2 = (1 - ease(rows / S_BLEND))[..., None, None]
            f[on_hills] = (hills[rows[on_hills] + S_MAX] * w2[on_hills] + f[on_hills] * (1 - w2[on_hills]))
        sum_ = np.zeros((H, W, 3), np.float32)
        for dt in (-0.17, 0.0, 0.17):      # the shutter: a grain moves too far in a frame to be drawn sharp
            g = f.copy()
            draw_grain(g, sheet, rs, lands, t + dt / FPS, s)
            sum_ += g
        put(n + shift, sum_ / 3, t0, RAW4)
    report("k2", setup)


# ---------------------------------------------------------------- 17-19.2 and 23-27: the near ghost

def breath(t, a, b, much=0.22):
    """The wander of the outline turns a little and comes back, so the figure is the key frame again at each end."""
    return much * np.sin(np.pi * np.clip((t - a) / (b - a), 0, 1))


class Near:
    def __init__(self):
        self.sheet = sheet = Sheet(seed=91)
        self.fine, self.coarse = mend_plate(sheet, *shroud_plate(sheet, 0.67))
        look, cry = near_ghost(0.22), near_ghost(1.0)
        # what the plate left bare is paper in both those poses. In any other pose that paper may show where it should be ink: closed over from round it
        self.hole = sm((np.minimum(look.reserve(), cry.reserve()) - 0.04) / 0.2)
        # and a dark line stands along the edge of that bare paper, where the water stopped: it belongs to the figure's edge at 0.22 and is a scar on the wash while the head hangs lower
        self.scar = sm((gaussian_filter(self.hole, 9.0) - 0.06) / 0.25)
        have = gaussian_filter(1 - self.scar, 26.0) + 1e-3
        self.fill_f = gaussian_filter(self.fine * (1 - self.scar), 26.0) / have
        self.fill_c = gaussian_filter(self.coarse * (1 - self.scar), 26.0) / have
        self.layers = mouth_layers(sheet, cry.mouth(), 0.67, warp=0.3, soften=1.6, seep=5.0)
        self.c1, self.ax1 = cry.mouth_place()

    def mouth_now(self, g, sx, sy):
        """The mouth of the howl, painted once, as far open as this and where the head is now."""
        c, ax = g.mouth_place()
        U1, V1, U, V = self.ax1, np.array((-self.ax1[1], self.ax1[0])), ax, np.array((-ax[1], ax[0]))
        a = U1[0] * U[0] / sy + V1[0] * V[0] / sx
        b = U1[0] * U[1] / sy + V1[0] * V[1] / sx
        d = U1[1] * U[0] / sy + V1[1] * V[0] / sx
        e = U1[1] * U[1] / sy + V1[1] * V[1] / sx
        tx = (a, b, self.c1[0] - a * c[0] - b * c[1], d, e, self.c1[1] - d * c[0] - e * c[1])
        return [np.maximum(affine(x, tx), 0) for x in self.layers]

    def frame(self, lift, drift=0.0, jaw=0.0):
        sheet = self.sheet
        g = Ghost(lift, scale=1.3, at=(1100, 1010), to=(1060, 1120), form="shroud", drift=drift)
        kept = g.carve()
        g.carve = lambda seed=9: kept
        poke = np.maximum(self.hole, self.scar * (1 - ease(lift / 0.22))) * (1 - kept[0])
        fine = self.fine * (1 - poke) + self.fill_f * poke
        coarse = self.coarse * (1 - poke) + self.fill_c * poke
        fine, coarse = lift_figure(sheet, g, fine, coarse)
        fine, coarse = tidy(fine, coarse)
        o = (lift - 0.22) / 0.78
        if o >= 1 - 1e-6 and not drift and not jaw:
            inner, mf, mc, black = self.layers
        elif o > 0.24:
            # the head has gone back before it opens, and then the jaw goes first and the width follows; a small round dark place on the head would be an eye
            inner, mf, mc, black = self.mouth_now(g, max(ease((o - 0.30) / 0.70) ** 1.1, 0.03) * (1 + jaw), max(ease((o - 0.24) / 0.70) ** 0.75, 0.05) * (1 + 0.6 * jaw))
            k = ease((o - 0.24) / 0.16)
            inner, mf, mc, black = inner * k, mf * k, mc * k, black * k
        else:
            inner = None
        if inner is not None:
            fine, coarse = fine * (1 - 0.9 * inner) + mf + black, coarse * (1 - 0.9 * inner) + mc
        return develop(sheet, fine, coarse, qing=qing_wash(sheet, g, 0))


MOUTH_AT = 0.22 + 0.78 * 0.24      # the lift where Near.frame's mouth starts to show


def shot_near():
    """One shot, one plate. The head comes up until the mouth is about to show; half a second either side of 鬼夜哭 the mouth opens; then it holds the howl, shaking very slightly in the head and the jaw, and is the key frame k4_cry again as the camera starts back."""
    t_s = time.time()
    near = Near()
    setup = time.time() - t_s
    a = ink_subs.at("而天雨粟") + 5.0
    o0, o1 = ink_subs.at("鬼夜哭") - 0.5, ink_subs.at("鬼夜哭") + 0.5
    b = ink_subs.at("鬼何以哭") - 3.0      # the pull back takes three seconds, and starts here as it always did
    for n in range(int(round(a * FPS)), int(round(b * FPS))):
        t = n / FPS
        if not wanted(n):
            continue
        t0 = time.time()
        if t < o1:
            lift = MOUTH_AT * ease((t - a - 0.15) / (o0 - a - 0.15)) if t < o0 else MOUTH_AT + (1 - MOUTH_AT) * ease((t - o0) / (o1 - o0))
            drift, jaw = breath(t, a, o0), 0.0
        else:
            # three rates that never line up, swelling and sinking, so it is a throat and not a motor
            on = ease((t - o1) / 0.35) * (1 - ease((t - b + 0.40) / 0.35)) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.9 * t))
            v = 0.55 * np.sin(2 * np.pi * 5.1 * t + 0.7) + 0.30 * np.sin(2 * np.pi * 7.9 * t + 2.1) + 0.15 * np.sin(2 * np.pi * 12.3 * t + 4.0)
            u = 0.55 * np.sin(2 * np.pi * 4.3 * t + 1.9) + 0.30 * np.sin(2 * np.pi * 6.7 * t + 0.3) + 0.15 * np.sin(2 * np.pi * 11.1 * t + 5.0)
            lift = 1 - on * 0.022 * (1 + v)      # two or three film pixels at the top of the head
            jaw = float(on * 0.08 * u)
            drift = breath(t, o1, b, 0.10)
        put(n, near.frame(float(lift), drift, jaw), t0, RAW4)
    report("near", setup)


# ---------------------------------------------------------------- 19.2-23.0: what it sees

BEATS = (((6, 4), (5, 3), (10, 0)), ((7, 4), (14, 6), (4, 3), (11, 0)))      # for each length of each stroke: frames it takes to grow, frames it then stands still


def shot_vision():
    t_s = time.time()
    sheet = Sheet(seed=93)
    m = Mud()
    m.char()

    def stage():
        s, sh, cav, broad = relief_film(m, CAM_K3)
        relief, lift = ink_of(s, sh, cav)
        return s, relief + np.clip(0.16 - broad * 2.4, 0, 0.5), lift

    s0, rel0, lift0 = stage()
    m.grown_stroke(0)
    s1, rel1, lift1 = stage()
    m.grown_stroke(1)
    _, rel2, lift2 = stage()
    fine, coarse, env = mud_wash(sheet, CAM_K3, "k3", (860, 500), (860, 460))
    present = sm((env - 0.24 + 0.14 * noise(sheet, 3.2, 81)) / 0.42)
    D_wash = fine + coarse
    ink, keep = painted_twig(sheet, STICK_DOWN)
    under = gaussian_filter(twig_shadow(STICK_DOWN["path"], STICK_DOWN["radius"], [1, 3, 3, 4, 3], light=(0.30, 0.78)), 3.0)
    old, strokes, fields = vision_fields(sheet, CAM_K3)
    new_all = np.maximum.reduce([f[0] for f in fields])
    near_max = gaussian_filter(np.maximum(old, new_all), 12).max()
    hairs = [vision_hair(sheet, k, *fields[k]) for k in range(2)]
    # how far along each stroke every pixel of the frame is: the lip of a cut reaches well past the mask
    yy, xx = _grid()
    along = []
    for k, (mask, U, q, r) in enumerate(fields):
        _, idx = cKDTree(q).query(np.stack([xx.ravel(), yy.ravel()], 1), workers=4)
        along.append(np.float32(strokes[k]["u"][idx].reshape(H, W)))
    glint = sm((s0 - 0.32) / 0.2) * gaussian_filter(old, 2) * sm((noise(sheet, 2.6, 96) + 0.2) / 0.8)
    wash = Wash(sheet, s=0.67)
    damp = gaussian_filter(old, 26)
    damp = damp / damp.max()
    front = [0.0, 0.0]
    wet_state = [0.0]

    def story():
        wash.load(lumps(sheet, damp * 1.5, 91, level=0.45, soft=0.25, amount=0.55), 0.30)
        wash.run(20, evap=1e-4)
        wash.load(old, 0.46, fine=2.1, coarse=0.5)
        # it goes wet at once: most of the running in the first frames
        runs = (100, 70, 55, 45, 35, 30, 25, 20, 20, 20)
        for i, n in enumerate(runs):
            wash.run(n, evap=1.2e-4)
            wet_state[0] = ease((i + 1) / 5)
            yield
        wash.load(old, 0.12, fine=0.6, coarse=0.1)
        for _ in range(2):
            wash.run(30, evap=2e-4)
            yield
        for k, (st, (mask, U, q, r)) in enumerate(zip(strokes, fields)):
            n = len(q)
            if k:
                for _ in range(8):      # the brush stops: nothing grows, the last of it goes on bleeding
                    wash.run(15, evap=5e-4)
                    yield
            mask = mask * (0.86 + 0.14 * np.clip(noise(sheet, 3.0, 110 + k), -1.5, 1.5))
            wet = mask * (1 - sm((U - 0.84) / 0.13))
            stops = st["stops"]
            beats = [0.0] + [at for at, _, _ in stops] + [1.0]
            for i, (a, b) in enumerate(zip(beats[:-1], beats[1:])):
                nf, pause = BEATS[k][i]
                cuts = np.linspace(a, b, nf + 1)
                run = max(int(round((16 + 90 * (b - a)) / nf)), 2)
                for j in range(nf):
                    wash.load(wet * np.clip((U - cuts[j]) * n / 3.0 + 0.5, 0, 1) * np.clip((cuts[j + 1] - U) * n / 3.0 + 0.5, 0, 1), 0.24, fine=1.55, coarse=0.26)
                    if i == 0 and j == 0:
                        wash.load(mask * np.exp(-(U * n / 26) ** 2), 0.05, fine=1.0, coarse=0.16)
                    wash.run(run, evap=2.4e-4)
                    front[k] = float(cuts[j + 1])
                    yield
                if i < len(stops):
                    at, much, size = stops[i]
                    wash.load(mask * np.exp(-((U - at) * n / (15 * size)) ** 2), 0.05, fine=0.85 * much, coarse=0.12 * much)
                    for _ in range(pause):
                        wash.run(6, evap=2.4e-4)
                        yield
        for i in range(17):      # the stroke has stopped where it broke: a beat, the last of the wet still going into the paper, slower and slower. Then the ghost
            wash.run(max(100 - 7 * i, 8), evap=7e-4)
            yield

    setup = time.time() - t_s
    print(f"setup {setup:.0f}s", flush=True)
    n = 576
    t0 = time.time()
    for _ in story():
        if wanted(n):
            f_, c_ = wash.density()
            extra = f_ + c_
            new = np.zeros((H, W), np.float32)
            rel, lift = rel0, lift0
            for k, (mask, U, q, r) in enumerate(fields):
                if front[k] <= 0:
                    continue
                L = len(q)
                rev = np.clip((front[k] - along[k]) * L / 10.0 + 0.5, 0, 1)
                new = np.maximum(new, mask * rev)
                extra = extra + hairs[k] * np.clip((front[k] - U) * L / 3.0 + 0.5, 0, 1)
                base_r, next_r = (rel0, rel1) if k == 0 else (rel1, rel2)
                base_l, next_l = (lift0, lift1) if k == 0 else (lift1, lift2)
                rel = rel + (next_r - base_r) * rev
                lift = lift + (next_l - base_l) * rev
            extra = extra * (1 - 0.75 * glint * wet_state[0])
            near = np.clip(gaussian_filter(np.maximum(old * wet_state[0], new), 12) / near_max * 2.0, 0, 1)
            relief = rel * (1 - near) + gaussian_filter(rel, 2.5) * near
            lift = lift * (1 - 0.7 * near)
            D = 0.55 * D_wash * (1 - 0.6 * lift) + relief * present + extra
            D = D * (1 - 0.5 * keep) + 0.28 * under * (0.4 + 0.6 * present) * (1 - keep) + ink
            put(n, develop(sheet, D, np.zeros_like(D), grain=0), t0)
        else:
            _times.append((n, time.time() - t0))
        n += 1
        t0 = time.time()
    assert n == 690, n
    report("vision", setup)


# ---------------------------------------------------------------- 27-30: away from it, and the others answer

Z0 = 1.3 / 0.36      # the near figure against the middle one of the ridge
C_FIX = np.array(((977.0 * Z0 - 1060) / (Z0 - 1), (609.69 * Z0 - 1120) / (Z0 - 1)))       # the one point of the frame that does not move
T_PULL0, T_PULL1 = 27.0, 29.5


def rises(R, x, tall, t, a, dur, seed, sheet):
    """Going up from where it stands on the ridge, on a wandering edge: 1 where the thinning has got to by now."""
    yy, _ = _grid()
    u = ease((t - a) / dur)
    y = R[int(np.clip(x, 0, W - 1))] + 30 - (tall + 60) * u
    return sm((yy - y + 30 * noise(sheet, 3.0, seed)) / 50.0) * (u > 0)


class Nearer:
    """A film-size plate looked at from closer than it was painted. Not the finished picture blown up: the ink is taken out of the plate as ink, its edges tightened in proportion to how far in we are and given back the last of their break by fibres fixed to the sheet at twice the film's resolution; the paper's tooth is sampled the same way; and only then is it put on the paper, at the frame's own pixels. At z 1 it is exactly the plate."""

    def __init__(self, sheet, seed=240):
        self.sheet = sheet
        self.dx, self.dy = band(2 * H, 2 * W, 0.8, 7.0, seed), band(2 * H, 2 * W, 0.8, 7.0, seed + 1)
        self.tooth = band(2 * H, 2 * W, 0.6, 3.0, seed + 2)
        self.rgb = [np.float32(sheet.rgb[..., i]) for i in range(3)]

    def __call__(self, fine, coarse, line, box, z):
        if z <= 1 + 1e-6:
            return develop(self.sheet, fine, coarse, line=line)
        up = lambda a: np.asarray(Image.fromarray(np.float32(a), "F").resize((W, H), Image.BICUBIC, box=box), np.float32)
        yy, xx = _grid()
        cx, cy = (box[0] + xx / z) * 2, (box[1] + yy / z) * 2      # where each pixel of the frame lies on the fine fields
        fixed = lambda a: map_coordinates(a, [np.clip(cy, 0, 2 * H - 1), np.clip(cx, 0, 2 * W - 1)], order=1)
        much = 1 - 1 / z
        dx, dy = fixed(self.dx) * z * much * 0.75, fixed(self.dy) * z * much * 0.75
        v = up(self.sheet.valley) + 0.8 * much * fixed(self.tooth)
        at = [yy + dy, xx + dx]
        sharp = lambda a, k: np.maximum(a + k * (a - gaussian_filter(a, 0.8 * z)), 0)
        pull = lambda a, k: sharp(map_coordinates(up(a), at, order=1, mode="nearest"), k * much)
        rgb = np.stack([up(c) for c in self.rgb], -1)
        return develop(SimpleNamespace(valley=v, rgb=rgb), pull(fine, 1.5), pull(coarse, 1.5), line=pull(line, 2.2) * (1 + 0.22 * much * np.clip(fixed(self.tooth), -2, 2)))


def shot_pull():
    t_s = time.time()
    P = k4_parts(0.67)
    sheet, R = P["sheet"], P["R"]
    mid, left = P["lifted"]
    yy, xx = _grid()
    inner, mf, mc, black = mouth_layers(sheet, np.maximum.reduce([g.mouth() for g in P["ghosts"]]), 0.67, warp=0.25, soften=0.8)
    on_left = sm((700 - xx) / 40)
    # the final picture settles what stays out on the bare paper
    full = P["fc"] * mid * left
    for k in P["traces"]:
        full = full * k
    tk = tidy_keep((full * (1 - 0.9 * inner) + mf + black + P["over_f"]) * P["floor"], (P["cc"] * mid * left * np.prod(P["traces"], axis=0) * (1 - 0.9 * inner) + mc + P["over_c"]) * P["floor"])
    near_key = Image.open(os.path.join(OUT, "k4_cry.png")).convert("RGB")
    near_half = near_key.resize((W // 2, H // 2), Image.LANCZOS)      # below a half, the key frame is sampled from this: no stair-steps as it gets small
    nearer = Nearer(sheet)
    paper_wander = gaussian_filter(noise(sheet, 3.2, 230), 60)      # slow: it is not to read as clouds of noise
    paper_wander = paper_wander / paper_wander.std()
    # who answers and when: x, how tall, start. The ones off the frame at the start wait until they are in it
    answer = [(5, 1266, 88, 28.05), (3, 1168, 128, 27.35), (4, 1216, 62, 27.6), (2, 676, 58, 27.85), (1, 240, 104, 28.65), (0, 196, 64, 28.9)]
    setup = time.time() - t_s
    print(f"setup {setup:.0f}s", flush=True)
    for n in range(810, 900):
        t = n / FPS
        if not wanted(n):
            continue
        t0 = time.time()
        fine, coarse = P["fc"] * mid, P["cc"] * mid
        g_left = rises(R, 438, 300, t, 28.25, 0.95, 201, sheet)
        k = 1 - (1 - left) * g_left
        fine, coarse = fine * k, coarse * k
        for i, x, tall, a in answer:
            if t > a:
                k = 1 - (1 - P["traces"][i]) * rises(R, x, tall, t, a, 0.8, 210 + i, sheet)
                fine, coarse = fine * k, coarse * k
        m_left = on_left * ease((t - 29.0) / 0.45)
        vis = 1 - on_left + m_left
        fine = fine * (1 - 0.9 * inner * vis) + (mf + black) * vis
        coarse = coarse * (1 - 0.9 * inner * vis) + mc * vis
        fine, coarse = (fine + P["over_f"]) * P["floor"] * tk, (coarse + P["over_c"]) * P["floor"] * tk
        # an even ratio: the same fraction smaller every frame
        z = Z0 * (1 / Z0) ** np.clip((t - T_PULL0) / (T_PULL1 - T_PULL0), 0, 1)
        p = np.log(Z0 / z) / np.log(Z0)
        cx, cy = C_FIX
        box = (cx - cx / z, cy - cy / z, cx + (W - cx) / z, cy + (H - cy) / z)
        f_w = np.float32(np.clip(nearer(fine, coarse, P["lines"], box, z), 0, 255))
        if p <= 0:
            f = np.float32(near_key)
        elif p >= 0.75:
            f = f_w
        else:
            # The near frame has bare paper all round, so as it gets smaller it is on paper, with no edge. The far ink does not come in everywhere at once.
            k_ = z / Z0
            if k_ > 0.6:
                f_n = np.float32(near_key.transform((W, H), Image.AFFINE, (1 / k_, 0, cx - cx / k_, 0, 1 / k_, cy - cy / k_), Image.BICUBIC))
            else:
                k2_ = k_ * 2
                f_n = np.float32(near_half.transform((W, H), Image.AFFINE, (1 / k2_, 0, (cx - cx / k_) / 2, 0, 1 / k2_, (cy - cy / k_) / 2), Image.BICUBIC))
            x0, x1, y0, y1 = cx - cx * k_, cx + (W - cx) * k_, cy - cy * k_, cy + (H - cy) * k_
            e = ease(np.minimum.reduce([xx - x0, x1 - xx, yy - y0, y1 - yy]) / 70.0)[..., None]
            f_n = sheet.rgb * (1 - e) + f_n * e
            M = (1 - ease((p - 0.03 + 0.05 * paper_wander) / 0.55))[..., None]
            # through density, so that where ink lies over ink it stays ink
            f = np.exp(np.log(np.maximum(f_n, 1.0)) * M + np.log(np.maximum(f_w, 1.0)) * (1 - M))
        put(n, f, t0)
    report("pull", setup)


# ---------------------------------------------------------------- the small frames

def source(n):
    """Up to the pull back, this version's frames. From there on, the same shots as before, the pull and the painting coming back, one run in seq_raw/ from its frame 810."""
    pull = int(round((ink_subs.at("鬼何以哭") - 3.0) * FPS))
    if n >= int(round(ink_subs.at("或曰：/文字") * FPS)):      # the slip and the ghost caught by its own character: film_myth_slip.py, film_myth_gui.py
        return os.path.join(MYTH, "seq_raw7", f"f{n:05d}.png")
    return os.path.join(RAW4, f"f{n:05d}.png") if n < pull else os.path.join(RAW, f"f{n - pull + 810:05d}.png")


def burn(first=150, last=None, encode=False, name="myth_low5"):
    import subprocess
    last = int(round(ink_subs.at("或曰") * FPS)) if last is None else last      # through 鬼何以哭
    os.makedirs(SEQ, exist_ok=True)
    t0 = time.time()
    missing = []
    prev_path, prev = None, None
    for n in range(first, last):
        path = source(n)
        if not os.path.exists(path):
            missing.append(n)
            continue
        if path != prev_path:
            prev, prev_path = np.asarray(Image.open(path).convert("RGB"), np.uint8), path
        f = ink_subs.apply(prev, n / FPS)
        Image.fromarray(f).resize((960, 540), Image.LANCZOS).save(os.path.join(SEQ, f"f{n:05d}.png"), compress_level=1)
    print(f"burnt {last - first - len(missing)} frames in {time.time() - t0:.0f}s; missing {missing[:12]}{'...' if len(missing) > 12 else ''}", flush=True)
    if encode and not missing:
        out = os.path.join(MYTH, name + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", str(first), "-i", os.path.join(SEQ, "f%05d.png"), "-frames:v", str(last - first), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", "-movflags", "+faststart", out], check=True)
        print(out, flush=True)
        # frames even in time, each with its second on it
        cols, rows, w_, h_ = 6, 5, 400, 225
        sheet_ = Image.new("RGB", (cols * w_, rows * h_), (255, 255, 255))
        for i, n in enumerate(np.linspace(first, last - 1, cols * rows).astype(int)):
            im = Image.open(os.path.join(SEQ, f"f{n:05d}.png")).convert("RGB").resize((w_, h_), Image.LANCZOS)
            ImageDraw.Draw(im).text((6, 4), f"{n / FPS:.1f}", fill=(150, 30, 20))
            sheet_.paste(im, ((i % cols) * w_, (i // cols) * h_))
        path = os.path.join(MYTH, name + "_sheet.png")
        sheet_.save(path)
        print(path, flush=True)


SHOTS = dict(k2=shot_k2, near=shot_near, pull=shot_pull)

if __name__ == "__main__":
    what = sys.argv[1]
    if "--only" in sys.argv:
        ONLY = set(int(v) for v in sys.argv[sys.argv.index("--only") + 1].split(","))
    if what == "burn":
        burn(encode="mp4" in sys.argv)
    elif what == "burn7":      # from 鬼何以哭 to just before 依我之见
        burn(int(round(ink_subs.at("鬼何以哭") * FPS)), int(round(ink_subs.at("依我之见") * FPS)), encode="mp4" in sys.argv, name="myth_low7")
    else:
        SHOTS[what]()
